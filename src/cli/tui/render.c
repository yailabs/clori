/* Project the typed client state into one transcript, one composer, and transient overlays. */
#define _XOPEN_SOURCE 700

#include "src/cli/tui/private.h"

#include <operator/registry.h>

#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <wchar.h>

typedef struct {
    char *output;
    size_t capacity, count;
    unsigned int columns, column;
    int color, unicode, failed;
    const char *reset, *strong, *accent, *violet, *dim;
    const char *success, *warning, *error, *selected;
} tui_frame;

static unsigned int text_width(const char *text, size_t count);

static void frame_raw(tui_frame *frame, const char *bytes, size_t count)
{
    if (frame->failed || count >= frame->capacity - frame->count) {
        frame->failed = 1;
        return;
    }
    memcpy(frame->output + frame->count, bytes, count);
    frame->count += count;
    frame->output[frame->count] = '\0';
}

static void frame_format(tui_frame *frame, const char *format, ...)
{
    char text[1024];
    va_list arguments;
    int count;
    va_start(arguments, format);
    count = vsnprintf(text, sizeof(text), format, arguments);
    va_end(arguments);
    if (count < 0) {
        frame->failed = 1;
        return;
    }
    if ((size_t)count >= sizeof(text)) count = (int)sizeof(text) - 1;
    frame_raw(frame, text, (size_t)count);
}

static void frame_style(tui_frame *frame, const char *style)
{
    if (frame->color && style && style[0]) frame_raw(frame, style, strlen(style));
}

static size_t codepoint(const char *text, size_t count, int *width)
{
    mbstate_t state;
    wchar_t value;
    size_t result;
    memset(&state, 0, sizeof(state));
    result = mbrtowc(&value, text, count, &state);
    if (result == (size_t)-1 || result == (size_t)-2 || !result) {
        *width = 1;
        return 1u;
    }
    *width = wcwidth(value);
    if (*width < 0) *width = 1;
    return result;
}

static void frame_text_n(tui_frame *frame, const char *text, size_t count,
                         unsigned int limit)
{
    size_t cursor = 0u;
    while (cursor < count && frame->column < limit) {
        unsigned char byte = (unsigned char)text[cursor];
        size_t extent = 1u;
        int width = 1;
        if (byte == '\n' || byte == '\r') break;
        if (byte == '\t') {
            unsigned int spaces = 4u - frame->column % 4u;
            while (spaces-- && frame->column < limit) {
                frame_raw(frame, " ", 1u);
                frame->column++;
            }
            cursor++;
            continue;
        }
        if (byte < 0x20u || byte == 0x7fu || byte == 0x1bu) {
            frame_raw(frame, "?", 1u);
            frame->column++;
            cursor++;
            continue;
        }
        if (byte >= 0x80u) extent = codepoint(text + cursor, count - cursor, &width);
        if (frame->column + (unsigned int)width > limit) break;
        frame_raw(frame, text + cursor, extent);
        frame->column += (unsigned int)width;
        cursor += extent;
    }
}

static void frame_text(tui_frame *frame, const char *text, unsigned int limit)
{
    if (text) frame_text_n(frame, text, strlen(text), limit);
}

static void frame_begin_line(tui_frame *frame, unsigned int row)
{
    frame_format(frame, "\033[0m\033[%u;1H\033[2K", row);
    frame->column = 0u;
}

static void frame_to_column(tui_frame *frame, unsigned int column,
                            unsigned int limit)
{
    while (frame->column < column && frame->column < limit) {
        frame_raw(frame, " ", 1u);
        frame->column++;
    }
}

static void frame_rule_until(tui_frame *frame, unsigned int end)
{
    const char *rule = frame->unicode ? "─" : "-";
    frame_style(frame, frame->dim);
    while (frame->column < end && frame->column + 1u < frame->columns) {
        frame_raw(frame, rule, strlen(rule));
        frame->column++;
    }
    frame_style(frame, frame->reset);
}

static unsigned int text_width(const char *text, size_t count)
{
    size_t cursor = 0u;
    unsigned int width = 0u;
    while (cursor < count) {
        int value = 1;
        size_t extent = (unsigned char)text[cursor] < 0x80u
                            ? 1u : codepoint(text + cursor, count - cursor, &value);
        if (text[cursor] == '\t') value = (int)(4u - width % 4u);
        if (text[cursor] == '\n' || text[cursor] == '\r') break;
        width += (unsigned int)(value > 0 ? value : 1);
        cursor += extent;
    }
    return width;
}

static const char *connection_name(yvex_tui_connection connection)
{
    if (connection == YVEX_TUI_CONNECTION_CONNECTED) return "connected";
    if (connection == YVEX_TUI_CONNECTION_INCOMPATIBLE) return "incompatible";
    if (connection == YVEX_TUI_CONNECTION_DISCONNECTED) return "disconnected";
    return "offline";
}

static const char *phase_name(yvex_client_generation_phase phase)
{
    switch (phase) {
    case YVEX_CLIENT_PHASE_IDLE: return "idle";
    case YVEX_CLIENT_PHASE_QUEUED: return "queued";
    case YVEX_CLIENT_PHASE_TOKENIZING: return "tokenizing";
    case YVEX_CLIENT_PHASE_PREFILL: return "prefill";
    case YVEX_CLIENT_PHASE_DECODE: return "decode";
    case YVEX_CLIENT_PHASE_COMPLETE: return "complete";
    case YVEX_CLIENT_PHASE_CANCELLED: return "cancelled";
    case YVEX_CLIENT_PHASE_FAILED: return "failed";
    case YVEX_CLIENT_PHASE_UNAVAILABLE: break;
    }
    return "unavailable";
}

static const yvex_server_engine_summary *active_engine(const yvex_tui_state *state)
{
    size_t index;
    if (!state->active_engine.alias[0] || !state->active_engine.generation) return NULL;
    for (index = 0u; index < state->engine_count; ++index)
        if (!strcmp(state->engines[index].alias, state->active_engine.alias) &&
            state->engines[index].generation == state->active_engine.generation)
            return &state->engines[index];
    return NULL;
}

static void format_bytes(char *output, size_t capacity,
                         unsigned long long bytes)
{
    static const char *const units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double value = (double)bytes;
    size_t unit = 0u;
    while (value >= 1024.0 && unit + 1u < sizeof(units) / sizeof(units[0])) {
        value /= 1024.0;
        unit++;
    }
    if (!unit)
        (void)snprintf(output, capacity, "%llu %s", bytes, units[unit]);
    else
        (void)snprintf(output, capacity, "%.2f %s", value, units[unit]);
}

static void render_top(tui_frame *frame, const yvex_tui_state *state)
{
    const yvex_server_engine_summary *engine = active_engine(state);
    unsigned int limit = frame->columns > 1u ? frame->columns - 1u : 1u;
    char right[384];
    unsigned int width;
    frame_begin_line(frame, 1u);
    frame_style(frame, frame->accent);
    frame_text(frame, "  >_ YVEX", limit);
    frame_style(frame, frame->reset);
    if (engine)
        (void)snprintf(right, sizeof(right), "%s  ·  %s", engine->alias,
                       state->active_session);
    else
        (void)snprintf(right, sizeof(right), "No model loaded  ·  %s",
                       state->active_session);
    width = text_width(right, strlen(right));
    if (width + frame->column + 3u < limit) {
        frame_to_column(frame, limit - width, limit);
        frame_style(frame, engine ? frame->dim : frame->warning);
        frame_text(frame, right, limit);
        frame_style(frame, frame->reset);
    }
}

static int transcript_activity(const yvex_tui_activity *activity)
{
    return activity->kind != YVEX_TUI_ACTIVITY_RUNTIME;
}

static const char *activity_marker(const tui_frame *frame,
                                   const yvex_tui_activity *activity)
{
    if (activity->kind == YVEX_TUI_ACTIVITY_USER) return frame->unicode ? "› " : "> ";
    if (activity->kind == YVEX_TUI_ACTIVITY_ERROR ||
        activity->channel == YVEX_CLIENT_STREAM_ERROR)
        return "! ";
    if (activity->channel == YVEX_CLIENT_STREAM_EXPLICIT_REASONING) return "· ";
    if (activity->channel == YVEX_CLIENT_STREAM_TOOL_CALL ||
        activity->channel == YVEX_CLIENT_STREAM_TOOL_RESULT)
        return frame->unicode ? "↳ " : "> ";
    return frame->unicode ? "● " : "* ";
}

static const char *activity_style(tui_frame *frame,
                                  const yvex_tui_activity *activity)
{
    if (activity->severity == YVEX_TUI_SEVERITY_ERROR) return frame->error;
    if (activity->severity == YVEX_TUI_SEVERITY_WARNING) return frame->warning;
    if (activity->kind == YVEX_TUI_ACTIVITY_USER) return frame->accent;
    if (activity->channel == YVEX_CLIENT_STREAM_EXPLICIT_REASONING) return frame->dim;
    return activity->kind == YVEX_TUI_ACTIVITY_GENERATION
               ? frame->strong : frame->dim;
}

static size_t line_extent(const char *text, size_t count,
                          unsigned int columns)
{
    size_t cursor = 0u;
    unsigned int width = 0u;
    while (cursor < count) {
        size_t extent = 1u;
        int value = 1;
        unsigned char byte = (unsigned char)text[cursor];
        if (byte == '\n' || byte == '\r') break;
        if (byte == '\t') value = (int)(4u - width % 4u);
        else if (byte >= 0x80u)
            extent = codepoint(text + cursor, count - cursor, &value);
        if (width + (unsigned int)value > columns) break;
        width += (unsigned int)value;
        cursor += extent;
    }
    return cursor;
}

static unsigned int activity_lines(const yvex_tui_activity *activity,
                                   unsigned int columns)
{
    const char *text = activity->text[0] ? activity->text : "…";
    size_t count = strlen(text), cursor = 0u;
    unsigned int lines = 0u;
    while (cursor < count) {
        size_t previous = cursor;
        cursor += line_extent(text + cursor, count - cursor, columns);
        if (cursor < count && text[cursor] == '\r') cursor++;
        if (cursor < count && text[cursor] == '\n') cursor++;
        if (cursor == previous) cursor++;
        lines++;
    }
    return lines ? lines : 1u;
}

static unsigned int render_activity(tui_frame *frame,
                                    const yvex_tui_activity *activity,
                                    unsigned int row, unsigned int last)
{
    const char *text = activity->text[0] ? activity->text : "…";
    size_t count = strlen(text), cursor = 0u;
    unsigned int line = 0u;
    unsigned int limit = frame->columns > 1u ? frame->columns - 1u : 1u;
    unsigned int wrap = limit > 6u ? limit - 6u : 1u;
    while (cursor < count && row <= last) {
        size_t previous = cursor;
        size_t extent = line_extent(text + cursor, count - cursor, wrap);
        frame_begin_line(frame, row++);
        frame_text(frame, "  ", limit);
        frame_style(frame, activity_style(frame, activity));
        frame_text(frame, line ? "  " : activity_marker(frame, activity), limit);
        frame_text_n(frame, text + cursor, extent, limit);
        frame_style(frame, frame->reset);
        cursor += extent;
        if (cursor < count && text[cursor] == '\r') cursor++;
        if (cursor < count && text[cursor] == '\n') cursor++;
        if (cursor == previous) cursor++;
        line++;
    }
    return row;
}

static unsigned int metrics_rows(const yvex_tui_state *state)
{
    const yvex_server_engine_summary *engine = active_engine(state);
    if (!state->last_turn.turn_available && !state->last_turn.profile_available &&
        !(engine && engine->prepared_bytes))
        return 0u;
    return state->last_turn.profile_available ? 3u : 2u;
}

static unsigned int render_metrics(tui_frame *frame, const yvex_tui_state *state,
                                   unsigned int row, unsigned int last)
{
    const yvex_tui_turn_observation *turn = &state->last_turn;
    const yvex_server_engine_summary *engine = active_engine(state);
    unsigned int limit = frame->columns > 1u ? frame->columns - 1u : 1u;
    char memory[64] = "";
    if (engine) format_bytes(memory, sizeof(memory), engine->prepared_bytes);
    if (row <= last) {
        frame_begin_line(frame, row++);
        frame_style(frame, frame->dim);
        frame_text(frame, "    ", limit);
        frame_format(frame, "%llu prompt · %llu generated · %.2f s total",
                     turn->prompt_tokens, turn->generated_tokens,
                     turn->total_seconds);
        if (engine && engine->prepared_bytes)
            frame_format(frame, " · %s prepared", memory);
    }
    if (row <= last) {
        frame_begin_line(frame, row++);
        frame_style(frame, frame->dim);
        frame_text(frame, "    ", limit);
        frame_format(frame, "%.2f tok/s prefill · %.2f tok/s decode",
                     turn->prefill_rate, turn->decode_rate);
    }
    if (turn->profile_available && row <= last) {
        char h2d[64];
        format_bytes(h2d, sizeof(h2d), turn->h2d_bytes);
        frame_begin_line(frame, row++);
        frame_style(frame, frame->dim);
        frame_text(frame, "    ", limit);
        frame_format(frame, "%llu kernels · %llu/%llu Tensor Core · %s H2D",
                     turn->kernel_launches, turn->tensor_core_launches,
                     turn->kernel_launches, h2d);
    }
    frame_style(frame, frame->reset);
    return row;
}

static size_t transcript_ordinal_at(const yvex_tui_state *state, size_t ordinal)
{
    size_t index, seen = 0u;
    for (index = 0u; index < state->activity_count; ++index) {
        size_t slot = (state->activity_start + index) % YVEX_TUI_ACTIVITY_CAP;
        if (!transcript_activity(&state->activities[slot])) continue;
        if (seen++ == ordinal) return slot;
    }
    return SIZE_MAX;
}

static size_t transcript_count(const yvex_tui_state *state)
{
    size_t count = 0u, index;
    for (index = 0u; index < state->activity_count; ++index) {
        size_t slot = (state->activity_start + index) % YVEX_TUI_ACTIVITY_CAP;
        if (transcript_activity(&state->activities[slot])) count++;
    }
    return count;
}

static void render_transcript(tui_frame *frame, const yvex_tui_state *state,
                              unsigned int first, unsigned int last)
{
    size_t total = transcript_count(state), end, start, ordinal;
    unsigned int available, used = 0u, metric_count;
    unsigned int row = first;
    if (last < first) return;
    available = last - first + 1u;
    end = total > state->activity_scroll ? total - state->activity_scroll : 0u;
    metric_count = state->activity_scroll ? 0u : metrics_rows(state);
    if (metric_count > available) metric_count = available;
    start = end;
    while (start) {
        size_t slot = transcript_ordinal_at(state, start - 1u);
        unsigned int lines;
        if (slot == SIZE_MAX) break;
        lines = activity_lines(&state->activities[slot],
                               frame->columns > 7u ? frame->columns - 7u : 1u);
        if (used + lines + metric_count > available) break;
        used += lines;
        start--;
    }
    if (!end && row <= last) {
        frame_begin_line(frame, row++);
        frame_style(frame, frame->strong);
        frame_text(frame, "  Ready to work", frame->columns - 1u);
        if (row <= last) {
            frame_begin_line(frame, row++);
            frame_style(frame, frame->dim);
            frame_text(frame, "  Type a request, / for commands, or Ctrl-O to choose a model.",
                       frame->columns - 1u);
        }
    }
    for (ordinal = start; ordinal < end && row <= last; ++ordinal) {
        size_t slot = transcript_ordinal_at(state, ordinal);
        if (slot != SIZE_MAX)
            row = render_activity(frame, &state->activities[slot], row, last);
    }
    if (!state->activity_scroll && row <= last)
        (void)render_metrics(frame, state, row, last);
}

static size_t slash_extent(const yvex_tui_state *state)
{
    const unsigned char *space = memchr(state->composer.bytes, ' ',
                                        state->composer.count);
    return space ? (size_t)(space - state->composer.bytes) : state->composer.count;
}

static int slash_match(const yvex_operator_descriptor *descriptor,
                       const yvex_tui_state *state)
{
    size_t count = slash_extent(state);
    return descriptor->slash_projection &&
           strcmp(descriptor->slash_projection, "none") &&
           strlen(descriptor->slash_projection) >= count &&
           !memcmp(descriptor->slash_projection, state->composer.bytes, count);
}

static void render_slash_overlay(tui_frame *frame, const yvex_tui_state *state,
                                 unsigned int first, unsigned int last)
{
    size_t index, matches = 0u, first_match = 0u, selected = state->slash_selected;
    unsigned int capacity = last >= first ? last - first + 1u : 0u;
    unsigned int rows = capacity > 9u ? 9u : capacity;
    unsigned int row = last + 1u > rows ? last + 1u - rows : first;
    unsigned int limit = frame->columns > 1u ? frame->columns - 1u : 1u;
    if (!rows) return;
    for (index = 0u; index < yvex_operator_descriptor_count; ++index)
        if (slash_match(&yvex_operator_descriptors[index], state)) matches++;
    if (matches > rows - 1u && selected >= rows - 1u)
        first_match = selected - rows + 2u;
    frame_begin_line(frame, row++);
    frame_style(frame, frame->violet);
    frame_text(frame, "  Commands", limit);
    frame_style(frame, frame->dim);
    frame_format(frame, "  %zu match%s · Tab completes", matches,
                 matches == 1u ? "" : "es");
    matches = 0u;
    for (index = 0u; index < yvex_operator_descriptor_count && row <= last; ++index) {
        const yvex_operator_descriptor *descriptor = &yvex_operator_descriptors[index];
        if (!slash_match(descriptor, state)) continue;
        if (matches++ < first_match) continue;
        frame_begin_line(frame, row++);
        frame_style(frame, matches - 1u == selected ? frame->selected : frame->accent);
        frame_text(frame, matches - 1u == selected ? "  › " : "    ", limit);
        frame_text(frame, descriptor->slash_projection, limit);
        frame_style(frame, frame->dim);
        frame_to_column(frame, limit > 48u ? 30u : 18u, limit);
        frame_text(frame, descriptor->summary, limit);
        frame_style(frame, frame->reset);
    }
}

static void render_overlay_header(tui_frame *frame, unsigned int row,
                                  const char *title, const char *hint)
{
    unsigned int limit = frame->columns > 1u ? frame->columns - 1u : 1u;
    frame_begin_line(frame, row);
    frame_style(frame, frame->violet);
    frame_text(frame, "  ", limit);
    frame_text(frame, title, limit);
    frame_style(frame, frame->dim);
    frame_text(frame, "  ", limit);
    frame_text(frame, hint, limit);
    frame_style(frame, frame->reset);
}

static void render_search(tui_frame *frame, unsigned int row, const char *query,
                          const char *placeholder, unsigned int *cursor_column)
{
    unsigned int limit = frame->columns > 1u ? frame->columns - 1u : 1u;
    frame_begin_line(frame, row);
    frame_style(frame, frame->accent);
    frame_text(frame, "  / ", limit);
    frame_style(frame, frame->reset);
    if (query[0]) frame_text(frame, query, limit);
    else {
        frame_style(frame, frame->dim);
        frame_text(frame, placeholder, limit);
        frame_style(frame, frame->reset);
    }
    *cursor_column = 5u + text_width(query, strlen(query));
    if (*cursor_column > limit) *cursor_column = limit;
}

static void render_model_overlay(tui_frame *frame, const yvex_tui_state *state,
                                 unsigned int first, unsigned int last,
                                 unsigned int *cursor_row,
                                 unsigned int *cursor_column)
{
    unsigned int row = first, limit = frame->columns - 1u;
    size_t visible, ordinal;
    render_overlay_header(frame, row++, "Select model",
                          "Enter loads · Tab profile · Ctrl-R remote · Esc closes");
    if (row <= last) {
        render_search(frame, row, state->model_search, "Search local models",
                      cursor_column);
        *cursor_row = row++;
    }
    if (row <= last) {
        frame_begin_line(frame, row++);
        frame_text(frame, "  ", limit);
        frame_rule_until(frame, limit > 3u ? limit - 2u : limit);
    }
    if (state->model_catalog_status == YVEX_TUI_MODEL_CATALOG_ERROR && row <= last) {
        frame_begin_line(frame, row);
        frame_style(frame, frame->error);
        frame_text(frame, "  Registry unavailable: ", limit);
        frame_text(frame, state->model_catalog_reason, limit);
        return;
    }
    visible = yvex_tui_model_visible_count(state);
    for (ordinal = state->model_viewport; ordinal < visible && row <= last;
         ++ordinal) {
        size_t index = yvex_tui_model_visible_at(state, ordinal);
        const yvex_tui_model_row *model;
        if (index == SIZE_MAX) continue;
        model = &state->models[index];
        frame_begin_line(frame, row++);
        frame_style(frame, index == state->selected_model
                               ? frame->selected : frame->reset);
        frame_text(frame, index == state->selected_model ? "  › " : "    ", limit);
        frame_text(frame, model->display_name, limit);
        frame_style(frame, model->startup_ready ? frame->success : frame->warning);
        frame_format(frame, "  %s", model->startup_ready ? "ready" : "blocked");
        frame_style(frame, frame->reset);
        if (index == state->selected_model && row <= last) {
            const yvex_model_runtime_profile_fact *profile =
                yvex_tui_launch_profile(state);
            frame_begin_line(frame, row++);
            frame_style(frame, state->focus == YVEX_TUI_FOCUS_MODEL_PROFILE
                                   ? frame->accent : frame->dim);
            frame_text(frame, "      profile  ", limit);
            frame_text(frame, profile ? profile->alias : "none", limit);
            if (profile && !profile->launchable && profile->blocker[0]) {
                frame_text(frame, "  ·  ", limit);
                frame_text(frame, profile->blocker, limit);
            }
            frame_style(frame, frame->reset);
        }
    }
    if (!visible && row <= last) {
        frame_begin_line(frame, row);
        frame_style(frame, frame->dim);
        frame_text(frame, "  No local model matches this search.", limit);
    }
}

static void render_session_overlay(tui_frame *frame, const yvex_tui_state *state,
                                   unsigned int first, unsigned int last,
                                   unsigned int *cursor_row,
                                   unsigned int *cursor_column)
{
    unsigned int row = first, limit = frame->columns - 1u;
    size_t visible, ordinal;
    render_overlay_header(frame, row++, "Attach session",
                          "Enter attaches · Esc closes");
    if (row <= last) {
        render_search(frame, row, state->session_search, "Search sessions",
                      cursor_column);
        *cursor_row = row++;
    }
    visible = yvex_tui_session_visible_count(state);
    for (ordinal = state->session_viewport; ordinal < visible && row <= last;
         ++ordinal) {
        size_t index = yvex_tui_session_visible_at(state, ordinal);
        const yvex_tui_session_row *session;
        if (index == SIZE_MAX) continue;
        session = &state->sessions[index];
        frame_begin_line(frame, row++);
        frame_style(frame, index == state->selected_session
                               ? frame->selected : frame->reset);
        frame_text(frame, index == state->selected_session ? "  › " : "    ", limit);
        frame_text(frame, session->name, limit);
        frame_style(frame, frame->dim);
        frame_format(frame, "  position %llu · %llu turns", session->position,
                     session->turns);
        frame_style(frame, frame->reset);
    }
    if (!visible && row <= last) {
        frame_begin_line(frame, row);
        frame_style(frame, frame->dim);
        frame_text(frame, "  No server session matches this search.", limit);
    }
}

static void render_remote_overlay(tui_frame *frame, const yvex_tui_state *state,
                                  unsigned int first, unsigned int last,
                                  unsigned int *cursor_row,
                                  unsigned int *cursor_column)
{
    unsigned int row = first, limit = frame->columns - 1u;
    size_t index;
    render_overlay_header(frame, row++, "Discover model",
                          "Enter searches · a acquires selected · Esc closes");
    if (row <= last) {
        render_search(frame, row, state->discover_query, "Search Hugging Face",
                      cursor_column);
        *cursor_row = row++;
    }
    if (state->remote_search_running && row <= last) {
        frame_begin_line(frame, row++);
        frame_style(frame, frame->warning);
        frame_text(frame, "  Searching remote catalog…", limit);
    }
    for (index = state->remote_viewport; index < state->remote_count && row <= last;
         ++index) {
        const yvex_remote_model *model = &state->remote_models[index];
        frame_begin_line(frame, row++);
        frame_style(frame, index == state->selected_remote
                               ? frame->selected : frame->reset);
        frame_text(frame, index == state->selected_remote ? "  › " : "    ", limit);
        frame_text(frame, model->repository, limit);
        frame_style(frame, frame->dim);
        if (model->family[0]) frame_format(frame, "  %s", model->family);
        frame_style(frame, frame->reset);
    }
    if (!state->remote_search_running && !state->remote_count &&
        state->remote_search_reason[0] && row <= last) {
        frame_begin_line(frame, row);
        frame_style(frame, frame->warning);
        frame_text(frame, "  ", limit);
        frame_text(frame, state->remote_search_reason, limit);
    }
}

static void render_help(tui_frame *frame, unsigned int first, unsigned int last)
{
    static const char *const lines[] = {
        "Keyboard shortcuts",
        "",
        "Enter              send or confirm",
        "Ctrl-J / Shift-Enter  insert newline",
        "Tab                complete command or move selector field",
        "Ctrl-O             choose model and runtime profile",
        "Ctrl-P or /        open slash commands in the composer",
        "PageUp / PageDown  scroll transcript",
        "Esc                close overlay or cancel active generation",
        "Ctrl-L             refresh typed runtime state",
        "Ctrl-D             exit",
    };
    unsigned int row = first, limit = frame->columns - 1u;
    size_t index;
    for (index = 0u; index < sizeof(lines) / sizeof(lines[0]) && row <= last;
         ++index, ++row) {
        frame_begin_line(frame, row);
        frame_style(frame, index == 0u ? frame->violet : frame->reset);
        frame_text(frame, "  ", limit);
        frame_text(frame, lines[index], limit);
    }
}

static size_t composer_line_count(const yvex_tui_composer *composer)
{
    size_t count = 1u, index;
    for (index = 0u; index < composer->count; ++index)
        if (composer->bytes[index] == '\n') count++;
    return count;
}

static size_t composer_cursor_line(const yvex_tui_composer *composer)
{
    size_t line = 0u, index;
    for (index = 0u; index < composer->cursor; ++index)
        if (composer->bytes[index] == '\n') line++;
    return line;
}

static void composer_bounds(const yvex_tui_composer *composer, size_t line,
                            size_t *begin, size_t *end)
{
    size_t index = 0u, current = 0u;
    while (index < composer->count && current < line)
        if (composer->bytes[index++] == '\n') current++;
    *begin = index;
    while (index < composer->count && composer->bytes[index] != '\n') index++;
    *end = index;
}

static unsigned int composer_rows(const yvex_tui_state *state)
{
    size_t count = composer_line_count(&state->composer);
    if (count > 8u) count = 8u;
    return (unsigned int)count;
}

static void render_composer(tui_frame *frame, const yvex_tui_state *state,
                            unsigned int first, unsigned int rows,
                            unsigned int *cursor_row,
                            unsigned int *cursor_column)
{
    const yvex_tui_composer *composer = &state->composer;
    size_t total = composer_line_count(composer);
    size_t active = composer_cursor_line(composer);
    size_t start = active >= rows ? active - rows + 1u : 0u;
    size_t line;
    unsigned int limit = frame->columns - 1u;
    if (start + rows > total && total > rows) start = total - rows;
    for (line = start; line < total && line < start + rows; ++line) {
        size_t begin, end, visible = 0u;
        unsigned int row = first + (unsigned int)(line - start);
        unsigned int prefix = 4u;
        composer_bounds(composer, line, &begin, &end);
        if (line == active && composer->cursor >= begin) {
            unsigned int available = limit > prefix ? limit - prefix : 1u;
            while (visible < composer->cursor - begin &&
                   text_width((const char *)composer->bytes + begin + visible,
                              composer->cursor - begin - visible) >= available - 1u)
                visible++;
        }
        frame_begin_line(frame, row);
        frame_style(frame, line == start ? frame->accent : frame->dim);
        frame_text(frame, line == start ? "  › " : "    ", limit);
        frame_style(frame, frame->reset);
        if (visible) frame_text(frame, frame->unicode ? "…" : "<", limit);
        frame_text_n(frame, (const char *)composer->bytes + begin + visible,
                     end - begin - visible, limit);
        if (!composer->count && line == 0u) {
            frame_style(frame, frame->dim);
            frame_text(frame, "Ask YVEX anything", limit);
            frame_style(frame, frame->reset);
        }
        if (line == active) {
            *cursor_row = row;
            *cursor_column = prefix +
                text_width((const char *)composer->bytes + begin + visible,
                           composer->cursor - begin - visible) + 1u;
            if (*cursor_column > limit) *cursor_column = limit;
        }
    }
}

static void render_composer_rule(tui_frame *frame, const yvex_tui_state *state,
                                 unsigned int row)
{
    char context[384];
    unsigned int limit = frame->columns - 1u, width;
    frame_begin_line(frame, row);
    frame_text(frame, "  ", limit);
    frame_rule_until(frame, limit);
    if (state->generation_active)
        (void)snprintf(context, sizeof(context), "%s · Esc to interrupt",
                       phase_name(state->generation_phase));
    else if (state->pending_count)
        (void)snprintf(context, sizeof(context), "%zu queued", state->pending_count);
    else
        (void)snprintf(context, sizeof(context), "%s", state->active_session);
    width = text_width(context, strlen(context));
    if (width + 2u < limit) {
        frame_begin_line(frame, row);
        frame_to_column(frame, limit - width, limit);
        frame_style(frame, state->generation_active ? frame->violet : frame->dim);
        frame_text(frame, context, limit);
        frame_style(frame, frame->reset);
    }
}

static void render_footer(tui_frame *frame, const yvex_tui_state *state,
                          unsigned int row)
{
    char left[384], right[192];
    unsigned int limit = frame->columns - 1u, width;
    const char *style;
    if (state->notice[0])
        (void)snprintf(left, sizeof(left), "%s", state->notice);
    else if (state->pending_review)
        (void)snprintf(left, sizeof(left), "queued draft restored for review");
    else if (state->console_available && state->console.context_capacity)
        (void)snprintf(left, sizeof(left), "%s · context %llu/%llu",
                       connection_name(state->connection), state->console.context_used,
                       state->console.context_capacity);
    else
        (void)snprintf(left, sizeof(left), "%s", connection_name(state->connection));
    (void)snprintf(right, sizeof(right), "? shortcuts · / commands");
    style = state->notice[0] && state->notice_severity == YVEX_TUI_SEVERITY_ERROR
                ? frame->error
            : state->notice[0] && state->notice_severity == YVEX_TUI_SEVERITY_WARNING
                ? frame->warning
            : state->connection == YVEX_TUI_CONNECTION_CONNECTED
                ? frame->success : frame->warning;
    frame_begin_line(frame, row);
    frame_text(frame, "  ", limit);
    frame_style(frame, style);
    frame_text(frame, left, limit);
    width = text_width(right, strlen(right));
    if (frame->column + width + 3u < limit) {
        frame_to_column(frame, limit - width, limit);
        frame_style(frame, frame->dim);
        frame_text(frame, right, limit);
    }
    frame_style(frame, frame->reset);
}

static void frame_styles(tui_frame *frame, int color)
{
    frame->color = color;
    frame->reset = color ? "\033[0m" : "";
    frame->strong = color ? "\033[1;38;5;255m" : "";
    frame->accent = color ? "\033[1;38;5;81m" : "";
    frame->violet = color ? "\033[1;38;5;141m" : "";
    frame->dim = color ? "\033[38;5;245m" : "";
    frame->success = color ? "\033[38;5;114m" : "";
    frame->warning = color ? "\033[38;5;221m" : "";
    frame->error = color ? "\033[38;5;203m" : "";
    frame->selected = color ? "\033[1;7m" : "";
}

static void render_overlay(tui_frame *frame, const yvex_tui_state *state,
                           unsigned int first, unsigned int last,
                           unsigned int *cursor_row,
                           unsigned int *cursor_column)
{
    if (state->overlay == YVEX_TUI_OVERLAY_SLASH)
        render_slash_overlay(frame, state, first, last);
    else if (state->overlay == YVEX_TUI_OVERLAY_MODEL)
        render_model_overlay(frame, state, first, last, cursor_row, cursor_column);
    else if (state->overlay == YVEX_TUI_OVERLAY_SESSION)
        render_session_overlay(frame, state, first, last, cursor_row, cursor_column);
    else if (state->overlay == YVEX_TUI_OVERLAY_REMOTE)
        render_remote_overlay(frame, state, first, last, cursor_row, cursor_column);
    else if (state->overlay == YVEX_TUI_OVERLAY_HELP)
        render_help(frame, first, last);
}

int yvex_tui_render(const yvex_tui_state *state, char *output, size_t capacity,
                    size_t *count, unsigned int *cursor_row,
                    unsigned int *cursor_column)
{
    tui_frame frame;
    unsigned int rows, composer_count, composer_first, rule_row, footer_row;
    unsigned int transcript_first = 3u, transcript_last;
    if (!state || !output || capacity < 128u || !count ||
        !cursor_row || !cursor_column)
        return YVEX_ERR_INVALID_ARG;
    memset(&frame, 0, sizeof(frame));
    frame.output = output;
    frame.capacity = capacity;
    frame.columns = state->terminal.columns ? state->terminal.columns : 80u;
    frame.unicode = state->terminal.unicode;
    frame_styles(&frame, state->terminal.color);
    {
        static const char begin[] = "\033[0m\033[?25l\033[H\033[2J";
        frame_raw(&frame, begin, sizeof(begin) - 1u);
    }
    rows = state->terminal.rows ? state->terminal.rows : 1u;
    composer_count = composer_rows(state);
    if (rows < composer_count + 3u) composer_count = rows > 3u ? rows - 3u : 1u;
    footer_row = rows;
    composer_first = rows > composer_count ? rows - composer_count : 1u;
    rule_row = composer_first > 1u ? composer_first - 1u : 1u;
    transcript_last = rule_row > transcript_first ? rule_row - 1u : transcript_first;
    render_top(&frame, state);
    if (rows > 1u) {
        frame_begin_line(&frame, 2u);
        frame_text(&frame, "  ", frame.columns - 1u);
        frame_rule_until(&frame, frame.columns - 1u);
    }
    if (rows >= 6u && frame.columns >= 20u)
        render_transcript(&frame, state, transcript_first, transcript_last);
    render_composer_rule(&frame, state, rule_row);
    render_composer(&frame, state, composer_first, composer_count,
                    cursor_row, cursor_column);
    if (rows > 1u) render_footer(&frame, state, footer_row);
    if (rows >= 6u && frame.columns >= 20u)
        render_overlay(&frame, state, transcript_first, transcript_last,
                       cursor_row, cursor_column);
    frame_style(&frame, frame.reset);
    frame_format(&frame, "\033[%u;%uH\033[?25h", *cursor_row, *cursor_column);
    if (frame.failed) return YVEX_ERR_BOUNDS;
    *count = frame.count;
    return YVEX_OK;
}
