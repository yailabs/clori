/* Project application state into one bounded ANSI frame without performing protocol work. */
#define _XOPEN_SOURCE 700

#include "src/cli/tui/private.h"

#include <operator/registry.h>

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
            unsigned int spaces = 4u - (frame->column % 4u);
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

static size_t text_line_bytes(const char *text, size_t count,
                              unsigned int maximum_columns)
{
    size_t cursor = 0u;
    unsigned int columns = 0u;
    while (cursor < count) {
        size_t extent = 1u;
        int width = 1;
        unsigned char byte = (unsigned char)text[cursor];
        if (byte == '\n' || byte == '\r') break;
        if (byte == '\t')
            width = (int)(4u - columns % 4u);
        else if (byte >= 0x80u)
            extent = codepoint(text + cursor, count - cursor, &width);
        if (columns + (unsigned int)width > maximum_columns) break;
        columns += (unsigned int)width;
        cursor += extent;
    }
    return cursor;
}

static void frame_begin_line(tui_frame *frame, unsigned int row)
{
    /* EL inherits the active background, so reset before clearing every row. */
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

static void frame_rule(tui_frame *frame, unsigned int row)
{
    const char *rule = frame->unicode ? "─" : "-";
    unsigned int index, limit = frame->columns > 1u ? frame->columns - 1u : 1u;
    frame_begin_line(frame, row);
    frame_style(frame, frame->dim);
    for (index = 0u; index < limit; ++index) frame_raw(frame, rule, strlen(rule));
    frame->column = limit;
    frame_style(frame, frame->reset);
}

static void frame_rule_until(tui_frame *frame, unsigned int end)
{
    const char *rule = frame->unicode ? "─" : "-";
    frame_style(frame, frame->dim);
    while (frame->column < end && frame->column < frame->columns - 1u) {
        frame_raw(frame, rule, strlen(rule));
        frame->column++;
    }
    frame_style(frame, frame->reset);
}

static const char *surface_name(yvex_tui_surface surface)
{
    if (surface == YVEX_TUI_SURFACE_MODELS ||
        surface == YVEX_TUI_SURFACE_MODEL_DETAIL)
        return "Models";
    if (surface == YVEX_TUI_SURFACE_SESSIONS) return "Sessions";
    if (surface == YVEX_TUI_SURFACE_RUNTIME) return "Runtime";
    return "Home";
}

static const yvex_server_engine_summary *active_engine(const yvex_tui_state *state);

static const char *backend_name(yvex_backend_kind backend)
{
    return backend == YVEX_BACKEND_KIND_CUDA ? "CUDA" : "CPU";
}

static const char *connection_name(yvex_tui_connection connection)
{
    if (connection == YVEX_TUI_CONNECTION_CONNECTED) return "ready";
    if (connection == YVEX_TUI_CONNECTION_INCOMPATIBLE) return "incompatible";
    if (connection == YVEX_TUI_CONNECTION_DISCONNECTED) return "disconnected";
    return "unavailable";
}

static const char *lifecycle_name(const yvex_tui_state *state)
{
    switch (state->runtime_lifecycle) {
    case YVEX_TUI_RUNTIME_LAUNCH_REQUESTED: return "launch requested";
    case YVEX_TUI_RUNTIME_LAUNCHING: return "starting";
    case YVEX_TUI_RUNTIME_WAITING_PROTOCOL: return "waiting for protocol";
    case YVEX_TUI_RUNTIME_ENGINE_LOADING: return "loading model";
    case YVEX_TUI_RUNTIME_CONNECTED_EXTERNAL: return "ready · external";
    case YVEX_TUI_RUNTIME_CONNECTED_OWNED: return "ready · launched here";
    case YVEX_TUI_RUNTIME_LAUNCH_FAILED: return "launch failed";
    case YVEX_TUI_RUNTIME_SHUTDOWN_REQUESTED: return "stopping";
    case YVEX_TUI_RUNTIME_STOPPED: return "stopped";
    case YVEX_TUI_RUNTIME_NONE: break;
    }
    return connection_name(state->connection);
}

static const char *launch_failure_name(yvex_tui_launch_failure failure)
{
    switch (failure) {
    case YVEX_TUI_LAUNCH_FAILURE_PREFLIGHT: return "PREFLIGHT FAILURE";
    case YVEX_TUI_LAUNCH_FAILURE_SPAWN: return "SPAWN FAILURE";
    case YVEX_TUI_LAUNCH_FAILURE_BOOTSTRAP: return "BOOTSTRAP FAILURE";
    case YVEX_TUI_LAUNCH_FAILURE_PROTOCOL: return "PROTOCOL FAILURE";
    case YVEX_TUI_LAUNCH_FAILURE_ENGINE_LOAD: return "MODEL LOAD FAILURE";
    case YVEX_TUI_LAUNCH_FAILURE_NONE: break;
    }
    return "RUNTIME LAUNCH FAILED";
}

static const yvex_tui_model_row *launch_model(const yvex_tui_state *state)
{
    if (!state->model_count || state->launch_selected_model >= state->model_count)
        return NULL;
    return &state->models[state->launch_selected_model];
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

static void render_top(tui_frame *frame, const yvex_tui_state *state)
{
    const yvex_server_engine_summary *engine = active_engine(state);
    unsigned int limit = frame->columns > 1u ? frame->columns - 1u : 1u;
    unsigned int right_width;
    char right[512];
    frame_begin_line(frame, 1u);
    frame_style(frame, frame->accent);
    frame_text(frame, "  YVEX", limit);
    frame_style(frame, frame->reset);
    if (state->surface != YVEX_TUI_SURFACE_HOME) {
        frame_style(frame, frame->dim);
        frame_text(frame, "  /  ", limit);
        frame_style(frame, frame->strong);
        frame_text(frame, surface_name(state->surface), limit);
        frame_style(frame, frame->reset);
    }
    if (state->runtime_available && engine)
        (void)snprintf(right, sizeof(right), "%s  ·  %s  ·  %s",
                       engine->alias, backend_name(engine->backend),
                       state->active_session);
    else if (state->runtime_available)
        (void)snprintf(right, sizeof(right), "%s  no model loaded",
                       lifecycle_name(state));
    else if (state->model_catalog_status == YVEX_TUI_MODEL_CATALOG_ERROR)
        (void)snprintf(right, sizeof(right), "%s  registry unavailable", lifecycle_name(state));
    else
        (void)snprintf(right, sizeof(right), "%s  %s", lifecycle_name(state),
                       yvex_tui_startup_model_count(state) ? "Enter to start" : "models required");
    right_width = text_width(right, strlen(right));
    if (right_width + 2u < limit) {
        unsigned int start = limit - right_width;
        frame_to_column(frame, start, limit);
        frame_style(frame, state->connection == YVEX_TUI_CONNECTION_CONNECTED
                               ? frame->dim
                           : state->runtime_lifecycle == YVEX_TUI_RUNTIME_LAUNCH_FAILED
                               ? frame->error : frame->warning);
        frame_text(frame, right, limit);
        frame_style(frame, frame->reset);
    }
}

static const char *activity_label(const yvex_tui_activity *activity)
{
    if (activity->kind == YVEX_TUI_ACTIVITY_USER) return "";
    if (activity->kind == YVEX_TUI_ACTIVITY_ERROR) return "error";
    if (activity->kind == YVEX_TUI_ACTIVITY_GENERATION) {
        if (activity->channel == YVEX_CLIENT_STREAM_EXPLICIT_REASONING) return "thinking";
        if (activity->channel == YVEX_CLIENT_STREAM_TOOL_CALL) return "tool";
        if (activity->channel == YVEX_CLIENT_STREAM_TOOL_RESULT) return "tool result";
        if (activity->channel == YVEX_CLIENT_STREAM_ERROR) return "error";
        return "";
    }
    return "";
}

static const char *activity_style(tui_frame *frame,
                                  const yvex_tui_activity *activity)
{
    if (activity->severity == YVEX_TUI_SEVERITY_ERROR) return frame->error;
    if (activity->severity == YVEX_TUI_SEVERITY_WARNING) return frame->warning;
    if (activity->severity == YVEX_TUI_SEVERITY_SUCCESS) return frame->success;
    if (activity->kind == YVEX_TUI_ACTIVITY_USER) return frame->accent;
    if (activity->channel == YVEX_CLIENT_STREAM_EXPLICIT_REASONING) return frame->dim;
    if (activity->kind == YVEX_TUI_ACTIVITY_GENERATION) return frame->strong;
    return frame->dim;
}

static const char *engine_execution_name(
    yvex_server_engine_kind kind, yvex_server_execution_strategy strategy)
{
    if (kind == YVEX_SERVER_ENGINE_MEDIA) return "media";
    if (strategy == YVEX_SERVER_EXECUTION_SPECULATIVE) return "speculative";
    if (strategy == YVEX_SERVER_EXECUTION_TARGET_ONLY) return "target-only";
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

static const char *engine_state_name(yvex_server_engine_state state)
{
    switch (state) {
    case YVEX_SERVER_ENGINE_UNLOADED: return "unloaded";
    case YVEX_SERVER_ENGINE_LOADING: return "loading";
    case YVEX_SERVER_ENGINE_LOADED: return "loaded";
    case YVEX_SERVER_ENGINE_DRAINING: return "draining";
    case YVEX_SERVER_ENGINE_UNLOADING: return "unloading";
    case YVEX_SERVER_ENGINE_FAILED: return "failed";
    }
    return "unknown";
}

static void inspector_field(tui_frame *frame, const char *label,
                            const char *value, unsigned int limit)
{
    char key[16];
    (void)snprintf(key, sizeof(key), "%-8.8s  ", label);
    frame_style(frame, frame->dim);
    frame_text(frame, key, limit);
    frame_style(frame, frame->reset);
    frame_text(frame, value, limit);
}

static int conversation_activity(const yvex_tui_activity *activity)
{
    return activity->kind == YVEX_TUI_ACTIVITY_USER ||
           activity->kind == YVEX_TUI_ACTIVITY_GENERATION;
}

static unsigned int conversation_line_count(const yvex_tui_activity *activity,
                                            unsigned int columns)
{
    const char *text = activity->text[0] ? activity->text : "…";
    size_t count = strlen(text), cursor = 0u;
    unsigned int lines = 0u;
    while (cursor < count) {
        size_t previous = cursor;
        size_t extent = text_line_bytes(text + cursor, count - cursor, columns);
        lines++;
        cursor += extent;
        if (cursor < count && text[cursor] == '\r') cursor++;
        if (cursor < count && text[cursor] == '\n') cursor++;
        if (cursor == previous) cursor++;
    }
    return lines ? lines : 1u;
}

static const char *conversation_marker(const tui_frame *frame,
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

static unsigned int render_conversation_activity(
    tui_frame *frame, const yvex_tui_activity *activity,
    unsigned int row, unsigned int last, unsigned int limit,
    unsigned int wrap_columns, unsigned int skip_lines)
{
    const char *text = activity->text[0] ? activity->text : "…";
    const char *label = activity_label(activity);
    size_t count = strlen(text), cursor = 0u;
    unsigned int line = 0u;
    while (cursor < count && row <= last) {
        size_t previous = cursor;
        size_t extent = text_line_bytes(text + cursor, count - cursor, wrap_columns);
        if (line >= skip_lines) {
            frame_begin_line(frame, row++);
            frame_text(frame, "  ", limit);
            frame_style(frame, activity_style(frame, activity));
            if (line == 0u)
                frame_text(frame, conversation_marker(frame, activity), limit);
            else if (line == skip_lines && skip_lines)
                frame_text(frame, frame->unicode ? "… " : "> ", limit);
            else
                frame_text(frame, "  ", limit);
            if (line == 0u && label[0]) {
                frame_text(frame, label, limit);
                frame_text(frame, "  ", limit);
            }
            frame_text_n(frame, text + cursor, extent, limit);
            frame_style(frame, frame->reset);
        }
        cursor += extent;
        if (cursor < count && text[cursor] == '\r') cursor++;
        if (cursor < count && text[cursor] == '\n') cursor++;
        if (cursor == previous) cursor++;
        line++;
    }
    return row;
}

static const yvex_tui_model_row *first_startup_model(const yvex_tui_state *state)
{
    size_t index;
    if (state->selected_model < state->model_count &&
        state->models[state->selected_model].startup_ready)
        return &state->models[state->selected_model];
    for (index = 0u; index < state->model_count; ++index)
        if (state->models[index].startup_ready) return &state->models[index];
    return NULL;
}

static void lifecycle_field(tui_frame *frame, unsigned int row,
                            const char *label, const char *value,
                            unsigned int value_column, unsigned int limit)
{
    frame_begin_line(frame, row);
    frame_style(frame, frame->dim);
    frame_text(frame, "      ", limit);
    frame_text(frame, label, limit);
    frame_to_column(frame, value_column, limit);
    frame_style(frame, frame->reset);
    frame_text(frame, value, limit);
}

static void render_lifecycle_state(tui_frame *frame, const yvex_tui_state *state,
                                   unsigned int first, unsigned int last,
                                   const char *surface)
{
    const yvex_tui_model_row *model = launch_model(state);
    const yvex_model_runtime_profile_fact *profile = yvex_tui_launch_profile(state);
    unsigned int row = first, limit = frame->columns > 1u ? frame->columns - 1u : 1u;
    char value[256];
    if (!model) model = first_startup_model(state);
    if (!profile && model && state->model_library) {
        size_t model_index = (size_t)(model - state->models);
        profile = yvex_model_library_profile_at(state->model_library, model_index, 0u);
    }
    frame_begin_line(frame, row++);
    frame_style(frame, frame->dim);
    frame_text(frame, "  ", limit);
    frame_text(frame, surface, limit);
    if (row <= last) {
        frame_begin_line(frame, row++);
        frame_text(frame, "  ", limit);
        frame_rule_until(frame, limit > 3u ? limit - 2u : limit);
    }
    if (row < last) row++;
    frame_begin_line(frame, row++);
    frame_text(frame, "      ", limit);
    if (state->model_catalog_status == YVEX_TUI_MODEL_CATALOG_ERROR) {
        frame_style(frame, frame->error);
        frame_text(frame, "MODEL REGISTRY UNAVAILABLE", limit);
    } else if (state->runtime_lifecycle == YVEX_TUI_RUNTIME_LAUNCH_FAILED) {
        frame_style(frame, frame->error);
        frame_text(frame, launch_failure_name(state->launch_failure), limit);
    } else if (state->runtime_lifecycle == YVEX_TUI_RUNTIME_LAUNCHING ||
               state->runtime_lifecycle == YVEX_TUI_RUNTIME_WAITING_PROTOCOL ||
               state->runtime_lifecycle == YVEX_TUI_RUNTIME_ENGINE_LOADING ||
               state->runtime_lifecycle == YVEX_TUI_RUNTIME_LAUNCH_REQUESTED) {
        frame_style(frame, frame->warning);
        frame_text(frame, "STARTING RUNTIME", limit);
    } else if (state->runtime_lifecycle == YVEX_TUI_RUNTIME_SHUTDOWN_REQUESTED) {
        frame_style(frame, frame->warning);
        frame_text(frame, "STOPPING RUNTIME", limit);
    } else if (state->runtime_available && !state->active_engine.alias[0]) {
        frame_style(frame, frame->accent);
        frame_text(frame, "HOST READY · NO MODEL LOADED", limit);
    } else {
        frame_style(frame, frame->warning);
        frame_text(frame, "RUNTIME OFFLINE", limit);
    }
    if (row <= last) {
        frame_begin_line(frame, row++);
        frame_style(frame, frame->dim);
        frame_text(frame,
                   state->model_catalog_status == YVEX_TUI_MODEL_CATALOG_ERROR
                       ? "      The canonical local registry could not be decoded."
                       : state->runtime_lifecycle == YVEX_TUI_RUNTIME_LAUNCH_FAILED
                           ? "      Runtime readiness was not established."
                       : state->runtime_available && !state->active_engine.alias[0]
                           ? "      The canonical host is reachable and ready to load a model."
                       : "      No resident YVEX runtime is currently reachable.",
                   limit);
    }
    if (state->model_catalog_status == YVEX_TUI_MODEL_CATALOG_ERROR) {
        if (row <= last)
            lifecycle_field(frame, row++, "Cause", state->model_catalog_reason,
                            24u, limit);
        if (row < last) row++;
        if (row <= last)
            lifecycle_field(frame, row++, "Recovery",
                            "r retry · Tab to Models for details", 24u, limit);
        if (state->composer.count && row <= last) {
            (void)snprintf(value, sizeof(value), "%zu-byte Home draft preserved",
                           state->composer.count);
            lifecycle_field(frame, row, "Draft", value, 24u, limit);
        }
        return;
    }
    if (state->connection_reason[0] && row <= last)
        lifecycle_field(frame, row++, "Cause", state->connection_reason, 24u, limit);
    if (row < last) row++;
    if (model && row <= last)
        lifecycle_field(frame, row++, "Model", model->display_name, 24u, limit);
    if (profile && row <= last) {
        (void)snprintf(value, sizeof(value), "%s · profile %s",
                       profile->backend[0] ? profile->backend : "profile default",
                       profile->alias);
        lifecycle_field(frame, row++, "Backend", value, 24u, limit);
    }
    if (profile && row <= last) {
        if (profile->context_capacity)
            (void)snprintf(value, sizeof(value), "%llu tokens · profile",
                           profile->context_capacity);
        else
            (void)snprintf(value, sizeof(value), "profile default");
        lifecycle_field(frame, row++, "Context", value, 24u, limit);
    }
    if (row < last) row++;
    if (state->runtime_lifecycle == YVEX_TUI_RUNTIME_LAUNCHING ||
        state->runtime_lifecycle == YVEX_TUI_RUNTIME_WAITING_PROTOCOL ||
        state->runtime_lifecycle == YVEX_TUI_RUNTIME_ENGINE_LOADING) {
        if (row <= last)
            lifecycle_field(frame, row++, "Launch canonical host", "done", 34u, limit);
        if (row <= last)
            lifecycle_field(frame, row++, "Enter server operation",
                            state->runtime_lifecycle == YVEX_TUI_RUNTIME_WAITING_PROTOCOL
                                ? "done" : "waiting", 34u, limit);
        if (row <= last)
            lifecycle_field(frame, row++, "Versioned protocol handshake",
                            state->runtime_lifecycle == YVEX_TUI_RUNTIME_ENGINE_LOADING
                                ? "done" : "waiting", 34u, limit);
        if (row <= last)
            lifecycle_field(frame, row++, "Load selected model",
                            state->runtime_lifecycle == YVEX_TUI_RUNTIME_ENGINE_LOADING
                                ? "waiting" : "pending", 34u, limit);
    } else if (state->runtime_lifecycle == YVEX_TUI_RUNTIME_LAUNCH_FAILED) {
        if (row <= last)
            lifecycle_field(frame, row++, "Cause",
                            state->launch_failure_reason[0]
                                ? state->launch_failure_reason
                                : "runtime launch did not establish readiness",
                            24u, limit);
        if (state->launch_diagnostic[0] && row <= last)
            lifecycle_field(frame, row++, "Diagnostic",
                            state->launch_diagnostic, 24u, limit);
        if ((state->launch_exec_error || state->launch_exit_status) && row <= last) {
            if (state->launch_exec_error)
                (void)snprintf(value, sizeof(value), "exec error %d",
                               state->launch_exec_error);
            else
                (void)snprintf(value, sizeof(value), "server exit status %d",
                               state->launch_exit_status);
            lifecycle_field(frame, row++, "Process", value, 24u, limit);
        }
        if (row <= last)
            lifecycle_field(frame, row++, "Recovery", "Enter to retry", 24u, limit);
    } else if (model && model->startup_ready && row <= last) {
        lifecycle_field(frame, row++, "Primary action",
                        state->runtime_available ? "Enter  Load Model"
                                                 : "Enter  Start Runtime",
                        24u, limit);
    } else if (model && row <= last) {
        lifecycle_field(frame, row++, "Launch blocker", model->startup_reason, 24u, limit);
    } else if (row <= last) {
        lifecycle_field(frame, row++, "Next action", "Tab to Models · r refresh", 24u, limit);
    }
    if (state->composer.count && row <= last) {
        (void)snprintf(value, sizeof(value), "%zu-byte Home draft preserved",
                       state->composer.count);
        lifecycle_field(frame, row, "Draft", value, 24u, limit);
    }
}

static void render_home(tui_frame *frame, const yvex_tui_state *state,
                        unsigned int first, unsigned int last)
{
    const yvex_server_engine_summary *engine = active_engine(state);
    size_t visible[YVEX_TUI_ACTIVITY_CAP], line_counts[YVEX_TUI_ACTIVITY_CAP];
    unsigned int row, body_rows, used = 0u, skip = 0u;
    unsigned int limit = frame->columns > 1u ? frame->columns - 1u : 1u;
    unsigned int content_limit = limit > 128u ? 128u : limit;
    unsigned int wrap_columns = content_limit > 18u ? content_limit - 18u : 1u;
    size_t end = state->activity_count > state->activity_scroll
                     ? state->activity_count - state->activity_scroll : 0u;
    size_t begin, count = 0u, ordinal;
    if (!state->runtime_available || !state->active_engine.alias[0]) {
        render_lifecycle_state(frame, state, first, last, "CHAT / RUNTIME CONTROL");
        return;
    }
    frame_begin_line(frame, first);
    frame_style(frame, frame->dim);
    frame_text(frame, "  CHAT", content_limit);
    frame_text(frame, "  ·  ", content_limit);
    frame_style(frame, frame->accent);
    frame_text(frame, state->active_session, content_limit);
    frame_style(frame, frame->reset);
    if (first + 1u <= last) {
        frame_begin_line(frame, first + 1u);
        frame_text(frame, "  ", content_limit);
        frame_rule_until(frame, content_limit > 3u ? content_limit - 2u : content_limit);
    }
    row = first + 2u;
    body_rows = row <= last ? last - row + 1u : 0u;
    for (ordinal = 0u; ordinal < end; ++ordinal) {
        size_t slot = (state->activity_start + ordinal) % YVEX_TUI_ACTIVITY_CAP;
        if (!conversation_activity(&state->activities[slot])) continue;
        visible[count] = slot;
        line_counts[count] = conversation_line_count(&state->activities[slot],
                                                     wrap_columns);
        count++;
    }
    if (!count) {
        if (body_rows > 9u) row += 2u;
        if (row <= last) {
            frame_begin_line(frame, row++);
            frame_style(frame, frame->strong);
            frame_text(frame, "  Ready to work", content_limit);
        }
        if (row <= last) {
            frame_begin_line(frame, row++);
            frame_style(frame, frame->accent);
            frame_text(frame, "  ", content_limit);
            frame_text(frame, engine->alias, content_limit);
        }
        if (row <= last) {
            char context[256];
            (void)snprintf(context, sizeof(context), "  %s · %s · session %s",
                           backend_name(engine->backend),
                           engine_execution_name(
                               engine->engine_kind, engine->execution_strategy),
                           state->active_session);
            frame_begin_line(frame, row++);
            frame_style(frame, frame->dim);
            frame_text(frame, context, content_limit);
        }
        if (row < last) row++;
        if (row <= last) {
            frame_begin_line(frame, row);
            frame_style(frame, frame->dim);
            frame_text(frame,
                       "  Type below to begin · Ctrl-O model · Ctrl-P commands",
                       content_limit);
        }
        return;
    }
    begin = count;
    while (begin) {
        unsigned int need = (unsigned int)line_counts[begin - 1u] + 1u;
        unsigned int remaining = body_rows > used ? body_rows - used : 0u;
        if (need > remaining) {
            if (remaining) {
                begin--;
                if (line_counts[begin] > remaining)
                    skip = (unsigned int)line_counts[begin] - remaining;
            }
            break;
        }
        begin--;
        used += need;
    }
    for (ordinal = begin; ordinal < count && row <= last; ++ordinal) {
        row = render_conversation_activity(frame, &state->activities[visible[ordinal]],
                                           row, last, content_limit, wrap_columns,
                                           ordinal == begin ? skip : 0u);
        if (ordinal + 1u < count && row <= last) {
            frame_begin_line(frame, row);
            row++;
        }
    }
}

static void render_model_inspector(tui_frame *frame, const yvex_tui_state *state,
                                   unsigned int offset, unsigned int limit)
{
    const yvex_tui_model_row *model;
    char text[256];
    if (!state->model_count || state->selected_model >= state->model_count) {
        if (!offset) {
            frame_style(frame, frame->violet);
            frame_text(frame, "SELECTED MODEL", limit);
        }
        return;
    }
    model = &state->models[state->selected_model];
    if (offset == 0u) {
        frame_style(frame, frame->violet);
        frame_text(frame, "SELECTED MODEL", limit);
    } else if (offset == 1u) {
        frame_style(frame, frame->strong);
        frame_text(frame, model->display_name, limit);
    } else if (offset == 2u) {
        (void)snprintf(text, sizeof(text), "%s / %s",
                       model->family[0] ? model->family : "unknown family",
                       model->model[0] ? model->model : "unknown model");
        frame_style(frame, frame->dim);
        frame_text(frame, text, limit);
    } else if (offset == 4u) {
        (void)snprintf(text, sizeof(text), "%llu", model->source_count);
        inspector_field(frame, "sources", text, limit);
    } else if (offset == 5u) {
        (void)snprintf(text, sizeof(text), "%llu", model->artifact_count);
        inspector_field(frame, "artifacts", text, limit);
    } else if (offset == 6u) {
        (void)snprintf(text, sizeof(text), "%llu · %llu launchable",
                       model->profile_count, model->launchable_profile_count);
        inspector_field(frame, "profiles", text, limit);
    } else if (offset == 8u) {
        inspector_field(frame, "source local", model->source_local ? "yes" : "no", limit);
    } else if (offset == 9u) {
        inspector_field(frame, "artifact ready", model->artifact_ready ? "yes" : "no", limit);
    } else if (offset == 10u) {
        frame_style(frame, frame->dim);
        frame_text(frame, "status  ", limit);
        frame_style(frame, model->startup_ready ? frame->success : frame->warning);
        frame_text(frame, model->resident ? "resident" :
                          model->startup_ready ? "launchable" : "startup blocked", limit);
        frame_style(frame, frame->reset);
    } else if (offset == 11u && !model->startup_ready) {
        frame_style(frame, frame->dim);
        frame_text(frame, model->startup_reason, limit);
    } else if (offset == 12u) {
        frame_style(frame, frame->dim);
        frame_text(frame, "Enter inspect   / search", limit);
    }
}

static void model_finish_line(tui_frame *frame, const yvex_tui_state *state,
                              unsigned int row, unsigned int first,
                              unsigned int table_limit, unsigned int limit)
{
    if (table_limit >= limit) return;
    frame_to_column(frame, table_limit, limit);
    frame_style(frame, frame->dim);
    frame_text(frame, frame->unicode ? "│  " : "|  ", limit);
    frame_style(frame, frame->reset);
    render_model_inspector(frame, state, row - first, limit);
}

static const yvex_model_library_entry *remote_local_model(
    const yvex_tui_state *state, const yvex_remote_model *remote,
    unsigned long long *model_index)
{
    unsigned long long index = 0u;
    if (!state->model_library ||
        !yvex_model_library_remote_match(state->model_library, remote, &index))
        return NULL;
    if (model_index) *model_index = index;
    return yvex_model_library_at(state->model_library, index);
}

static int remote_acquisition_admitted(const yvex_remote_model *remote)
{
    return remote && remote->support_stage >= YVEX_MODEL_SUPPORT_SOURCE_INGEST &&
           remote->resolved_revision[0];
}

static void detail_field(tui_frame *frame, unsigned int row, const char *label,
                         const char *value, unsigned int limit);

static void render_remote_detail(tui_frame *frame, const yvex_tui_state *state,
                                 unsigned int first, unsigned int last)
{
    const yvex_remote_model *remote;
    const yvex_model_library_entry *local;
    unsigned long long local_index = 0u;
    int resident = 0;
    unsigned int limit = frame->columns > 1u ? frame->columns - 1u : 1u;
    char text[256];
    if (state->selected_remote >= state->remote_count) return;
    remote = &state->remote_models[state->selected_remote];
    local = remote_local_model(state, remote, &local_index);
    if (local && local_index < state->model_count)
        resident = state->models[local_index].resident;
    frame_begin_line(frame, first);
    frame_style(frame, frame->accent);
    frame_text(frame, "  REMOTE MODEL", limit);
    frame_style(frame, frame->strong);
    frame_text(frame, "  ", limit);
    frame_text(frame, remote->repository, limit);
    if (first + 2u <= last)
        detail_field(frame, first + 2u, "PROVIDER", "Hugging Face", limit);
    if (first + 3u <= last)
        detail_field(frame, first + 3u, "repository", remote->repository, limit);
    if (first + 4u <= last)
        detail_field(frame, first + 4u, "revision",
                     remote->resolved_revision[0] ? remote->resolved_revision
                                                   : remote->revision_reference, limit);
    if (first + 5u <= last)
        detail_field(frame, first + 5u, "architecture", remote->architecture, limit);
    if (first + 6u <= last)
        detail_field(frame, first + 6u, "pipeline", remote->pipeline, limit);
    if (first + 8u <= last)
        detail_field(frame, first + 8u, "REMOTE", "available", limit);
    if (first + 9u <= last)
        detail_field(frame, first + 9u, "YVEX SUPPORT",
                     yvex_model_support_stage_name(remote->support_stage), limit);
    if (first + 10u <= last) {
        (void)snprintf(text, sizeof(text), "%s%s",
                       remote->gated_known && remote->gated ? "gated" : "public",
                       remote_acquisition_admitted(remote) ? " · acquisition admitted" : "");
        detail_field(frame, first + 10u, "ACCESS", text, limit);
    }
    if (first + 12u <= last) {
        (void)snprintf(text, sizeof(text), "%s · artifact %s · profile %s · runtime %s",
                       local && local->source_local ? "source local" : "source absent",
                       local && local->artifact_ready ? "ready" : "absent",
                       local && local->profile_launchable ? "launchable" : "absent",
                       resident ? "resident" : "stopped");
        detail_field(frame, first + 12u, "LOCAL STATE", text, limit);
    }
    if (first + 14u <= last)
        detail_field(frame, first + 14u, "ACTION",
                     state->acquisition_running
                         ? "acquisition running through canonical model operation"
                     : state->acquisition_exit_known && state->acquisition_exit_status
                         ? "acquisition failed; bounded diagnostic below"
                     : remote_acquisition_admitted(remote)
                         ? "a acquire through canonical model operation · Esc back"
                         : remote->support_reason[0]
                             ? remote->support_reason
                             : "no immutable supported acquisition target · Esc back", limit);
    if (first + 15u <= last && state->acquisition_diagnostic[0])
        detail_field(frame, first + 15u, "DIAGNOSTIC",
                     state->acquisition_diagnostic, limit);
}

static void render_discover(tui_frame *frame, const yvex_tui_state *state,
                            unsigned int first, unsigned int last)
{
    unsigned int row, limit = frame->columns > 1u ? frame->columns - 1u : 1u;
    char text[512];
    if (state->remote_detail) {
        render_remote_detail(frame, state, first, last);
        return;
    }
    frame_begin_line(frame, first);
    frame_style(frame, frame->accent);
    frame_text(frame, "  MODELS / DISCOVER", limit);
    frame_style(frame, frame->dim);
    frame_text(frame, "  l Library   [Hugging Face]", limit);
    if (first + 1u <= last) {
        frame_begin_line(frame, first + 1u);
        frame_style(frame, state->focus == YVEX_TUI_FOCUS_MODEL_SEARCH
                               ? frame->accent : frame->dim);
        frame_text(frame, "  / ", limit);
        frame_style(frame, frame->reset);
        frame_text(frame, state->discover_query_count ? state->discover_query
                                                      : "search public remote models", limit);
    }
    if (state->remote_search_running) {
        if (first + 3u <= last) {
            frame_begin_line(frame, first + 3u);
            frame_style(frame, frame->violet);
            frame_text(frame, "      SEARCHING HUGGING FACE", limit);
        }
        if (first + 4u <= last) {
            frame_begin_line(frame, first + 4u);
            frame_style(frame, frame->dim);
            frame_text(frame, "      Official provider CLI · bounded result and timeout", limit);
        }
        return;
    }
    if (!state->remote_count) {
        if (first + 3u <= last) {
            frame_begin_line(frame, first + 3u);
            frame_style(frame, state->remote_search_reason[0]
                                   ? frame->warning : frame->dim);
            (void)snprintf(text, sizeof(text), "      %s",
                           state->remote_search_reason[0]
                               ? "REMOTE SEARCH UNAVAILABLE"
                               : "ENTER A QUERY TO DISCOVER MODELS");
            frame_text(frame, text, limit);
        }
        if (first + 5u <= last && state->remote_search_reason[0])
            lifecycle_field(frame, first + 5u, "Provider",
                            state->remote_search_reason, 24u, limit);
        return;
    }
    for (row = first + 2u; row <= last; ++row) {
        size_t index = state->remote_viewport;
        if (row > first + 2u) index += (size_t)(row - first - 3u);
        frame_begin_line(frame, row);
        if (row == first + 2u) {
            frame_style(frame, frame->dim);
            frame_text(frame, "   REPOSITORY                         REMOTE  YVEX       LOCAL       ACTION", limit);
        } else if (index < state->remote_count) {
            const yvex_remote_model *remote = &state->remote_models[index];
            const yvex_model_library_entry *local = remote_local_model(state, remote, NULL);
            int selected = index == state->selected_remote;
            (void)snprintf(text, sizeof(text), "%c  %-34.34s %-7s %-10s %-11s %s",
                           selected ? '>' : ' ', remote->repository, "yes",
                           yvex_model_support_stage_name(remote->support_stage),
                           local && local->source_local ? "source local" : "not local",
                           remote_acquisition_admitted(remote) ? "acquire" : "not admitted");
            frame_style(frame, selected ? frame->selected : frame->reset);
            frame_text(frame, text, limit);
            frame_style(frame, frame->reset);
        }
    }
}

static void render_models(tui_frame *frame, const yvex_tui_state *state,
                          unsigned int first, unsigned int last)
{
    unsigned int row, limit = frame->columns > 1u ? frame->columns - 1u : 1u;
    unsigned int table_limit = state->terminal.layout == YVEX_TUI_LAYOUT_COMPACT
                                   ? limit
                                   : state->terminal.layout == YVEX_TUI_LAYOUT_STANDARD
                                         ? limit * 72u / 100u : limit * 76u / 100u;
    size_t visible = yvex_tui_model_visible_count(state);
    size_t viewport = state->model_viewport;
    char text[512];
    if (state->models_mode == YVEX_TUI_MODELS_DISCOVER) {
        render_discover(frame, state, first, last);
        return;
    }
    if (state->model_catalog_status == YVEX_TUI_MODEL_CATALOG_ERROR) {
        frame_begin_line(frame, first);
        frame_style(frame, frame->accent);
        frame_text(frame, "  MODELS / LIBRARY", limit);
        if (first + 1u <= last) {
            frame_begin_line(frame, first + 1u);
            frame_text(frame, "  ", limit);
            frame_rule_until(frame, limit > 3u ? limit - 2u : limit);
        }
        if (first + 3u <= last) {
            frame_begin_line(frame, first + 3u);
            frame_style(frame, frame->error);
            frame_text(frame, "      MODEL REGISTRY UNAVAILABLE", limit);
        }
        if (first + 4u <= last) {
            frame_begin_line(frame, first + 4u);
            frame_style(frame, frame->dim);
            frame_text(frame, "      The catalog was not treated as an empty registry.", limit);
        }
        if (first + 6u <= last)
            lifecycle_field(frame, first + 6u, "Cause",
                            state->model_catalog_reason, 24u, limit);
        if (first + 8u <= last)
            lifecycle_field(frame, first + 8u, "Recovery",
                            "r retry · d Discover on Hugging Face", 24u, limit);
        return;
    }
    if (!state->model_count) {
        frame_begin_line(frame, first);
        frame_style(frame, frame->accent);
        frame_text(frame, "  MODELS / LIBRARY", limit);
        if (first + 1u <= last) {
            frame_begin_line(frame, first + 1u);
            frame_text(frame, "  ", limit);
            frame_rule_until(frame, limit > 3u ? limit - 2u : limit);
        }
        if (first + 3u <= last) {
            frame_begin_line(frame, first + 3u);
            frame_style(frame, frame->warning);
            frame_text(frame, "      NO REGISTERED MODELS", limit);
        }
        if (first + 4u <= last) {
            frame_begin_line(frame, first + 4u);
            frame_style(frame, frame->dim);
            frame_text(frame, "      The canonical local model registry has no rows.", limit);
        }
        if (first + 6u <= last)
            lifecycle_field(frame, first + 6u, "Next action",
                            "d Discover remote models · r refresh", 24u, limit);
        return;
    }
    frame_begin_line(frame, first);
    frame_style(frame, frame->accent);
    frame_text(frame, "  MODELS / LIBRARY", table_limit);
    frame_style(frame, frame->dim);
    (void)snprintf(text, sizeof(text), "  d Discover · %zu visible / %zu logical",
                   visible, state->model_count);
    frame_text(frame, text, table_limit);
    model_finish_line(frame, state, first, first, table_limit, limit);
    if (first + 1u <= last) {
        frame_begin_line(frame, first + 1u);
        frame_style(frame, state->focus == YVEX_TUI_FOCUS_MODEL_SEARCH
                               ? frame->accent : frame->dim);
        frame_text(frame, "  / ", table_limit);
        frame_style(frame, frame->reset);
        frame_text(frame, state->model_search_count
                              ? state->model_search : "search logical model library",
                   table_limit);
        model_finish_line(frame, state, first + 1u, first, table_limit, limit);
    }
    for (row = first + 2u; row <= last; ++row) {
        size_t ordinal = viewport;
        if (row > first + 2u) ordinal += (size_t)(row - first - 3u);
        frame_begin_line(frame, row);
        if (row == first + 2u) {
            frame_style(frame, frame->dim);
            if (state->terminal.layout == YVEX_TUI_LAYOUT_COMPACT)
                frame_text(frame, "   MODEL                    FAMILY       STATUS", table_limit);
            else if (state->terminal.layout == YVEX_TUI_LAYOUT_STANDARD)
                frame_text(frame, "   MODEL                 FAMILY      ARTIFACTS PROFILES", table_limit);
            else
                frame_text(frame,
                           "   MODEL                 FAMILY      SOURCES ARTIFACTS PROFILES RUNTIME STATUS",
                           table_limit);
            frame_style(frame, frame->reset);
        } else if (ordinal < visible) {
            size_t model_index = yvex_tui_model_visible_at(state, ordinal);
            const yvex_tui_model_row *model = &state->models[model_index];
            int selected = model_index == state->selected_model;
            frame_style(frame, selected ? frame->selected : frame->reset);
            if (state->terminal.layout == YVEX_TUI_LAYOUT_COMPACT)
                (void)snprintf(text, sizeof(text), "%c  %-24.24s %-12.12s %s",
                               selected ? '>' : ' ', model->display_name, model->family,
                               model->resident ? "resident" :
                               model->startup_ready ? "launchable" : "blocked");
            else if (state->terminal.layout == YVEX_TUI_LAYOUT_STANDARD)
                (void)snprintf(text, sizeof(text), "%c  %-21.21s %-11.11s %9llu %8llu",
                               selected ? '>' : ' ', model->display_name, model->family,
                               model->artifact_count, model->profile_count);
            else
                (void)snprintf(text, sizeof(text),
                               "%c  %-21.21s %-11.11s %7llu %9llu %8llu %-7s %s",
                               selected ? '>' : ' ', model->display_name, model->family,
                               model->source_count, model->artifact_count, model->profile_count,
                               model->resident ? "running" : "stopped",
                               model->startup_ready ? "launchable" : "blocked");
            frame_text(frame, text, table_limit);
            frame_style(frame, frame->reset);
        } else if (!visible && row == first + 3u) {
            frame_style(frame, frame->warning);
            frame_text(frame, "   No registry models match this search", table_limit);
            frame_style(frame, frame->reset);
        }
        model_finish_line(frame, state, row, first, table_limit, limit);
    }
}

static void detail_field(tui_frame *frame, unsigned int row, const char *label,
                         const char *value, unsigned int limit)
{
    frame_begin_line(frame, row);
    frame_style(frame, frame->dim);
    frame_text(frame, "  ", limit);
    frame_text(frame, label, limit);
    frame_to_column(frame, 18u, limit);
    frame_style(frame, frame->reset);
    frame_text(frame, value && value[0] ? value : "unavailable", limit);
}

static void render_model_detail(tui_frame *frame, const yvex_tui_state *state,
                                unsigned int first, unsigned int last)
{
    const yvex_tui_model_row *model;
    unsigned long long index, maximum;
    unsigned int row;
    unsigned int limit = frame->columns > 1u ? frame->columns - 1u : 1u;
    char label[32], text[512];
    if (!state->model_count || state->selected_model >= state->model_count) {
        frame_begin_line(frame, first);
        frame_text(frame, "  No selected registry model", limit);
        return;
    }
    model = &state->models[state->selected_model];
    frame_begin_line(frame, first);
    frame_style(frame, frame->accent);
    frame_text(frame, "  MODEL", limit);
    frame_style(frame, frame->strong);
    frame_text(frame, "  ", limit);
    frame_text(frame, model->display_name, limit);
    row = first + 2u;
    if (row <= last) detail_field(frame, row++, "IDENTITY / family", model->family, limit);
    if (row <= last) detail_field(frame, row++, "identity / model", model->model, limit);
    if (row <= last) detail_field(frame, row++, "identity / key", model->identity, limit);

    (void)snprintf(text, sizeof(text), "%llu local · %llu remote",
                   model->source_count, model->remote_count);
    if (row < last) detail_field(frame, ++row, "SOURCES", text, limit);
    maximum = state->terminal.layout == YVEX_TUI_LAYOUT_COMPACT ? 1u : 2u;
    for (index = 0u; index < model->source_count && index < maximum && row < last; ++index) {
        const yvex_local_source_record *source = yvex_model_library_source_at(
            state->model_library, state->selected_model, index);
        (void)snprintf(label, sizeof(label), "source %llu", index + 1u);
        (void)snprintf(text, sizeof(text), "%s · %s@%s · %s",
                       source->provider, source->repository, source->revision,
                       source->acquisition_state[0] ? source->acquisition_state : "unknown");
        detail_field(frame, ++row, label, text, limit);
    }
    if (model->source_count > maximum && row < last) {
        (void)snprintf(text, sizeof(text), "%llu more source installations",
                       model->source_count - maximum);
        detail_field(frame, ++row, "source", text, limit);
    }

    (void)snprintf(text, sizeof(text), "%llu · %s", model->artifact_count,
                   model->artifact_ready ? "execution-ready present"
                                         : "no executable artifact");
    if (row < last) detail_field(frame, ++row, "ARTIFACTS", text, limit);
    maximum = state->terminal.layout == YVEX_TUI_LAYOUT_COMPACT ? 1u :
              state->terminal.layout == YVEX_TUI_LAYOUT_STANDARD ? 2u : 3u;
    for (index = 0u; index < model->artifact_count && index < maximum && row < last; ++index) {
        const yvex_model_artifact_fact *artifact = yvex_model_library_artifact_at(
            state->model_library, state->selected_model, index);
        (void)snprintf(label, sizeof(label), "artifact %llu", index + 1u);
        (void)snprintf(text, sizeof(text), "%s · %s · %s · %s",
                       artifact->format, artifact->artifact_class,
                       artifact->physical_variant,
                       artifact->execution_ready ? "execution-ready" : "not execution-ready");
        detail_field(frame, ++row, label, text, limit);
    }
    if (model->artifact_count > maximum && row < last) {
        (void)snprintf(text, sizeof(text), "%llu more physical variants",
                       model->artifact_count - maximum);
        detail_field(frame, ++row, "artifact", text, limit);
    }

    (void)snprintf(text, sizeof(text), "%llu · %llu launchable",
                   model->profile_count, model->launchable_profile_count);
    if (row < last) detail_field(frame, ++row, "RUNTIME PROFILES", text, limit);
    maximum = state->terminal.layout == YVEX_TUI_LAYOUT_COMPACT ? 2u :
              state->terminal.layout == YVEX_TUI_LAYOUT_STANDARD ? 6u : 12u;
    for (index = 0u; index < model->profile_count && index < maximum && row < last; ++index) {
        const yvex_model_runtime_profile_fact *profile = yvex_model_library_profile_at(
            state->model_library, state->selected_model, index);
        (void)snprintf(label, sizeof(label), "profile %llu", index + 1u);
        (void)snprintf(text, sizeof(text), "%s · %s · %llu ctx · %s",
                       profile->alias, profile->backend, profile->context_capacity,
                       profile->launchable ? "launchable" : "blocked");
        detail_field(frame, ++row, label, text, limit);
    }
    if (model->profile_count > maximum && row < last) {
        (void)snprintf(text, sizeof(text), "%llu more profiles",
                       model->profile_count - maximum);
        detail_field(frame, ++row, "profile", text, limit);
    }
    if (row < last)
        detail_field(frame, ++row, "RUNTIME",
                     model->resident ? "resident" : "stopped", limit);
    if (row < last)
        detail_field(frame, ++row, "STATUS", model->startup_ready
                         ? "launchable; protocol readiness remains authoritative"
                         : model->startup_reason, limit);
}

static void render_sessions(tui_frame *frame, const yvex_tui_state *state,
                            unsigned int first, unsigned int last)
{
    unsigned int row, limit = frame->columns > 1u ? frame->columns - 1u : 1u;
    size_t visible = yvex_tui_session_visible_count(state);
    char text[512];
    if (!state->runtime_available || !state->active_engine.alias[0]) {
        render_lifecycle_state(frame, state, first, last, "SESSIONS / SERVER OWNED");
        return;
    }
    frame_begin_line(frame, first);
    frame_style(frame, frame->accent);
    frame_text(frame, "  SESSIONS", limit);
    frame_style(frame, frame->dim);
    (void)snprintf(text, sizeof(text), "  %zu visible / %zu server-owned",
                   visible, state->session_count);
    frame_text(frame, text, limit);
    if (first + 1u <= last) {
        frame_begin_line(frame, first + 1u);
        frame_style(frame, state->focus == YVEX_TUI_FOCUS_SESSION_SEARCH
                               ? frame->accent : frame->dim);
        frame_text(frame, "  / ", limit);
        frame_style(frame, frame->reset);
        frame_text(frame, state->session_search_count
                              ? state->session_search : "search server sessions", limit);
    }
    if (!state->session_count) {
        if (first + 3u <= last) {
            frame_begin_line(frame, first + 3u);
            frame_style(frame, frame->dim);
            frame_text(frame, "      NO MATERIALIZED SESSIONS", limit);
        }
        if (first + 4u <= last) {
            frame_begin_line(frame, first + 4u);
            frame_text(frame, "      The runtime owns session creation and lifecycle.", limit);
        }
        if (first + 6u <= last)
            lifecycle_field(frame, first + 6u, "Action",
                            "Ctrl-P · /session-new", 24u, limit);
        return;
    }
    for (row = first + 2u; row <= last; ++row) {
        size_t ordinal = state->session_viewport;
        size_t index;
        if (row > first + 2u) ordinal += (size_t)(row - first - 3u);
        frame_begin_line(frame, row);
        if (row == first + 2u) {
            frame_style(frame, frame->dim);
            frame_text(frame, "   NAME                    STATE        POSITION  TURNS  CONTEXT       KV", limit);
            frame_style(frame, frame->reset);
        } else if (ordinal < visible) {
            index = yvex_tui_session_visible_at(state, ordinal);
            const yvex_tui_session_row *session = &state->sessions[index];
            int selected = index == state->selected_session;
            char kv[32];
            if (session->kv_used_available)
                (void)snprintf(kv, sizeof(kv), "%.1f MiB",
                               (double)session->kv_used_bytes / 1048576.0);
            else
                (void)snprintf(kv, sizeof(kv), "unavailable");
            frame_style(frame, selected ? frame->selected : frame->reset);
            (void)snprintf(text, sizeof(text), "%c  %-23.23s %-12.12s %8llu %6llu %10llu  %s%s",
                           selected ? '>' : ' ', session->name,
                           yvex_server_session_state_name(session->state),
                           session->position, session->turns, session->context_used,
                           kv, !strcmp(session->name, state->active_session) ? "  attached" : "");
            frame_text(frame, text, limit);
            frame_style(frame, frame->reset);
        } else if (!visible && row == first + 3u) {
            frame_style(frame, frame->warning);
            frame_text(frame, "   No server sessions match this search", limit);
        }
    }
}

static void runtime_metric(tui_frame *frame, unsigned int row,
                           const char *label, const char *value,
                           unsigned int limit)
{
    frame_begin_line(frame, row);
    frame_style(frame, frame->dim);
    frame_text(frame, "  ", limit);
    frame_text(frame, label, limit);
    frame_to_column(frame, 28u, limit);
    frame_style(frame, frame->reset);
    frame_text(frame, value, limit);
}

static void render_runtime(tui_frame *frame, const yvex_tui_state *state,
                           unsigned int first, unsigned int last)
{
    const yvex_server_engine_summary *engine = active_engine(state);
    unsigned int row = first, limit = frame->columns > 1u ? frame->columns - 1u : 1u;
    char value[256];
    if (!state->runtime_available || !state->active_engine.alias[0]) {
        render_lifecycle_state(frame, state, first, last, "RUNTIME / LIFECYCLE");
        return;
    }
    frame_begin_line(frame, row++);
    frame_style(frame, frame->accent);
    frame_text(frame, "  RUNTIME / TELEMETRY", limit);
    if (row <= last) runtime_metric(frame, row++, "state", lifecycle_name(state), limit);
    if (row <= last) runtime_metric(frame, row++, "socket", state->runtime.socket_path, limit);
    (void)snprintf(value, sizeof(value), "%llu loaded / %llu known",
                   state->runtime.loaded_engine_count, state->runtime.engine_count);
    if (row <= last) runtime_metric(frame, row++, "engines", value, limit);
    if (engine) {
        (void)snprintf(value, sizeof(value), "%s@%llu · %s",
                       engine->alias, engine->generation,
                       engine_state_name(engine->state));
        if (row <= last) runtime_metric(frame, row++, "active model", value, limit);
        (void)snprintf(value, sizeof(value), "%s · %s",
                       backend_name(engine->backend),
                       engine_execution_name(
                           engine->engine_kind, engine->execution_strategy));
        if (row <= last) runtime_metric(frame, row++, "execution", value, limit);
    } else if (row <= last) {
        runtime_metric(frame, row++, "active model", "none loaded", limit);
    }
    (void)snprintf(value, sizeof(value), "%.1f s", (double)state->runtime.metrics.uptime_ns / 1000000000.0);
    if (row <= last) runtime_metric(frame, row++, "uptime", value, limit);
    row++;
    (void)snprintf(value, sizeof(value), "%.2f GiB mapped · %.2f GiB host · %.2f GiB device",
                   (double)(engine ? engine->mapped_package_bytes : 0u) / 1073741824.0,
                   (double)(engine ? engine->resident_host_bytes : 0u) / 1073741824.0,
                   (double)(engine ? engine->resident_device_bytes : 0u) / 1073741824.0);
    if (row <= last) runtime_metric(frame, row++, "memory / residency", value, limit);
    (void)snprintf(value, sizeof(value), "%.2f GiB current · %.2f GiB peak",
                   (double)state->runtime.metrics.current_rss_bytes / 1073741824.0,
                   (double)state->runtime.metrics.peak_rss_bytes / 1073741824.0);
    if (row <= last) runtime_metric(frame, row++, "process RSS", value, limit);
    row++;
    (void)snprintf(value, sizeof(value), "%llu/%llu queued · %llu active",
                   state->runtime.metrics.queue_depth,
                   state->runtime.metrics.queue_capacity,
                   state->runtime.metrics.active_requests);
    if (row <= last) runtime_metric(frame, row++, "scheduling", value, limit);
    (void)snprintf(value, sizeof(value), "%llu complete · %llu failed · %llu cancelled",
                   state->runtime.metrics.completed_requests,
                   state->runtime.metrics.failed_requests,
                   state->runtime.metrics.cancelled_requests);
    if (row <= last) runtime_metric(frame, row++, "requests", value, limit);
    (void)snprintf(value, sizeof(value), "%llu active · %llu total",
                   state->runtime.metrics.active_sessions,
                   state->runtime.metrics.total_sessions);
    if (row <= last) runtime_metric(frame, row++, "sessions", value, limit);
    row++;
    (void)snprintf(value, sizeof(value), "%s%s",
                   state->runtime.openai_listener_enabled ? "enabled" : "disabled",
                   state->runtime.openai_listener_ready ? " · ready" : "");
    if (row <= last) runtime_metric(frame, row++, "OpenAI adapter", value, limit);
    (void)snprintf(value, sizeof(value), "%llu", state->runtime.metrics.telemetry_dropped);
    if (row <= last) runtime_metric(frame, row++, "telemetry dropped", value, limit);
    if (row < last) row++;
    if (row <= last) {
        frame_begin_line(frame, row++);
        frame_style(frame, frame->violet);
        frame_text(frame, "  LIFECYCLE ACTIONS", limit);
    }
    if (row <= last) {
        frame_begin_line(frame, row++);
        frame_text(frame, "  ", limit);
        frame_text(frame, state->runtime_action == 0u ? "› " : "  ", limit);
        frame_style(frame, state->runtime_action == 0u ? frame->selected : frame->dim);
        frame_text(frame, " Stop Runtime ", limit);
        frame_style(frame, frame->reset);
        frame_text(frame, state->runtime_action == 1u ? "  › " : "    ", limit);
        frame_style(frame, state->runtime_action == 1u &&
                               yvex_tui_startup_model_count(state)
                           ? frame->selected : frame->dim);
        frame_text(frame, yvex_tui_startup_model_count(state)
                              ? " Restart Runtime… " : " Restart needs model profile ", limit);
        frame_style(frame, frame->reset);
    }
    if (row + 1u <= last && state->event_count) {
        size_t index, shown = state->event_count < (size_t)(last - row) ?
                                  state->event_count : (size_t)(last - row);
        frame_begin_line(frame, row++);
        frame_style(frame, frame->violet);
        frame_text(frame, "  RECENT STRUCTURED EVENTS", limit);
        for (index = state->event_count - shown; index < state->event_count && row <= last;
             ++index, ++row) {
            size_t slot = (state->event_start + index) % YVEX_TUI_EVENT_CAP;
            const yvex_server_event *event = &state->events[slot];
            frame_begin_line(frame, row);
            (void)snprintf(value, sizeof(value), "  %6llu  %-30s  %s",
                           event->sequence, yvex_server_event_kind_name(event->kind),
                           event->phase);
            frame_text(frame, value, limit);
        }
    }
}

static int folded_contains(const char *text, const char *query)
{
    size_t start, offset, text_count = strlen(text), query_count = strlen(query);
    if (!query_count) return 1;
    for (start = 0u; start + query_count <= text_count; ++start) {
        for (offset = 0u; offset < query_count; ++offset) {
            unsigned char left = (unsigned char)text[start + offset];
            unsigned char right = (unsigned char)query[offset];
            if (left >= 'A' && left <= 'Z') left = (unsigned char)(left + 32u);
            if (right >= 'A' && right <= 'Z') right = (unsigned char)(right + 32u);
            if (left != right) break;
        }
        if (offset == query_count) return 1;
    }
    return 0;
}

static void render_palette(tui_frame *frame, const yvex_tui_state *state,
                           unsigned int first, unsigned int last)
{
    const char *query = (const char *)state->command.bytes;
    size_t index;
    unsigned int row = first, limit = frame->columns > 1u ? frame->columns - 1u : 1u;
    if (query[0] == '/') query++;
    frame_begin_line(frame, row++);
    frame_style(frame, frame->violet);
    frame_text(frame, "  COMMAND PALETTE", limit);
    frame_style(frame, frame->dim);
    frame_text(frame, "  registry-owned operations", limit);
    for (index = 0u; index < yvex_operator_descriptor_count && row <= last; ++index) {
        const yvex_operator_descriptor *descriptor = &yvex_operator_descriptors[index];
        int unavailable;
        if (!strcmp(descriptor->slash_projection, "none") ||
            descriptor->visibility == YVEX_OPERATOR_VISIBILITY_REMOVED ||
            (!folded_contains(descriptor->slash_projection, query) &&
             !folded_contains(descriptor->summary, query)))
            continue;
        unavailable = !strcmp(descriptor->daemon_requirement, "required") &&
                      state->connection != YVEX_TUI_CONNECTION_CONNECTED;
        frame_begin_line(frame, row++);
        frame_style(frame, unavailable ? frame->dim : frame->accent);
        frame_text(frame, "  ", limit);
        frame_text(frame, descriptor->slash_projection, limit);
        frame_to_column(frame, 20u, limit);
        frame_style(frame, frame->reset);
        frame_text(frame, descriptor->summary, limit);
        if (unavailable) {
            frame_style(frame, frame->warning);
            frame_text(frame, "  [runtime unavailable]", limit);
        }
    }
    if (row == first + 1u && row <= last) {
        frame_begin_line(frame, row);
        frame_style(frame, frame->dim);
        frame_text(frame, "  No contextual operation matches", limit);
    }
}

static void launch_option(tui_frame *frame, unsigned int row,
                          const yvex_tui_state *state, unsigned int field,
                          const char *label, const char *value,
                          unsigned int limit)
{
    frame_begin_line(frame, row);
    frame_text(frame, "      ", limit);
    frame_style(frame, state->launch_field == field ? frame->accent : frame->dim);
    frame_text(frame, state->launch_field == field ? "› " : "  ", limit);
    frame_text(frame, label, limit);
    frame_to_column(frame, 28u, limit);
    frame_style(frame, state->launch_field == field ? frame->selected : frame->reset);
    frame_text(frame, " ", limit);
    frame_text(frame, value, limit);
    frame_text(frame, " ", limit);
    frame_style(frame, frame->reset);
}

static void render_launch_overlay(tui_frame *frame, const yvex_tui_state *state,
                                  unsigned int first, unsigned int last)
{
    const yvex_tui_model_row *model = launch_model(state);
    const yvex_model_runtime_profile_fact *profile = yvex_tui_launch_profile(state);
    int selecting_model = state->runtime_available && state->active_engine.alias[0] &&
                          !state->restart_pending;
    unsigned int row = first, limit = frame->columns > 1u ? frame->columns - 1u : 1u;
    frame_begin_line(frame, row++);
    frame_style(frame, frame->violet);
    frame_text(frame, state->restart_pending ? "  RESTART RUNTIME"
                          : selecting_model ? "  SELECT MODEL"
                                            : "  START RUNTIME", limit);
    frame_style(frame, frame->dim);
    frame_text(frame, selecting_model ? "  canonical model and runtime profile"
                                      : "  separate canonical server process", limit);
    if (row <= last) {
        frame_begin_line(frame, row++);
        frame_text(frame, "  ", limit);
        frame_rule_until(frame, limit > 3u ? limit - 2u : limit);
    }
    if (row < last) row++;
    if (model && row <= last)
        launch_option(frame, row++, state, 0u, "Model", model->display_name, limit);
    if (profile && row <= last)
        launch_option(frame, row++, state, 1u, "Profile", profile->alias, limit);
    if (row < last) row++;
    if (row <= last)
        launch_option(frame, row++, state, 2u, "Action",
                      state->restart_pending ? "Stop then launch"
                          : selecting_model ? "Load Model" : "Start Runtime", limit);
    if (row < last) row++;
    if (row <= last) {
        frame_begin_line(frame, row++);
        frame_style(frame, frame->dim);
        frame_text(frame, "      Tab fields   ↑↓ model/profile   Enter start   Esc cancel", limit);
    }
    if (row <= last) {
        frame_begin_line(frame, row);
        frame_style(frame, frame->dim);
        frame_text(frame, "      Readiness is established only by protocol handshake.", limit);
    }
}

static void render_help(tui_frame *frame, unsigned int first, unsigned int last)
{
    static const char *const lines[] = {
        "  CONTEXTUAL HELP",
        "",
        "  Tab / Shift-Tab    move between Home, Models, Sessions, Runtime",
        "  Up / Down or j/k   move the active list or activity viewport",
        "  Enter              inspect selection or submit the composer",
        "  Ctrl-J             insert a composer newline",
        "  Ctrl-O             choose a model/runtime profile from Home",
        "  /                  search Models or discover registry operations",
        "  Ctrl-P             open command palette",
        "  Esc                close context or return",
        "  Ctrl-C             cancel active work, clear draft, or exit when idle",
        "  Ctrl-L / r         refresh authoritative snapshots",
    };
    unsigned int row = first, limit = frame->columns > 1u ? frame->columns - 1u : 1u;
    size_t index;
    for (index = 0u; index < sizeof(lines) / sizeof(lines[0]) && row <= last;
         ++index, ++row) {
        frame_begin_line(frame, row);
        frame_style(frame, index == 0u ? frame->violet : frame->reset);
        frame_text(frame, lines[index], limit);
    }
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

static size_t next_boundary(const unsigned char *bytes, size_t count, size_t cursor)
{
    if (cursor >= count) return count;
    cursor++;
    while (cursor < count && (bytes[cursor] & 0xc0u) == 0x80u) cursor++;
    return cursor;
}

static void render_composer(tui_frame *frame, const yvex_tui_state *state,
                            unsigned int row, unsigned int *cursor_column)
{
    const unsigned char *bytes = state->composer.bytes;
    size_t count = state->composer.count, cursor = state->composer.cursor;
    size_t line_start = cursor, line_end = cursor, visible_start;
    const char *prefix = "  › ", *placeholder = "Ask YVEX anything";
    int editable = 1;
    unsigned int prefix_width;
    unsigned int limit = frame->columns > 1u ? frame->columns - 1u : 1u;
    unsigned int available;
    if (state->overlay == YVEX_TUI_OVERLAY_PALETTE) {
        bytes = state->command.bytes;
        count = state->command.count;
        cursor = state->command.cursor;
        prefix = "  : ";
        placeholder = "Search canonical operator operations";
    } else if (state->overlay == YVEX_TUI_OVERLAY_RUNTIME_LAUNCH) {
        bytes = (const unsigned char *)"";
        count = 0u;
        cursor = 0u;
        editable = 0;
        placeholder = "Configure launch above · Enter advances";
    } else if (state->surface == YVEX_TUI_SURFACE_MODELS ||
               state->surface == YVEX_TUI_SURFACE_MODEL_DETAIL) {
        int discover = state->surface == YVEX_TUI_SURFACE_MODELS &&
                       state->models_mode == YVEX_TUI_MODELS_DISCOVER;
        bytes = (const unsigned char *)(discover ? state->discover_query
                                                 : state->model_search);
        count = discover ? state->discover_query_count : state->model_search_count;
        cursor = count;
        prefix = "  / ";
        editable = discover || state->model_catalog_status != YVEX_TUI_MODEL_CATALOG_ERROR;
        placeholder = discover ? "Search Hugging Face models"
                      : editable ? "Search canonical models"
                               : "Registry unavailable · r retry";
    } else if (state->surface == YVEX_TUI_SURFACE_SESSIONS) {
        bytes = (const unsigned char *)state->session_search;
        count = state->session_search_count;
        cursor = count;
        prefix = "  / ";
        placeholder = state->runtime_available && state->active_engine.alias[0]
                          ? "Search server sessions"
                      : state->runtime_available ? "Enter to load a model"
                                                 : "Enter to start runtime";
        editable = state->runtime_available && state->active_engine.alias[0];
    } else if (state->surface == YVEX_TUI_SURFACE_RUNTIME) {
        bytes = (const unsigned char *)"";
        count = cursor = 0u;
        editable = 0;
        placeholder = state->runtime_available && state->active_engine.alias[0]
                          ? "←→ lifecycle action · Enter apply"
                      : state->runtime_available
                          ? "Enter  Load Model"
                      : state->model_catalog_status == YVEX_TUI_MODEL_CATALOG_ERROR
                          ? "Registry unavailable · r retry"
                          : "Enter  Start Runtime";
    } else if (!state->runtime_available || !state->active_engine.alias[0]) {
        bytes = (const unsigned char *)"";
        count = cursor = 0u;
        editable = 0;
        placeholder = state->model_catalog_status == YVEX_TUI_MODEL_CATALOG_ERROR
                          ? "Registry unavailable · r retry"
                      : yvex_tui_startup_model_count(state)
                          ? state->runtime_available ? "Enter  Load Model"
                                                     : "Enter  Start Runtime"
                          : "No startup-ready model · Tab to Models";
    }
    prefix_width = text_width(prefix, strlen(prefix));
    available = limit > prefix_width ? limit - prefix_width : 1u;
    while (line_start && bytes[line_start - 1u] != '\n') line_start--;
    while (line_end < count && bytes[line_end] != '\n') line_end++;
    visible_start = line_start;
    while (visible_start < cursor &&
           text_width((const char *)bytes + visible_start, cursor - visible_start) >= available - 1u)
        visible_start = next_boundary(bytes, cursor, visible_start);
    frame_begin_line(frame, row);
    frame_style(frame, editable ? frame->accent : frame->dim);
    frame_text(frame, prefix, limit);
    frame_style(frame, frame->reset);
    if (visible_start > line_start) {
        frame_text(frame, frame->unicode ? "…" : "<", limit);
        prefix_width++;
    }
    frame_text_n(frame, (const char *)bytes + visible_start,
                 line_end - visible_start, limit);
    if (!count) {
        frame_style(frame, frame->dim);
        frame_text(frame, placeholder, limit);
        frame_style(frame, frame->reset);
    }
    *cursor_column = prefix_width +
                     text_width((const char *)bytes + visible_start,
                                cursor - visible_start) + 1u;
    if (*cursor_column > limit) *cursor_column = limit;
}

static void render_composer_rule(tui_frame *frame, const yvex_tui_state *state,
                                 unsigned int row)
{
    const yvex_server_engine_summary *engine = active_engine(state);
    const char *label = "COMPOSE", *context = state->active_session;
    char home_context[512];
    unsigned int limit = frame->columns > 1u ? frame->columns - 1u : 1u;
    unsigned int context_width, right;
    if (state->overlay == YVEX_TUI_OVERLAY_PALETTE) {
        label = "COMMAND";
        context = "operator registry";
    } else if (state->overlay == YVEX_TUI_OVERLAY_RUNTIME_LAUNCH) {
        label = "LAUNCH";
        context = "canonical server host";
    } else if (state->surface == YVEX_TUI_SURFACE_MODELS ||
               state->surface == YVEX_TUI_SURFACE_MODEL_DETAIL) {
        int discover = state->surface == YVEX_TUI_SURFACE_MODELS &&
                       state->models_mode == YVEX_TUI_MODELS_DISCOVER;
        label = discover ? "REMOTE SEARCH" : "MODEL SEARCH";
        context = discover ? "Hugging Face"
                  : state->model_catalog_status == YVEX_TUI_MODEL_CATALOG_ERROR
                      ? "registry unavailable" : "local registry";
    } else if (state->surface == YVEX_TUI_SURFACE_SESSIONS) {
        label = "SESSION SEARCH";
        context = state->runtime_available ? "server owned" : "runtime offline";
    } else if (state->surface == YVEX_TUI_SURFACE_RUNTIME) {
        label = "RUNTIME ACTION";
        context = lifecycle_name(state);
    } else if (!state->runtime_available || !state->active_engine.alias[0]) {
        label = "RUNTIME ACTION";
        context = lifecycle_name(state);
    } else if (state->surface == YVEX_TUI_SURFACE_HOME && engine) {
        label = "";
        (void)snprintf(home_context, sizeof(home_context), "%s · %s%s%s",
                       engine->alias, state->active_session,
                       state->generation_active ? " · " : "",
                       state->generation_active ? phase_name(state->generation_phase) : "");
        context = home_context;
    }
    context_width = text_width(context, strlen(context));
    right = limit > context_width + 2u ? limit - context_width : limit;
    frame_begin_line(frame, row);
    frame_text(frame, "  ", limit);
    if (label[0]) {
        frame_style(frame, frame->accent);
        frame_text(frame, label, limit);
        frame_text(frame, " ", limit);
    }
    frame_rule_until(frame, right > 2u ? right - 2u : right);
    if (right > frame->column) frame_to_column(frame, right, limit);
    frame_style(frame, state->generation_active ? frame->violet : frame->dim);
    frame_text(frame, context, limit);
    frame_style(frame, frame->reset);
}

static void render_footer(tui_frame *frame, const yvex_tui_state *state,
                          unsigned int row)
{
    unsigned int limit = frame->columns > 1u ? frame->columns - 1u : 1u;
    const char *hint = state->surface == YVEX_TUI_SURFACE_MODELS
                           ? (state->models_mode == YVEX_TUI_MODELS_DISCOVER
                                  ? "l library   / query   Enter inspect/search   a acquire"
                              : state->model_catalog_status == YVEX_TUI_MODEL_CATALOG_ERROR
                                  ? "r retry   Tab navigate   ? help"
                                  : "↑↓ select   / search   Enter inspect   d discover")
                       : state->surface == YVEX_TUI_SURFACE_SESSIONS
                           ? "↑↓ select   / search   Enter attach   Ctrl-P commands"
                       : state->surface == YVEX_TUI_SURFACE_RUNTIME
                           ? "←→ action   Enter apply   r refresh   ? help"
                           : !state->runtime_available || !state->active_engine.alias[0]
                               ? "Enter start/load   Tab navigate   Ctrl-P commands   ? help"
                               : "Enter send   Ctrl-O model   Ctrl-P commands   Tab views";
    char status[256];
    const char *status_style;
    unsigned int hint_width = text_width(hint, strlen(hint));
    unsigned int status_width;
    if (state->notice[0])
        (void)snprintf(status, sizeof(status), "%s", state->notice);
    else
        (void)snprintf(status, sizeof(status), "%s  %s%s%s",
                       lifecycle_name(state),
                       state->runtime_available ? state->active_session : "",
                       state->generation_active ? "  " : "",
                       state->generation_active ? phase_name(state->generation_phase) : "");
    status_style = state->notice[0]
                       ? state->notice_severity == YVEX_TUI_SEVERITY_ERROR
                             ? frame->error
                         : state->notice_severity == YVEX_TUI_SEVERITY_WARNING
                             ? frame->warning : frame->dim
                       : state->connection == YVEX_TUI_CONNECTION_CONNECTED
                             ? frame->success : frame->warning;
    status_width = text_width(status, strlen(status));
    frame_begin_line(frame, row);
    frame_text(frame, "  ", limit);
    frame_style(frame, status_style);
    frame_text(frame, status, limit);
    frame_style(frame, frame->dim);
    if (hint_width + status_width + 6u < limit) {
        frame_to_column(frame, limit - hint_width, limit);
        frame_text(frame, hint, limit);
    } else if (hint_width + 2u < limit) {
        frame_begin_line(frame, row);
        frame_text(frame, "  ", limit);
        frame_text(frame, hint, limit);
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
    /* Reverse video follows the operator's foreground/background theme. */
    frame->selected = color ? "\033[1;7m" : "";
}

int yvex_tui_render(const yvex_tui_state *state, char *output, size_t capacity,
                    size_t *count, unsigned int *cursor_row,
                    unsigned int *cursor_column)
{
    tui_frame frame;
    unsigned int first = 3u, last, composer_row, footer_row;
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
        /* ED also inherits the active background; reset to terminal defaults first. */
        static const char begin[] = "\033[0m\033[?25l\033[H\033[2J";
        frame_raw(&frame, begin, sizeof(begin) - 1u);
    }
    if (state->terminal.rows < 6u || frame.columns < 20u) {
        composer_row = state->terminal.rows > 1u ? state->terminal.rows - 1u : 1u;
        footer_row = state->terminal.rows ? state->terminal.rows : 1u;
        render_top(&frame, state);
        if (composer_row > 1u)
            render_composer_rule(&frame, state, composer_row - 1u);
        render_composer(&frame, state, composer_row, cursor_column);
        render_footer(&frame, state, footer_row);
        *cursor_row = composer_row;
    } else {
        composer_row = state->terminal.rows - 1u;
        footer_row = state->terminal.rows;
        last = state->terminal.rows - 3u;
        render_top(&frame, state);
        frame_rule(&frame, 2u);
        if (state->surface == YVEX_TUI_SURFACE_MODELS)
            render_models(&frame, state, first, last);
        else if (state->surface == YVEX_TUI_SURFACE_MODEL_DETAIL)
            render_model_detail(&frame, state, first, last);
        else if (state->surface == YVEX_TUI_SURFACE_SESSIONS)
            render_sessions(&frame, state, first, last);
        else if (state->surface == YVEX_TUI_SURFACE_RUNTIME)
            render_runtime(&frame, state, first, last);
        else
            render_home(&frame, state, first, last);
        if (state->overlay == YVEX_TUI_OVERLAY_PALETTE)
            render_palette(&frame, state, first, last);
        else if (state->overlay == YVEX_TUI_OVERLAY_HELP)
            render_help(&frame, first, last);
        else if (state->overlay == YVEX_TUI_OVERLAY_RUNTIME_LAUNCH)
            render_launch_overlay(&frame, state, first, last);
        render_composer_rule(&frame, state, state->terminal.rows - 2u);
        render_composer(&frame, state, composer_row, cursor_column);
        render_footer(&frame, state, footer_row);
        *cursor_row = composer_row;
    }
    frame_style(&frame, frame.reset);
    frame_format(&frame, "\033[%u;%uH\033[?25h", *cursor_row, *cursor_column);
    if (frame.failed) return YVEX_ERR_BOUNDS;
    *count = frame.count;
    return YVEX_OK;
}
