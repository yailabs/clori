/* Own deterministic client-side state transitions and bounded projection storage. */
#include "src/cli/tui/private.h"

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static yvex_tui_layout layout_for(unsigned int columns)
{
    if (columns < 92u) return YVEX_TUI_LAYOUT_COMPACT;
    if (columns < 132u) return YVEX_TUI_LAYOUT_STANDARD;
    return YVEX_TUI_LAYOUT_WIDE;
}

static void text_copy(char *output, size_t capacity, const char *input)
{
    size_t count;
    if (!capacity) return;
    if (!input) input = "";
    count = strlen(input);
    if (count >= capacity) count = capacity - 1u;
    while (count && ((unsigned char)input[count] & 0xc0u) == 0x80u) count--;
    memcpy(output, input, count);
    output[count] = '\0';
}

static size_t utf8_previous(const unsigned char *bytes, size_t cursor)
{
    if (!cursor) return 0u;
    cursor--;
    while (cursor && (bytes[cursor] & 0xc0u) == 0x80u) cursor--;
    return cursor;
}

static size_t utf8_next(const unsigned char *bytes, size_t count, size_t cursor)
{
    size_t width;
    if (cursor >= count) return count;
    if (bytes[cursor] < 0x80u) return cursor + 1u;
    if ((bytes[cursor] & 0xe0u) == 0xc0u) width = 2u;
    else if ((bytes[cursor] & 0xf0u) == 0xe0u) width = 3u;
    else if ((bytes[cursor] & 0xf8u) == 0xf0u) width = 4u;
    else width = 1u;
    return cursor + width <= count ? cursor + width : count;
}

static int utf8_valid(const unsigned char *bytes, size_t count)
{
    size_t cursor = 0u;
    while (cursor < count) {
        unsigned char first = bytes[cursor];
        size_t width, index;
        unsigned int value;
        if (first < 0x80u) {
            cursor++;
            continue;
        }
        if (first >= 0xc2u && first <= 0xdfu) {
            width = 2u;
            value = first & 0x1fu;
        } else if (first >= 0xe0u && first <= 0xefu) {
            width = 3u;
            value = first & 0x0fu;
        } else if (first >= 0xf0u && first <= 0xf4u) {
            width = 4u;
            value = first & 0x07u;
        } else {
            return 0;
        }
        if (cursor + width > count) return 0;
        for (index = 1u; index < width; ++index) {
            if ((bytes[cursor + index] & 0xc0u) != 0x80u) return 0;
            value = (value << 6u) | (bytes[cursor + index] & 0x3fu);
        }
        if ((width == 3u && value < 0x800u) ||
            (width == 4u && value < 0x10000u) ||
            value > 0x10ffffu || (value >= 0xd800u && value <= 0xdfffu))
            return 0;
        cursor += width;
    }
    return 1;
}

static void composer_multiline(yvex_tui_composer *composer)
{
    composer->multiline = memchr(composer->bytes, '\n', composer->count) != NULL;
}

void yvex_tui_state_init(yvex_tui_state *state, unsigned int rows,
                         unsigned int columns, const char *session)
{
    const char *language;
    if (!state) return;
    memset(state, 0, sizeof(*state));
    state->terminal.rows = rows;
    state->terminal.columns = columns;
    state->terminal.layout = layout_for(columns);
    state->terminal.color = getenv("NO_COLOR") == NULL;
    language = getenv("LC_ALL");
    if (!language || !language[0]) language = getenv("LC_CTYPE");
    if (!language || !language[0]) language = getenv("LANG");
    state->terminal.unicode = language &&
        (strstr(language, "UTF-8") || strstr(language, "utf8") ||
         strstr(language, "UTF8"));
    state->surface = YVEX_TUI_SURFACE_HOME;
    state->prior_surface = YVEX_TUI_SURFACE_HOME;
    state->focus = YVEX_TUI_FOCUS_COMPOSER;
    state->connection = YVEX_TUI_CONNECTION_UNAVAILABLE;
    state->runtime_lifecycle = YVEX_TUI_RUNTIME_NONE;
    state->model_catalog_status = YVEX_TUI_MODEL_CATALOG_UNLOADED;
    state->reasoning_policy = YVEX_REASONING_DISABLED;
    state->generation_phase = YVEX_CLIENT_PHASE_UNAVAILABLE;
    state->next_activity_order = 1u;
    state->redraw = 1;
    text_copy(state->active_session, sizeof(state->active_session),
              session && session[0] ? session : "main");
    yvex_tui_activity_add(state, YVEX_TUI_ACTIVITY_SYSTEM,
                          YVEX_TUI_SEVERITY_INFO,
                          YVEX_CLIENT_STREAM_CONTROL_EVENT,
                          "Official terminal client started");
}

void yvex_tui_state_close(yvex_tui_state *state)
{
    if (!state) return;
    free(state->models);
    yvex_model_library_close(state->model_library);
    state->models = NULL;
    state->model_library = NULL;
    state->model_count = 0u;
    state->model_capacity = 0u;
}

void yvex_tui_state_resize(yvex_tui_state *state, unsigned int rows,
                           unsigned int columns)
{
    if (!state) return;
    state->terminal.rows = rows;
    state->terminal.columns = columns;
    state->terminal.layout = layout_for(columns);
    state->redraw = 1;
}

void yvex_tui_activity_add(yvex_tui_state *state, yvex_tui_activity_kind kind,
                           yvex_tui_severity severity,
                           yvex_client_stream_channel channel,
                           const char *text)
{
    size_t slot;
    yvex_tui_activity *activity;
    if (!state) return;
    if (state->activity_count == YVEX_TUI_ACTIVITY_CAP) {
        state->activity_start = (state->activity_start + 1u) % YVEX_TUI_ACTIVITY_CAP;
        state->activity_count--;
    }
    slot = (state->activity_start + state->activity_count) % YVEX_TUI_ACTIVITY_CAP;
    activity = &state->activities[slot];
    memset(activity, 0, sizeof(*activity));
    activity->kind = kind;
    activity->severity = severity;
    activity->channel = channel;
    activity->order = state->next_activity_order++;
    text_copy(activity->session, sizeof(activity->session), state->active_session);
    text_copy(activity->text, sizeof(activity->text), text);
    state->activity_count++;
    state->activity_scroll = 0u;
    state->redraw = 1;
}

void yvex_tui_state_connection(yvex_tui_state *state,
                               yvex_tui_connection connection,
                               const char *reason)
{
    size_t model_index;
    int changed;
    if (!state) return;
    changed = state->connection != connection ||
              strcmp(state->connection_reason, reason ? reason : "") != 0;
    state->connection = connection;
    text_copy(state->connection_reason, sizeof(state->connection_reason), reason);
    if (connection == YVEX_TUI_CONNECTION_CONNECTED) {
        if (state->runtime_lifecycle == YVEX_TUI_RUNTIME_LAUNCHING ||
            state->runtime_lifecycle == YVEX_TUI_RUNTIME_WAITING_PROTOCOL)
            state->runtime_lifecycle = YVEX_TUI_RUNTIME_CONNECTED_OWNED;
        else if (state->runtime_lifecycle != YVEX_TUI_RUNTIME_ENGINE_LOADING &&
                 state->runtime_lifecycle != YVEX_TUI_RUNTIME_CONNECTED_OWNED)
            state->runtime_lifecycle = YVEX_TUI_RUNTIME_CONNECTED_EXTERNAL;
    } else {
        state->generation_active = 0;
        state->generation_phase = YVEX_CLIENT_PHASE_UNAVAILABLE;
        state->runtime_available = 0;
        state->console_available = 0;
        state->engine_count = 0u;
        memset(&state->active_engine, 0, sizeof(state->active_engine));
        for (model_index = 0u; model_index < state->model_count; ++model_index) {
            state->models[model_index].resident = 0;
            state->models[model_index].runtime_count = 0u;
        }
        if (connection == YVEX_TUI_CONNECTION_INCOMPATIBLE &&
            (state->runtime_lifecycle == YVEX_TUI_RUNTIME_LAUNCHING ||
             state->runtime_lifecycle == YVEX_TUI_RUNTIME_WAITING_PROTOCOL))
            yvex_tui_runtime_launch_failed(
                state, YVEX_TUI_LAUNCH_FAILURE_PROTOCOL,
                reason && reason[0] ? reason : "local protocol handshake was incompatible",
                0, 0, NULL);
        else if (state->runtime_lifecycle == YVEX_TUI_RUNTIME_SHUTDOWN_REQUESTED)
            state->runtime_lifecycle = YVEX_TUI_RUNTIME_STOPPED;
        else if (state->runtime_lifecycle != YVEX_TUI_RUNTIME_LAUNCH_REQUESTED &&
                 state->runtime_lifecycle != YVEX_TUI_RUNTIME_LAUNCHING &&
                 state->runtime_lifecycle != YVEX_TUI_RUNTIME_WAITING_PROTOCOL &&
                 state->runtime_lifecycle != YVEX_TUI_RUNTIME_LAUNCH_FAILED)
            state->runtime_lifecycle = YVEX_TUI_RUNTIME_NONE;
    }
    if (changed) {
        const char *message = connection == YVEX_TUI_CONNECTION_CONNECTED
                                  ? "Runtime protocol handshake accepted"
                              : connection == YVEX_TUI_CONNECTION_INCOMPATIBLE
                                  ? "Resident runtime protocol is incompatible"
                              : state->runtime_lifecycle == YVEX_TUI_RUNTIME_WAITING_PROTOCOL
                                  ? "Server process running; waiting for local protocol"
                                  : "Resident runtime unavailable; draft preserved";
        yvex_tui_activity_add(state, YVEX_TUI_ACTIVITY_RUNTIME,
                              connection == YVEX_TUI_CONNECTION_CONNECTED
                                  ? YVEX_TUI_SEVERITY_SUCCESS
                                  : YVEX_TUI_SEVERITY_WARNING,
                              YVEX_CLIENT_STREAM_CONTROL_EVENT, message);
    }
    state->redraw = 1;
}

static void runtime_summary(yvex_tui_state *state,
                            const yvex_server_summary *runtime)
{
    state->runtime = *runtime;
    state->runtime_available = 1;
    if (state->runtime_lifecycle != YVEX_TUI_RUNTIME_CONNECTED_OWNED)
        state->runtime_lifecycle = YVEX_TUI_RUNTIME_CONNECTED_EXTERNAL;
    state->launch_exit_status = 0;
    state->launch_exec_error = 0;
    state->launch_failure = YVEX_TUI_LAUNCH_FAILURE_NONE;
    state->launch_failure_reason[0] = '\0';
    state->launch_diagnostic[0] = '\0';
}

int yvex_tui_composer_insert(yvex_tui_composer *composer,
                             const unsigned char *bytes, size_t count)
{
    if (!composer || (!bytes && count) ||
        count >= YVEX_TUI_COMPOSER_CAP - composer->count ||
        !utf8_valid(bytes, count))
        return 0;
    memmove(composer->bytes + composer->cursor + count,
            composer->bytes + composer->cursor,
            composer->count - composer->cursor);
    memcpy(composer->bytes + composer->cursor, bytes, count);
    composer->cursor += count;
    composer->count += count;
    composer->bytes[composer->count] = '\0';
    composer_multiline(composer);
    return 1;
}

void yvex_tui_composer_left(yvex_tui_composer *composer)
{
    if (composer) composer->cursor = utf8_previous(composer->bytes, composer->cursor);
}

void yvex_tui_composer_right(yvex_tui_composer *composer)
{
    if (composer)
        composer->cursor = utf8_next(composer->bytes, composer->count,
                                     composer->cursor);
}

void yvex_tui_composer_home(yvex_tui_composer *composer)
{
    if (!composer) return;
    while (composer->cursor && composer->bytes[composer->cursor - 1u] != '\n')
        composer->cursor = utf8_previous(composer->bytes, composer->cursor);
}

void yvex_tui_composer_end(yvex_tui_composer *composer)
{
    if (!composer) return;
    while (composer->cursor < composer->count &&
           composer->bytes[composer->cursor] != '\n')
        composer->cursor = utf8_next(composer->bytes, composer->count,
                                     composer->cursor);
}

void yvex_tui_composer_erase(yvex_tui_composer *composer, int backward)
{
    size_t begin, end;
    if (!composer) return;
    begin = composer->cursor;
    end = composer->cursor;
    if (backward && begin) begin = utf8_previous(composer->bytes, begin);
    else if (!backward && end < composer->count)
        end = utf8_next(composer->bytes, composer->count, end);
    if (begin == end) return;
    memmove(composer->bytes + begin, composer->bytes + end,
            composer->count - end);
    composer->count -= end - begin;
    composer->cursor = begin;
    composer->bytes[composer->count] = '\0';
    composer_multiline(composer);
}

void yvex_tui_composer_clear(yvex_tui_composer *composer)
{
    if (!composer) return;
    composer->count = 0u;
    composer->cursor = 0u;
    composer->multiline = 0;
    composer->bytes[0] = '\0';
    composer->history_cursor = composer->history_count;
}

void yvex_tui_composer_history_push(yvex_tui_composer *composer)
{
    size_t slot, count;
    if (!composer || !composer->count) return;
    if (composer->history_count &&
        composer->count < YVEX_TUI_HISTORY_TEXT_CAP &&
        !memcmp(composer->history[composer->history_count - 1u],
                composer->bytes, composer->count) &&
        composer->history[composer->history_count - 1u][composer->count] == '\0')
        return;
    if (composer->history_count == YVEX_TUI_HISTORY_CAP) {
        memmove(composer->history, composer->history + 1,
                (YVEX_TUI_HISTORY_CAP - 1u) * sizeof(composer->history[0]));
        composer->history_count--;
    }
    slot = composer->history_count++;
    count = composer->count;
    if (count >= YVEX_TUI_HISTORY_TEXT_CAP) count = YVEX_TUI_HISTORY_TEXT_CAP - 1u;
    while (count && (composer->bytes[count] & 0xc0u) == 0x80u) count--;
    memcpy(composer->history[slot], composer->bytes, count);
    composer->history[slot][count] = '\0';
    composer->history_cursor = composer->history_count;
}

void yvex_tui_composer_history_move(yvex_tui_composer *composer, int direction)
{
    const char *entry;
    size_t count;
    if (!composer || !composer->history_count) return;
    if (direction < 0 && composer->history_cursor)
        composer->history_cursor--;
    else if (direction > 0 && composer->history_cursor < composer->history_count)
        composer->history_cursor++;
    if (composer->history_cursor == composer->history_count) {
        yvex_tui_composer_clear(composer);
        return;
    }
    entry = composer->history[composer->history_cursor];
    count = strlen(entry);
    memcpy(composer->bytes, entry, count + 1u);
    composer->count = count;
    composer->cursor = count;
    composer_multiline(composer);
}

static yvex_tui_session_row *session_upsert(yvex_tui_state *state,
                                            const char *name)
{
    size_t index;
    for (index = 0u; index < state->session_count; ++index)
        if (!strcmp(state->sessions[index].name, name)) return &state->sessions[index];
    if (state->session_count == YVEX_TUI_SESSION_CAP) return NULL;
    memset(&state->sessions[state->session_count], 0,
           sizeof(state->sessions[state->session_count]));
    text_copy(state->sessions[state->session_count].name,
              sizeof(state->sessions[state->session_count].name), name);
    return &state->sessions[state->session_count++];
}

static void session_message(yvex_tui_state *state,
                            const yvex_client_message *message)
{
    yvex_tui_session_row *row = session_upsert(state, message->session_name);
    if (!row) return;
    row->state = message->session_state;
    row->position = message->final_position;
    row->turns = message->turn_count;
    row->context_used = message->context_used;
    row->kv_used_bytes = message->kv_used_bytes;
    row->kv_used_available = message->kv_used_available;
}

static void console_message(yvex_tui_state *state,
                            const yvex_client_message *message)
{
    yvex_tui_session_row *row;
    runtime_summary(state, &message->runtime);
    state->console = message->console;
    state->partial_turn = message->partial_turn;
    state->console_available = 1;
    state->reasoning_policy = message->console.reasoning_policy;
    state->generation_phase = message->console.generation_phase;
    state->generation_active = message->console.generation_phase >= YVEX_CLIENT_PHASE_QUEUED &&
                               message->console.generation_phase <= YVEX_CLIENT_PHASE_DECODE;
    if (message->console.session_name[0]) {
        row = session_upsert(state, message->console.session_name);
        if (row) {
            row->state = message->console.session_state;
            row->position = message->console.position;
            row->turns = message->console.turn_count;
            row->context_used = message->console.context_used;
            row->kv_used_bytes = message->console.kv_used_bytes;
            row->kv_used_available = message->console.kv_used_available;
            row->attached = message->console.attached;
        }
    }
}

static void event_message(yvex_tui_state *state, const yvex_server_event *event)
{
    size_t slot;
    const char *name = yvex_server_event_kind_name(event->kind);
    if (state->event_count == YVEX_TUI_EVENT_CAP) {
        state->event_start = (state->event_start + 1u) % YVEX_TUI_EVENT_CAP;
        state->event_count--;
    }
    slot = (state->event_start + state->event_count) % YVEX_TUI_EVENT_CAP;
    state->events[slot] = *event;
    state->event_count++;
    if (event->kind == YVEX_SERVER_EVENT_PROCESS_START && state->launched_pid > 0 &&
        event->process_id == (unsigned long long)state->launched_pid)
        state->runtime_lifecycle = YVEX_TUI_RUNTIME_CONNECTED_OWNED;
    if (event->kind == YVEX_SERVER_EVENT_REQUEST_QUEUED)
        state->generation_phase = YVEX_CLIENT_PHASE_QUEUED;
    else if (event->kind == YVEX_SERVER_EVENT_TOKENIZER_COMPLETED)
        state->generation_phase = YVEX_CLIENT_PHASE_TOKENIZING;
    else if (event->kind >= YVEX_SERVER_EVENT_PREFILL_STARTED &&
             event->kind <= YVEX_SERVER_EVENT_PREFILL_COMPLETED)
        state->generation_phase = YVEX_CLIENT_PHASE_PREFILL;
    else if (event->kind >= YVEX_SERVER_EVENT_GENERATION_FIRST_TOKEN &&
             event->kind <= YVEX_SERVER_EVENT_GENERATION_PROFILE)
        state->generation_phase = YVEX_CLIENT_PHASE_DECODE;
    if (event->kind != YVEX_SERVER_EVENT_GENERATION_FRAGMENT &&
        event->kind != YVEX_SERVER_EVENT_GENERATION_PROGRESS &&
        event->kind != YVEX_SERVER_EVENT_PREFILL_PROGRESS)
        yvex_tui_activity_add(state, YVEX_TUI_ACTIVITY_RUNTIME,
                              event->severity >= YVEX_SERVER_SEVERITY_ERROR
                                  ? YVEX_TUI_SEVERITY_ERROR
                                  : YVEX_TUI_SEVERITY_INFO,
                              YVEX_CLIENT_STREAM_CONTROL_EVENT,
                              name ? name : "runtime event");
}

static void engine_message(yvex_tui_state *state,
                           const yvex_server_engine_summary *engine)
{
    size_t index, model_index;
    int was_loaded = 0, is_new;

    for (index = 0u; index < state->engine_count; ++index)
        if (strcmp(state->engines[index].alias, engine->alias) == 0) break;
    is_new = index == state->engine_count;
    if (!is_new) was_loaded = state->engines[index].state == YVEX_SERVER_ENGINE_LOADED;
    if (index == state->engine_count && state->engine_count < YVEX_TUI_ENGINE_CAP)
        state->engine_count++;
    if (index >= YVEX_TUI_ENGINE_CAP) return;
    state->engines[index] = *engine;
    if (engine->state == YVEX_SERVER_ENGINE_LOADED &&
        (!state->active_engine.alias[0] ||
         strcmp(state->launch_request.profile, engine->alias) == 0)) {
        text_copy(state->active_engine.alias, sizeof(state->active_engine.alias),
                  engine->alias);
        state->active_engine.generation = engine->generation;
        state->engine_load_requested = 0;
        state->runtime_lifecycle = state->launched_pid > 0
                                       ? YVEX_TUI_RUNTIME_CONNECTED_OWNED
                                       : YVEX_TUI_RUNTIME_CONNECTED_EXTERNAL;
    } else if (strcmp(state->active_engine.alias, engine->alias) == 0 &&
               state->active_engine.generation == engine->generation &&
               engine->state != YVEX_SERVER_ENGINE_LOADED) {
        memset(&state->active_engine, 0, sizeof(state->active_engine));
        state->console_available = 0;
    }
    for (model_index = 0u; model_index < state->model_count; ++model_index) {
        int matches = state->models[model_index].runtime_target[0] && engine->target_id[0] &&
                      strcmp(state->models[model_index].runtime_target, engine->target_id) == 0;
        if (matches && !was_loaded && engine->state == YVEX_SERVER_ENGINE_LOADED) {
            state->models[model_index].resident = 1;
            state->models[model_index].runtime_count++;
        } else if (matches && was_loaded && engine->state != YVEX_SERVER_ENGINE_LOADED &&
                   state->models[model_index].runtime_count) {
            state->models[model_index].runtime_count--;
            state->models[model_index].resident = state->models[model_index].runtime_count != 0u;
        }
    }
}

static void fragment_message(yvex_tui_state *state,
                             const yvex_client_message *message)
{
    yvex_tui_activity *activity = NULL;
    size_t slot, used, index;
    if (state->activity_count) {
        slot = (state->activity_start + state->activity_count - 1u) %
               YVEX_TUI_ACTIVITY_CAP;
        if (state->activities[slot].kind == YVEX_TUI_ACTIVITY_GENERATION &&
            state->activities[slot].channel == message->stream_channel)
            activity = &state->activities[slot];
    }
    if (!activity) {
        yvex_tui_activity_add(state, YVEX_TUI_ACTIVITY_GENERATION,
                              message->stream_channel == YVEX_CLIENT_STREAM_ERROR
                                  ? YVEX_TUI_SEVERITY_ERROR
                                  : YVEX_TUI_SEVERITY_INFO,
                              message->stream_channel, "");
        slot = (state->activity_start + state->activity_count - 1u) %
               YVEX_TUI_ACTIVITY_CAP;
        activity = &state->activities[slot];
    }
    used = strlen(activity->text);
    for (index = 0u; index < message->byte_count &&
                     used + 1u < sizeof(activity->text); ++index) {
        unsigned char byte = message->bytes[index];
        if (byte == '\033' || (byte < 0x20u && byte != '\n' && byte != '\t'))
            activity->text[used++] = '?';
        else
            activity->text[used++] = (char)byte;
    }
    activity->text[used] = '\0';
    state->redraw = 1;
}

void yvex_tui_state_message(yvex_tui_state *state,
                            const yvex_cli_interactive_event *event)
{
    const yvex_client_message *message;
    if (!state || !event) return;
    if (event->kind == YVEX_CLI_INTERACTIVE_CONNECTION) {
        yvex_tui_state_connection(state, event->connection, event->reason);
        return;
    }
    if (event->kind == YVEX_CLI_INTERACTIVE_FAILURE) {
        yvex_tui_activity_add(state, YVEX_TUI_ACTIVITY_ERROR,
                              YVEX_TUI_SEVERITY_ERROR,
                              YVEX_CLIENT_STREAM_ERROR, event->reason);
        return;
    }
    message = &event->message;
    if (message->kind == YVEX_CLIENT_MESSAGE_STATUS) {
        runtime_summary(state, &message->runtime);
    } else if (message->kind == YVEX_CLIENT_MESSAGE_ENGINE) {
        engine_message(state, &message->engine);
    } else if (message->kind == YVEX_CLIENT_MESSAGE_CONSOLE_STATUS) {
        console_message(state, message);
    } else if (message->kind == YVEX_CLIENT_MESSAGE_SESSION) {
        session_message(state, message);
    } else if (message->kind == YVEX_CLIENT_MESSAGE_EVENT) {
        event_message(state, &message->event);
    } else if (message->kind == YVEX_CLIENT_MESSAGE_TURN_STARTED) {
        state->generation_active = 1;
        state->generation_phase = message->generation_phase;
        yvex_tui_activity_add(state, YVEX_TUI_ACTIVITY_RUNTIME,
                              YVEX_TUI_SEVERITY_INFO,
                              YVEX_CLIENT_STREAM_CONTROL_EVENT,
                              "Generation accepted");
    } else if (message->kind == YVEX_CLIENT_MESSAGE_FRAGMENT) {
        fragment_message(state, message);
    } else if (message->kind == YVEX_CLIENT_MESSAGE_TURN_COMPLETE) {
        state->generation_active = 0;
        state->generation_phase = YVEX_CLIENT_PHASE_COMPLETE;
        session_message(state, message);
        yvex_tui_activity_add(state, YVEX_TUI_ACTIVITY_RUNTIME,
                              YVEX_TUI_SEVERITY_SUCCESS,
                              YVEX_CLIENT_STREAM_CONTROL_EVENT,
                              "Generation complete");
    } else if (message->kind == YVEX_CLIENT_MESSAGE_ERROR) {
        int cancelled = message->status == YVEX_ERR_CANCELLED ||
                        message->failure_class == YVEX_CLIENT_FAILURE_CLIENT_CANCELLED;
        if (event->operation == YVEX_CLIENT_OP_ENGINE_LOAD) {
            state->engine_load_requested = 0;
            yvex_tui_runtime_launch_failed(
                state, YVEX_TUI_LAUNCH_FAILURE_ENGINE_LOAD,
                message->reason[0] ? message->reason : "model engine load failed",
                0, 0, NULL);
            yvex_tui_activity_add(state, YVEX_TUI_ACTIVITY_ERROR,
                                  YVEX_TUI_SEVERITY_ERROR,
                                  YVEX_CLIENT_STREAM_ERROR,
                                  message->reason[0] ? message->reason
                                                     : "Model engine load failed");
            state->redraw = 1;
            return;
        }
        state->generation_active = 0;
        state->generation_phase = cancelled ? YVEX_CLIENT_PHASE_CANCELLED
                                            : YVEX_CLIENT_PHASE_FAILED;
        state->partial_turn = message->partial_turn;
        yvex_tui_activity_add(state, cancelled ? YVEX_TUI_ACTIVITY_RUNTIME
                                               : YVEX_TUI_ACTIVITY_ERROR,
                              cancelled ? YVEX_TUI_SEVERITY_WARNING
                                        : YVEX_TUI_SEVERITY_ERROR,
                              cancelled ? YVEX_CLIENT_STREAM_CONTROL_EVENT
                                        : YVEX_CLIENT_STREAM_ERROR,
                              message->reason[0] ? message->reason
                                                 : "Request failed");
    } else if (message->kind == YVEX_CLIENT_MESSAGE_ACK) {
        if (event->operation == YVEX_CLIENT_OP_ENGINE_LIST && !state->engine_count)
            memset(&state->active_engine, 0, sizeof(state->active_engine));
        if (event->operation == YVEX_CLIENT_OP_SESSION_ATTACH && event->session[0])
            text_copy(state->active_session, sizeof(state->active_session),
                      event->session);
        yvex_tui_activity_add(state, YVEX_TUI_ACTIVITY_RUNTIME,
                              YVEX_TUI_SEVERITY_SUCCESS,
                              YVEX_CLIENT_STREAM_CONTROL_EVENT,
                              message->reason[0] ? message->reason : "Operation complete");
    }
    state->redraw = 1;
}

static void model_copy(yvex_tui_model_row *row,
                       const yvex_model_library_entry *entry,
                       const yvex_model_library *library, size_t model_index)
{
    unsigned long long index;
    memset(row, 0, sizeof(*row));
    text_copy(row->identity, sizeof(row->identity), entry->identity);
    text_copy(row->display_name, sizeof(row->display_name), entry->display_name);
    text_copy(row->family, sizeof(row->family), entry->family);
    text_copy(row->model, sizeof(row->model), entry->model);
    text_copy(row->runtime_target, sizeof(row->runtime_target), entry->runtime_target);
    row->remote_count = entry->remote_count;
    row->source_count = entry->source_count;
    row->artifact_count = entry->artifact_count;
    row->profile_count = entry->profile_count;
    row->launchable_profile_count = entry->launchable_profile_count;
    row->remote_available = entry->remote_available;
    row->source_local = entry->source_local;
    row->artifact_ready = entry->artifact_ready;
    row->startup_ready = entry->profile_launchable;
    if (!row->startup_ready)
        for (index = 0u; index < entry->profile_count; ++index) {
            const yvex_model_runtime_profile_fact *profile =
                yvex_model_library_profile_at(library, model_index, index);
            if (profile && profile->blocker[0]) {
                text_copy(row->startup_reason, sizeof(row->startup_reason), profile->blocker);
                break;
            }
        }
    if (!row->startup_reason[0] && !row->profile_count)
        text_copy(row->startup_reason, sizeof(row->startup_reason),
                  "no runtime profile is registered");
}

static size_t preferred_launchable_profile(const yvex_tui_state *state,
                                           size_t model_index);

static void models_publish(yvex_tui_state *state, yvex_tui_model_row *models,
                           yvex_model_library *library, size_t count,
                           const char *selected_identity)
{
    size_t index;
    free(state->models);
    yvex_model_library_close(state->model_library);
    state->models = models;
    state->model_library = library;
    state->model_count = count;
    state->model_capacity = count;
    state->model_catalog_status = count ? YVEX_TUI_MODEL_CATALOG_READY
                                        : YVEX_TUI_MODEL_CATALOG_EMPTY;
    state->model_catalog_reason[0] = '\0';
    state->selected_model = 0u;
    if (selected_identity && selected_identity[0]) {
        for (index = 0u; index < count; ++index)
            if (!strcmp(models[index].identity, selected_identity)) {
                state->selected_model = index;
                break;
            }
    } else {
        for (index = 0u; index < count; ++index)
            if (models[index].startup_ready) {
                state->selected_model = index;
                break;
            }
    }
    state->launch_selected_model = state->selected_model;
    state->launch_selected_profile =
        preferred_launchable_profile(state, state->launch_selected_model);
    if (state->launch_selected_profile == SIZE_MAX)
        state->launch_selected_profile = 0u;
    if (state->model_viewport >= count) state->model_viewport = 0u;
    state->redraw = 1;
}

static int models_publish_library(yvex_tui_state *state,
                                  yvex_model_library *library,
                                  const char *selected_identity,
                                  yvex_error *err)
{
    yvex_tui_model_row *models = NULL;
    unsigned long long count = yvex_model_library_count(library);
    size_t index;
    if (count > SIZE_MAX / sizeof(*models) ||
        (count && !(models = calloc((size_t)count, sizeof(*models))))) {
        yvex_model_library_close(library);
        yvex_error_set(err, YVEX_ERR_NOMEM, "cli.tui.models",
                       "model catalog allocation failed");
        return YVEX_ERR_NOMEM;
    }
    for (index = 0u; index < (size_t)count; ++index)
        model_copy(&models[index], yvex_model_library_at(library, index), library, index);
    models_publish(state, models, library, (size_t)count, selected_identity);
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_tui_models_load(yvex_tui_state *state, const char *registry_path,
                         yvex_error *err)
{
    yvex_local_catalog_options options;
    yvex_model_library *library = NULL;
    char selected_identity[YVEX_MODEL_LIBRARY_ID_CAP] = "";
    int rc;
    if (!state) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "cli.tui.models",
                       "application state is required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (state->model_count)
        text_copy(selected_identity, sizeof(selected_identity),
                  state->models[state->selected_model].identity);
    memset(&options, 0, sizeof(options));
    options.registry_path = registry_path;
    rc = yvex_model_library_open(&library, &options, err);
    if (rc != YVEX_OK) {
        state->model_catalog_status = YVEX_TUI_MODEL_CATALOG_ERROR;
        text_copy(state->model_catalog_reason, sizeof(state->model_catalog_reason),
                  yvex_error_message(err));
        state->redraw = 1;
        return rc;
    }
    return models_publish_library(state, library, selected_identity, err);
}

size_t yvex_tui_startup_model_count(const yvex_tui_state *state)
{
    size_t index, count = 0u;
    if (!state || state->model_catalog_status == YVEX_TUI_MODEL_CATALOG_ERROR)
        return 0u;
    for (index = 0u; index < state->model_count; ++index)
        if (state->models[index].startup_ready) count++;
    return count;
}

const yvex_model_runtime_profile_fact *yvex_tui_launch_profile(
    const yvex_tui_state *state)
{
    if (!state || !state->model_library ||
        state->launch_selected_model >= state->model_count)
        return NULL;
    return yvex_model_library_profile_at(
        state->model_library, state->launch_selected_model,
        state->launch_selected_profile);
}

static size_t preferred_launchable_profile(const yvex_tui_state *state,
                                           size_t model_index)
{
    unsigned long long count, index;
    if (!state || !state->model_library || model_index >= state->model_count)
        return SIZE_MAX;
    count = yvex_model_library_profile_count(state->model_library, model_index);
    /* The v5 registry has no explicit preferred-profile field.  Registration is
     * append ordered, so default to the newest admissible profile while keeping
     * every older profile available for explicit selection. */
    for (index = count; index > 0u; --index) {
        const yvex_model_runtime_profile_fact *profile =
            yvex_model_library_profile_at(state->model_library, model_index,
                                          index - 1u);
        if (profile && profile->launchable) return (size_t)(index - 1u);
    }
    return SIZE_MAX;
}

void yvex_tui_runtime_launch_open(yvex_tui_state *state, size_t model_index,
                                  int restart)
{
    size_t index;
    if (!state || !state->model_count) return;
    if (model_index >= state->model_count ||
        !state->models[model_index].startup_ready) {
        for (index = 0u; index < state->model_count; ++index)
            if (state->models[index].startup_ready) break;
        if (index == state->model_count) return;
        model_index = index;
    }
    state->launch_selected_model = model_index;
    state->launch_selected_profile =
        preferred_launchable_profile(state, model_index);
    if (state->launch_selected_profile == SIZE_MAX) return;
    text_copy(state->launch_request.profile, sizeof(state->launch_request.profile),
              yvex_tui_launch_profile(state)->alias);
    state->launch_field = 0u;
    state->restart_pending = restart != 0;
    state->overlay = YVEX_TUI_OVERLAY_RUNTIME_LAUNCH;
    state->focus = YVEX_TUI_FOCUS_RUNTIME_LAUNCH;
    state->redraw = 1;
}

void yvex_tui_runtime_launch_started(yvex_tui_state *state, pid_t pid,
                                     unsigned long long started_ns)
{
    if (!state) return;
    state->launched_pid = pid;
    state->launch_started_ns = started_ns;
    state->runtime_lifecycle = YVEX_TUI_RUNTIME_LAUNCHING;
    state->launch_failure = YVEX_TUI_LAUNCH_FAILURE_NONE;
    state->launch_failure_reason[0] = '\0';
    state->launch_diagnostic[0] = '\0';
    state->overlay = YVEX_TUI_OVERLAY_NONE;
    state->focus = YVEX_TUI_FOCUS_CONTENT;
    state->redraw = 1;
}

void yvex_tui_runtime_launch_failed(yvex_tui_state *state,
                                    yvex_tui_launch_failure failure,
                                    const char *reason, int exec_error,
                                    int exit_status, const char *diagnostic)
{
    if (!state) return;
    state->runtime_lifecycle = YVEX_TUI_RUNTIME_LAUNCH_FAILED;
    state->launch_failure = failure;
    text_copy(state->launch_failure_reason, sizeof(state->launch_failure_reason), reason);
    text_copy(state->launch_diagnostic, sizeof(state->launch_diagnostic), diagnostic);
    state->launch_exec_error = exec_error;
    state->launch_exit_status = exit_status;
    state->redraw = 1;
}

void yvex_tui_runtime_stop_requested(yvex_tui_state *state, int restart)
{
    if (!state) return;
    state->runtime_lifecycle = YVEX_TUI_RUNTIME_SHUTDOWN_REQUESTED;
    state->restart_pending = restart != 0;
    state->redraw = 1;
}

void yvex_tui_remote_search_started(yvex_tui_state *state)
{
    if (!state) return;
    memset(state->remote_models, 0, sizeof(state->remote_models));
    state->remote_count = 0u;
    state->remote_search_reason[0] = '\0';
    state->remote_search_running = 1;
    state->remote_detail = 0;
    state->selected_remote = 0u;
    state->remote_viewport = 0u;
    state->focus = YVEX_TUI_FOCUS_CONTENT;
    state->redraw = 1;
}

void yvex_tui_remote_search_publish(yvex_tui_state *state,
                                    const yvex_remote_model *results,
                                    size_t result_count,
                                    const char *reason)
{
    if (!state || (!results && result_count)) return;
    if (result_count > YVEX_TUI_REMOTE_CAP) result_count = YVEX_TUI_REMOTE_CAP;
    memset(state->remote_models, 0, sizeof(state->remote_models));
    if (result_count)
        memcpy(state->remote_models, results, result_count * sizeof(*results));
    state->remote_count = result_count;
    text_copy(state->remote_search_reason, sizeof(state->remote_search_reason), reason);
    state->remote_search_running = 0;
    state->remote_detail = 0;
    state->selected_remote = 0u;
    state->remote_viewport = 0u;
    state->focus = YVEX_TUI_FOCUS_CONTENT;
    state->redraw = 1;
}

static int text_contains_folded(const char *text, const char *query)
{
    size_t text_count = strlen(text), query_count = strlen(query), start, offset;
    if (!query_count) return 1;
    if (query_count > text_count) return 0;
    for (start = 0u; start + query_count <= text_count; ++start) {
        for (offset = 0u; offset < query_count; ++offset)
            if (tolower((unsigned char)text[start + offset]) !=
                tolower((unsigned char)query[offset]))
                break;
        if (offset == query_count) return 1;
    }
    return 0;
}

size_t yvex_tui_model_visible_count(const yvex_tui_state *state)
{
    size_t count = 0u, index;
    if (!state) return 0u;
    for (index = 0u; index < state->model_count; ++index) {
        const yvex_tui_model_row *row = &state->models[index];
        if (text_contains_folded(row->display_name, state->model_search) ||
            text_contains_folded(row->family, state->model_search) ||
            text_contains_folded(row->model, state->model_search) ||
            text_contains_folded(row->runtime_target, state->model_search))
            count++;
    }
    return count;
}

size_t yvex_tui_model_visible_at(const yvex_tui_state *state, size_t ordinal)
{
    size_t count = 0u, index;
    if (!state) return SIZE_MAX;
    for (index = 0u; index < state->model_count; ++index) {
        const yvex_tui_model_row *row = &state->models[index];
        if (!(text_contains_folded(row->display_name, state->model_search) ||
              text_contains_folded(row->family, state->model_search) ||
              text_contains_folded(row->model, state->model_search) ||
              text_contains_folded(row->runtime_target, state->model_search)))
            continue;
        if (count++ == ordinal) return index;
    }
    return SIZE_MAX;
}

size_t yvex_tui_session_visible_count(const yvex_tui_state *state)
{
    size_t count = 0u, index;
    if (!state) return 0u;
    for (index = 0u; index < state->session_count; ++index)
        if (text_contains_folded(state->sessions[index].name,
                                 state->session_search))
            count++;
    return count;
}

size_t yvex_tui_session_visible_at(const yvex_tui_state *state,
                                   size_t ordinal)
{
    size_t count = 0u, index;
    if (!state) return SIZE_MAX;
    for (index = 0u; index < state->session_count; ++index) {
        if (!text_contains_folded(state->sessions[index].name,
                                  state->session_search))
            continue;
        if (count++ == ordinal) return index;
    }
    return SIZE_MAX;
}

void yvex_tui_selection_move(yvex_tui_state *state, int direction,
                             size_t page_rows)
{
    size_t count, ordinal = 0u, index;
    if (!state || !direction) return;
    if (state->surface == YVEX_TUI_SURFACE_MODELS) {
        if (state->models_mode == YVEX_TUI_MODELS_DISCOVER) {
            count = state->remote_count;
            if (!count) return;
            ordinal = state->selected_remote;
            if (direction < 0 && ordinal) ordinal--;
            else if (direction > 0 && ordinal + 1u < count) ordinal++;
            state->selected_remote = ordinal;
            if (ordinal < state->remote_viewport) state->remote_viewport = ordinal;
            if (page_rows && ordinal >= state->remote_viewport + page_rows)
                state->remote_viewport = ordinal - page_rows + 1u;
            state->redraw = 1;
            return;
        }
        count = yvex_tui_model_visible_count(state);
        for (index = 0u; index < count; ++index)
            if (yvex_tui_model_visible_at(state, index) == state->selected_model) {
                ordinal = index;
                break;
            }
        if (direction < 0 && ordinal) ordinal--;
        else if (direction > 0 && ordinal + 1u < count) ordinal++;
        index = yvex_tui_model_visible_at(state, ordinal);
        if (index != SIZE_MAX) state->selected_model = index;
        if (ordinal < state->model_viewport) state->model_viewport = ordinal;
        if (page_rows && ordinal >= state->model_viewport + page_rows)
            state->model_viewport = ordinal - page_rows + 1u;
    } else if (state->surface == YVEX_TUI_SURFACE_SESSIONS && state->session_count) {
        count = yvex_tui_session_visible_count(state);
        for (index = 0u; index < count; ++index)
            if (yvex_tui_session_visible_at(state, index) == state->selected_session) {
                ordinal = index;
                break;
            }
        if (direction < 0 && ordinal) ordinal--;
        else if (direction > 0 && ordinal + 1u < count) ordinal++;
        index = yvex_tui_session_visible_at(state, ordinal);
        if (index != SIZE_MAX) state->selected_session = index;
        if (ordinal < state->session_viewport)
            state->session_viewport = ordinal;
        if (page_rows && ordinal >= state->session_viewport + page_rows)
            state->session_viewport = ordinal - page_rows + 1u;
    } else if (state->surface == YVEX_TUI_SURFACE_HOME) {
        if (direction < 0 && state->activity_scroll + 1u < state->activity_count)
            state->activity_scroll++;
        else if (direction > 0 && state->activity_scroll)
            state->activity_scroll--;
    }
    state->redraw = 1;
}
