/* Normalize terminal byte streams into deterministic navigation and composer transitions. */
#include "src/cli/tui/private.h"

#include <string.h>

static size_t utf8_expected(unsigned char first);

static void surface_cycle(yvex_tui_state *state, int backwards)
{
    static const yvex_tui_surface surfaces[] = {
        YVEX_TUI_SURFACE_HOME,
        YVEX_TUI_SURFACE_MODELS,
        YVEX_TUI_SURFACE_SESSIONS,
        YVEX_TUI_SURFACE_RUNTIME,
    };
    size_t index;
    yvex_tui_surface current = state->surface == YVEX_TUI_SURFACE_MODEL_DETAIL
                                   ? YVEX_TUI_SURFACE_MODELS
                                   : state->surface;
    for (index = 0u; index < sizeof(surfaces) / sizeof(surfaces[0]); ++index)
        if (surfaces[index] == current) break;
    if (backwards)
        index = index ? index - 1u : sizeof(surfaces) / sizeof(surfaces[0]) - 1u;
    else
        index = (index + 1u) % (sizeof(surfaces) / sizeof(surfaces[0]));
    state->prior_surface = state->surface;
    state->surface = surfaces[index];
    state->focus = state->surface == YVEX_TUI_SURFACE_HOME
                       ? YVEX_TUI_FOCUS_COMPOSER
                       : YVEX_TUI_FOCUS_CONTENT;
    state->overlay = YVEX_TUI_OVERLAY_NONE;
    state->redraw = 1;
}

static void escape_action(yvex_tui_state *state)
{
    if (state->overlay != YVEX_TUI_OVERLAY_NONE) {
        if (state->overlay == YVEX_TUI_OVERLAY_PALETTE)
            yvex_tui_composer_clear(&state->command);
        state->overlay = YVEX_TUI_OVERLAY_NONE;
        state->focus = state->surface == YVEX_TUI_SURFACE_HOME
                           ? YVEX_TUI_FOCUS_COMPOSER
                           : YVEX_TUI_FOCUS_CONTENT;
    } else if (state->focus == YVEX_TUI_FOCUS_MODEL_SEARCH ||
               state->focus == YVEX_TUI_FOCUS_SESSION_SEARCH) {
        state->focus = YVEX_TUI_FOCUS_CONTENT;
    } else if (state->surface == YVEX_TUI_SURFACE_MODELS &&
               state->models_mode == YVEX_TUI_MODELS_DISCOVER &&
               state->remote_detail) {
        state->remote_detail = 0;
    } else if (state->surface == YVEX_TUI_SURFACE_MODEL_DETAIL) {
        state->surface = YVEX_TUI_SURFACE_MODELS;
        state->focus = YVEX_TUI_FOCUS_CONTENT;
    } else if (state->focus == YVEX_TUI_FOCUS_COMPOSER && state->composer.count) {
        yvex_tui_composer_clear(&state->composer);
    } else {
        state->focus = YVEX_TUI_FOCUS_COMPOSER;
    }
    state->redraw = 1;
}

static void search_insert(yvex_tui_state *state, const unsigned char *bytes,
                          size_t count)
{
    int discover = state->focus == YVEX_TUI_FOCUS_MODEL_SEARCH &&
                   state->models_mode == YVEX_TUI_MODELS_DISCOVER;
    char *search = state->focus == YVEX_TUI_FOCUS_SESSION_SEARCH
                       ? state->session_search
                   : discover ? state->discover_query : state->model_search;
    size_t *search_count = state->focus == YVEX_TUI_FOCUS_SESSION_SEARCH
                               ? &state->session_search_count
                           : discover ? &state->discover_query_count
                                      : &state->model_search_count;
    size_t capacity = state->focus == YVEX_TUI_FOCUS_SESSION_SEARCH
                          ? sizeof(state->session_search)
                      : discover ? sizeof(state->discover_query)
                                 : sizeof(state->model_search);
    if (*search_count + count >= capacity) return;
    memcpy(search + *search_count, bytes, count);
    *search_count += count;
    search[*search_count] = '\0';
    if (state->focus == YVEX_TUI_FOCUS_SESSION_SEARCH) {
        state->session_viewport = 0u;
        if (yvex_tui_session_visible_count(state))
            state->selected_session = yvex_tui_session_visible_at(state, 0u);
    } else if (!discover) {
        state->model_viewport = 0u;
        if (yvex_tui_model_visible_count(state))
            state->selected_model = yvex_tui_model_visible_at(state, 0u);
    }
    state->redraw = 1;
}

static void search_erase(yvex_tui_state *state)
{
    int discover = state->focus == YVEX_TUI_FOCUS_MODEL_SEARCH &&
                   state->models_mode == YVEX_TUI_MODELS_DISCOVER;
    char *search = state->focus == YVEX_TUI_FOCUS_SESSION_SEARCH
                       ? state->session_search
                   : discover ? state->discover_query : state->model_search;
    size_t *search_count = state->focus == YVEX_TUI_FOCUS_SESSION_SEARCH
                               ? &state->session_search_count
                           : discover ? &state->discover_query_count
                                      : &state->model_search_count;
    size_t cursor = *search_count;
    if (!cursor) return;
    cursor--;
    while (cursor && ((unsigned char)search[cursor] & 0xc0u) == 0x80u)
        cursor--;
    *search_count = cursor;
    search[cursor] = '\0';
    if (state->focus == YVEX_TUI_FOCUS_SESSION_SEARCH) {
        state->session_viewport = 0u;
        if (yvex_tui_session_visible_count(state))
            state->selected_session = yvex_tui_session_visible_at(state, 0u);
    } else if (!discover) {
        state->model_viewport = 0u;
        if (yvex_tui_model_visible_count(state))
            state->selected_model = yvex_tui_model_visible_at(state, 0u);
    }
    state->redraw = 1;
}

static void insert_text(yvex_tui_state *state, const unsigned char *bytes,
                        size_t count)
{
    if (state->focus == YVEX_TUI_FOCUS_MODEL_SEARCH ||
        state->focus == YVEX_TUI_FOCUS_SESSION_SEARCH)
        search_insert(state, bytes, count);
    else if (state->overlay == YVEX_TUI_OVERLAY_PALETTE) {
        if (yvex_tui_composer_insert(&state->command, bytes, count)) {
            state->focus = YVEX_TUI_FOCUS_PALETTE;
            state->redraw = 1;
        }
    } else if (state->surface == YVEX_TUI_SURFACE_HOME &&
               yvex_tui_composer_insert(&state->composer, bytes, count)) {
        state->focus = YVEX_TUI_FOCUS_COMPOSER;
        state->redraw = 1;
    }
}

static void launch_model_move(yvex_tui_state *state, int direction)
{
    size_t index = state->launch_selected_model;
    if (!state->model_count) return;
    do {
        if (direction < 0) index = index ? index - 1u : state->model_count - 1u;
        else index = (index + 1u) % state->model_count;
        if (state->models[index].startup_ready) {
            yvex_tui_runtime_launch_open(state, index,
                                         state->restart_pending);
            return;
        }
    } while (index != state->launch_selected_model);
}

static void launch_profile_move(yvex_tui_state *state, int direction)
{
    size_t start, index, count;
    if (!state || state->launch_selected_model >= state->model_count) return;
    count = (size_t)state->models[state->launch_selected_model].profile_count;
    if (!count) return;
    start = state->launch_selected_profile;
    index = start;
    do {
        index = direction < 0 ? (index ? index - 1u : count - 1u)
                              : (index + 1u) % count;
        state->launch_selected_profile = index;
        if (yvex_tui_launch_profile(state) && yvex_tui_launch_profile(state)->launchable) {
            state->redraw = 1;
            return;
        }
    } while (index != start);
    state->launch_selected_profile = start;
}

static yvex_tui_composer *active_editor(yvex_tui_state *state)
{
    return state->overlay == YVEX_TUI_OVERLAY_PALETTE
               ? &state->command : &state->composer;
}

static yvex_tui_input_action normal_ascii(yvex_tui_state *state,
                                          unsigned char byte)
{
    if (byte == '\r') {
        if (state->overlay == YVEX_TUI_OVERLAY_RUNTIME_LAUNCH) {
            if (state->launch_field < 2u) {
                state->launch_field++;
                state->redraw = 1;
                return YVEX_TUI_INPUT_NONE;
            }
            return YVEX_TUI_INPUT_SUBMIT;
        }
        if (state->focus == YVEX_TUI_FOCUS_MODEL_SEARCH ||
            state->focus == YVEX_TUI_FOCUS_SESSION_SEARCH) {
            if (state->focus == YVEX_TUI_FOCUS_MODEL_SEARCH &&
                state->models_mode == YVEX_TUI_MODELS_DISCOVER)
                return YVEX_TUI_INPUT_REMOTE_SEARCH;
            state->focus = YVEX_TUI_FOCUS_CONTENT;
            state->redraw = 1;
            return YVEX_TUI_INPUT_NONE;
        }
        if (state->focus == YVEX_TUI_FOCUS_CONTENT &&
            state->surface == YVEX_TUI_SURFACE_MODELS &&
            state->models_mode == YVEX_TUI_MODELS_DISCOVER &&
            state->remote_count) {
            state->remote_detail = !state->remote_detail;
            state->redraw = 1;
            return YVEX_TUI_INPUT_NONE;
        }
        if (state->focus == YVEX_TUI_FOCUS_CONTENT &&
            state->surface == YVEX_TUI_SURFACE_MODELS && state->model_count) {
            state->prior_surface = state->surface;
            state->surface = YVEX_TUI_SURFACE_MODEL_DETAIL;
            state->redraw = 1;
            return YVEX_TUI_INPUT_NONE;
        }
        if (state->surface == YVEX_TUI_SURFACE_MODEL_DETAIL &&
            state->model_count && !state->models[state->selected_model].resident &&
            state->models[state->selected_model].startup_ready) {
            yvex_tui_runtime_launch_open(state, state->selected_model, 0);
            return YVEX_TUI_INPUT_NONE;
        }
        if ((state->surface == YVEX_TUI_SURFACE_HOME ||
             state->surface == YVEX_TUI_SURFACE_RUNTIME) &&
            !state->active_engine.alias[0] &&
            state->runtime_lifecycle != YVEX_TUI_RUNTIME_LAUNCHING &&
            state->runtime_lifecycle != YVEX_TUI_RUNTIME_WAITING_PROTOCOL &&
            state->runtime_lifecycle != YVEX_TUI_RUNTIME_ENGINE_LOADING &&
            yvex_tui_startup_model_count(state)) {
            yvex_tui_runtime_launch_open(state, state->selected_model, 0);
            return YVEX_TUI_INPUT_NONE;
        }
        return YVEX_TUI_INPUT_SUBMIT;
    }
    if (byte == '\n') {
        insert_text(state, &byte, 1u);
        return YVEX_TUI_INPUT_NONE;
    }
    if (byte == '\t') {
        if (state->overlay == YVEX_TUI_OVERLAY_RUNTIME_LAUNCH) {
            state->launch_field = (state->launch_field + 1u) % 3u;
            state->redraw = 1;
            return YVEX_TUI_INPUT_NONE;
        }
        surface_cycle(state, 0);
        return YVEX_TUI_INPUT_NONE;
    }
    if (byte == 0x04u) return YVEX_TUI_INPUT_EXIT;
    if (byte == 0x0cu) {
        state->redraw = 1;
        return YVEX_TUI_INPUT_REFRESH;
    }
    if (byte == 'r' && state->focus == YVEX_TUI_FOCUS_CONTENT)
        return YVEX_TUI_INPUT_REFRESH;
    if (state->surface == YVEX_TUI_SURFACE_MODELS &&
        state->focus == YVEX_TUI_FOCUS_CONTENT && byte == 'd') {
        state->models_mode = YVEX_TUI_MODELS_DISCOVER;
        state->focus = YVEX_TUI_FOCUS_MODEL_SEARCH;
        state->remote_detail = 0;
        state->redraw = 1;
        return YVEX_TUI_INPUT_NONE;
    }
    if (state->surface == YVEX_TUI_SURFACE_MODELS &&
        state->focus == YVEX_TUI_FOCUS_CONTENT && byte == 'l' &&
        state->models_mode == YVEX_TUI_MODELS_DISCOVER) {
        state->models_mode = YVEX_TUI_MODELS_LIBRARY;
        state->focus = YVEX_TUI_FOCUS_CONTENT;
        state->remote_detail = 0;
        state->redraw = 1;
        return YVEX_TUI_INPUT_NONE;
    }
    if (state->surface == YVEX_TUI_SURFACE_MODELS &&
        state->models_mode == YVEX_TUI_MODELS_DISCOVER &&
        state->focus == YVEX_TUI_FOCUS_CONTENT && byte == 'a' &&
        state->remote_count)
        return YVEX_TUI_INPUT_ACQUIRE;
    if (byte == 0x10u && state->overlay == YVEX_TUI_OVERLAY_NONE) {
        state->overlay = YVEX_TUI_OVERLAY_PALETTE;
        state->focus = YVEX_TUI_FOCUS_PALETTE;
        yvex_tui_composer_clear(&state->command);
        insert_text(state, (const unsigned char *)"/", 1u);
        return YVEX_TUI_INPUT_NONE;
    }
    if (byte == 0x7fu || byte == 0x08u) {
        if (state->focus == YVEX_TUI_FOCUS_MODEL_SEARCH ||
            state->focus == YVEX_TUI_FOCUS_SESSION_SEARCH)
            search_erase(state);
        else if (state->overlay == YVEX_TUI_OVERLAY_PALETTE) {
            yvex_tui_composer_erase(&state->command, 1);
            state->redraw = 1;
        }
        else {
            yvex_tui_composer_erase(&state->composer, 1);
            state->redraw = 1;
        }
        return YVEX_TUI_INPUT_NONE;
    }
    if (byte == '/' && state->overlay == YVEX_TUI_OVERLAY_NONE &&
        (state->surface == YVEX_TUI_SURFACE_MODELS ||
         state->surface == YVEX_TUI_SURFACE_SESSIONS)) {
        state->focus = state->surface == YVEX_TUI_SURFACE_MODELS
                           ? YVEX_TUI_FOCUS_MODEL_SEARCH
                           : YVEX_TUI_FOCUS_SESSION_SEARCH;
        state->redraw = 1;
        return YVEX_TUI_INPUT_NONE;
    }
    if (byte == '/' && state->overlay == YVEX_TUI_OVERLAY_NONE &&
        state->surface == YVEX_TUI_SURFACE_RUNTIME) {
        state->overlay = YVEX_TUI_OVERLAY_PALETTE;
        state->focus = YVEX_TUI_FOCUS_PALETTE;
        yvex_tui_composer_clear(&state->command);
    }
    if (byte == '?' && state->overlay == YVEX_TUI_OVERLAY_NONE) {
        state->overlay = YVEX_TUI_OVERLAY_HELP;
        state->focus = YVEX_TUI_FOCUS_PALETTE;
        state->redraw = 1;
        return YVEX_TUI_INPUT_NONE;
    }
    if (state->focus == YVEX_TUI_FOCUS_CONTENT &&
        (state->surface == YVEX_TUI_SURFACE_MODELS ||
         state->surface == YVEX_TUI_SURFACE_SESSIONS) &&
        (byte == 'j' || byte == 'k')) {
        yvex_tui_selection_move(state, byte == 'j' ? 1 : -1, 8u);
        return YVEX_TUI_INPUT_NONE;
    }
    if (byte >= 0x20u) insert_text(state, &byte, 1u);
    return YVEX_TUI_INPUT_NONE;
}

static yvex_tui_input_action sequence_action(yvex_tui_input *input,
                                             yvex_tui_state *state)
{
    const unsigned char *sequence = input->sequence;
    size_t count = input->sequence_count;
    if (count == 2u && sequence[0] == '[') {
        if (state->overlay == YVEX_TUI_OVERLAY_RUNTIME_LAUNCH) {
            if ((sequence[1] == 'A' || sequence[1] == 'B') &&
                state->launch_field == 0u)
                launch_model_move(state, sequence[1] == 'A' ? -1 : 1);
            else if ((sequence[1] == 'A' || sequence[1] == 'B') &&
                     state->launch_field == 1u)
                launch_profile_move(state, sequence[1] == 'A' ? -1 : 1);
            else if (sequence[1] == 'Z') {
                state->launch_field = state->launch_field
                                          ? state->launch_field - 1u : 2u;
            }
            state->redraw = 1;
            return YVEX_TUI_INPUT_NONE;
        }
        if (state->surface == YVEX_TUI_SURFACE_RUNTIME &&
            state->focus == YVEX_TUI_FOCUS_CONTENT &&
            state->connection == YVEX_TUI_CONNECTION_CONNECTED &&
            (sequence[1] == 'C' || sequence[1] == 'D')) {
            state->runtime_action = sequence[1] == 'C' ? 1u : 0u;
            state->redraw = 1;
            return YVEX_TUI_INPUT_NONE;
        }
        if (sequence[1] == 'A') {
            if (state->focus == YVEX_TUI_FOCUS_CONTENT ||
                state->focus == YVEX_TUI_FOCUS_MODEL_SEARCH ||
                state->focus == YVEX_TUI_FOCUS_SESSION_SEARCH)
                yvex_tui_selection_move(state, -1, 8u);
            else if (state->overlay == YVEX_TUI_OVERLAY_PALETTE)
                yvex_tui_composer_history_move(&state->command, -1);
            else
                yvex_tui_composer_history_move(&state->composer, -1);
        } else if (sequence[1] == 'B') {
            if (state->focus == YVEX_TUI_FOCUS_CONTENT ||
                state->focus == YVEX_TUI_FOCUS_MODEL_SEARCH ||
                state->focus == YVEX_TUI_FOCUS_SESSION_SEARCH)
                yvex_tui_selection_move(state, 1, 8u);
            else if (state->overlay == YVEX_TUI_OVERLAY_PALETTE)
                yvex_tui_composer_history_move(&state->command, 1);
            else
                yvex_tui_composer_history_move(&state->composer, 1);
        } else if (sequence[1] == 'C') {
            yvex_tui_composer_right(active_editor(state));
        } else if (sequence[1] == 'D') {
            yvex_tui_composer_left(active_editor(state));
        } else if (sequence[1] == 'H') {
            yvex_tui_composer_home(active_editor(state));
        } else if (sequence[1] == 'F') {
            yvex_tui_composer_end(active_editor(state));
        } else if (sequence[1] == 'Z') {
            surface_cycle(state, 1);
        }
        state->redraw = 1;
        return YVEX_TUI_INPUT_NONE;
    }
    if (count == 3u && sequence[0] == '[' && sequence[2] == '~') {
        if (sequence[1] == '1' || sequence[1] == '7')
            yvex_tui_composer_home(active_editor(state));
        else if (sequence[1] == '4' || sequence[1] == '8')
            yvex_tui_composer_end(active_editor(state));
        else if (sequence[1] == '3')
            yvex_tui_composer_erase(active_editor(state), 0);
        else if (sequence[1] == '5')
            yvex_tui_selection_move(state, -1, 8u);
        else if (sequence[1] == '6')
            yvex_tui_selection_move(state, 1, 8u);
        state->redraw = 1;
        return YVEX_TUI_INPUT_NONE;
    }
    if (count == 5u && !memcmp(sequence, "[200~", 5u)) {
        input->paste = 1;
        return YVEX_TUI_INPUT_NONE;
    }
    escape_action(state);
    return YVEX_TUI_INPUT_NONE;
}

static int sequence_complete(const unsigned char *sequence, size_t count)
{
    if (!count) return 0;
    if (sequence[0] != '[') return 1;
    if (count == 2u && ((sequence[1] >= 'A' && sequence[1] <= 'D') ||
                       sequence[1] == 'F' || sequence[1] == 'H' ||
                       sequence[1] == 'Z'))
        return 1;
    if (sequence[count - 1u] == '~') return 1;
    return count == YVEX_TUI_INPUT_SEQUENCE_CAP;
}

static int paste_end_prefix(const unsigned char *sequence, size_t count)
{
    static const unsigned char ending[] = "\033[201~";
    return count <= sizeof(ending) - 1u && !memcmp(sequence, ending, count);
}

static void paste_byte(yvex_tui_input *input, yvex_tui_state *state,
                       unsigned char byte)
{
    static const unsigned char ending[] = "\033[201~";
    if (!input->sequence_count && byte != '\033') {
        if (byte == '\r') byte = '\n';
        if (byte < 0x80u) insert_text(state, &byte, 1u);
        else {
            input->utf8_expected = utf8_expected(byte);
            if (input->utf8_expected) {
                input->utf8[0] = byte;
                input->utf8_count = 1u;
            }
        }
        return;
    }
    if (input->sequence_count < sizeof(input->sequence))
        input->sequence[input->sequence_count++] = byte;
    if (paste_end_prefix(input->sequence, input->sequence_count)) {
        if (input->sequence_count == sizeof(ending) - 1u) {
            input->sequence_count = 0u;
            input->paste = 0;
        }
        return;
    }
    insert_text(state, (const unsigned char *)"?", 1u);
    if (input->sequence_count > 1u)
        insert_text(state, input->sequence + 1u, input->sequence_count - 1u);
    input->sequence_count = 0u;
}

static size_t utf8_expected(unsigned char first)
{
    if (first >= 0xc2u && first <= 0xdfu) return 2u;
    if (first >= 0xe0u && first <= 0xefu) return 3u;
    if (first >= 0xf0u && first <= 0xf4u) return 4u;
    return 0u;
}

yvex_tui_input_action yvex_tui_input_byte(yvex_tui_input *input,
                                           yvex_tui_state *state,
                                           unsigned char byte)
{
    if (!input || !state) return YVEX_TUI_INPUT_NONE;
    if (input->utf8_expected) {
        input->utf8[input->utf8_count++] = byte;
        if (input->utf8_count == input->utf8_expected) {
            insert_text(state, input->utf8, input->utf8_count);
            input->utf8_count = 0u;
            input->utf8_expected = 0u;
        }
        return YVEX_TUI_INPUT_NONE;
    }
    if (input->paste) {
        paste_byte(input, state, byte);
        return YVEX_TUI_INPUT_NONE;
    }
    if (input->escape) {
        if (input->sequence_count < sizeof(input->sequence))
            input->sequence[input->sequence_count++] = byte;
        if (sequence_complete(input->sequence, input->sequence_count)) {
            yvex_tui_input_action action = sequence_action(input, state);
            input->escape = 0;
            input->sequence_count = 0u;
            return action;
        }
        return YVEX_TUI_INPUT_NONE;
    }
    if (byte == '\033') {
        input->escape = 1;
        input->sequence_count = 0u;
        return YVEX_TUI_INPUT_NONE;
    }
    if (byte >= 0x80u) {
        input->utf8_expected = utf8_expected(byte);
        if (!input->utf8_expected) return YVEX_TUI_INPUT_NONE;
        input->utf8[0] = byte;
        input->utf8_count = 1u;
        return YVEX_TUI_INPUT_NONE;
    }
    return normal_ascii(state, byte);
}

yvex_tui_input_action yvex_tui_input_flush(yvex_tui_input *input,
                                            yvex_tui_state *state)
{
    if (!input || !state) return YVEX_TUI_INPUT_NONE;
    if (input->escape && !input->sequence_count) {
        input->escape = 0;
        escape_action(state);
    }
    return YVEX_TUI_INPUT_NONE;
}
