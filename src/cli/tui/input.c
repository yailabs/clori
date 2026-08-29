/* Normalize terminal byte streams into deterministic composer and overlay transitions. */
#include "src/cli/tui/private.h"

#include <operator/registry.h>

#include <string.h>

static size_t utf8_expected(unsigned char first);

static int slash_active(const yvex_tui_state *state)
{
    return state->composer.count && state->composer.bytes[0] == '/' &&
           memchr(state->composer.bytes, '\n', state->composer.count) == NULL;
}

static void slash_sync(yvex_tui_state *state)
{
    if (state->overlay != YVEX_TUI_OVERLAY_NONE &&
        state->overlay != YVEX_TUI_OVERLAY_SLASH)
        return;
    state->overlay = slash_active(state) ? YVEX_TUI_OVERLAY_SLASH
                                         : YVEX_TUI_OVERLAY_NONE;
    state->slash_selected = 0u;
}

static yvex_tui_input_action escape_action(yvex_tui_state *state)
{
    if (state->overlay != YVEX_TUI_OVERLAY_NONE) {
        state->overlay = YVEX_TUI_OVERLAY_NONE;
        state->focus = YVEX_TUI_FOCUS_COMPOSER;
        state->redraw = 1;
        return YVEX_TUI_INPUT_NONE;
    }
    if (state->generation_active) return YVEX_TUI_INPUT_CANCEL;
    if (state->composer.count) {
        yvex_tui_composer_clear(&state->composer);
        state->pending_review = 0;
    }
    state->focus = YVEX_TUI_FOCUS_COMPOSER;
    state->redraw = 1;
    return YVEX_TUI_INPUT_NONE;
}

static void search_reset_selection(yvex_tui_state *state)
{
    if (state->overlay == YVEX_TUI_OVERLAY_MODEL) {
        state->model_viewport = 0u;
        if (yvex_tui_model_visible_count(state)) {
            state->selected_model = yvex_tui_model_visible_at(state, 0u);
            yvex_tui_runtime_launch_open(state, state->selected_model,
                                         state->restart_pending);
        }
    } else if (state->overlay == YVEX_TUI_OVERLAY_SESSION) {
        state->session_viewport = 0u;
        if (yvex_tui_session_visible_count(state))
            state->selected_session = yvex_tui_session_visible_at(state, 0u);
    }
}

static void search_insert(yvex_tui_state *state, const unsigned char *bytes,
                          size_t count)
{
    char *search;
    size_t *used, capacity;
    if (state->overlay == YVEX_TUI_OVERLAY_MODEL) {
        search = state->model_search;
        used = &state->model_search_count;
        capacity = sizeof(state->model_search);
    } else if (state->overlay == YVEX_TUI_OVERLAY_SESSION) {
        search = state->session_search;
        used = &state->session_search_count;
        capacity = sizeof(state->session_search);
    } else {
        search = state->discover_query;
        used = &state->discover_query_count;
        capacity = sizeof(state->discover_query);
    }
    if (*used + count >= capacity) return;
    memcpy(search + *used, bytes, count);
    *used += count;
    search[*used] = '\0';
    search_reset_selection(state);
    state->redraw = 1;
}

static void search_erase(yvex_tui_state *state)
{
    char *search;
    size_t *used, cursor;
    if (state->overlay == YVEX_TUI_OVERLAY_MODEL) {
        search = state->model_search;
        used = &state->model_search_count;
    } else if (state->overlay == YVEX_TUI_OVERLAY_SESSION) {
        search = state->session_search;
        used = &state->session_search_count;
    } else {
        search = state->discover_query;
        used = &state->discover_query_count;
    }
    cursor = *used;
    if (!cursor) return;
    cursor--;
    while (cursor && ((unsigned char)search[cursor] & 0xc0u) == 0x80u)
        cursor--;
    *used = cursor;
    search[cursor] = '\0';
    search_reset_selection(state);
    state->redraw = 1;
}

static void insert_text(yvex_tui_state *state, const unsigned char *bytes,
                        size_t count)
{
    if (state->overlay == YVEX_TUI_OVERLAY_MODEL ||
        state->overlay == YVEX_TUI_OVERLAY_SESSION ||
        state->overlay == YVEX_TUI_OVERLAY_REMOTE) {
        if (state->overlay == YVEX_TUI_OVERLAY_MODEL) {
            state->launch_field = 0u;
            state->focus = YVEX_TUI_FOCUS_MODEL_SEARCH;
        }
        search_insert(state, bytes, count);
        return;
    }
    if (state->overlay == YVEX_TUI_OVERLAY_HELP) return;
    if (yvex_tui_composer_insert(&state->composer, bytes, count)) {
        state->focus = YVEX_TUI_FOCUS_COMPOSER;
        slash_sync(state);
        state->redraw = 1;
    }
}

static void launch_profile_move(yvex_tui_state *state, int direction)
{
    size_t count, index;
    if (!state->model_library ||
        state->launch_selected_model >= state->model_count)
        return;
    count = (size_t)yvex_model_library_profile_count(
        state->model_library, state->launch_selected_model);
    if (!count) return;
    index = state->launch_selected_profile;
    if (index >= count) index = 0u;
    index = direction < 0 ? (index ? index - 1u : count - 1u)
                          : (index + 1u) % count;
    state->launch_selected_profile = index;
    state->redraw = 1;
}

static int slash_prefix(const char *projection, const unsigned char *query,
                        size_t count)
{
    return projection && strcmp(projection, "none") &&
           strlen(projection) >= count && !memcmp(projection, query, count);
}

static size_t slash_match_count(const yvex_tui_state *state)
{
    size_t extent = state->composer.count, index, count = 0u;
    const unsigned char *space = memchr(state->composer.bytes, ' ', extent);
    if (space) extent = (size_t)(space - state->composer.bytes);
    for (index = 0u; index < yvex_operator_descriptor_count; ++index)
        if (slash_prefix(yvex_operator_descriptors[index].slash_projection,
                         state->composer.bytes, extent))
            count++;
    return count;
}

static const yvex_operator_descriptor *slash_match_at(
    const yvex_tui_state *state, size_t ordinal)
{
    size_t extent = state->composer.count, index, count = 0u;
    const unsigned char *space = memchr(state->composer.bytes, ' ', extent);
    if (space) extent = (size_t)(space - state->composer.bytes);
    for (index = 0u; index < yvex_operator_descriptor_count; ++index) {
        const yvex_operator_descriptor *descriptor =
            &yvex_operator_descriptors[index];
        if (!slash_prefix(descriptor->slash_projection,
                          state->composer.bytes, extent))
            continue;
        if (count++ == ordinal) return descriptor;
    }
    return NULL;
}

static void slash_move(yvex_tui_state *state, int direction)
{
    size_t count = slash_match_count(state);
    if (!count) return;
    if (state->slash_selected >= count) state->slash_selected = 0u;
    if (direction < 0)
        state->slash_selected = state->slash_selected
                                    ? state->slash_selected - 1u : count - 1u;
    else
        state->slash_selected = (state->slash_selected + 1u) % count;
    state->redraw = 1;
}

static void slash_complete(yvex_tui_state *state)
{
    const yvex_operator_descriptor *descriptor =
        slash_match_at(state, state->slash_selected);
    if (!descriptor) return;
    yvex_tui_composer_clear(&state->composer);
    (void)yvex_tui_composer_insert(
        &state->composer,
        (const unsigned char *)descriptor->slash_projection,
        strlen(descriptor->slash_projection));
    if (descriptor->slash_argument_count)
        (void)yvex_tui_composer_insert(&state->composer,
                                       (const unsigned char *)" ", 1u);
    slash_sync(state);
    state->redraw = 1;
}

static yvex_tui_input_action normal_ascii(yvex_tui_state *state,
                                          unsigned char byte)
{
    if (byte == '\r') {
        if (state->overlay == YVEX_TUI_OVERLAY_HELP) {
            state->overlay = YVEX_TUI_OVERLAY_NONE;
            state->focus = YVEX_TUI_FOCUS_COMPOSER;
            state->redraw = 1;
            return YVEX_TUI_INPUT_NONE;
        }
        if (state->overlay == YVEX_TUI_OVERLAY_REMOTE &&
            !state->remote_count)
            return YVEX_TUI_INPUT_REMOTE_SEARCH;
        return YVEX_TUI_INPUT_SUBMIT;
    }
    if (byte == '\n') {
        if (state->overlay == YVEX_TUI_OVERLAY_NONE ||
            state->overlay == YVEX_TUI_OVERLAY_SLASH)
            insert_text(state, &byte, 1u);
        return YVEX_TUI_INPUT_NONE;
    }
    if (byte == '\t') {
        if (state->overlay == YVEX_TUI_OVERLAY_SLASH)
            slash_complete(state);
        else if (state->overlay == YVEX_TUI_OVERLAY_MODEL) {
            state->launch_field = (state->launch_field + 1u) % 2u;
            state->focus = state->launch_field
                               ? YVEX_TUI_FOCUS_MODEL_PROFILE
                               : YVEX_TUI_FOCUS_MODEL_SEARCH;
            state->redraw = 1;
        }
        return YVEX_TUI_INPUT_NONE;
    }
    if (byte == 0x04u) return YVEX_TUI_INPUT_EXIT;
    if (byte == 0x0cu) {
        state->redraw = 1;
        return YVEX_TUI_INPUT_REFRESH;
    }
    if (byte == 0x0fu) {
        yvex_tui_runtime_launch_open(state, state->selected_model, 0);
        return YVEX_TUI_INPUT_NONE;
    }
    if (byte == 0x10u && state->overlay == YVEX_TUI_OVERLAY_NONE) {
        if (!state->composer.count)
            insert_text(state, (const unsigned char *)"/", 1u);
        return YVEX_TUI_INPUT_NONE;
    }
    if (byte == 0x12u && state->overlay == YVEX_TUI_OVERLAY_MODEL) {
        state->overlay = YVEX_TUI_OVERLAY_REMOTE;
        state->focus = YVEX_TUI_FOCUS_REMOTE_SEARCH;
        state->redraw = 1;
        return YVEX_TUI_INPUT_NONE;
    }
    if (byte == 0x7fu || byte == 0x08u) {
        if (state->overlay == YVEX_TUI_OVERLAY_MODEL ||
            state->overlay == YVEX_TUI_OVERLAY_SESSION ||
            state->overlay == YVEX_TUI_OVERLAY_REMOTE)
            search_erase(state);
        else {
            yvex_tui_composer_erase(&state->composer, 1);
            slash_sync(state);
            state->redraw = 1;
        }
        return YVEX_TUI_INPUT_NONE;
    }
    if (byte == '?' && state->overlay == YVEX_TUI_OVERLAY_NONE &&
        !state->composer.count) {
        state->overlay = YVEX_TUI_OVERLAY_HELP;
        state->focus = YVEX_TUI_FOCUS_OVERLAY;
        state->redraw = 1;
        return YVEX_TUI_INPUT_NONE;
    }
    if (byte == 'a' && state->overlay == YVEX_TUI_OVERLAY_REMOTE &&
        state->remote_count)
        return YVEX_TUI_INPUT_ACQUIRE;
    if (byte >= 0x20u) insert_text(state, &byte, 1u);
    return YVEX_TUI_INPUT_NONE;
}

static yvex_tui_input_action sequence_action(yvex_tui_input *input,
                                             yvex_tui_state *state)
{
    const unsigned char *sequence = input->sequence;
    size_t count = input->sequence_count;
    if (count == 5u && !memcmp(sequence, "[200~", 5u)) {
        input->paste = 1;
        return YVEX_TUI_INPUT_NONE;
    }
    if (count == 6u && !memcmp(sequence, "[13;2u", 6u)) {
        const unsigned char newline = '\n';
        insert_text(state, &newline, 1u);
        return YVEX_TUI_INPUT_NONE;
    }
    if (count == 2u && sequence[0] == '[') {
        if (sequence[1] == 'A' || sequence[1] == 'B') {
            int direction = sequence[1] == 'A' ? -1 : 1;
            if (state->overlay == YVEX_TUI_OVERLAY_SLASH)
                slash_move(state, direction);
            else if (state->overlay == YVEX_TUI_OVERLAY_MODEL &&
                     state->launch_field == 1u)
                launch_profile_move(state, direction);
            else if (state->overlay != YVEX_TUI_OVERLAY_NONE)
                yvex_tui_selection_move(state, direction, 8u);
            else if (!state->composer.multiline)
                yvex_tui_composer_history_move(&state->composer, direction);
        } else if (sequence[1] == 'C') {
            yvex_tui_composer_right(&state->composer);
        } else if (sequence[1] == 'D') {
            yvex_tui_composer_left(&state->composer);
        } else if (sequence[1] == 'H') {
            yvex_tui_composer_home(&state->composer);
        } else if (sequence[1] == 'F') {
            yvex_tui_composer_end(&state->composer);
        } else if (sequence[1] == 'Z' &&
                   state->overlay == YVEX_TUI_OVERLAY_MODEL) {
            state->launch_field = state->launch_field ? 0u : 1u;
        }
        state->redraw = 1;
        return YVEX_TUI_INPUT_NONE;
    }
    if (count == 3u && sequence[0] == '[' && sequence[2] == '~') {
        if (sequence[1] == '1' || sequence[1] == '7')
            yvex_tui_composer_home(&state->composer);
        else if (sequence[1] == '4' || sequence[1] == '8')
            yvex_tui_composer_end(&state->composer);
        else if (sequence[1] == '3')
            yvex_tui_composer_erase(&state->composer, 0);
        else if (sequence[1] == '5')
            yvex_tui_selection_move(state, -1, 8u);
        else if (sequence[1] == '6')
            yvex_tui_selection_move(state, 1, 8u);
        state->redraw = 1;
        return YVEX_TUI_INPUT_NONE;
    }
    return escape_action(state);
}

static int sequence_complete(const unsigned char *sequence, size_t count)
{
    if (!count) return 0;
    if (sequence[0] != '[') return 1;
    if (count == 2u && ((sequence[1] >= 'A' && sequence[1] <= 'D') ||
                       sequence[1] == 'F' || sequence[1] == 'H' ||
                       sequence[1] == 'Z'))
        return 1;
    if (sequence[count - 1u] == '~' || sequence[count - 1u] == 'u') return 1;
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
        return escape_action(state);
    }
    return YVEX_TUI_INPUT_NONE;
}
