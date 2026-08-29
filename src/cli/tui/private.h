/*
 * Keep terminal mechanics, deterministic projection state, and protocol ingestion separate.
 * The resident server remains the only owner of hosted model, session, and request lifetimes.
 */
#ifndef SRC_CLI_TUI_PRIVATE_H_INCLUDED
#define SRC_CLI_TUI_PRIVATE_H_INCLUDED

#include <signal.h>
#include <stddef.h>
#include <sys/types.h>
#include <termios.h>

#include <yvex/catalog.h>
#include <yvex/registry.h>
#include <yvex/server.h>

#include "src/cli/io/private.h"

#define YVEX_TUI_ACTIVITY_CAP 96u
#define YVEX_TUI_ACTIVITY_TEXT_CAP 4096u
#define YVEX_TUI_COMPOSER_CAP 65536u
#define YVEX_TUI_HISTORY_CAP 32u
#define YVEX_TUI_HISTORY_TEXT_CAP 4096u
#define YVEX_TUI_PENDING_CAP 8u
#define YVEX_TUI_PENDING_TEXT_CAP 4096u
#define YVEX_TUI_SESSION_CAP 128u
#define YVEX_TUI_EVENT_CAP 32u
#define YVEX_TUI_ENGINE_CAP 32u
#define YVEX_TUI_REMOTE_CAP 32u
#define YVEX_TUI_INPUT_SEQUENCE_CAP 16u
#define YVEX_TUI_EXECUTABLE_CAP 4096u
#define YVEX_TUI_LAUNCH_ARG_CAP 12u
#define YVEX_TUI_ACQUIRE_ARG_CAP 16u
#define YVEX_TUI_BOOTSTRAP_DIAGNOSTIC_CAP 2048u

typedef enum {
    YVEX_TUI_LAYOUT_COMPACT = 0,
    YVEX_TUI_LAYOUT_STANDARD,
    YVEX_TUI_LAYOUT_WIDE
} yvex_tui_layout;

typedef enum {
    YVEX_TUI_FOCUS_COMPOSER = 0,
    YVEX_TUI_FOCUS_OVERLAY,
    YVEX_TUI_FOCUS_MODEL_SEARCH,
    YVEX_TUI_FOCUS_SESSION_SEARCH,
    YVEX_TUI_FOCUS_REMOTE_SEARCH,
    YVEX_TUI_FOCUS_MODEL_PROFILE
} yvex_tui_focus;

typedef enum {
    YVEX_TUI_OVERLAY_NONE = 0,
    YVEX_TUI_OVERLAY_SLASH,
    YVEX_TUI_OVERLAY_HELP,
    YVEX_TUI_OVERLAY_MODEL,
    YVEX_TUI_OVERLAY_SESSION,
    YVEX_TUI_OVERLAY_REMOTE
} yvex_tui_overlay;

typedef enum {
    YVEX_TUI_RUNTIME_NONE = 0,
    YVEX_TUI_RUNTIME_LAUNCH_REQUESTED,
    YVEX_TUI_RUNTIME_LAUNCHING,
    YVEX_TUI_RUNTIME_WAITING_PROTOCOL,
    YVEX_TUI_RUNTIME_ENGINE_LOADING,
    YVEX_TUI_RUNTIME_CONNECTED_EXTERNAL,
    YVEX_TUI_RUNTIME_CONNECTED_OWNED,
    YVEX_TUI_RUNTIME_LAUNCH_FAILED,
    YVEX_TUI_RUNTIME_SHUTDOWN_REQUESTED,
    YVEX_TUI_RUNTIME_STOPPED
} yvex_tui_runtime_lifecycle;

typedef enum {
    YVEX_TUI_LAUNCH_FAILURE_NONE = 0,
    YVEX_TUI_LAUNCH_FAILURE_PREFLIGHT,
    YVEX_TUI_LAUNCH_FAILURE_SPAWN,
    YVEX_TUI_LAUNCH_FAILURE_BOOTSTRAP,
    YVEX_TUI_LAUNCH_FAILURE_PROTOCOL,
    YVEX_TUI_LAUNCH_FAILURE_ENGINE_LOAD
} yvex_tui_launch_failure;

typedef enum {
    YVEX_TUI_MODEL_CATALOG_UNLOADED = 0,
    YVEX_TUI_MODEL_CATALOG_READY,
    YVEX_TUI_MODEL_CATALOG_EMPTY,
    YVEX_TUI_MODEL_CATALOG_ERROR
} yvex_tui_model_catalog_status;

typedef yvex_cli_interactive_connection yvex_tui_connection;

#define YVEX_TUI_CONNECTION_UNAVAILABLE YVEX_CLI_INTERACTIVE_UNAVAILABLE
#define YVEX_TUI_CONNECTION_CONNECTED YVEX_CLI_INTERACTIVE_CONNECTED
#define YVEX_TUI_CONNECTION_DISCONNECTED YVEX_CLI_INTERACTIVE_DISCONNECTED
#define YVEX_TUI_CONNECTION_INCOMPATIBLE YVEX_CLI_INTERACTIVE_INCOMPATIBLE

typedef enum {
    YVEX_TUI_ACTIVITY_SYSTEM = 0,
    YVEX_TUI_ACTIVITY_USER,
    YVEX_TUI_ACTIVITY_RUNTIME,
    YVEX_TUI_ACTIVITY_GENERATION,
    YVEX_TUI_ACTIVITY_ERROR
} yvex_tui_activity_kind;

typedef enum {
    YVEX_TUI_SEVERITY_INFO = 0,
    YVEX_TUI_SEVERITY_SUCCESS,
    YVEX_TUI_SEVERITY_WARNING,
    YVEX_TUI_SEVERITY_ERROR
} yvex_tui_severity;

typedef struct {
    unsigned int rows, columns;
    yvex_tui_layout layout;
    int color, unicode;
} yvex_tui_terminal_view;

typedef struct {
    yvex_tui_activity_kind kind;
    yvex_tui_severity severity;
    yvex_client_stream_channel channel;
    unsigned long long order;
    char session[YVEX_SERVER_SESSION_NAME_CAP];
    char correlation[YVEX_SERVER_ID_CAP];
    char text[YVEX_TUI_ACTIVITY_TEXT_CAP];
} yvex_tui_activity;

typedef struct {
    char name[YVEX_SERVER_SESSION_NAME_CAP];
    yvex_server_session_state state;
    unsigned long long position, turns, context_used, kv_used_bytes;
    int attached, kv_used_available;
} yvex_tui_session_row;

typedef struct {
    char identity[YVEX_MODEL_LIBRARY_ID_CAP];
    char display_name[YVEX_MODEL_LIBRARY_NAME_CAP];
    char family[64], model[YVEX_MODEL_LIBRARY_NAME_CAP];
    char runtime_target[YVEX_MODEL_LIBRARY_NAME_CAP];
    unsigned long long remote_count, source_count, artifact_count;
    unsigned long long profile_count, launchable_profile_count, runtime_count;
    int remote_available, source_local, artifact_ready, startup_ready, resident;
    char startup_reason[160];
} yvex_tui_model_row;

typedef struct {
    char profile[YVEX_MODEL_LIBRARY_NAME_CAP];
} yvex_tui_launch_request;

typedef struct {
    char executable[YVEX_TUI_EXECUTABLE_CAP];
    char console[8];
    char *argv[YVEX_TUI_LAUNCH_ARG_CAP];
    int argc;
} yvex_tui_launch_command;

typedef struct {
    char executable[YVEX_TUI_EXECUTABLE_CAP];
    char repository[YVEX_REMOTE_REPOSITORY_CAP];
    char family[YVEX_REMOTE_FAMILY_CAP];
    char name[YVEX_REMOTE_NAME_CAP];
    char revision[YVEX_REMOTE_REVISION_CAP];
    char *argv[YVEX_TUI_ACQUIRE_ARG_CAP];
    int argc;
} yvex_tui_acquire_command;

typedef struct {
    char executable[YVEX_TUI_EXECUTABLE_CAP];
    pid_t pid;
    int exec_read_fd, diagnostic_read_fd;
    int running, exec_confirmed, exec_error;
    int exit_known, exit_status;
    size_t diagnostic_count;
    int diagnostic_truncated;
    char diagnostic[YVEX_TUI_BOOTSTRAP_DIAGNOSTIC_CAP];
    unsigned long long started_ns;
} yvex_tui_launcher;

typedef struct {
    unsigned char bytes[YVEX_TUI_COMPOSER_CAP];
    size_t count, cursor;
    char history[YVEX_TUI_HISTORY_CAP][YVEX_TUI_HISTORY_TEXT_CAP];
    size_t history_count, history_cursor;
    int multiline;
} yvex_tui_composer;

typedef struct {
    unsigned char bytes[YVEX_TUI_PENDING_TEXT_CAP];
    size_t count;
    char session[YVEX_SERVER_SESSION_NAME_CAP];
    char engine[YVEX_MODEL_LIBRARY_NAME_CAP];
    unsigned long long engine_generation;
} yvex_tui_pending_message;

typedef struct {
    char turn_id[YVEX_SERVER_ID_CAP];
    unsigned long long prompt_tokens, reused_tokens, prefill_tokens;
    unsigned long long generated_tokens;
    unsigned long long h2d_bytes, d2h_bytes, d2d_bytes;
    unsigned long long kernel_launches, queue_synchronizations;
    unsigned long long device_synchronizations, accelerated_matrix_launches;
    unsigned long long graph_launches, graph_captures, graph_replays;
    unsigned long long attention_calls, attention_cache_hits;
    unsigned long long attention_cache_misses, moe_row_expert_pairs;
    unsigned long long moe_weight_bytes, output_rows, logits_d2h_bytes;
    double queue_seconds, prefill_seconds, first_token_seconds;
    double decode_seconds, publication_seconds, total_seconds;
    double prefill_rate, decode_rate;
    double synchronization_seconds, attention_seconds, moe_seconds;
    double output_seconds;
    int turn_available, profile_available;
} yvex_tui_turn_observation;

typedef struct {
    yvex_tui_terminal_view terminal;
    yvex_tui_focus focus;
    yvex_tui_overlay overlay;
    yvex_tui_connection connection;
    yvex_tui_runtime_lifecycle runtime_lifecycle;
    yvex_server_summary runtime;
    yvex_console_status console;
    yvex_client_partial_turn partial_turn;
    int runtime_available, console_available;
    char connection_reason[YVEX_SERVER_REASON_CAP];
    char active_session[YVEX_SERVER_SESSION_NAME_CAP];
    yvex_cli_engine_binding active_engine;
    yvex_server_engine_summary engines[YVEX_TUI_ENGINE_CAP];
    size_t engine_count;
    yvex_tui_session_row sessions[YVEX_TUI_SESSION_CAP];
    size_t session_count, selected_session, session_viewport;
    yvex_tui_model_row *models;
    yvex_model_library *model_library;
    size_t model_count, model_capacity, selected_model, model_viewport;
    yvex_tui_model_catalog_status model_catalog_status;
    char model_catalog_reason[256];
    char model_search[128];
    size_t model_search_count;
    char discover_query[128];
    size_t discover_query_count, selected_remote, remote_viewport;
    yvex_remote_model remote_models[YVEX_TUI_REMOTE_CAP];
    size_t remote_count;
    char remote_search_reason[YVEX_SERVER_REASON_CAP];
    int remote_search_running, remote_detail;
    int acquisition_running, acquisition_exit_known, acquisition_exit_status;
    char acquisition_diagnostic[512];
    char session_search[128];
    size_t session_search_count;
    yvex_tui_activity activities[YVEX_TUI_ACTIVITY_CAP];
    size_t activity_start, activity_count, activity_scroll;
    unsigned long long next_activity_order;
    yvex_server_event events[YVEX_TUI_EVENT_CAP];
    size_t event_start, event_count;
    yvex_tui_turn_observation last_turn;
    yvex_tui_composer composer;
    size_t slash_selected;
    yvex_tui_pending_message pending[YVEX_TUI_PENDING_CAP];
    size_t pending_start, pending_count;
    int pending_review;
    yvex_tui_launch_request launch_request;
    size_t launch_selected_model;
    size_t launch_selected_profile;
    unsigned int launch_field, runtime_action;
    pid_t launched_pid;
    int launch_exit_status, launch_exec_error, restart_pending;
    int engine_load_requested;
    yvex_tui_launch_failure launch_failure;
    char launch_failure_reason[YVEX_SERVER_REASON_CAP];
    char launch_diagnostic[YVEX_TUI_BOOTSTRAP_DIAGNOSTIC_CAP];
    unsigned long long launch_started_ns;
    yvex_reasoning_policy reasoning_policy;
    yvex_client_generation_phase generation_phase;
    yvex_tui_severity notice_severity;
    char notice[256];
    int generation_active, submit_after_launch, redraw, shutdown_requested;
    unsigned long long maximum_new_tokens;
} yvex_tui_state;

typedef struct {
    int input_fd, output_fd;
    int signal_read_fd, signal_write_fd;
    struct termios saved_termios;
    struct sigaction saved_winch, saved_interrupt, saved_terminate;
    volatile sig_atomic_t resize_pending;
    volatile sig_atomic_t interrupt_pending;
    volatile sig_atomic_t terminate_pending;
    int termios_saved, termios_changed, signals_installed;
    int alternate_screen, cursor_hidden, paste_enabled;
} yvex_tui_terminal;

typedef enum {
    YVEX_TUI_INPUT_NONE = 0,
    YVEX_TUI_INPUT_SUBMIT,
    YVEX_TUI_INPUT_CANCEL,
    YVEX_TUI_INPUT_REMOTE_SEARCH,
    YVEX_TUI_INPUT_ACQUIRE,
    YVEX_TUI_INPUT_EXIT,
    YVEX_TUI_INPUT_REFRESH
} yvex_tui_input_action;

typedef struct {
    unsigned char sequence[YVEX_TUI_INPUT_SEQUENCE_CAP];
    size_t sequence_count;
    unsigned char utf8[4];
    size_t utf8_count, utf8_expected;
    int escape, paste;
} yvex_tui_input;

int yvex_tui_terminal_open(yvex_tui_terminal *terminal, int input_fd,
                           int output_fd, yvex_error *err);
void yvex_tui_terminal_close(yvex_tui_terminal *terminal);
int yvex_tui_terminal_dimensions(yvex_tui_terminal *terminal,
                                 unsigned int *rows, unsigned int *columns);
int yvex_tui_terminal_signal_fd(const yvex_tui_terminal *terminal);
void yvex_tui_terminal_take_signals(yvex_tui_terminal *terminal, int *resize,
                                    int *interrupt, int *terminate);
int yvex_tui_terminal_write(yvex_tui_terminal *terminal,
                            const char *bytes, size_t count);

void yvex_tui_state_init(yvex_tui_state *state, unsigned int rows,
                         unsigned int columns, const char *session);
void yvex_tui_state_close(yvex_tui_state *state);
void yvex_tui_state_resize(yvex_tui_state *state, unsigned int rows,
                           unsigned int columns);
void yvex_tui_state_connection(yvex_tui_state *state,
                               yvex_tui_connection connection,
                               const char *reason);
void yvex_tui_state_message(yvex_tui_state *state,
                            const yvex_cli_interactive_event *event);
void yvex_tui_activity_add(yvex_tui_state *state, yvex_tui_activity_kind kind,
                           yvex_tui_severity severity,
                           yvex_client_stream_channel channel,
                           const char *text);
int yvex_tui_composer_insert(yvex_tui_composer *composer,
                             const unsigned char *bytes, size_t count);
void yvex_tui_composer_left(yvex_tui_composer *composer);
void yvex_tui_composer_right(yvex_tui_composer *composer);
void yvex_tui_composer_home(yvex_tui_composer *composer);
void yvex_tui_composer_end(yvex_tui_composer *composer);
void yvex_tui_composer_erase(yvex_tui_composer *composer, int backward);
void yvex_tui_composer_clear(yvex_tui_composer *composer);
void yvex_tui_composer_history_push(yvex_tui_composer *composer);
void yvex_tui_composer_history_move(yvex_tui_composer *composer, int direction);
int yvex_tui_pending_enqueue(yvex_tui_state *state);
const yvex_tui_pending_message *yvex_tui_pending_front(
    const yvex_tui_state *state);
void yvex_tui_pending_pop(yvex_tui_state *state);
void yvex_tui_pending_restore(yvex_tui_state *state);
int yvex_tui_models_load(yvex_tui_state *state, const char *registry_path,
                         yvex_error *err);
size_t yvex_tui_model_visible_count(const yvex_tui_state *state);
size_t yvex_tui_model_visible_at(const yvex_tui_state *state, size_t ordinal);
size_t yvex_tui_session_visible_count(const yvex_tui_state *state);
size_t yvex_tui_session_visible_at(const yvex_tui_state *state,
                                   size_t ordinal);
void yvex_tui_selection_move(yvex_tui_state *state, int direction,
                             size_t page_rows);
size_t yvex_tui_startup_model_count(const yvex_tui_state *state);
const yvex_model_runtime_profile_fact *yvex_tui_launch_profile(
    const yvex_tui_state *state);
void yvex_tui_runtime_launch_open(yvex_tui_state *state, size_t model_index,
                                  int restart);
void yvex_tui_runtime_launch_started(yvex_tui_state *state, pid_t pid,
                                     unsigned long long started_ns);
void yvex_tui_runtime_launch_failed(yvex_tui_state *state,
                                    yvex_tui_launch_failure failure,
                                    const char *reason, int exec_error,
                                    int exit_status, const char *diagnostic);
void yvex_tui_runtime_stop_requested(yvex_tui_state *state, int restart);
void yvex_tui_remote_search_started(yvex_tui_state *state);
void yvex_tui_remote_search_publish(yvex_tui_state *state,
                                    const yvex_remote_model *results,
                                    size_t result_count,
                                    const char *reason);

int yvex_tui_launcher_open(yvex_tui_launcher *launcher,
                           const char *executable_hint, yvex_error *err);
void yvex_tui_launcher_close(yvex_tui_launcher *launcher);
int yvex_tui_launch_prepare(const char *executable,
                            yvex_tui_launch_command *command, yvex_error *err);
int yvex_tui_launcher_start(yvex_tui_launcher *launcher,
                            unsigned long long started_ns, yvex_error *err);
int yvex_tui_launcher_exec_fd(const yvex_tui_launcher *launcher);
int yvex_tui_launcher_exec_take(yvex_tui_launcher *launcher);
int yvex_tui_launcher_diagnostic_fd(const yvex_tui_launcher *launcher);
int yvex_tui_launcher_diagnostic_take(yvex_tui_launcher *launcher);
int yvex_tui_launcher_reap(yvex_tui_launcher *launcher);
int yvex_tui_acquire_prepare(const char *executable,
                             const yvex_remote_model *remote,
                             yvex_tui_acquire_command *command,
                             yvex_error *err);

yvex_tui_input_action yvex_tui_input_byte(yvex_tui_input *input,
                                           yvex_tui_state *state,
                                           unsigned char byte);
yvex_tui_input_action yvex_tui_input_flush(yvex_tui_input *input,
                                            yvex_tui_state *state);

int yvex_tui_render(const yvex_tui_state *state, char *output, size_t capacity,
                    size_t *count, unsigned int *cursor_row,
                    unsigned int *cursor_column);

int yvex_tui_run(const char *executable, const char *model, const char *session,
                 unsigned long long maximum_new_tokens);
int yvex_tui_dispatch(int argc, char **argv);

#endif
