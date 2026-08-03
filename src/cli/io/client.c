/*
 * Runtime-facing commands are deliberately thin local-protocol clients. Even though the yvex ELF
 * also contains finite offline-engine adapters, this lane cannot open artifacts, initialize CUDA,
 * or call generation directly; every hosted operation crosses yvexd's protocol boundary.
 *
 * The file also owns the linear interactive console. Operation identity and argument schemas come
 * from the compiled registry; terminal state and rendering remain client-owned projections.
 */
#define _POSIX_C_SOURCE 200809L
#define _XOPEN_SOURCE 700

#include <build_commit.h>
#include <operator/registry.h>
#include "src/cli/input/private.h"
#include "src/cli/io/private.h"
#include "src/cli/private.h"

#include <yvex/server.h>

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

#define CLIENT_REPL_LINE_MAX 65536u
#define CLIENT_REPL_HISTORY_MAX 64u

typedef struct {
    unsigned long long maximum_new_tokens, seed, top_k;
    double temperature, top_p, min_p, typical_p;
    int stochastic, seed_present;
} client_turn_options;
typedef struct {
    char session[YVEX_SERVER_SESSION_NAME_CAP];
    atomic_int done, interrupts, force_exit;
    pthread_t thread;
    sigset_t previous_mask;
    int ready;
} client_turn_signals;
typedef struct {
    char *entry[CLIENT_REPL_HISTORY_MAX];
    size_t count;
} client_repl_history;
typedef struct {
    char name[YVEX_SERVER_SESSION_NAME_CAP];
    char artifact[PATH_MAX];
    char binding[PATH_MAX];
    char target[128];
    char backend[8];
    char mode[16];
    unsigned long long context;
} client_model_config;
static volatile sig_atomic_t repl_signal_state;
static int console_status(const char *session_name);
static int console_status_fetch(const char *session_name,
                                yvex_client_message *message,
                                yvex_error *err);
static void render_console_status(const yvex_client_message *message, int startup);

static void discovery_json_string(FILE *output, const char *value)
{
    const unsigned char *cursor = (const unsigned char *)value;
    fputc('"', output);
    while (*cursor) {
        unsigned char byte = *cursor++;
        if (byte == '"' || byte == '\\') fprintf(output, "\\%c", (int)byte);
        else if (byte == '\b') fputs("\\b", output);
        else if (byte == '\f') fputs("\\f", output);
        else if (byte == '\n') fputs("\\n", output);
        else if (byte == '\r') fputs("\\r", output);
        else if (byte == '\t') fputs("\\t", output);
        else if (byte < 0x20u) fprintf(output, "\\u%04x", (unsigned int)byte);
        else fputc((int)byte, output);
    }
    fputc('"', output);
}
static void discovery_json_list(FILE *output, const char *value, int delimiter)
{
    const char *cursor = value;
    int first = 1;
    fputc('[', output);
    if (strcmp(value, "none")) {
        while (*cursor) {
            const char *end = strchr(cursor, delimiter);
            size_t extent = end ? (size_t)(end - cursor) : strlen(cursor);
            char item[512];
            if (extent >= sizeof(item)) extent = sizeof(item) - 1u;
            memcpy(item, cursor, extent);
            item[extent] = '\0';
            if (!first) fputc(',', output);
            discovery_json_string(output, item);
            first = 0;
            if (!end) break;
            cursor = end + 1;
        }
    }
    fputc(']', output);
}
static const char *visibility_name(yvex_operator_visibility visibility)
{
    switch (visibility) {
    case YVEX_OPERATOR_VISIBILITY_PRODUCT_DEFAULT: return "product-default";
    case YVEX_OPERATOR_VISIBILITY_PRODUCT_ADVANCED: return "product-advanced";
    case YVEX_OPERATOR_VISIBILITY_ENGINEERING: return "engineering";
    case YVEX_OPERATOR_VISIBILITY_AUTOMATION: return "automation";
    case YVEX_OPERATOR_VISIBILITY_API_ONLY: return "API-only";
    case YVEX_OPERATOR_VISIBILITY_TEST_ONLY: return "test-only";
    case YVEX_OPERATOR_VISIBILITY_REMOVED: return "removed";
    }
    return "unknown";
}
static const char *plane_name(yvex_operator_plane plane)
{
    static const char *const names[] = {
        "Compile", "Execute", "Inspect", "Integrate", "Profile", "Run", "System"};
    return (unsigned int)plane < sizeof(names) / sizeof(names[0]) ? names[plane]
                                                                  : "Unknown";
}
static const char *lane_name(yvex_operator_lane lane)
{
    switch (lane) {
    case YVEX_OPERATOR_LANE_RUNTIME_CLIENT: return "runtime-client";
    case YVEX_OPERATOR_LANE_OFFLINE_ENGINE: return "offline-engine";
    case YVEX_OPERATOR_LANE_DAEMON_ENTRYPOINT: return "daemon-entrypoint";
    case YVEX_OPERATOR_LANE_REPL_LOCAL: return "REPL-local";
    case YVEX_OPERATOR_LANE_API_ONLY: return "API-only";
    case YVEX_OPERATOR_LANE_TEST_ONLY: return "test-only";
    }
    return "unknown";
}
static int descriptor_has_prefix(const yvex_operator_descriptor *descriptor,
                                 size_t count, const char *const *path)
{
    size_t index;
    if (count > descriptor->command_word_count) return 0;
    for (index = 0u; index < count; ++index)
        if (strcmp(path[index], descriptor->command_words[index])) return 0;
    return 1;
}
static void render_leaf_usage(FILE *output,
                              const yvex_operator_descriptor *descriptor)
{
    size_t index;
    fprintf(output, "usage: yvex %s", descriptor->command_path);
    for (index = 0u; index < descriptor->argument_count; ++index) {
        const yvex_operator_argument_descriptor *argument = &descriptor->arguments[index];
        if (!strcmp(argument->multiplicity, "many"))
            fprintf(output, " [%s ...]", argument->name);
        else if (argument->required)
            fprintf(output, " %s", argument->name);
        else
            fprintf(output, " [%s]", argument->name);
    }
    if (descriptor->flag_count) fputs(" [options]", output);
    fputc('\n', output);
}
static void render_leaf_help(const yvex_operator_descriptor *descriptor)
{
    size_t index;
    render_leaf_usage(stdout, descriptor);
    printf("\n%s\n\noperation: %s\nplane: %s\nvisibility: %s\nlane: %s\n",
           descriptor->summary, descriptor->operation_id, plane_name(descriptor->plane),
           visibility_name(descriptor->visibility), lane_name(descriptor->lane));
    if (descriptor->flag_count) {
        puts("\noptions:");
        for (index = 0u; index < descriptor->flag_count; ++index) {
            const yvex_operator_flag_descriptor *flag = &descriptor->flags[index];
            printf("  %-24s%s%s\n", flag->name, flag->takes_value ? " VALUE" : "",
                   strcmp(flag->aliases, "none") ? "  (alias available)" : "");
        }
    }
}
void yvex_client_render_usage_error(const yvex_operator_descriptor *operation)
{
    render_leaf_usage(stderr, operation);
}
static void render_discovery_aliases(size_t operation_index)
{
    size_t index;
    int first = 1;
    fputc('[', stdout);
    for (index = 0u; index < yvex_operator_alias_count; ++index) {
        if (yvex_operator_aliases[index].operation_index != operation_index) continue;
        if (!first) fputc(',', stdout);
        discovery_json_string(stdout, yvex_operator_aliases[index].path);
        first = 0;
    }
    fputc(']', stdout);
}
static void render_discovery_arguments(
    const yvex_operator_argument_descriptor *arguments, size_t count)
{
    size_t index;
#define JSON_ARGUMENT_FIELD(name, value) \
    do { fputs("\"" name "\":", stdout); discovery_json_string(stdout, (value)); } while (0)
    fputc('[', stdout);
    for (index = 0u; index < count; ++index) {
        const yvex_operator_argument_descriptor *argument = &arguments[index];
        if (index) fputc(',', stdout);
        fputc('{', stdout); JSON_ARGUMENT_FIELD("name", argument->name);
        fputc(',', stdout); JSON_ARGUMENT_FIELD("type", argument->value_type);
        printf(",\"required\":%s,", argument->required ? "true" : "false");
        JSON_ARGUMENT_FIELD("multiplicity", argument->multiplicity);
        fputc(',', stdout); JSON_ARGUMENT_FIELD("range", argument->range);
        fputs(",\"enum_values\":", stdout);
        discovery_json_list(stdout, argument->enum_values, '|');
        fputc(',', stdout);
        JSON_ARGUMENT_FIELD("completion_provider", argument->completion_provider);
        fputc(',', stdout);
        JSON_ARGUMENT_FIELD("sensitive_display", argument->sensitive_display);
        fputc(',', stdout); JSON_ARGUMENT_FIELD("validator", argument->validator);
        fputc('}', stdout);
    }
    fputc(']', stdout);
#undef JSON_ARGUMENT_FIELD
}
static void render_discovery_operation(size_t operation_index,
                                       const yvex_operator_descriptor *descriptor)
{
    size_t index;
#define JSON_FIELD(name, value) \
    do { fputs("\"" name "\":", stdout); discovery_json_string(stdout, (value)); } while (0)
    fputc('{', stdout);
    printf("\"schema_version\":%u,", descriptor->schema_version);
    JSON_FIELD("operation_id", descriptor->operation_id);
    fputc(',', stdout); JSON_FIELD("command_path", descriptor->command_path);
    fputs(",\"aliases\":", stdout); render_discovery_aliases(operation_index);
    fputc(',', stdout); JSON_FIELD("visibility", visibility_name(descriptor->visibility));
    fputc(',', stdout); JSON_FIELD("plane", plane_name(descriptor->plane));
    fputc(',', stdout); JSON_FIELD("lane", lane_name(descriptor->lane));
    fputc(',', stdout); JSON_FIELD("summary", descriptor->summary);
    fputs(",\"arguments\":", stdout);
    render_discovery_arguments(descriptor->arguments, descriptor->argument_count);
    fputs(",\"slash_arguments\":", stdout);
    render_discovery_arguments(descriptor->slash_arguments,
                               descriptor->slash_argument_count);
    fputs(",\"flags\":[", stdout);
    for (index = 0u; index < descriptor->flag_count; ++index) {
        const yvex_operator_flag_descriptor *flag = &descriptor->flags[index];
        if (index) fputc(',', stdout);
        fputc('{', stdout); JSON_FIELD("name", flag->name);
        fputs(",\"aliases\":", stdout); discovery_json_list(stdout, flag->aliases, '|');
        fputc(',', stdout); JSON_FIELD("type", flag->value_type);
        printf(",\"takes_value\":%s,", flag->takes_value ? "true" : "false");
        printf("\"required\":%s,", flag->required ? "true" : "false");
        JSON_FIELD("multiplicity", flag->multiplicity);
        fputc(',', stdout); JSON_FIELD("default_provider", flag->default_provider);
        fputc(',', stdout); JSON_FIELD("range", flag->range);
        fputs(",\"enum_values\":", stdout); discovery_json_list(stdout, flag->enum_values, '|');
        fputs(",\"conflicts\":", stdout); discovery_json_list(stdout, flag->conflicts, '|');
        fputs(",\"dependencies\":", stdout);
        discovery_json_list(stdout, flag->dependencies, '|');
        fputc(',', stdout); JSON_FIELD("environment", flag->environment);
        fputc(',', stdout); JSON_FIELD("config", flag->config);
        fputc(',', stdout); JSON_FIELD("protocol_field", flag->protocol_field);
        fputc(',', stdout); JSON_FIELD("output_interaction", flag->output_interaction);
        fputc(',', stdout); JSON_FIELD("deprecation", flag->deprecation);
        fputc(',', stdout); JSON_FIELD("validator", flag->validator);
        fputc('}', stdout);
    }
    fputs("],\"default_providers\":", stdout);
    discovery_json_list(stdout, descriptor->default_providers, '|');
    fputs(",\"validators\":", stdout);
    discovery_json_list(stdout, descriptor->validator_ids, '|');
    fputs(",\"input_schema\":", stdout); discovery_json_string(stdout, descriptor->input_schema);
    fputs(",\"result_schema\":", stdout); discovery_json_string(stdout, descriptor->result_schema);
    fputs(",\"side_effects\":", stdout); discovery_json_string(stdout, descriptor->side_effects);
    fputs(",\"tty_policy\":", stdout); discovery_json_string(stdout, descriptor->tty_policy);
    fputs(",\"requirements\":{", stdout); JSON_FIELD("daemon", descriptor->daemon_requirement);
    fputc(',', stdout); JSON_FIELD("model", descriptor->model_requirement);
    fputc(',', stdout); JSON_FIELD("artifact", descriptor->artifact_requirement);
    fputc(',', stdout); JSON_FIELD("backend", descriptor->backend_requirement);
    fputs("},\"output_schemas\":[", stdout); discovery_json_string(stdout, descriptor->result_schema);
    fputs("],\"adapter_id\":", stdout); discovery_json_string(stdout, descriptor->adapter_id);
    fputs(",\"renderer_id\":", stdout); discovery_json_string(stdout, descriptor->renderer_id);
    fputs(",\"completion_provider\":", stdout);
    discovery_json_string(stdout, descriptor->completion_provider);
    fputs(",\"projections\":{\"cli\":", stdout);
    fputs(descriptor->cli_projection ? "true" : "false", stdout);
    fputs(",\"slash\":", stdout); discovery_json_string(stdout, descriptor->slash_projection);
    fputs(",\"protocol\":", stdout); discovery_json_string(stdout, descriptor->protocol_operation);
    fputs(",\"future_tui\":", stdout); discovery_json_string(stdout, descriptor->future_tui_projection);
    fputs("},\"deprecation\":", stdout); discovery_json_string(stdout, descriptor->deprecation_state);
    fputs(",\"superseded_by\":", stdout);
    discovery_json_list(stdout, descriptor->superseded_by, '|');
    fputs(",\"test_owner\":", stdout); discovery_json_string(stdout, descriptor->test_owner);
    fputs(",\"documentation_owner\":", stdout);
    discovery_json_string(stdout, descriptor->documentation_owner);
    fputc('}', stdout);
#undef JSON_FIELD
}
static void render_discovery_json(void)
{
    size_t index;
    fputs("{\"schema\":", stdout); discovery_json_string(stdout, YVEX_COMMAND_DISCOVERY_SCHEMA);
    fputs(",\"registry_identity\":", stdout); discovery_json_string(stdout, yvex_operator_registry_identity);
    fputs(",\"build_commit\":", stdout); discovery_json_string(stdout, YVEX_BUILD_COMMIT);
    fputs(",\"operations\":[", stdout);
    for (index = 0u; index < yvex_operator_descriptor_count; ++index) {
        if (index) fputc(',', stdout);
        render_discovery_operation(index, &yvex_operator_descriptors[index]);
    }
    fputs("]}\n", stdout);
}
int yvex_client_render_help_path(size_t path_count, const char *const *path,
                                 int advanced, int json)
{
    const yvex_operator_descriptor *exact = NULL;
    size_t index, matches = 0u;
    if (json) {
        render_discovery_json();
        return 0;
    }
    for (index = 0u; index < yvex_operator_descriptor_count; ++index) {
        const yvex_operator_descriptor *descriptor = &yvex_operator_descriptors[index];
        if (!descriptor->cli_projection || !descriptor_has_prefix(descriptor, path_count, path))
            continue;
        if (descriptor->command_word_count == path_count) exact = descriptor;
        matches++;
    }
    if (!matches) {
        fprintf(stderr, "yvex: unknown help path");
        for (index = 0u; index < path_count; ++index) fprintf(stderr, " %s", path[index]);
        fputc('\n', stderr);
        return 2;
    }
    if (exact && matches == 1u) {
        render_leaf_help(exact);
        return 0;
    }
    puts(path_count ? "YVEX command namespace\n" : "YVEX local inference\n");
    for (index = 0u; index < yvex_operator_descriptor_count; ++index) {
        const yvex_operator_descriptor *descriptor = &yvex_operator_descriptors[index];
        int visible = path_count != 0u ||
                      descriptor->visibility == YVEX_OPERATOR_VISIBILITY_PRODUCT_DEFAULT ||
                      (advanced && (descriptor->visibility == YVEX_OPERATOR_VISIBILITY_PRODUCT_ADVANCED ||
                                    descriptor->visibility == YVEX_OPERATOR_VISIBILITY_ENGINEERING));
        if (descriptor->cli_projection && visible &&
            descriptor_has_prefix(descriptor, path_count, path))
            printf("  yvex %-42s %s%s\n", descriptor->command_path, descriptor->summary,
                   descriptor->visibility == YVEX_OPERATOR_VISIBILITY_ENGINEERING ? " [engineering]" : "");
    }
    if (!advanced) puts("\nUse `yvex help --advanced` for advanced and engineering commands.");
    return 0;
}
static int client_error(const yvex_error *err)
{
    fprintf(stderr, "yvex: %s\n", yvex_error_message(err));
    if (yvex_error_code(err) == YVEX_ERR_IO)
        fprintf(stderr, "hint: run `yvex model list`, select one model, then use "
                        "`yvex runtime start`\n");
    return 1;
}
static void request_init(yvex_client_request *request,
                         yvex_client_operation operation)
{
    static unsigned long long next_request = 1u;
    yvex_provider_request defaults;
    yvex_provider_request_default(&defaults);
    memset(request, 0, sizeof(*request));
    request->schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    request->operation = operation;
    request->request_number = next_request++;
    request->maximum_new_tokens = defaults.maximum_output_tokens;
    request->stochastic = defaults.sampling.stochastic;
    request->seed_present = defaults.sampling.seed_present;
    request->seed = defaults.sampling.seed;
    request->temperature = defaults.sampling.temperature;
    request->top_k = defaults.sampling.top_k;
    request->top_p = defaults.sampling.top_p;
    request->min_p = defaults.sampling.min_p;
    request->typical_p = defaults.sampling.typical_p;
}
static void turn_options_init(client_turn_options *options)
{
    yvex_provider_request defaults;
    yvex_provider_request_default(&defaults);
    memset(options, 0, sizeof(*options));
    options->maximum_new_tokens = defaults.maximum_output_tokens;
    options->stochastic = defaults.sampling.stochastic;
    options->seed_present = defaults.sampling.seed_present;
    options->seed = defaults.sampling.seed;
    options->temperature = defaults.sampling.temperature;
    options->top_k = defaults.sampling.top_k;
    options->top_p = defaults.sampling.top_p;
    options->min_p = defaults.sampling.min_p;
    options->typical_p = defaults.sampling.typical_p;
}
static int parse_u64(const char *text, unsigned long long *value, int allow_zero)
{
    char *end = NULL;
    unsigned long long parsed;
    errno = 0;
    parsed = strtoull(text, &end, 10);
    if (errno || !end || *end || (!allow_zero && !parsed)) return 0;
    *value = parsed;
    return 1;
}
static int parse_double(const char *text, double *value)
{
    char *end = NULL;
    double parsed;
    errno = 0;
    parsed = strtod(text, &end);
    if (errno || !end || *end || !isfinite(parsed)) return 0;
    *value = parsed;
    return 1;
}
static int request_open(yvex_client **client,
                        const yvex_client_request *request, yvex_error *err)
{
    int rc = yvex_client_connect(client, NULL, err);
    if (rc == YVEX_OK) rc = yvex_client_send(*client, request, err);
    if (rc != YVEX_OK) yvex_client_close(client);
    return rc;
}
static int cancellation_request(const char *session)
{
    yvex_client_request request;
    yvex_client_message message;
    yvex_client *client = NULL;
    yvex_error err;
    int rc;
    request_init(&request, YVEX_CLIENT_OP_GENERATION_CANCEL);
    (void)snprintf(request.session_name, sizeof(request.session_name), "%s",
                   session);
    rc = request_open(&client, &request, &err);
    if (rc == YVEX_OK) rc = yvex_client_receive(client, &message, &err);
    yvex_client_close(&client);
    return rc == YVEX_OK && message.kind == YVEX_CLIENT_MESSAGE_ACK;
}
static void *turn_signal_main(void *opaque)
{
    client_turn_signals *state = opaque;
    sigset_t signals;
    int number;
    (void)sigemptyset(&signals);
    (void)sigaddset(&signals, SIGINT);
    (void)sigaddset(&signals, SIGUSR1);
    while (sigwait(&signals, &number) == 0) {
        if (number == SIGUSR1) break;
        if (atomic_fetch_add_explicit(&state->interrupts, 1,
                                      memory_order_acq_rel) > 0) {
            atomic_store_explicit(&state->force_exit, 1, memory_order_release);
            continue;
        }
        while (!atomic_load_explicit(&state->done, memory_order_acquire)) {
            struct timespec delay = {0, 10000000L};
            if (cancellation_request(state->session)) break;
            (void)nanosleep(&delay, NULL);
        }
    }
    return NULL;
}
static void turn_signals_open(client_turn_signals *state,
                              const char *session)
{
    sigset_t signals;
    memset(state, 0, sizeof(*state));
    (void)snprintf(state->session, sizeof(state->session), "%s", session);
    atomic_init(&state->done, 0);
    atomic_init(&state->interrupts, 0);
    atomic_init(&state->force_exit, 0);
    (void)sigemptyset(&signals);
    (void)sigaddset(&signals, SIGINT);
    (void)sigaddset(&signals, SIGUSR1);
    if (pthread_sigmask(SIG_BLOCK, &signals, &state->previous_mask) != 0)
        return;
    if (pthread_create(&state->thread, NULL, turn_signal_main, state) == 0)
        state->ready = 1;
    else
        (void)pthread_sigmask(SIG_SETMASK, &state->previous_mask, NULL);
}
static int turn_signals_close(client_turn_signals *state)
{
    int result = 0;
    if (!state->ready) return 0;
    atomic_store_explicit(&state->done, 1, memory_order_release);
    (void)pthread_kill(state->thread, SIGUSR1);
    (void)pthread_join(state->thread, NULL);
    (void)pthread_sigmask(SIG_SETMASK, &state->previous_mask, NULL);
    if (atomic_load_explicit(&state->force_exit, memory_order_acquire))
        result = 2;
    else if (atomic_load_explicit(&state->interrupts, memory_order_acquire))
        result = 1;
    return result;
}
static const char *backend_name(yvex_backend_kind backend)
{
    return backend == YVEX_BACKEND_KIND_CUDA ? "CUDA" : "CPU";
}
static const char *generation_mode_name(yvex_server_generation_mode mode)
{
    return mode == YVEX_SERVER_GENERATION_DSPARK ? "DSpark" : "target-only";
}
static void render_status(const yvex_server_summary *status, int json)
{
    if (json) {
        printf("{\"protocol\":%u,\"status\":%u,\"target\":\"%s\","
               "\"backend\":%u,\"generation_mode\":\"%s\","
               "\"ready\":%s,\"uptime_ns\":%llu,"
               "\"model_open_count\":%llu,\"model_close_count\":%llu,"
               "\"artifact_open_count\":%llu,\"binding_open_count\":%llu,"
               "\"materialization_count\":%llu,\"residency_build_count\":%llu,"
               "\"output_head_upload_count\":%llu,\"sessions\":%llu,"
               "\"active_sessions\":%llu,\"total_sessions\":%llu,"
               "\"queue_depth\":%llu,\"queue_capacity\":%llu,"
               "\"active_requests\":%llu,\"completed_requests\":%llu,"
               "\"failed_requests\":%llu,\"cancelled_requests\":%llu,"
               "\"openai_enabled\":%s,\"openai_ready\":%s,"
               "\"openai_port\":%u,\"active_http_requests\":%llu,"
               "\"completed_http_requests\":%llu,"
               "\"failed_http_requests\":%llu,"
               "\"cancelled_http_requests\":%llu,"
               "\"telemetry_dropped\":%llu,\"rss_bytes\":%llu,"
               "\"peak_rss_bytes\":%llu,\"mapped_artifact_bytes\":%llu,"
               "\"resident_host_bytes\":%llu,\"resident_device_bytes\":%llu,"
               "\"model_identity\":\"%s\",\"binding_identity\":\"%s\","
               "\"artifact_identity\":\"%s\",\"variant_identity\":\"%s\"}\n",
               YVEX_LOCAL_PROTOCOL_VERSION, (unsigned int)status->status,
               status->target_id, (unsigned int)status->backend,
               status->generation_mode == YVEX_SERVER_GENERATION_DSPARK
                   ? "dspark" : "target-only",
               status->runtime_ready ? "true" : "false",
               status->metrics.uptime_ns, status->metrics.model_open_count,
               status->metrics.model_close_count,
               status->metrics.artifact_open_count,
               status->metrics.binding_open_count,
               status->metrics.materialization_count,
               status->metrics.residency_build_count,
               status->metrics.output_head_upload_count, status->session_count,
               status->metrics.active_sessions, status->metrics.total_sessions,
               status->metrics.queue_depth, status->metrics.queue_capacity,
               status->metrics.active_requests,
               status->metrics.completed_requests,
               status->metrics.failed_requests,
               status->metrics.cancelled_requests,
               status->openai_listener_enabled ? "true" : "false",
               status->openai_listener_ready ? "true" : "false",
               (unsigned int)status->openai_port,
               status->metrics.active_http_requests,
               status->metrics.completed_http_requests,
               status->metrics.failed_http_requests,
               status->metrics.cancelled_http_requests,
               status->metrics.telemetry_dropped,
               status->metrics.current_rss_bytes,
               status->metrics.peak_rss_bytes,
               status->metrics.mapped_artifact_bytes,
               status->metrics.resident_host_bytes,
               status->metrics.resident_device_bytes,
               status->runtime_model_identity,
               status->runtime_binding_identity,
               status->artifact_identity,
               status->physical_variant_identity);
        return;
    }
    {
        yvex_cli_terminal_style style;
        int ready = status->status == YVEX_SERVER_STATUS_READY;
        yvex_cli_terminal_style_get(stdout, &style);
        printf("%sYVEX runtime%s · %s%s%s · %s · %s · %s · %llu session%s · "
               "queue %llu/%llu · model opened %llu×",
               style.strong, style.reset, ready ? style.success : style.warning,
               ready ? "● ready" : "● starting", style.reset,
               status->target_id[0] ? status->target_id : "no model",
               backend_name(status->backend), generation_mode_name(status->generation_mode),
               status->session_count, status->session_count == 1u ? "" : "s",
               status->metrics.queue_depth, status->metrics.queue_capacity,
               status->metrics.model_open_count);
        if (status->openai_listener_enabled)
            printf(" · OpenAI %s%s%s 127.0.0.1:%u · %llu active/%llu completed",
                   status->openai_listener_ready ? style.success : style.warning,
                   status->openai_listener_ready ? "ready" : "starting", style.reset,
                   (unsigned int)status->openai_port,
                   status->metrics.active_http_requests,
                   status->metrics.completed_http_requests);
        else
            printf(" · OpenAI %sdisabled%s", style.dim, style.reset);
        printf(" · %smemory %.2f GiB host/%.2f GiB device/%.2f GiB RSS%s\n",
               style.dim,
               (double)status->metrics.resident_host_bytes / 1073741824.0,
               (double)status->metrics.resident_device_bytes / 1073741824.0,
               (double)status->metrics.current_rss_bytes / 1073741824.0,
               style.reset);
    }
}
static int runtime_summary_fetch(yvex_server_summary *summary, yvex_error *err)
{
    yvex_client_request request;
    yvex_client_message message;
    yvex_client *client = NULL;
    int rc;
    request_init(&request, YVEX_CLIENT_OP_RUNTIME_STATUS);
    rc = request_open(&client, &request, err);
    if (rc == YVEX_OK) rc = yvex_client_receive(client, &message, err);
    if (rc == YVEX_OK && message.kind == YVEX_CLIENT_MESSAGE_STATUS)
        *summary = message.runtime;
    else if (rc == YVEX_OK) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "client.status",
                       "daemon returned an unexpected response");
        rc = YVEX_ERR_FORMAT;
    }
    yvex_client_close(&client);
    return rc;
}
static int runtime_status(int json)
{
    yvex_server_summary summary;
    yvex_error err;
    int rc = runtime_summary_fetch(&summary, &err);
    if (rc == YVEX_OK) render_status(&summary, json);
    return rc == YVEX_OK ? 0 : client_error(&err);
}
static int runtime_model(void)
{
    yvex_server_summary summary;
    yvex_error err;
    int rc = runtime_summary_fetch(&summary, &err);
    if (rc == YVEX_OK) {
        yvex_cli_terminal_style style;
        yvex_cli_terminal_style_get(stdout, &style);
        printf("%slive runtime model%s · %s · %s · %s · model %s · variant %s · "
               "artifact %s · binding %s\n",
               style.strong, style.reset, summary.target_id, backend_name(summary.backend),
               generation_mode_name(summary.generation_mode),
               summary.runtime_model_identity, summary.physical_variant_identity,
               summary.artifact_identity, summary.runtime_binding_identity);
    }
    return rc == YVEX_OK ? 0 : client_error(&err);
}
static int runtime_memory(void)
{
    yvex_server_summary summary;
    yvex_error err;
    int rc = runtime_summary_fetch(&summary, &err);
    if (rc == YVEX_OK) {
        yvex_cli_terminal_style style;
        yvex_cli_terminal_style_get(stdout, &style);
        printf("%sruntime memory%s · %.2f GiB host · %.2f GiB device · "
               "%.2f GiB mapped · %.2f GiB RSS · %.2f GiB peak RSS\n",
               style.strong, style.reset,
               (double)summary.metrics.resident_host_bytes / 1073741824.0,
               (double)summary.metrics.resident_device_bytes / 1073741824.0,
               (double)summary.metrics.mapped_artifact_bytes / 1073741824.0,
               (double)summary.metrics.current_rss_bytes / 1073741824.0,
               (double)summary.metrics.peak_rss_bytes / 1073741824.0);
    }
    return rc == YVEX_OK ? 0 : client_error(&err);
}
static int runtime_events(int projection)
{
    yvex_client_request request;
    yvex_client_message message;
    yvex_client *client = NULL;
    yvex_server_summary summary;
    yvex_error err;
    yvex_cli_terminal_style style;
    char json[2048];
    int rc;
    if (!projection) {
        rc = runtime_summary_fetch(&summary, &err);
        if (rc != YVEX_OK) return client_error(&err);
        render_status(&summary, 0);
        yvex_cli_terminal_style_get(stdout, &style);
        printf("%swatch%s · operational history and live events · Ctrl-C to stop\n\n",
               style.accent, style.reset);
    }
    request_init(&request, projection ? YVEX_CLIENT_OP_RUNTIME_TRACE
                                      : YVEX_CLIENT_OP_RUNTIME_WATCH);
    request.trace_level = projection ? YVEX_SERVER_TRACE_FULL
                                     : YVEX_SERVER_TRACE_STAGES;
    rc = request_open(&client, &request, &err);
    while (rc == YVEX_OK) {
        rc = yvex_client_receive(client, &message, &err);
        if (rc != YVEX_OK) break;
        if (message.kind != YVEX_CLIENT_MESSAGE_EVENT) continue;
        if (projection < 2)
            (void)yvex_cli_out_server_event(&message.event, projection == 1);
        else if (yvex_server_event_json(&message.event, json, sizeof(json), &err) == YVEX_OK) {
            fputs(json, stdout);
            fflush(stdout);
        }
        if (message.event.kind == YVEX_SERVER_EVENT_RUNTIME_SHUTDOWN_COMPLETE)
            break;
    }
    yvex_client_close(&client);
    return rc == YVEX_OK ? 0 : client_error(&err);
}
static int administration(yvex_client_operation operation,
                          const char *session_name, int render_mode)
{
    yvex_client_request request;
    yvex_client_message message;
    yvex_client *client = NULL;
    yvex_error err;
    int rc;
    request_init(&request, operation);
    if (session_name)
        snprintf(request.session_name, sizeof(request.session_name), "%s",
                 session_name);
    rc = request_open(&client, &request, &err);
    while (rc == YVEX_OK) {
        rc = yvex_client_receive(client, &message, &err);
        if (rc != YVEX_OK) break;
        if (message.kind == YVEX_CLIENT_MESSAGE_ERROR) {
            yvex_error_set(&err, (yvex_status)message.status, "client.request",
                           message.reason);
            rc = message.status;
            break;
        }
        if (render_mode >= 0 && message.kind == YVEX_CLIENT_MESSAGE_SESSION)
            printf("%-20s %-10s position=%llu turns=%llu\n",
                   message.session_name,
                   yvex_server_session_state_name(message.session_state),
                   message.final_position, message.turn_count);
        else if (message.kind == YVEX_CLIENT_MESSAGE_ACK) {
            if (!render_mode)
                printf("%s\n", message.reason[0] ? message.reason : "ok");
            break;
        }
        if (render_mode <= 0) break;
    }
    yvex_client_close(&client);
    return rc == YVEX_OK ? 0 : client_error(&err);
}
static int generation_turn(const char *session_name,
                           const unsigned char *prompt,
                           unsigned long long prompt_bytes,
                           const client_turn_options *options,
                           int conversation,
                           unsigned long long context_capacity)
{
    yvex_client_request request;
    yvex_client_message message;
    yvex_client *client = NULL;
    client_turn_signals signals;
    yvex_error err;
    yvex_cli_terminal_style style;
    int rc, started = 0, last_newline = 1, progress_active = 0;
    yvex_cli_terminal_style_get(stdout, &style);
    request_init(&request, YVEX_CLIENT_OP_GENERATION_TURN);
    snprintf(request.session_name, sizeof(request.session_name), "%s",
             session_name);
    request.prompt = prompt;
    request.prompt_bytes = prompt_bytes;
    request.maximum_new_tokens = options->maximum_new_tokens;
    request.stochastic = options->stochastic;
    request.seed_present = options->seed_present;
    request.seed = options->seed;
    request.temperature = options->temperature;
    request.top_k = options->top_k;
    request.top_p = options->top_p;
    request.min_p = options->min_p;
    request.typical_p = options->typical_p;
    turn_signals_open(&signals, session_name);
    rc = request_open(&client, &request, &err);
    while (rc == YVEX_OK) {
        rc = yvex_client_receive(client, &message, &err);
        if (rc != YVEX_OK) break;
        if (message.kind == YVEX_CLIENT_MESSAGE_TURN_STARTED) {
            continue;
        } else if (message.kind == YVEX_CLIENT_MESSAGE_EVENT) {
            if (conversation && message.event.kind == YVEX_SERVER_EVENT_PREFILL_STARTED) {
                printf("\r\033[2K%sprocessing %llu input tokens · 0/%llu · 0%%%s",
                       style.accent, message.event.value_a, message.event.value_a,
                       style.reset);
                fflush(stdout);
                progress_active = 1;
            } else if (conversation &&
                       message.event.kind == YVEX_SERVER_EVENT_PREFILL_PROGRESS) {
                printf("\r\033[2K%sprocessing %llu input tokens · %llu/%llu · %.1f%%%s",
                       style.accent, message.event.value_b, message.event.value_a,
                       message.event.value_b,
                       message.event.value_b ? 100.0 * (double)message.event.value_a /
                                                   (double)message.event.value_b : 0.0,
                       style.reset);
                fflush(stdout);
            } else if (conversation &&
                       message.event.kind == YVEX_SERVER_EVENT_PREFILL_COMPLETED) {
                printf("\r\033[2K%sprocessing %llu input tokens · %llu/%llu · 100%%%s\n",
                       style.success, message.event.value_a, message.event.value_a,
                       message.event.value_a, style.reset);
                fflush(stdout);
                progress_active = 0;
            }
        } else if (message.kind == YVEX_CLIENT_MESSAGE_FRAGMENT) {
            if (progress_active) {
                fputs("\r\033[2K", stdout);
                progress_active = 0;
            }
            if (message.byte_count)
                (void)fwrite(message.bytes, 1u, (size_t)message.byte_count,
                             stdout);
            if (message.byte_count)
                last_newline = message.bytes[message.byte_count - 1u] == '\n';
            fflush(stdout);
            started = 1;
        } else if (message.kind == YVEX_CLIENT_MESSAGE_TURN_COMPLETE) {
            if (progress_active) fputs("\r\033[2K", stdout);
            if (started && !last_newline) putchar('\n');
            printf("%sprefill%s %llu new/%llu prompt/%llu reused · %.2f s · %.2f tok/s · "
                   "%sgeneration%s %llu tokens · %.2f s · %.2f tok/s · TTFT %.2f s",
                   style.accent, style.reset,
                   message.prefill_tokens, message.prompt_tokens,
                   message.reused_tokens, message.prefill_seconds,
                   message.prefill_rate, style.success, style.reset, message.generated_tokens,
                   message.decode_seconds, message.decode_rate,
                   message.first_token_seconds);
            if (message.generation_mode == YVEX_SERVER_GENERATION_DSPARK)
                printf(" · %sDSpark%s %llu proposed/%llu accepted/%llu rejected/%llu verified",
                       style.accent, style.reset,
                       message.proposed_tokens, message.accepted_draft_tokens,
                       message.rejected_draft_tokens,
                       message.target_verification_count);
            if (context_capacity)
                printf(" · context %llu/%llu", message.context_used, context_capacity);
            else
                printf(" · context %llu", message.context_used);
            printf(" · stop %s · %ssession %s%s\n",
                   yvex_cli_out_stop_reason(message.stop_reason),
                   style.dim, message.session_name, style.reset);
            break;
        } else if (message.kind == YVEX_CLIENT_MESSAGE_ERROR) {
            if (progress_active) fputs("\r\033[2K", stdout);
            if (started && !last_newline) putchar('\n');
            yvex_error_set(&err, (yvex_status)message.status, "client.turn",
                           message.reason);
            rc = message.status;
            break;
        }
    }
    yvex_client_close(&client);
    {
        int interrupted = turn_signals_close(&signals);
        if (interrupted) {
            if (conversation) {
                const char *text = interrupted == 2 ? "cancelled · leaving chat"
                                   : message.session_state == YVEX_SERVER_SESSION_PARTIAL
                                       ? "cancelled · session partial · use /reset"
                                       : "cancelled";
                printf("%s%s%s\n", style.warning, text, style.reset);
            }
            return interrupted == 2 ? 131 : 130;
        }
    }
    return rc == YVEX_OK ? 0 : client_error(&err);
}
static int session_ensure(const char *name)
{
    yvex_client_request request;
    yvex_client_message message;
    yvex_client *client = NULL;
    yvex_error err;
    int rc;
    request_init(&request, YVEX_CLIENT_OP_SESSION_SHOW);
    snprintf(request.session_name, sizeof(request.session_name), "%s", name);
    rc = request_open(&client, &request, &err);
    if (rc == YVEX_OK) rc = yvex_client_receive(client, &message, &err);
    yvex_client_close(&client);
    if (rc == YVEX_OK && message.kind != YVEX_CLIENT_MESSAGE_ERROR) return 0;
    return administration(YVEX_CLIENT_OP_SESSION_NEW, name, -1);
}

static void repl_history_push(client_repl_history *history, const char *line)
{
    char *copy;
    if (!line[0] || (history->count &&
                     !strcmp(history->entry[history->count - 1u], line)))
        return;
    copy = strdup(line);
    if (!copy) return;
    if (history->count == CLIENT_REPL_HISTORY_MAX) {
        free(history->entry[0]);
        memmove(history->entry, history->entry + 1,
                (CLIENT_REPL_HISTORY_MAX - 1u) * sizeof(history->entry[0]));
        history->count--;
    }
    history->entry[history->count++] = copy;
}

static void repl_history_close(client_repl_history *history)
{
    size_t index;
    for (index = 0u; index < history->count; ++index) free(history->entry[index]);
    memset(history, 0, sizeof(*history));
}

static void repl_signal_handler(int number)
{
    sig_atomic_t interrupts = repl_signal_state & 3;
    if (number == SIGWINCH)
        repl_signal_state |= 4;
    else if (interrupts < 2)
        repl_signal_state = (repl_signal_state & ~3) | (interrupts + 1);
}

static void repl_redraw(const char *prompt, const char *line, size_t count)
{
    fputs("\r\033[2K", stdout);
    fputs(prompt, stdout);
    if (count) (void)fwrite(line, 1u, count, stdout);
    fflush(stdout);
}

static int repl_replace_line(char **line, size_t *count, size_t *capacity,
                             const char *replacement, const char *prompt)
{
    size_t needed = strlen(replacement) + 1u;
    char *grown;
    if (needed > CLIENT_REPL_LINE_MAX + 1u) return 0;
    if (needed > *capacity) {
        grown = realloc(*line, needed);
        if (!grown) return 0;
        *line = grown;
        *capacity = needed;
    }
    memcpy(*line, replacement, needed);
    *count = needed - 1u;
    repl_redraw(prompt, *line, *count);
    return 1;
}

static int repl_complete_slash(char **line, size_t *count, size_t *capacity,
                               const char *prompt)
{
    const yvex_operator_descriptor *match = NULL;
    size_t index, matches = 0u;
    if (!*line || !*count || (*line)[0] != '/' || strchr(*line, ' ')) return 0;
    for (index = 0u; index < yvex_operator_descriptor_count; ++index) {
        const yvex_operator_descriptor *candidate = &yvex_operator_descriptors[index];
        if (strcmp(candidate->slash_projection, "none") &&
            !strncmp(candidate->slash_projection, *line, *count)) {
            match = candidate;
            matches++;
        }
    }
    if (matches == 1u) {
        char replacement[128];
        (void)snprintf(replacement, sizeof(replacement), "%s%s",
                       match->slash_projection, match->argument_count ? " " : "");
        return repl_replace_line(line, count, capacity, replacement, prompt);
    }
    if (matches > 1u) return 1;
    return 0;
}

static int repl_append_byte(char **line, size_t *count, size_t *capacity,
                            unsigned char byte)
{
    char *grown;
    size_t next;
    if (*count >= CLIENT_REPL_LINE_MAX) return 0;
    if (*count + 1u >= *capacity) {
        next = *capacity ? *capacity * 2u : 256u;
        if (next > CLIENT_REPL_LINE_MAX + 1u) next = CLIENT_REPL_LINE_MAX + 1u;
        grown = realloc(*line, next);
        if (!grown) return 0;
        *line = grown;
        *capacity = next;
    }
    (*line)[(*count)++] = (char)byte;
    (*line)[*count] = '\0';
    return 1;
}

static void repl_backspace(char *line, size_t *count)
{
    if (!*count) return;
    (*count)--;
    while (*count && (((unsigned char)line[*count] & 0xc0u) == 0x80u))
        (*count)--;
    line[*count] = '\0';
}

static int repl_read_line(const char *prompt, const client_repl_history *history,
                          char **output, size_t *output_count)
{
    struct termios saved, raw;
    char *line = NULL;
    size_t count = 0u, capacity = 0u, selected = history->count;
    int paste = 0, result = -1;
    unsigned char byte;
    if (tcgetattr(STDIN_FILENO, &saved) != 0) return -1;
    raw = saved;
    raw.c_lflag &= (tcflag_t)~(ICANON | ECHO);
    raw.c_iflag &= (tcflag_t)~(ICRNL | IXON);
    raw.c_cc[VMIN] = 1;
    raw.c_cc[VTIME] = 0;
    if (tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw) != 0) return -1;
    fputs("\033[?2004h", stdout);
    repl_redraw(prompt, "", 0u);
    for (;;) {
        ssize_t got = read(STDIN_FILENO, &byte, 1u);
        if (got < 0 && errno == EINTR) {
            if ((repl_signal_state & 3) >= 2) {
                result = 0;
                break;
            }
            if ((repl_signal_state & 3) == 1) {
                fputs("^C\r\n", stdout);
                result = -2;
                break;
            }
            if (repl_signal_state & 4) {
                struct winsize window;
                (void)ioctl(STDOUT_FILENO, TIOCGWINSZ, &window);
                repl_signal_state &= ~4;
                repl_redraw(prompt, line ? line : "", count);
            }
            continue;
        }
        if (got <= 0) {
            result = 0;
            break;
        }
        if (byte == '\r' || byte == '\n') {
            if (paste) {
                if (!repl_append_byte(&line, &count, &capacity, '\n')) break;
                fputs("\r\n... ", stdout);
                fflush(stdout);
                continue;
            }
            fputs("\r\n", stdout);
            result = 1;
            break;
        }
        if (byte == 4u && !paste) {
            fputs("\r\n", stdout);
            result = 0;
            break;
        }
        if (byte == '\t' && !paste) {
            if (!repl_complete_slash(&line, &count, &capacity, prompt)) fputc('\a', stdout);
            fflush(stdout);
            continue;
        }
        if (byte == 8u || byte == 127u) {
            repl_backspace(line, &count);
            repl_redraw(prompt, line ? line : "", count);
            continue;
        }
        if (byte == 27u) {
            unsigned char sequence[5];
            size_t length = 0u;
            while (length < sizeof(sequence) && read(STDIN_FILENO, &sequence[length], 1u) == 1) {
                length++;
                if ((length == 2u && (sequence[1] == 'A' || sequence[1] == 'B')) ||
                    (length == 5u && sequence[4] == '~'))
                    break;
            }
            if (length == 5u && !memcmp(sequence, "[200~", 5u)) paste = 1;
            else if (length == 5u && !memcmp(sequence, "[201~", 5u)) paste = 0;
            else if (!paste && length == 2u && sequence[0] == '[' && sequence[1] == 'A' && selected) {
                selected--;
                if (!repl_replace_line(&line, &count, &capacity,
                                       history->entry[selected], prompt))
                    break;
            } else if (!paste && length == 2u && sequence[0] == '[' &&
                       sequence[1] == 'B' && selected < history->count) {
                selected++;
                if (!repl_replace_line(&line, &count, &capacity,
                                       selected == history->count ? "" : history->entry[selected],
                                       prompt))
                    break;
            }
            continue;
        }
        if (!repl_append_byte(&line, &count, &capacity, byte)) break;
        (void)fwrite(&byte, 1u, 1u, stdout);
        fflush(stdout);
    }
    fputs("\033[?2004l", stdout);
    (void)tcsetattr(STDIN_FILENO, TCSAFLUSH, &saved);
    if (result == 1) {
        if (!line) {
            line = calloc(1u, 1u);
            if (!line) result = -1;
        }
        *output = line;
        *output_count = count;
    } else {
        free(line);
    }
    return result;
}

static int repl_switch_session(char current[YVEX_SERVER_SESSION_NAME_CAP],
                               const char *next, int create)
{
    yvex_cli_terminal_style style;
    if (!next || !next[0] || strlen(next) >= YVEX_SERVER_SESSION_NAME_CAP) return 0;
    if (!strcmp(current, next)) return 1;
    if (create && administration(YVEX_CLIENT_OP_SESSION_NEW, next, -1) != 0) return 0;
    if (administration(YVEX_CLIENT_OP_SESSION_ATTACH, next, -1) != 0) return 0;
    (void)administration(YVEX_CLIENT_OP_SESSION_DETACH, current, -1);
    (void)snprintf(current, YVEX_SERVER_SESSION_NAME_CAP, "%s", next);
    yvex_cli_terminal_style_get(stdout, &style);
    printf("%ssession%s · %s\n", style.success, style.reset, current);
    return 1;
}

static const yvex_operator_descriptor *slash_descriptor(const char *line,
                                                         const char **argument)
{
    const char *end = strchr(line, ' ');
    size_t extent = end ? (size_t)(end - line) : strlen(line), index;
    *argument = end ? end + 1 : NULL;
    while (*argument && **argument == ' ') (*argument)++;
    if (*argument && !**argument) *argument = NULL;
    for (index = 0u; index < yvex_operator_descriptor_count; ++index) {
        const yvex_operator_descriptor *descriptor = &yvex_operator_descriptors[index];
        if (strcmp(descriptor->slash_projection, "none") &&
            strlen(descriptor->slash_projection) == extent &&
            !memcmp(descriptor->slash_projection, line, extent))
            return descriptor;
    }
    return NULL;
}

static int repl_command(const char *line, char current[YVEX_SERVER_SESSION_NAME_CAP],
                        unsigned long long *generated_session)
{
    const yvex_operator_descriptor *descriptor;
    yvex_cli_operator_invocation invocation;
    const char *argument;
    char generated[YVEX_SERVER_SESSION_NAME_CAP];
    int result = 1, status;
    if (line[0] != '/') return 0;
    descriptor = slash_descriptor(line, &argument);
    if (!descriptor) {
        yvex_cli_terminal_style style;
        yvex_cli_terminal_style_get(stdout, &style);
        printf("%sunknown command:%s %.*s\n", style.error, style.reset,
               (int)(strchr(line, ' ') ? (size_t)(strchr(line, ' ') - line) : strlen(line)),
               line);
        return 1;
    }
    status = yvex_cli_operator_slash_parse(descriptor, argument, &invocation);
    if (status) {
        yvex_cli_terminal_style style;
        yvex_cli_terminal_style_get(stdout, &style);
        printf("%sinvalid arguments for %s:%s %s\n", style.error,
               descriptor->slash_projection, style.reset, invocation.message);
        yvex_cli_operator_invocation_close(&invocation);
        return 1;
    }
    argument = invocation.argument_count ? invocation.arguments[0] : NULL;
    if (descriptor->lane == YVEX_OPERATOR_LANE_REPL_LOCAL) {
        result = descriptor->repl_adapter == YVEX_OPERATOR_REPL_QUIT ? 2 : 1;
        yvex_cli_operator_invocation_close(&invocation);
        return result;
    }
    switch (descriptor->runtime_adapter) {
    case YVEX_OPERATOR_RUNTIME_HELP:
        if (invocation.argument_count)
            (void)yvex_client_render_help_path(invocation.argument_count,
                                                invocation.arguments, 0, 0);
        else
            yvex_cli_out_repl_catalog(0);
        break;
    case YVEX_OPERATOR_RUNTIME_CONSOLE_STATUS:
        (void)console_status(current);
        break;
    case YVEX_OPERATOR_RUNTIME_RUNTIME_STATUS:
        (void)runtime_status(0);
        break;
    case YVEX_OPERATOR_RUNTIME_RUNTIME_MODEL:
        (void)runtime_model();
        break;
    case YVEX_OPERATOR_RUNTIME_RUNTIME_MEMORY:
        (void)runtime_memory();
        break;
    case YVEX_OPERATOR_RUNTIME_SESSION_LIST:
        (void)administration(YVEX_CLIENT_OP_SESSION_LIST, NULL, 1);
        break;
    case YVEX_OPERATOR_RUNTIME_SESSION_SHOW:
        (void)administration(YVEX_CLIENT_OP_SESSION_SHOW, argument ? argument : current, 0);
        break;
    case YVEX_OPERATOR_RUNTIME_SESSION_NEW:
        if (!argument) {
            (void)snprintf(generated, sizeof(generated), "chat-%llu", (*generated_session)++);
            argument = generated;
        }
        (void)repl_switch_session(current, argument, 1);
        break;
    case YVEX_OPERATOR_RUNTIME_SESSION_ATTACH:
        (void)repl_switch_session(current, argument, 0);
        break;
    case YVEX_OPERATOR_RUNTIME_SESSION_DETACH:
        result = 2;
        break;
    case YVEX_OPERATOR_RUNTIME_SESSION_RESET:
        (void)administration(YVEX_CLIENT_OP_SESSION_RESET, current, 0);
        break;
    case YVEX_OPERATOR_RUNTIME_SESSION_CLOSE:
        (void)administration(YVEX_CLIENT_OP_SESSION_CLOSE, current, 0);
        result = 3;
        break;
    case YVEX_OPERATOR_RUNTIME_SESSION_CANCEL:
        {
            yvex_cli_terminal_style style;
            int cancelled = cancellation_request(current);
            yvex_cli_terminal_style_get(stdout, &style);
            printf("%s%s%s\n", cancelled ? style.warning : style.dim,
                   cancelled ? "cancel requested" : "no active turn", style.reset);
        }
        break;
    default:
        {
            yvex_cli_terminal_style style;
            yvex_cli_terminal_style_get(stdout, &style);
            printf("%scommand unavailable in this console%s\n", style.warning, style.reset);
        }
        break;
    }
    yvex_cli_operator_invocation_close(&invocation);
    return result;
}

static int chat(const char *session_name, unsigned long long maximum_new_tokens)
{
    client_turn_options options;
    client_repl_history history;
    yvex_client_message status;
    yvex_error err;
    yvex_cli_terminal_style style;
    struct sigaction action, prior_interrupt, prior_resize;
    char current[YVEX_SERVER_SESSION_NAME_CAP];
    char prompt[64];
    unsigned long long generated_session = 1u;
    int closed = 0;
    if (!isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO)) {
        fprintf(stderr, "yvex: chat requires a terminal; use `yvex run TEXT`\n");
        return 2;
    }
    memset(&history, 0, sizeof(history));
    memset(&action, 0, sizeof(action));
    action.sa_handler = repl_signal_handler;
    (void)sigemptyset(&action.sa_mask);
    repl_signal_state = 0;
    if (sigaction(SIGINT, &action, &prior_interrupt) != 0) return 1;
    if (sigaction(SIGWINCH, &action, &prior_resize) != 0) {
        (void)sigaction(SIGINT, &prior_interrupt, NULL);
        return 1;
    }
    turn_options_init(&options);
    options.maximum_new_tokens = maximum_new_tokens;
    (void)snprintf(current, sizeof(current), "%s", session_name);
    if (session_ensure(current) != 0) {
        (void)sigaction(SIGINT, &prior_interrupt, NULL);
        (void)sigaction(SIGWINCH, &prior_resize, NULL);
        return 1;
    }
    if (administration(YVEX_CLIENT_OP_SESSION_ATTACH, current, -1) != 0) {
        (void)sigaction(SIGINT, &prior_interrupt, NULL);
        (void)sigaction(SIGWINCH, &prior_resize, NULL);
        return 1;
    }
    if (console_status_fetch(current, &status, &err) != YVEX_OK) {
        (void)administration(YVEX_CLIENT_OP_SESSION_DETACH, current, -1);
        (void)sigaction(SIGINT, &prior_interrupt, NULL);
        (void)sigaction(SIGWINCH, &prior_resize, NULL);
        return client_error(&err);
    }
    render_console_status(&status, 1);
    yvex_cli_out_repl_catalog(1);
    yvex_cli_terminal_style_get(stdout, &style);
    (void)snprintf(prompt, sizeof(prompt), "%syvex>%s ", style.accent, style.reset);
    for (;;) {
        char *line = NULL;
        size_t count = 0u;
        int input = repl_read_line(prompt, &history, &line, &count);
        if (input == -2) continue;
        if (input <= 0) break;
        repl_signal_state &= ~3;
        if (!count) {
            free(line);
            continue;
        }
        if (line[0] == '/') {
            int command = repl_command(line, current, &generated_session);
            free(line);
            if (command == 3) {
                closed = 1;
                break;
            }
            if (command == 2) break;
            continue;
        }
        repl_history_push(&history, line);
        if (generation_turn(current, (const unsigned char *)line,
                            (unsigned long long)count, &options, 1,
                            status.console.context_capacity) == 131) {
            free(line);
            break;
        }
        free(line);
    }
    repl_history_close(&history);
    if (!closed) (void)administration(YVEX_CLIENT_OP_SESSION_DETACH, current, -1);
    (void)sigaction(SIGINT, &prior_interrupt, NULL);
    (void)sigaction(SIGWINCH, &prior_resize, NULL);
    return 0;
}

static int chat_command(int argc, char **argv)
{
    const char *session = "main";
    yvex_provider_request defaults;
    unsigned long long maximum_new_tokens;
    int index, saw_session = 0, saw_maximum = 0;
    yvex_provider_request_default(&defaults);
    maximum_new_tokens = defaults.maximum_output_tokens;
    for (index = 2; index < argc; ++index) {
        if (!strcmp(argv[index], "--session") && !saw_session && index + 1 < argc) {
            session = argv[++index];
            saw_session = 1;
        } else if (!strcmp(argv[index], "--max-new-tokens") && !saw_maximum &&
                   index + 1 < argc) {
            if (!parse_u64(argv[++index], &maximum_new_tokens, 0)) return 2;
            saw_maximum = 1;
        } else {
            return 2;
        }
    }
    return chat(session, maximum_new_tokens);
}

/* Parse and execute one complete one-shot policy without inferring strategy. */
static int run_command(int argc, char **argv)
{
    client_turn_options options;
    char ephemeral[YVEX_SERVER_SESSION_NAME_CAP];
    const char *session = NULL, *prompt = NULL;
    int index, owns_session = 0;
    turn_options_init(&options);
    for (index = 2; index < argc; ++index) {
        const char *argument = argv[index];
        if (!strcmp(argument, "--session") && index + 1 < argc)
            session = argv[++index];
        else if (!strcmp(argument, "--max-new-tokens") && index + 1 < argc) {
            if (!parse_u64(argv[++index], &options.maximum_new_tokens, 0)) return 2;
        } else if (!strcmp(argument, "--strategy") && index + 1 < argc) {
            const char *strategy = argv[++index];
            if (!strcmp(strategy, "greedy")) options.stochastic = 0;
            else if (!strcmp(strategy, "stochastic")) options.stochastic = 1;
            else return 2;
        } else if (!strcmp(argument, "--seed") && index + 1 < argc) {
            if (!parse_u64(argv[++index], &options.seed, 1)) return 2;
            options.seed_present = 1;
        } else if (!strcmp(argument, "--temperature") && index + 1 < argc) {
            if (!parse_double(argv[++index], &options.temperature)) return 2;
        } else if (!strcmp(argument, "--top-k") && index + 1 < argc) {
            if (!parse_u64(argv[++index], &options.top_k, 1)) return 2;
        } else if (!strcmp(argument, "--top-p") && index + 1 < argc) {
            if (!parse_double(argv[++index], &options.top_p)) return 2;
        } else if (!strcmp(argument, "--min-p") && index + 1 < argc) {
            if (!parse_double(argv[++index], &options.min_p)) return 2;
        } else if (!strcmp(argument, "--typical-p") && index + 1 < argc) {
            if (!parse_double(argv[++index], &options.typical_p)) return 2;
        } else if (argument[0] == '-') {
            fprintf(stderr, "yvex: unknown run option: %s\n", argument);
            return 2;
        } else if (!prompt)
            prompt = argument;
        else {
            fprintf(stderr, "yvex: run accepts one prompt argument\n");
            return 2;
        }
    }
    if (!prompt || options.temperature <= 0.0 || options.top_p <= 0.0 ||
        options.top_p > 1.0 || options.min_p < 0.0 || options.min_p > 1.0 ||
        options.typical_p <= 0.0 || options.typical_p > 1.0 ||
        (options.stochastic && !options.seed_present) ||
        (!options.stochastic &&
         (options.seed_present || options.temperature != 1.0 || options.top_k ||
          options.top_p != 1.0 || options.min_p != 0.0 ||
          options.typical_p != 1.0))) {
        fprintf(stderr,
                "yvex: run requires one prompt and an explicit valid strategy policy\n");
        return 2;
    }
    if (!session) {
        (void)snprintf(ephemeral, sizeof(ephemeral), "run-%lu",
                       (unsigned long)getpid());
        session = ephemeral;
        owns_session = 1;
        if (administration(YVEX_CLIENT_OP_SESSION_NEW, session, -1) != 0) return 1;
    }
    {
        int status = generation_turn(session, (const unsigned char *)prompt,
                                     (unsigned long long)strlen(prompt),
                                     &options, 0, 0u);
        if (owns_session)
            (void)administration(YVEX_CLIENT_OP_SESSION_CLOSE, session, -1);
        return status;
    }
}

static int model_config_paths(char directory[PATH_MAX], char path[PATH_MAX])
{
    const char *base = getenv("XDG_CONFIG_HOME");
    char fallback[PATH_MAX];
    int count;
    if (!base || !base[0]) {
        const char *home = getenv("HOME");
        count = home ? snprintf(fallback, sizeof(fallback), "%s/.config", home) : -1;
        if (!home || home[0] != '/' || count <= 0 || (size_t)count >= sizeof(fallback))
            return 0;
        base = fallback;
    }
    count = snprintf(directory, PATH_MAX, "%s/yvex", base);
    if (base[0] != '/' || count <= 0 || count >= PATH_MAX) return 0;
    count = snprintf(path, PATH_MAX, "%s/model.conf", directory);
    if (count <= 0 || count >= PATH_MAX)
        return 0;
    return 1;
}

static int model_config_directory(const char *directory)
{
    struct stat status;
    char parent[PATH_MAX], resolved[PATH_MAX], *slash;
    (void)snprintf(parent, sizeof(parent), "%s", directory);
    slash = strrchr(parent, '/');
    if (!slash || slash == parent) return 0;
    *slash = '\0';
    if (lstat(parent, &status) != 0 || !S_ISDIR(status.st_mode) ||
        status.st_uid != geteuid() || !realpath(parent, resolved) ||
        strcmp(parent, resolved))
        return 0;
    if (mkdir(directory, 0700) != 0 && errno != EEXIST) return 0;
    if (lstat(directory, &status) != 0 || !S_ISDIR(status.st_mode) ||
        S_ISLNK(status.st_mode) || status.st_uid != geteuid())
        return 0;
    /* Selection contains absolute model paths. Harden a user-owned YVEX directory left behind by
     * an ordinary permissive umask before publishing the mode-0600 snapshot. */
    if ((status.st_mode & 0077u) != 0u &&
        (chmod(directory, 0700) != 0 || lstat(directory, &status) != 0 ||
         (status.st_mode & 0077u) != 0u))
        return 0;
    return 1;
}

static int model_config_write(const client_model_config *config)
{
    char directory[PATH_MAX], path[PATH_MAX], temporary[PATH_MAX];
    FILE *output = NULL;
    int fd = -1, ok = 0, count;
    count = model_config_paths(directory, path)
                ? snprintf(temporary, sizeof(temporary), "%s/.model.%lu", directory,
                           (unsigned long)getpid())
                : -1;
    if (!model_config_paths(directory, path) || !model_config_directory(directory) ||
        count <= 0 || (size_t)count >= sizeof(temporary))
        return 0;
    fd = open(temporary, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
    if (fd < 0) return 0;
    output = fdopen(fd, "w");
    if (!output) {
        (void)close(fd);
        (void)unlink(temporary);
        return 0;
    }
    fd = -1;
    ok = fprintf(output,
                 "name\t%s\nartifact\t%s\nbinding\t%s\ntarget\t%s\nbackend\t%s\nmode\t%s\ncontext\t%llu\n",
                 config->name, config->artifact, config->binding, config->target,
                 config->backend, config->mode, config->context) > 0 &&
         fflush(output) == 0 && fsync(fileno(output)) == 0;
    if (fclose(output) != 0) ok = 0;
    output = NULL;
    if (ok) ok = rename(temporary, path) == 0;
    if (!ok) (void)unlink(temporary);
    return ok;
}

static int model_config_read(client_model_config *config)
{
    char directory[PATH_MAX], path[PATH_MAX], line[PATH_MAX + 32u];
    struct stat status;
    FILE *input;
    int fd, fields = 0;
    if (!config || !model_config_paths(directory, path) ||
        lstat(path, &status) != 0 || !S_ISREG(status.st_mode) ||
        S_ISLNK(status.st_mode) || status.st_uid != geteuid() ||
        (status.st_mode & 0077u) != 0u)
        return 0;
    fd = open(path, O_RDONLY | O_NOFOLLOW);
    if (fd < 0 || fstat(fd, &status) != 0 || !S_ISREG(status.st_mode)) {
        if (fd >= 0) (void)close(fd);
        return 0;
    }
    input = fdopen(fd, "r");
    if (!input) {
        (void)close(fd);
        return 0;
    }
    memset(config, 0, sizeof(*config));
    while (fgets(line, sizeof(line), input)) {
        char *value = strchr(line, '\t'), *newline;
        if (!value || !(newline = strchr(value + 1, '\n')) || newline[1]) {
            fields = -1;
            break;
        }
        *value++ = '\0';
        *newline = '\0';
        if (!strcmp(line, "name") && !config->name[0] && value[0] &&
            strlen(value) < sizeof(config->name) &&
            snprintf(config->name, sizeof(config->name), "%s", value) > 0)
            fields++;
        else if (!strcmp(line, "artifact") && !config->artifact[0] && value[0] &&
                 strlen(value) < sizeof(config->artifact) &&
                 snprintf(config->artifact, sizeof(config->artifact), "%s", value) > 0)
            fields++;
        else if (!strcmp(line, "binding") && !config->binding[0] && value[0] &&
                 strlen(value) < sizeof(config->binding) &&
                 snprintf(config->binding, sizeof(config->binding), "%s", value) > 0)
            fields++;
        else if (!strcmp(line, "target") && !config->target[0] && value[0] &&
                 strlen(value) < sizeof(config->target) &&
                 snprintf(config->target, sizeof(config->target), "%s", value) > 0)
            fields++;
        else if (!strcmp(line, "backend") && !config->backend[0] && value[0] &&
                 strlen(value) < sizeof(config->backend) &&
                 snprintf(config->backend, sizeof(config->backend), "%s", value) > 0)
            fields++;
        else if (!strcmp(line, "mode") && !config->mode[0] && value[0] &&
                 strlen(value) < sizeof(config->mode) &&
                 snprintf(config->mode, sizeof(config->mode), "%s", value) > 0)
            fields++;
        else if (!strcmp(line, "context") && !config->context &&
                 parse_u64(value, &config->context, 0))
            fields++;
        else {
            fields = -1;
            break;
        }
    }
    if (ferror(input) || fclose(input) != 0) fields = -1;
    return fields == 7 && config->artifact[0] == '/' && config->binding[0] == '/' &&
           (!strcmp(config->backend, "cpu") || !strcmp(config->backend, "cuda")) &&
           (!strcmp(config->mode, "target-only") || !strcmp(config->mode, "dspark"));
}

static int model_select_command(int argc, char **argv)
{
    yvex_model_registry_options options;
    yvex_model_registry *registry = NULL;
    const yvex_model_registry_entry *entry;
    client_model_config config;
    yvex_error err;
    int rc;

    memset(&config, 0, sizeof(config));
    memset(&options, 0, sizeof(options));
    yvex_error_clear(&err);
    if (argc != 4 || !argv[3][0] || strlen(argv[3]) >= sizeof(config.name))
        return 2;
    rc = yvex_model_registry_open(&registry, &options, &err);
    if (rc != YVEX_OK) {
        fprintf(stderr, "yvex: model registry is unavailable: %s\n"
                        "hint: use `yvex model registry add --help` to register a startup profile\n",
                yvex_error_message(&err));
        return 1;
    }
    entry = yvex_model_registry_find(registry, argv[3]);
    if (!entry) {
        fprintf(stderr, "yvex: model is not registered: %s\n"
                        "hint: inspect available profiles with `yvex model list`\n",
                argv[3]);
        yvex_model_registry_close(registry);
        return 1;
    }
    if (strlen(entry->path) >= sizeof(config.artifact) ||
        strlen(entry->runtime_binding) >= sizeof(config.binding) ||
        strlen(entry->runtime_target) >= sizeof(config.target) ||
        strlen(entry->runtime_backend) >= sizeof(config.backend) ||
        strlen(entry->runtime_mode) >= sizeof(config.mode)) {
        fprintf(stderr, "yvex: registered startup profile exceeds client configuration limits\n");
        yvex_model_registry_close(registry);
        return 1;
    }
    rc = yvex_model_registry_startup_validate(entry, &err);
    if (rc != YVEX_OK) {
        fprintf(stderr, "yvex: model cannot be selected: %s\n"
                        "hint: `yvex model show %s` reports its startup profile\n",
                yvex_error_message(&err), argv[3]);
        yvex_model_registry_close(registry);
        return 1;
    }
    if (snprintf(config.name, sizeof(config.name), "%s", entry->alias) <= 0 ||
        snprintf(config.artifact, sizeof(config.artifact), "%s", entry->path) <= 0 ||
        snprintf(config.binding, sizeof(config.binding), "%s", entry->runtime_binding) <= 0 ||
        snprintf(config.target, sizeof(config.target), "%s", entry->runtime_target) <= 0 ||
        snprintf(config.backend, sizeof(config.backend), "%s", entry->runtime_backend) <= 0 ||
        snprintf(config.mode, sizeof(config.mode), "%s", entry->runtime_mode) <= 0) {
        yvex_model_registry_close(registry);
        return 1;
    }
    config.context = entry->runtime_context;
    yvex_model_registry_close(registry);
    if (!model_config_write(&config)) {
        fprintf(stderr, "yvex: selected model configuration could not be written safely\n");
        return 1;
    }
    {
        yvex_cli_terminal_style style;
        yvex_cli_terminal_style_get(stdout, &style);
        printf("%sselected model:%s %s · target %s · %s · %s · context %llu · "
               "%srestart runtime to apply%s\n",
               style.success, style.reset, config.name, config.target,
               !strcmp(config.backend, "cuda") ? "CUDA" : "CPU",
               !strcmp(config.mode, "dspark") ? "DSpark" : "target-only",
               config.context, style.dim, style.reset);
    }
    return 0;
}

static int model_config_show(void)
{
    client_model_config config;
    if (!model_config_read(&config)) {
        fprintf(stderr,
                "yvex: no selected model\n"
                "hint: run `yvex model list`, then `yvex model select NAME`\n");
        return 1;
    }
    {
        yvex_cli_terminal_style style;
        yvex_cli_terminal_style_get(stdout, &style);
        printf("%sselected model:%s %s · target %s · backend=%s · mode=%s · context=%llu\n",
               style.strong, style.reset, config.name, config.target, config.backend,
               config.mode, config.context);
    }
    return 0;
}

static int exec_sibling_vector(const char *binary, char *const arguments[])
{
    char executable[PATH_MAX], sibling[PATH_MAX];
    ssize_t count;
    count = readlink("/proc/self/exe", executable, sizeof(executable) - 1u);
    if (count > 0 && (size_t)count < sizeof(executable)) {
        char *slash;
        executable[count] = '\0';
        slash = strrchr(executable, '/');
        if (slash) {
            *slash = '\0';
            if (snprintf(sibling, sizeof(sibling), "%s/%s", executable,
                         binary) > 0)
                execv(sibling, arguments);
        }
    }
    execvp(binary, arguments);
    fprintf(stderr, "yvex: cannot execute %s: %s\n", binary, strerror(errno));
    return 1;
}

static int exec_sibling(const char *binary, int argc, char **argv, int skip)
{
    char **arguments = calloc((size_t)argc + 1u, sizeof(*arguments));
    int index, out = 0, status;
    if (!arguments) return 1;
    arguments[out++] = (char *)binary;
    for (index = skip; index < argc; ++index) arguments[out++] = argv[index];
    arguments[out] = NULL;
    status = exec_sibling_vector(binary, arguments);
    free(arguments);
    return status;
}

static int runtime_start(int argc, char **argv)
{
    client_model_config config;
    char context[32];
    char *arguments[16];
    int count = 0;
    if (argc > 3) return exec_sibling("yvexd", argc, argv, 3);
    if (!model_config_read(&config)) {
        fprintf(stderr,
                "yvex: no selected model\n"
                "hint: run `yvex model list`, then `yvex model select NAME`\n");
        return 1;
    }
    (void)snprintf(context, sizeof(context), "%llu", config.context);
    arguments[count++] = "yvexd";
    arguments[count++] = "--model";
    arguments[count++] = config.artifact;
    arguments[count++] = "--runtime-binding";
    arguments[count++] = config.binding;
    arguments[count++] = "--target";
    arguments[count++] = config.target;
    arguments[count++] = "--backend";
    arguments[count++] = config.backend;
    arguments[count++] = "--generation-mode";
    arguments[count++] = config.mode;
    arguments[count++] = "--context";
    arguments[count++] = context;
    arguments[count] = NULL;
    {
        yvex_cli_terminal_style style;
        yvex_cli_terminal_style_get(stdout, &style);
        printf("%sYVEX runtime%s · %sloading selected model%s %s · target %s · %s · %s · "
               "context %llu\n",
               style.strong, style.reset, style.accent, style.reset, config.name,
               config.target, !strcmp(config.backend, "cuda") ? "CUDA" : "CPU",
               !strcmp(config.mode, "dspark") ? "DSpark" : "target-only", config.context);
        (void)fflush(stdout);
    }
    return exec_sibling_vector("yvexd", arguments);
}

static int help_command(int argc, char **argv, size_t consumed)
{
    const char *path[16];
    size_t count = 0u, index;
    int advanced = 0, json = 0;
    for (index = consumed + 1u; index < (size_t)argc; ++index) {
        if (!strcmp(argv[index], "--advanced")) advanced = 1;
        else if (!strcmp(argv[index], "--json")) json = 1;
        else if (strcmp(argv[index], "--help") && strcmp(argv[index], "-h")) {
            if (count == sizeof(path) / sizeof(path[0])) return 2;
            path[count++] = argv[index];
        }
    }
    if (json && (advanced || count)) {
        fprintf(stderr, "yvex: help --json is the complete deterministic discovery document\n");
        return 2;
    }
    return yvex_client_render_help_path(count, path, advanced, json);
}

static int console_status_fetch(const char *session_name,
                                yvex_client_message *message,
                                yvex_error *err)
{
    yvex_client_request request;
    yvex_client *client = NULL;
    int rc;
    request_init(&request, YVEX_CLIENT_OP_CONSOLE_STATUS);
    (void)snprintf(request.session_name, sizeof(request.session_name), "%s", session_name);
    rc = request_open(&client, &request, err);
    if (rc == YVEX_OK) rc = yvex_client_receive(client, message, err);
    if (rc == YVEX_OK && message->kind == YVEX_CLIENT_MESSAGE_ERROR) {
        yvex_error_set(err, (yvex_status)message->status, "client.console-status",
                       message->reason);
        rc = message->status;
    } else if (rc == YVEX_OK && message->kind != YVEX_CLIENT_MESSAGE_CONSOLE_STATUS) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "client.console-status",
                       "daemon returned an unexpected console status response");
        rc = YVEX_ERR_FORMAT;
    }
    yvex_client_close(&client);
    return rc;
}

static void render_console_status(const yvex_client_message *message, int startup)
{
    const yvex_console_status *status = &message->console;
    const char *target = message->runtime.target_id[0] ? message->runtime.target_id
                                                       : status->live_model_identity;
    yvex_cli_terminal_style style;
    yvex_cli_terminal_style_get(stdout, &style);
    if (startup)
        printf("%sYVEX %s%s · protocol %u · ", style.strong, yvex_version_string(),
               style.reset, YVEX_LOCAL_PROTOCOL_VERSION);
    else
        printf("%sconsole%s · ", style.strong, style.reset);
    printf("%s · %s · %s · variant %.12s · %s%s%s · %s · "
           "session %s · position %llu · turns %llu · context %llu/%llu",
           target, backend_name(status->backend),
           generation_mode_name(message->runtime.generation_mode),
           status->physical_variant_identity,
           status->runtime_ready ? style.success : style.warning,
           status->runtime_ready ? "● ready" : "● not ready", style.reset,
           status->attached ? "attached to resident runtime" : "detached from runtime",
           status->session_name,
           status->position, status->turn_count, status->context_used,
           status->context_capacity);
    if (status->kv_used_available)
        printf(" · KV %.2f MiB", (double)status->kv_used_bytes / 1048576.0);
    if (startup) {
        printf(" · %smemory %.2f GiB host/%.2f GiB device%s", style.dim,
               (double)message->runtime.metrics.resident_host_bytes / 1073741824.0,
               (double)message->runtime.metrics.resident_device_bytes / 1073741824.0,
               style.reset);
        if (message->runtime.openai_listener_enabled)
            printf(" · OpenAI %s%s%s 127.0.0.1:%u",
                   message->runtime.openai_listener_ready ? style.success : style.warning,
                   message->runtime.openai_listener_ready ? "ready" : "starting",
                   style.reset, (unsigned int)message->runtime.openai_port);
        else
            printf(" · OpenAI %sdisabled%s", style.dim, style.reset);
    } else {
        printf(" · live %.12s", status->live_model_identity);
        if (status->selected_model_available)
            printf(" · selected %.12s", status->selected_model_identity);
    }
    putchar('\n');
}

static int console_status(const char *session_name)
{
    yvex_client_message message;
    yvex_error err;
    int rc = console_status_fetch(session_name, &message, &err);
    if (rc == YVEX_OK) render_console_status(&message, 0);
    return rc == YVEX_OK ? 0 : client_error(&err);
}

int yvex_client_dispatch(const yvex_operator_descriptor *operation, int argc,
                         char **argv, size_t consumed)
{
    const char *name = consumed + 1u < (size_t)argc ? argv[consumed + 1u] : NULL;
    switch (operation->runtime_adapter) {
    case YVEX_OPERATOR_RUNTIME_CHAT: return chat_command(argc, argv);
    case YVEX_OPERATOR_RUNTIME_RUN: return run_command(argc, argv);
    case YVEX_OPERATOR_RUNTIME_RUNTIME_START: return runtime_start(argc, argv);
    case YVEX_OPERATOR_RUNTIME_RUNTIME_STATUS:
        return runtime_status(argc > 3 && !strcmp(argv[3], "--json"));
    case YVEX_OPERATOR_RUNTIME_RUNTIME_MODEL: return runtime_model();
    case YVEX_OPERATOR_RUNTIME_RUNTIME_MEMORY: return runtime_memory();
    case YVEX_OPERATOR_RUNTIME_RUNTIME_WATCH: return runtime_events(0);
    case YVEX_OPERATOR_RUNTIME_RUNTIME_TRACE:
        return runtime_events(argc > 3 && !strcmp(argv[3], "--json") ? 2 : 1);
    case YVEX_OPERATOR_RUNTIME_RUNTIME_STOP:
        return administration(YVEX_CLIENT_OP_RUNTIME_STOP, NULL, 0);
    case YVEX_OPERATOR_RUNTIME_SESSION_NEW:
        return administration(YVEX_CLIENT_OP_SESSION_NEW, name, 0);
    case YVEX_OPERATOR_RUNTIME_SESSION_LIST:
        return administration(YVEX_CLIENT_OP_SESSION_LIST, NULL, 1);
    case YVEX_OPERATOR_RUNTIME_SESSION_SHOW:
        return administration(YVEX_CLIENT_OP_SESSION_SHOW, name, 0);
    case YVEX_OPERATOR_RUNTIME_SESSION_ATTACH:
        return administration(YVEX_CLIENT_OP_SESSION_ATTACH, name, 0);
    case YVEX_OPERATOR_RUNTIME_SESSION_DETACH:
        return administration(YVEX_CLIENT_OP_SESSION_DETACH, name, 0);
    case YVEX_OPERATOR_RUNTIME_SESSION_RESET:
        return administration(YVEX_CLIENT_OP_SESSION_RESET, name, 0);
    case YVEX_OPERATOR_RUNTIME_SESSION_CLOSE:
        return administration(YVEX_CLIENT_OP_SESSION_CLOSE, name, 0);
    case YVEX_OPERATOR_RUNTIME_SESSION_CANCEL:
        puts(cancellation_request(name) ? "cancel requested" : "no active turn");
        return 0;
    case YVEX_OPERATOR_RUNTIME_MODEL_SELECTED: return model_config_show();
    case YVEX_OPERATOR_RUNTIME_MODEL_SELECT: return model_select_command(argc, argv);
    case YVEX_OPERATOR_RUNTIME_HELP: return help_command(argc, argv, consumed);
    case YVEX_OPERATOR_RUNTIME_COMPLETION:
        return yvex_cli_completion_command(argc, argv, consumed);
    case YVEX_OPERATOR_RUNTIME_VERSION:
        printf("yvex %s protocol=%u registry=%s commit=%s\n", yvex_version_string(),
               YVEX_LOCAL_PROTOCOL_VERSION, yvex_operator_registry_identity,
               YVEX_BUILD_COMMIT);
        return 0;
    case YVEX_OPERATOR_RUNTIME_CONSOLE_STATUS: return console_status(name ? name : "main");
    case YVEX_OPERATOR_RUNTIME_COUNT: break;
    }
    fprintf(stderr, "yvex: unbound runtime adapter: %s\n", operation->adapter_id);
    return 2;
}
