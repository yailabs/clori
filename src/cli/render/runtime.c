/* Render typed host facts as ordinary scrollback-safe human tables. */
#include "src/cli/render/private.h"

#include <string.h>

typedef struct {
    const char *key, *value;
    yvex_cli_table_tone tone;
} host_status_pair;

static void host_bytes(char out[32], unsigned long long bytes)
{
    static const char *const units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double value = (double)bytes;
    unsigned int unit = 0u;
    while (value >= 1024.0 && unit + 1u < sizeof(units) / sizeof(units[0])) {
        value /= 1024.0;
        unit++;
    }
    if (unit) snprintf(out, 32u, "%.2f %s", value, units[unit]);
    else snprintf(out, 32u, "%llu B", bytes);
}

static void host_duration(char out[32], unsigned long long nanoseconds)
{
    unsigned long long seconds = nanoseconds / 1000000000ull;
    unsigned long long days = seconds / 86400ull;
    unsigned long long hours = (seconds % 86400ull) / 3600ull;
    unsigned long long minutes = (seconds % 3600ull) / 60ull;
    seconds %= 60ull;
    if (days) snprintf(out, 32u, "%llud %02llu:%02llu:%02llu", days, hours,
                       minutes, seconds);
    else snprintf(out, 32u, "%02llu:%02llu:%02llu", hours, minutes, seconds);
}

static void host_section(FILE *fp, const char *title,
                         const host_status_pair *pairs, size_t count)
{
    static const yvex_cli_table_column columns[] = {
        {"", 10u, 20u, YVEX_CLI_TABLE_LEFT, 0},
        {"", 12u, 100u, YVEX_CLI_TABLE_LEFT, 1}
    };
    yvex_cli_table_cell cells[8][2];
    yvex_cli_table_row rows[8];
    size_t index;
    if (!count || count > 8u) return;
    yvex_cli_out_writef(fp, "%s\n", title);
    for (index = 0u; index < count; ++index) {
        cells[index][0] = (yvex_cli_table_cell){pairs[index].key,
                                                YVEX_CLI_TABLE_DIM};
        cells[index][1] = (yvex_cli_table_cell){pairs[index].value,
                                                pairs[index].tone};
        rows[index] = (yvex_cli_table_row){cells[index], NULL,
                                           YVEX_CLI_TABLE_PLAIN};
    }
    (void)yvex_cli_table_render(fp, columns, 2u, rows, count);
    yvex_cli_out_char(fp, '\n');
}

static void host_status_render_human(FILE *fp,
                                     const yvex_server_summary *status)
{
    char uptime[32], workers[24], queue[48], loaded[24], known[24], capacity[24];
    char active_sessions[24], total_sessions[24];
    char rss[32], peak_rss[32], mapped[32], host[32], device[32];
    char openai[64];
    int ready = status->status == YVEX_SERVER_STATUS_READY && status->host_ready;
    host_status_pair host_pairs[4], model_pairs[3], session_pairs[2];
    host_status_pair memory_pairs[5], endpoint_pairs[2];
    host_duration(uptime, status->metrics.uptime_ns);
    snprintf(workers, sizeof(workers), "%llu", status->worker_count);
    snprintf(queue, sizeof(queue), "%llu / %llu", status->metrics.queue_depth,
             status->metrics.queue_capacity);
    snprintf(loaded, sizeof(loaded), "%llu", status->loaded_engine_count);
    snprintf(known, sizeof(known), "%llu", status->engine_count);
    snprintf(capacity, sizeof(capacity), "%llu", status->maximum_engines);
    snprintf(active_sessions, sizeof(active_sessions), "%llu",
             status->metrics.active_sessions);
    snprintf(total_sessions, sizeof(total_sessions), "%llu",
             status->metrics.total_sessions);
    host_bytes(rss, status->metrics.current_rss_bytes);
    host_bytes(peak_rss, status->metrics.peak_rss_bytes);
    host_bytes(mapped, status->metrics.mapped_artifact_bytes);
    host_bytes(host, status->metrics.resident_host_bytes);
    host_bytes(device, status->metrics.resident_device_bytes);
    if (status->openai_listener_enabled)
        snprintf(openai, sizeof(openai), "127.0.0.1:%u (%s)",
                 (unsigned int)status->openai_port,
                 status->openai_listener_ready ? "ready" : "starting");
    else snprintf(openai, sizeof(openai), "%s", "disabled");
    host_pairs[0] = (host_status_pair){"State", ready ? "ready" : "starting",
        ready ? YVEX_CLI_TABLE_SUCCESS : YVEX_CLI_TABLE_WARNING};
    host_pairs[1] = (host_status_pair){"Uptime", uptime, YVEX_CLI_TABLE_PLAIN};
    host_pairs[2] = (host_status_pair){"Workers", workers, YVEX_CLI_TABLE_PLAIN};
    host_pairs[3] = (host_status_pair){"Queue", queue, YVEX_CLI_TABLE_PLAIN};
    model_pairs[0] = (host_status_pair){"Loaded", loaded,
        status->loaded_engine_count ? YVEX_CLI_TABLE_SUCCESS : YVEX_CLI_TABLE_WARNING};
    model_pairs[1] = (host_status_pair){"Known engines", known, YVEX_CLI_TABLE_PLAIN};
    model_pairs[2] = (host_status_pair){"Capacity", capacity, YVEX_CLI_TABLE_PLAIN};
    session_pairs[0] = (host_status_pair){"Active", active_sessions,
                                          YVEX_CLI_TABLE_PLAIN};
    session_pairs[1] = (host_status_pair){"Total", total_sessions,
                                          YVEX_CLI_TABLE_PLAIN};
    memory_pairs[0] = (host_status_pair){"RSS", rss, YVEX_CLI_TABLE_PLAIN};
    memory_pairs[1] = (host_status_pair){"Peak RSS", peak_rss, YVEX_CLI_TABLE_PLAIN};
    memory_pairs[2] = (host_status_pair){"Mapped", mapped, YVEX_CLI_TABLE_PLAIN};
    memory_pairs[3] = (host_status_pair){"Host resident", host, YVEX_CLI_TABLE_PLAIN};
    memory_pairs[4] = (host_status_pair){"Device resident", device,
                                         YVEX_CLI_TABLE_PLAIN};
    endpoint_pairs[0] = (host_status_pair){"Native", status->socket_path,
                                           YVEX_CLI_TABLE_DIM};
    endpoint_pairs[1] = (host_status_pair){"OpenAI", openai,
        status->openai_listener_ready ? YVEX_CLI_TABLE_SUCCESS
                                      : YVEX_CLI_TABLE_WARNING};
    host_section(fp, "HOST", host_pairs, 4u);
    host_section(fp, "MODELS", model_pairs, 3u);
    host_section(fp, "SESSIONS", session_pairs, 2u);
    host_section(fp, "MEMORY", memory_pairs, 5u);
    host_section(fp, "ENDPOINTS", endpoint_pairs, 2u);
}

static void host_status_render_json(FILE *fp,
                                    const yvex_server_summary *status)
{
    yvex_cli_out_writef(
        fp,
        "{\"schema\":\"yvex.host.status.v1\",\"protocol\":%u,"
        "\"status\":%u,\"host_ready\":%s,"
        "\"engine_count\":%llu,\"loaded_engine_count\":%llu,"
        "\"draining_engine_count\":%llu,\"maximum_engines\":%llu,"
        "\"workers\":%llu,\"session_count\":%llu,\"requests\":%llu,"
        "\"uptime_ns\":%llu,\"model_open_count\":%llu,"
        "\"model_close_count\":%llu,\"artifact_open_count\":%llu,"
        "\"binding_open_count\":%llu,\"materialization_count\":%llu,"
        "\"residency_build_count\":%llu,\"output_head_upload_count\":%llu,"
        "\"sessions\":%llu,\"active_sessions\":%llu,"
        "\"total_sessions\":%llu,\"queue_depth\":%llu,"
        "\"queue_capacity\":%llu,\"active_requests\":%llu,"
        "\"completed_requests\":%llu,\"failed_requests\":%llu,"
        "\"cancelled_requests\":%llu,\"openai_enabled\":%s,"
        "\"openai_ready\":%s,\"openai_port\":%u,"
        "\"active_http_requests\":%llu,\"completed_http_requests\":%llu,"
        "\"failed_http_requests\":%llu,\"cancelled_http_requests\":%llu,"
        "\"telemetry_dropped\":%llu,\"rss_bytes\":%llu,"
        "\"peak_rss_bytes\":%llu,\"mapped_artifact_bytes\":%llu,"
        "\"resident_host_bytes\":%llu,\"resident_device_bytes\":%llu}\n",
        YVEX_LOCAL_PROTOCOL_VERSION, (unsigned int)status->status,
        status->host_ready ? "true" : "false", status->engine_count,
        status->loaded_engine_count, status->draining_engine_count,
        status->maximum_engines, status->worker_count, status->session_count,
        status->request_count, status->metrics.uptime_ns,
        status->metrics.model_open_count, status->metrics.model_close_count,
        status->metrics.artifact_open_count, status->metrics.binding_open_count,
        status->metrics.materialization_count,
        status->metrics.residency_build_count,
        status->metrics.output_head_upload_count, status->session_count,
        status->metrics.active_sessions, status->metrics.total_sessions,
        status->metrics.queue_depth, status->metrics.queue_capacity,
        status->metrics.active_requests, status->metrics.completed_requests,
        status->metrics.failed_requests, status->metrics.cancelled_requests,
        status->openai_listener_enabled ? "true" : "false",
        status->openai_listener_ready ? "true" : "false",
        (unsigned int)status->openai_port,
        status->metrics.active_http_requests,
        status->metrics.completed_http_requests,
        status->metrics.failed_http_requests,
        status->metrics.cancelled_http_requests, status->metrics.telemetry_dropped,
        status->metrics.current_rss_bytes, status->metrics.peak_rss_bytes,
        status->metrics.mapped_artifact_bytes,
        status->metrics.resident_host_bytes,
        status->metrics.resident_device_bytes);
}

void yvex_cli_host_status_render(FILE *fp, const yvex_server_summary *status,
                                 int json)
{
    if (json) host_status_render_json(fp, status);
    else host_status_render_human(fp, status);
}

void yvex_cli_host_memory_render(FILE *fp, const yvex_server_summary *status,
                                 int json)
{
    char rss[32], peak_rss[32], mapped[32], host[32], device[32];
    host_status_pair pairs[5];
    if (!fp || !status) return;
    if (json) {
        yvex_cli_out_writef(
            fp,
            "{\"schema\":\"yvex.host.memory.v1\",\"rss_bytes\":%llu,"
            "\"peak_rss_bytes\":%llu,\"mapped_artifact_bytes\":%llu,"
            "\"resident_host_bytes\":%llu,\"resident_device_bytes\":%llu}\n",
            status->metrics.current_rss_bytes, status->metrics.peak_rss_bytes,
            status->metrics.mapped_artifact_bytes,
            status->metrics.resident_host_bytes,
            status->metrics.resident_device_bytes);
        return;
    }
    host_bytes(rss, status->metrics.current_rss_bytes);
    host_bytes(peak_rss, status->metrics.peak_rss_bytes);
    host_bytes(mapped, status->metrics.mapped_artifact_bytes);
    host_bytes(host, status->metrics.resident_host_bytes);
    host_bytes(device, status->metrics.resident_device_bytes);
    pairs[0] = (host_status_pair){"RSS", rss, YVEX_CLI_TABLE_PLAIN};
    pairs[1] = (host_status_pair){"Peak RSS", peak_rss, YVEX_CLI_TABLE_PLAIN};
    pairs[2] = (host_status_pair){"Mapped", mapped, YVEX_CLI_TABLE_PLAIN};
    pairs[3] = (host_status_pair){"Host resident", host, YVEX_CLI_TABLE_PLAIN};
    pairs[4] = (host_status_pair){"Device resident", device,
                                  YVEX_CLI_TABLE_PLAIN};
    host_section(fp, "MEMORY", pairs, 5u);
}

int yvex_cli_session_table_render(FILE *fp,
                                  const yvex_cli_session_table_fact *facts,
                                  size_t count)
{
    static const yvex_cli_table_column columns[] = {
        {"SESSION", 12u, 32u, YVEX_CLI_TABLE_LEFT, 0},
        {"STATE", 8u, 14u, YVEX_CLI_TABLE_LEFT, 0},
        {"POSITION", 8u, 14u, YVEX_CLI_TABLE_RIGHT, 0},
        {"TURNS", 5u, 10u, YVEX_CLI_TABLE_RIGHT, 0}
    };
    yvex_cli_table_cell *cells;
    yvex_cli_table_row *rows;
    char (*numbers)[2][24];
    size_t index;
    int rc;
    if (!count) {
        yvex_cli_out_fputs("no sessions known to this engine\n", fp);
        return YVEX_OK;
    }
    cells = calloc(count * 4u, sizeof(*cells));
    rows = calloc(count, sizeof(*rows));
    numbers = calloc(count, sizeof(*numbers));
    if (!cells || !rows || !numbers) {
        free(cells);
        free(rows);
        free(numbers);
        return YVEX_ERR_NOMEM;
    }
    for (index = 0u; index < count; ++index) {
        snprintf(numbers[index][0], sizeof(numbers[index][0]), "%llu",
                 facts[index].position);
        snprintf(numbers[index][1], sizeof(numbers[index][1]), "%llu",
                 facts[index].turns);
        cells[index * 4u + 0u] = (yvex_cli_table_cell){
            facts[index].name, YVEX_CLI_TABLE_ACCENT};
        cells[index * 4u + 1u] = (yvex_cli_table_cell){
            facts[index].state, facts[index].ready ? YVEX_CLI_TABLE_SUCCESS
                                                   : YVEX_CLI_TABLE_WARNING};
        cells[index * 4u + 2u] = (yvex_cli_table_cell){
            numbers[index][0], YVEX_CLI_TABLE_PLAIN};
        cells[index * 4u + 3u] = (yvex_cli_table_cell){
            numbers[index][1], YVEX_CLI_TABLE_PLAIN};
        rows[index] = (yvex_cli_table_row){&cells[index * 4u], NULL,
                                           YVEX_CLI_TABLE_PLAIN};
    }
    rc = yvex_cli_table_render(fp, columns, 4u, rows, count);
    free(cells);
    free(rows);
    free(numbers);
    return rc;
}

static void session_json_fact(FILE *fp,
                              const yvex_cli_session_table_fact *fact)
{
    yvex_cli_out_fputs("{\"name\":", fp);
    yvex_cli_out_json_string(fp, fact->name);
    yvex_cli_out_fputs(",\"state\":", fp);
    yvex_cli_out_json_string(fp, fact->state);
    yvex_cli_out_writef(fp, ",\"position\":%llu,\"turns\":%llu}",
                        fact->position, fact->turns);
}

int yvex_cli_session_json_render(FILE *fp,
                                 const yvex_cli_session_table_fact *facts,
                                 size_t count, int list)
{
    size_t index;
    if (!fp || (count && !facts)) return YVEX_ERR_INVALID_ARG;
    if (!list) {
        yvex_cli_out_fputs("{\"schema\":\"yvex.session.v1\",\"session\":", fp);
        if (count) session_json_fact(fp, &facts[0]);
        else yvex_cli_out_fputs("null", fp);
        yvex_cli_out_fputs("}\n", fp);
        return YVEX_OK;
    }
    yvex_cli_out_fputs("{\"schema\":\"yvex.session.list.v1\",\"sessions\":[", fp);
    for (index = 0u; index < count; ++index) {
        if (index) yvex_cli_out_fputs(",", fp);
        session_json_fact(fp, &facts[index]);
    }
    yvex_cli_out_fputs("]}\n", fp);
    return YVEX_OK;
}

void yvex_cli_session_table_fact_set(yvex_cli_session_table_fact *fact,
                                     const yvex_client_message *message)
{
    if (!fact || !message) return;
    memset(fact, 0, sizeof(*fact));
    snprintf(fact->name, sizeof(fact->name), "%s", message->session_name);
    snprintf(fact->state, sizeof(fact->state), "%s",
             yvex_server_session_state_name(message->session_state));
    fact->position = message->final_position;
    fact->turns = message->turn_count;
    fact->ready = message->session_state == YVEX_SERVER_SESSION_READY;
}
