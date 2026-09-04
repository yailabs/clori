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

static void host_resource_bytes(
    char out[32], const yvex_execution_resource_summary *resource,
    unsigned long long availability, unsigned long long bytes)
{
    if (resource && (resource->available & availability))
        host_bytes(out, bytes);
    else
        snprintf(out, 32u, "%s", "not reported");
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

static const char *engine_kind_name(yvex_server_engine_kind kind)
{
    if (kind == YVEX_SERVER_ENGINE_TEXT) return "text";
    if (kind == YVEX_SERVER_ENGINE_MEDIA) return "media";
    return "none";
}

static const char *engine_strategy_name(yvex_server_execution_strategy strategy)
{
    if (strategy == YVEX_SERVER_EXECUTION_TARGET_ONLY) return "target-only";
    if (strategy == YVEX_SERVER_EXECUTION_SPECULATIVE) return "speculative";
    return "n/a";
}

static const char *engine_backend_name(yvex_backend_kind backend)
{
    return backend == YVEX_BACKEND_KIND_CUDA ? "CUDA" : "CPU";
}

static const char *resource_placement_name(
    yvex_execution_resource_placement placement)
{
    switch (placement) {
    case YVEX_EXECUTION_PLACEMENT_EXPLICIT_HOST: return "explicit-host";
    case YVEX_EXECUTION_PLACEMENT_MANAGED_UNIFIED: return "managed-unified";
    case YVEX_EXECUTION_PLACEMENT_ARTIFACT_MAPPED: return "artifact-mapped";
    case YVEX_EXECUTION_PLACEMENT_ARTIFACT_MAPPED_DEVICE_ADDRESSABLE:
        return "mapped-device-addressable";
    case YVEX_EXECUTION_PLACEMENT_COMPOSITE: return "composite";
    case YVEX_EXECUTION_PLACEMENT_UNKNOWN: break;
    }
    return "unknown";
}

static const char *truth_name(int value)
{
    return value ? "yes" : "no";
}

static void resource_json(FILE *fp,
                          const yvex_execution_resource_summary *resource)
{
    yvex_cli_out_writef(
        fp,
        "{\"schema_version\":%u,\"available\":%llu,\"placement\":",
        resource->schema_version, resource->available);
    yvex_cli_out_json_string(fp, resource_placement_name(resource->placement));
    yvex_cli_out_writef(
        fp,
        ",\"physical_residency_known\":%s,\"unified_memory\":%s,"
        "\"components\":%llu,\"model_artifact_bytes\":%llu,"
        "\"model_mapped_bytes\":%llu,\"model_prepared_bytes\":%llu,"
        "\"model_explicit_host_bytes\":%llu,"
        "\"model_explicit_device_bytes\":%llu,"
        "\"model_device_addressable_bytes\":%llu,"
        "\"session_attention_allocated_bytes\":%llu,"
        "\"session_attention_resident_bytes\":%llu,"
        "\"session_attention_virtual_bytes\":%llu,"
        "\"session_attention_page_table_bytes\":%llu,"
        "\"session_recurrent_state_bytes\":%llu,"
        "\"session_convolution_state_bytes\":%llu,"
        "\"session_candidate_state_bytes\":%llu,"
        "\"session_physical_state_bytes\":%llu,"
        "\"activation_arena_current_bytes\":%llu,"
        "\"activation_arena_peak_bytes\":%llu,"
        "\"workspace_current_bytes\":%llu,\"workspace_peak_bytes\":%llu,"
        "\"transient_current_bytes\":%llu,\"transient_peak_bytes\":%llu,"
        "\"process_rss_current_bytes\":%llu,"
        "\"process_rss_peak_bytes\":%llu,\"logical_upload_bytes\":%llu,"
        "\"logical_download_bytes\":%llu}",
        (resource->available &
         YVEX_EXECUTION_RESOURCE_PHYSICAL_RESIDENCY_AVAILABLE) ? "true" : "false",
        (resource->available & YVEX_EXECUTION_RESOURCE_UNIFIED_MEMORY)
            ? "true" : "false",
        resource->component_count, resource->model_artifact_bytes,
        resource->model_mapped_bytes, resource->model_prepared_bytes,
        resource->model_explicit_host_bytes,
        resource->model_explicit_device_bytes,
        resource->model_device_addressable_bytes,
        resource->session_attention_allocated_bytes,
        resource->session_attention_resident_bytes,
        resource->session_attention_virtual_bytes,
        resource->session_attention_page_table_bytes,
        resource->session_recurrent_state_bytes,
        resource->session_convolution_state_bytes,
        resource->session_candidate_state_bytes,
        resource->session_physical_state_bytes,
        resource->activation_arena_current_bytes,
        resource->activation_arena_peak_bytes,
        resource->workspace_current_bytes, resource->workspace_peak_bytes,
        resource->transient_current_bytes, resource->transient_peak_bytes,
        resource->process_rss_current_bytes,
        resource->process_rss_peak_bytes, resource->logical_upload_bytes,
        resource->logical_download_bytes);
}

void yvex_cli_engine_render(FILE *fp,
                            const yvex_server_engine_summary *engine,
                            int json)
{
    const yvex_execution_capacity_summary *capacity = &engine->capacity;
    const yvex_execution_resource_summary *resource = &engine->resources;
    char mapped[32], prepared[32], addressable[32], explicit_device[32];
    char state[32], workspace[32];
    if (!fp || !engine) return;
    if (json) {
        yvex_cli_out_fputs("{\"alias\":", fp);
        yvex_cli_out_json_string(fp, engine->alias);
        yvex_cli_out_writef(fp, ",\"generation\":%llu,\"state\":",
                            engine->generation);
        yvex_cli_out_json_string(fp, engine_state_name(engine->state));
        yvex_cli_out_writef(fp, ",\"backend\":%u,\"engine_kind\":",
                            (unsigned int)engine->backend);
        yvex_cli_out_json_string(fp, engine_kind_name(engine->engine_kind));
        yvex_cli_out_fputs(",\"execution_strategy\":", fp);
        yvex_cli_out_json_string(fp,
                                 engine_strategy_name(engine->execution_strategy));
        yvex_cli_out_writef(
            fp,
            ",\"execution_ready\":%s,\"continuous_batching\":%s,"
            "\"context_capacity\":%llu,\"prefill_chunk_tokens\":%llu,"
            "\"maximum_new_tokens\":%llu,\"maximum_sessions\":%llu,"
            "\"configured_physical_sequence_width\":%llu,\"active_work\":%llu,"
            "\"sessions\":%llu,\"mapped_package_bytes\":%llu,"
            "\"resident_host_bytes\":%llu,\"resident_device_bytes\":%llu,"
            "\"prepared_bytes\":%llu,\"capacity\":{"
            "\"sessions\":%llu,\"runnable_work\":%llu,"
            "\"physical_sequence_width\":%llu,"
            "\"cooperative_scheduling\":%s,"
            "\"compatible_operation_batching\":%s,"
            "\"continuous_batching\":%s},\"resources\":",
            engine->execution_ready ? "true" : "false",
            engine->continuous_batching_ready ? "true" : "false",
            engine->context_capacity, engine->prefill_chunk_tokens,
            engine->maximum_new_tokens, engine->maximum_sessions,
            engine->concurrent_sequences, engine->active_work,
            engine->session_count, engine->mapped_package_bytes,
            engine->resident_host_bytes, engine->resident_device_bytes,
            engine->prepared_bytes, capacity->session_capacity,
            capacity->runnable_work_capacity,
            capacity->physical_sequence_width,
            capacity->cooperative_scheduling_ready ? "true" : "false",
            capacity->compatible_operation_batching_ready ? "true" : "false",
            capacity->continuous_batching_ready ? "true" : "false");
        resource_json(fp, resource);
        yvex_cli_out_fputs(",\"target\":", fp);
        yvex_cli_out_json_string(fp, engine->target_id);
        yvex_cli_out_fputs(",\"model_identity\":", fp);
        yvex_cli_out_json_string(fp, engine->runtime_model_identity);
        yvex_cli_out_fputs(",\"specialization_identity\":", fp);
        yvex_cli_out_json_string(fp, engine->specialization_identity);
        yvex_cli_out_char(fp, '}');
        return;
    }
    host_resource_bytes(mapped, resource, YVEX_EXECUTION_RESOURCE_MODEL_AVAILABLE,
                        resource->model_mapped_bytes);
    host_resource_bytes(prepared, resource,
                        YVEX_EXECUTION_RESOURCE_MODEL_AVAILABLE,
                        resource->model_prepared_bytes);
    host_resource_bytes(addressable, resource,
                        YVEX_EXECUTION_RESOURCE_MODEL_AVAILABLE,
                        resource->model_device_addressable_bytes);
    host_resource_bytes(explicit_device, resource,
                        YVEX_EXECUTION_RESOURCE_MODEL_AVAILABLE,
                        resource->model_explicit_device_bytes);
    host_resource_bytes(state, resource,
                        YVEX_EXECUTION_RESOURCE_SESSION_AVAILABLE,
                        resource->session_physical_state_bytes);
    host_resource_bytes(workspace, resource,
                        YVEX_EXECUTION_RESOURCE_WORKSPACE_AVAILABLE,
                        resource->workspace_current_bytes);
    yvex_cli_out_writef(
        fp, "%s  generation %llu  %s  %s/%s/%s\n",
        engine->alias, engine->generation, engine_state_name(engine->state),
        engine_backend_name(engine->backend),
        engine_kind_name(engine->engine_kind),
        engine_strategy_name(engine->execution_strategy));
    yvex_cli_out_writef(
        fp,
        "  capacity  sessions %llu/%llu · runnable %llu · physical width %llu"
        " · cooperative %s · compatible batching %s · continuous batching %s\n",
        engine->session_count, capacity->session_capacity,
        capacity->runnable_work_capacity, capacity->physical_sequence_width,
        truth_name(capacity->cooperative_scheduling_ready),
        truth_name(capacity->compatible_operation_batching_ready),
        truth_name(capacity->continuous_batching_ready));
    yvex_cli_out_writef(
        fp,
        "  model     mapped %s · prepared %s · device-addressable %s · "
        "explicit device %s\n",
        mapped, prepared, addressable, explicit_device);
    yvex_cli_out_writef(
        fp,
        "  runtime   session state %s · workspace %s · placement %s · "
        "physical residency %s\n",
        state, workspace, resource_placement_name(resource->placement),
        (resource->available &
         YVEX_EXECUTION_RESOURCE_PHYSICAL_RESIDENCY_AVAILABLE)
            ? "measured" : "not measured");
}

static void host_resource_sections(
    FILE *fp, const yvex_execution_resource_summary *resource)
{
    char artifact[32], mapped[32], prepared[32], addressable[32];
    char explicit_host[32], explicit_device[32], attention[32], recurrent[32];
    char convolution[32], candidate[32], physical[32], arena[32];
    char workspace[32], workspace_peak[32], transient[32], transient_peak[32];
    char rss[32], peak_rss[32];
    host_status_pair model[8], session[8], process[4];
    host_resource_bytes(artifact, resource,
                        YVEX_EXECUTION_RESOURCE_MODEL_AVAILABLE,
                        resource->model_artifact_bytes);
    host_resource_bytes(mapped, resource, YVEX_EXECUTION_RESOURCE_MODEL_AVAILABLE,
                        resource->model_mapped_bytes);
    host_resource_bytes(prepared, resource,
                        YVEX_EXECUTION_RESOURCE_MODEL_AVAILABLE,
                        resource->model_prepared_bytes);
    host_resource_bytes(addressable, resource,
                        YVEX_EXECUTION_RESOURCE_MODEL_AVAILABLE,
                        resource->model_device_addressable_bytes);
    host_resource_bytes(explicit_host, resource,
                        YVEX_EXECUTION_RESOURCE_MODEL_AVAILABLE,
                        resource->model_explicit_host_bytes);
    host_resource_bytes(explicit_device, resource,
                        YVEX_EXECUTION_RESOURCE_MODEL_AVAILABLE,
                        resource->model_explicit_device_bytes);
    host_resource_bytes(attention, resource,
                        YVEX_EXECUTION_RESOURCE_SESSION_AVAILABLE,
                        resource->session_attention_allocated_bytes);
    host_resource_bytes(recurrent, resource,
                        YVEX_EXECUTION_RESOURCE_SESSION_AVAILABLE,
                        resource->session_recurrent_state_bytes);
    host_resource_bytes(convolution, resource,
                        YVEX_EXECUTION_RESOURCE_SESSION_AVAILABLE,
                        resource->session_convolution_state_bytes);
    host_resource_bytes(candidate, resource,
                        YVEX_EXECUTION_RESOURCE_SESSION_AVAILABLE,
                        resource->session_candidate_state_bytes);
    host_resource_bytes(physical, resource,
                        YVEX_EXECUTION_RESOURCE_SESSION_AVAILABLE,
                        resource->session_physical_state_bytes);
    host_resource_bytes(arena, resource, YVEX_EXECUTION_RESOURCE_ARENA_AVAILABLE,
                        resource->activation_arena_current_bytes);
    host_resource_bytes(workspace, resource,
                        YVEX_EXECUTION_RESOURCE_WORKSPACE_AVAILABLE,
                        resource->workspace_current_bytes);
    host_resource_bytes(workspace_peak, resource,
                        YVEX_EXECUTION_RESOURCE_WORKSPACE_AVAILABLE,
                        resource->workspace_peak_bytes);
    host_resource_bytes(transient, resource,
                        YVEX_EXECUTION_RESOURCE_TRANSIENT_AVAILABLE,
                        resource->transient_current_bytes);
    host_resource_bytes(transient_peak, resource,
                        YVEX_EXECUTION_RESOURCE_TRANSIENT_AVAILABLE,
                        resource->transient_peak_bytes);
    host_resource_bytes(rss, resource, YVEX_EXECUTION_RESOURCE_PROCESS_AVAILABLE,
                        resource->process_rss_current_bytes);
    host_resource_bytes(peak_rss, resource,
                        YVEX_EXECUTION_RESOURCE_PROCESS_AVAILABLE,
                        resource->process_rss_peak_bytes);
    model[0] = (host_status_pair){"Artifact", artifact, YVEX_CLI_TABLE_PLAIN};
    model[1] = (host_status_pair){"Mapped", mapped, YVEX_CLI_TABLE_PLAIN};
    model[2] = (host_status_pair){"Prepared", prepared, YVEX_CLI_TABLE_PLAIN};
    model[3] = (host_status_pair){"Device-addressable", addressable,
                                  YVEX_CLI_TABLE_PLAIN};
    model[4] = (host_status_pair){"Explicit host", explicit_host,
                                  YVEX_CLI_TABLE_PLAIN};
    model[5] = (host_status_pair){"Explicit device", explicit_device,
                                  YVEX_CLI_TABLE_PLAIN};
    model[6] = (host_status_pair){"Placement",
        resource_placement_name(resource->placement), YVEX_CLI_TABLE_DIM};
    model[7] = (host_status_pair){"Physical pages",
        (resource->available &
         YVEX_EXECUTION_RESOURCE_PHYSICAL_RESIDENCY_AVAILABLE)
            ? "measured" : "not measured",
        (resource->available &
         YVEX_EXECUTION_RESOURCE_PHYSICAL_RESIDENCY_AVAILABLE)
            ? YVEX_CLI_TABLE_PLAIN : YVEX_CLI_TABLE_WARNING};
    session[0] = (host_status_pair){"Attention state", attention,
                                    YVEX_CLI_TABLE_PLAIN};
    session[1] = (host_status_pair){"Recurrent state", recurrent,
                                    YVEX_CLI_TABLE_PLAIN};
    session[2] = (host_status_pair){"Convolution state", convolution,
                                    YVEX_CLI_TABLE_PLAIN};
    session[3] = (host_status_pair){"Candidate state", candidate,
                                    YVEX_CLI_TABLE_PLAIN};
    session[4] = (host_status_pair){"Physical state", physical,
                                    YVEX_CLI_TABLE_PLAIN};
    session[5] = (host_status_pair){"Activation arena", arena,
                                    YVEX_CLI_TABLE_PLAIN};
    session[6] = (host_status_pair){"Workspace", workspace,
                                    YVEX_CLI_TABLE_PLAIN};
    session[7] = (host_status_pair){"Workspace peak", workspace_peak,
                                    YVEX_CLI_TABLE_PLAIN};
    process[0] = (host_status_pair){"RSS", rss, YVEX_CLI_TABLE_PLAIN};
    process[1] = (host_status_pair){"Peak RSS", peak_rss,
                                    YVEX_CLI_TABLE_PLAIN};
    process[2] = (host_status_pair){"Transient", transient,
                                    YVEX_CLI_TABLE_PLAIN};
    process[3] = (host_status_pair){"Transient peak", transient_peak,
                                    YVEX_CLI_TABLE_PLAIN};
    host_section(fp, "MODEL MEMORY", model, 8u);
    host_section(fp, "SESSION MEMORY", session, 8u);
    host_section(fp, "PROCESS MEMORY", process, 4u);
}

static void host_resource_summary_section(
    FILE *fp, const yvex_execution_resource_summary *resource)
{
    char rss[32], peak_rss[32], mapped[32], prepared[32], addressable[32];
    char explicit_device[32], state[32], workspace[32], workspace_peak[32];
    char rss_value[80], model_value[256], session_value[80], work_value[80];
    char placement_value[128];
    host_status_pair pairs[5];
    host_resource_bytes(rss, resource, YVEX_EXECUTION_RESOURCE_PROCESS_AVAILABLE,
                        resource->process_rss_current_bytes);
    host_resource_bytes(peak_rss, resource,
                        YVEX_EXECUTION_RESOURCE_PROCESS_AVAILABLE,
                        resource->process_rss_peak_bytes);
    host_resource_bytes(mapped, resource, YVEX_EXECUTION_RESOURCE_MODEL_AVAILABLE,
                        resource->model_mapped_bytes);
    host_resource_bytes(prepared, resource,
                        YVEX_EXECUTION_RESOURCE_MODEL_AVAILABLE,
                        resource->model_prepared_bytes);
    host_resource_bytes(addressable, resource,
                        YVEX_EXECUTION_RESOURCE_MODEL_AVAILABLE,
                        resource->model_device_addressable_bytes);
    host_resource_bytes(explicit_device, resource,
                        YVEX_EXECUTION_RESOURCE_MODEL_AVAILABLE,
                        resource->model_explicit_device_bytes);
    host_resource_bytes(state, resource,
                        YVEX_EXECUTION_RESOURCE_SESSION_AVAILABLE,
                        resource->session_physical_state_bytes);
    host_resource_bytes(workspace, resource,
                        YVEX_EXECUTION_RESOURCE_WORKSPACE_AVAILABLE,
                        resource->workspace_current_bytes);
    host_resource_bytes(workspace_peak, resource,
                        YVEX_EXECUTION_RESOURCE_WORKSPACE_AVAILABLE,
                        resource->workspace_peak_bytes);
    snprintf(rss_value, sizeof(rss_value), "%s current · %s peak", rss,
             peak_rss);
    snprintf(model_value, sizeof(model_value),
             "%s mapped · %s prepared · %s device-addressable · %s explicit device",
             mapped, prepared, addressable, explicit_device);
    snprintf(session_value, sizeof(session_value), "%s physical", state);
    snprintf(work_value, sizeof(work_value), "%s current · %s peak", workspace,
             workspace_peak);
    snprintf(placement_value, sizeof(placement_value), "%s · physical pages %s",
             resource_placement_name(resource->placement),
             (resource->available &
              YVEX_EXECUTION_RESOURCE_PHYSICAL_RESIDENCY_AVAILABLE)
                 ? "measured" : "not measured");
    pairs[0] = (host_status_pair){"RSS", rss_value, YVEX_CLI_TABLE_PLAIN};
    pairs[1] = (host_status_pair){"Model", model_value, YVEX_CLI_TABLE_PLAIN};
    pairs[2] = (host_status_pair){"Session", session_value,
                                  YVEX_CLI_TABLE_PLAIN};
    pairs[3] = (host_status_pair){"Workspace", work_value,
                                  YVEX_CLI_TABLE_PLAIN};
    pairs[4] = (host_status_pair){"Placement", placement_value,
                                  YVEX_CLI_TABLE_DIM};
    host_section(fp, "MEMORY", pairs, 5u);
}

static void host_status_render_human(FILE *fp,
                                     const yvex_server_summary *status)
{
    char uptime[32], workers[24], queue[48], loaded[24], known[24], capacity[24];
    char active_sessions[24], total_sessions[24];
    char openai[64];
    int ready = status->status == YVEX_SERVER_STATUS_READY && status->host_ready;
    host_status_pair host_pairs[4], model_pairs[3], session_pairs[2];
    host_status_pair endpoint_pairs[2];
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
    endpoint_pairs[0] = (host_status_pair){"Native", status->socket_path,
                                           YVEX_CLI_TABLE_DIM};
    endpoint_pairs[1] = (host_status_pair){"OpenAI", openai,
        status->openai_listener_ready ? YVEX_CLI_TABLE_SUCCESS
                                      : YVEX_CLI_TABLE_WARNING};
    host_section(fp, "HOST", host_pairs, 4u);
    host_section(fp, "MODELS", model_pairs, 3u);
    host_section(fp, "SESSIONS", session_pairs, 2u);
    host_resource_summary_section(fp, &status->metrics.resources);
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
        "\"resident_host_bytes\":%llu,\"resident_device_bytes\":%llu,"
        "\"resources\":",
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
    resource_json(fp, &status->metrics.resources);
    yvex_cli_out_fputs("}\n", fp);
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
    if (!fp || !status) return;
    if (json) {
        yvex_cli_out_writef(
            fp,
            "{\"schema\":\"yvex.host.memory.v2\",\"rss_bytes\":%llu,"
            "\"peak_rss_bytes\":%llu,\"mapped_artifact_bytes\":%llu,"
            "\"resident_host_bytes\":%llu,\"resident_device_bytes\":%llu,"
            "\"resources\":",
            status->metrics.current_rss_bytes, status->metrics.peak_rss_bytes,
            status->metrics.mapped_artifact_bytes,
            status->metrics.resident_host_bytes,
            status->metrics.resident_device_bytes);
        resource_json(fp, &status->metrics.resources);
        yvex_cli_out_fputs("}\n", fp);
        return;
    }
    host_resource_sections(fp, &status->metrics.resources);
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
