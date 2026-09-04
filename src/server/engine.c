/* Own loaded model-engine generations independently from the persistent server host. */
#include "src/server/private.h"

#include <assert.h>
#include <limits.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include <yvex/internal/core.h>
#include <yvex/internal/engine_scheduler.h>
#include <yvex/internal/tokenizer.h>

#define ENGINE_INTERACTIVE_PREFILL_CHUNK 64u
#define ENGINE_BATCHED_PREFILL_FLOOR 4u

typedef struct {
    yvex_server_engine_state state;
    unsigned long long generation, active_work;
    yvex_server_engine_options options;
    char alias[YVEX_SERVER_MODEL_ALIAS_CAP];
    char artifact_path[YVEX_PATH_CAP];
    char runtime_binding_path[YVEX_PATH_CAP];
    char target_id[128];
    yvex_model_engine *model;
    server_request_queue *request_queue;
    server_session_registry *sessions;
    server_media_registry *media;
    yvex_server_engine_summary summary;
    unsigned long long artifact_bytes, mapped_package_bytes, prepared_bytes;
    unsigned long long model_resident_host_bytes;
    unsigned long long model_resident_device_bytes;
    yvex_runtime_residency_summary model_residency;
    unsigned long long model_component_count;
    unsigned long long runnable_sequences, logical_runnable_capacity;
    unsigned long long physical_sequence_width;
    int compatible_operation_batching, telemetry_opened;
} server_engine;

typedef struct {
    server_engine_manager *manager;
    server_engine *engine;
    yvex_runtime_lifecycle_phase phase;
    unsigned long long phase_started_ns, last_emit_ns, last_completed;
    int initialized;
} engine_load_progress;

struct server_engine_manager {
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    server_engine *engines;
    unsigned long long capacity, next_generation;
    unsigned long long request_capacity, request_workers;
    server_request_queue_execute request_execute;
    void *request_context;
    server_telemetry *telemetry;
    int mutex_ready, condition_ready, closing;
};

static int engine_refuse(yvex_error *err, yvex_status status,
                         const char *reason)
{
    yvex_error_set(err, status, "server.engine", reason);
    return status;
}

static int alias_valid(const char *alias)
{
    size_t index, count;
    if (!alias || !alias[0]) return 0;
    count = strlen(alias);
    if (count >= YVEX_SERVER_MODEL_ALIAS_CAP) return 0;
    for (index = 0u; index < count; ++index) {
        unsigned char byte = (unsigned char)alias[index];
        if (!((byte >= 'a' && byte <= 'z') ||
              (byte >= 'A' && byte <= 'Z') ||
              (byte >= '0' && byte <= '9') || byte == '-' || byte == '_' ||
              byte == '.'))
            return 0;
    }
    return 1;
}

static void options_rebind(server_engine *engine)
{
    engine->options.alias = engine->alias;
    engine->options.target_id = engine->target_id;
    engine->options.artifact_path = engine->artifact_path[0]
                                        ? engine->artifact_path : NULL;
    engine->options.runtime_binding_path = engine->runtime_binding_path[0]
                                               ? engine->runtime_binding_path : NULL;
}

static unsigned long long adaptive_prefill_chunk(
    unsigned long long context_capacity,
    unsigned long long concurrent_sequences)
{
    unsigned long long selected = concurrent_sequences > 1ull
                                      ? concurrent_sequences
                                      : ENGINE_INTERACTIVE_PREFILL_CHUNK;
    if (concurrent_sequences > 1ull && selected < ENGINE_BATCHED_PREFILL_FLOOR)
        selected = ENGINE_BATCHED_PREFILL_FLOOR;
    return selected < context_capacity ? selected : context_capacity;
}

static yvex_execution_work_unit engine_load_work_unit(
    yvex_runtime_lifecycle_phase phase)
{
    if (phase == YVEX_RUNTIME_LIFECYCLE_ARTIFACT_HASH)
        return YVEX_EXECUTION_WORK_BYTES;
    if (phase == YVEX_RUNTIME_LIFECYCLE_RESIDENCY)
        return YVEX_EXECUTION_WORK_TENSORS;
    return YVEX_EXECUTION_WORK_OPERATIONS;
}

static int engine_load_denominator_known(yvex_runtime_lifecycle_phase phase)
{
    return phase == YVEX_RUNTIME_LIFECYCLE_ARTIFACT_HASH ||
           phase == YVEX_RUNTIME_LIFECYCLE_RESIDENCY;
}

static yvex_runtime_trace_policy engine_trace_policy(
    yvex_server_trace_level level)
{
    if (level == YVEX_SERVER_TRACE_FULL) return YVEX_RUNTIME_TRACE_FULL;
    if (level >= YVEX_SERVER_TRACE_STAGES) return YVEX_RUNTIME_TRACE_STAGES;
    return YVEX_RUNTIME_TRACE_SUMMARY;
}

static int engine_load_progress_emit(engine_load_progress *progress,
                                     yvex_runtime_lifecycle_phase phase,
                                     unsigned long long completed,
                                     unsigned long long total,
                                     unsigned long long now)
{
    static const char *const names[YVEX_RUNTIME_LIFECYCLE_COUNT] = {
        "artifact-open", "artifact-verification", "artifact-admission",
        "binding-validation", "materialization", "model-seal", "residency",
        "backend-open", "workspace-prepare", "graph-warmup", "graph-capture",
        "graph-instantiate", "execution", "publication", "cleanup"};
    yvex_execution_measurement measurement = {0};
    server_event_scope scope;
    yvex_error ignored;
    unsigned long long elapsed = now > progress->phase_started_ns
                                     ? now - progress->phase_started_ns : 0ull;
    double rate = elapsed && completed
                      ? (double)completed * 1000000000.0 / (double)elapsed
                      : 0.0;
    measurement.schema_version = YVEX_EXECUTION_MEASUREMENT_SCHEMA_V1;
    measurement.scope = YVEX_EXECUTION_SCOPE_MODEL_LIFECYCLE;
    measurement.clock = YVEX_EXECUTION_CLOCK_HOST_WALL;
    measurement.composition = YVEX_EXECUTION_COMPOSITION_NESTED;
    measurement.work_unit = engine_load_work_unit(phase);
    measurement.completed_units = completed;
    if (total && engine_load_denominator_known(phase)) {
        measurement.available |=
            YVEX_EXECUTION_MEASUREMENT_DENOMINATOR_AVAILABLE;
        measurement.total_units = total;
    }
    if (elapsed) {
        measurement.available |=
            YVEX_EXECUTION_MEASUREMENT_DURATION_AVAILABLE;
        measurement.duration_ns = elapsed;
    }
    if (elapsed && completed) {
        measurement.available |=
            YVEX_EXECUTION_MEASUREMENT_CUMULATIVE_RATE_AVAILABLE;
        measurement.cumulative_rate = rate;
    }
    server_event_scope_from_engine(&scope, &progress->engine->summary);
    yvex_error_clear(&ignored);
    (void)yvex_server_telemetry_emit_provider(
        progress->manager->telemetry, &scope,
        YVEX_SERVER_EVENT_ENGINE_LOAD_PROGRESS, YVEX_SERVER_SEVERITY_INFO,
        NULL, NULL, NULL, names[phase], completed,
        engine_load_denominator_known(phase) ? total : 0ull, phase,
        (double)elapsed / 1000000000.0, rate, NULL, NULL, &measurement, NULL,
        &ignored);
    return 1;
}

static int engine_load_progress_observe(
    void *opaque, yvex_runtime_lifecycle_phase phase,
    unsigned long long completed, unsigned long long total)
{
    engine_load_progress *progress = opaque;
    unsigned long long now = yvex_core_monotonic_ns();
    int phase_changed;
    int emit;
    if (!progress || phase >= YVEX_RUNTIME_LIFECYCLE_COUNT ||
        (total && completed > total))
        return 1;
    phase_changed = !progress->initialized || progress->phase != phase;
    if (phase_changed) {
        progress->phase = phase;
        progress->phase_started_ns = now;
        progress->last_completed = 0ull;
        progress->initialized = 1;
    }
    emit = phase_changed || (total && completed == total) ||
           now - progress->last_emit_ns >= 1000000000ull ||
           (completed >= progress->last_completed &&
            completed - progress->last_completed >= 64ull * 1024ull * 1024ull);
    if (!emit) return 1;
    progress->last_emit_ns = now;
    progress->last_completed = completed;
    return engine_load_progress_emit(progress, phase, completed, total, now);
}

static int options_admit(server_engine *engine,
                         const yvex_server_engine_options *options,
                         const yvex_server_media_options *media,
                         yvex_error *err)
{
    int text;
    if (!engine || !options ||
        options->schema_version != YVEX_SERVER_ENGINE_SCHEMA_CURRENT ||
        !alias_valid(options->alias) || !options->target_id ||
        strlen(options->target_id) >= sizeof(engine->target_id) ||
        (options->backend != YVEX_BACKEND_KIND_CPU &&
         options->backend != YVEX_BACKEND_KIND_CUDA) ||
        options->engine_kind == YVEX_SERVER_ENGINE_NONE ||
        options->engine_kind > YVEX_SERVER_ENGINE_MEDIA ||
        options->execution_strategy > YVEX_SERVER_EXECUTION_SPECULATIVE ||
        options->trace_level > YVEX_SERVER_TRACE_FULL ||
        !options->maximum_output_bytes || !options->maximum_sessions ||
        !options->concurrent_sequences ||
        options->concurrent_sequences > options->maximum_sessions)
        return engine_refuse(err, YVEX_ERR_INVALID_ARG,
                             "complete bounded engine options are required");
    text = options->engine_kind == YVEX_SERVER_ENGINE_TEXT;
    if ((text && (!options->artifact_path || !options->runtime_binding_path ||
                  !options->context_capacity || !options->maximum_new_tokens || media ||
                  options->execution_strategy ==
                      YVEX_SERVER_EXECUTION_NOT_APPLICABLE)) ||
        (!text && (!media || options->artifact_path ||
                   options->runtime_binding_path || options->context_capacity ||
                   options->prefill_chunk_tokens || options->maximum_new_tokens ||
                   options->execution_strategy !=
                       YVEX_SERVER_EXECUTION_NOT_APPLICABLE)))
        return engine_refuse(err, YVEX_ERR_INVALID_ARG,
                             "engine package and execution kind disagree");
    if ((options->artifact_path &&
         strlen(options->artifact_path) >= sizeof(engine->artifact_path)) ||
        (options->runtime_binding_path &&
         strlen(options->runtime_binding_path) >=
             sizeof(engine->runtime_binding_path)))
        return engine_refuse(err, YVEX_ERR_BOUNDS,
                             "engine package path exceeds its bound");
    memset(engine, 0, sizeof(*engine));
    engine->options = *options;
    if (text && !engine->options.prefill_chunk_tokens)
        engine->options.prefill_chunk_tokens = adaptive_prefill_chunk(
            engine->options.context_capacity,
            engine->options.concurrent_sequences);
    yvex_core_text_copy(engine->alias, sizeof(engine->alias), options->alias);
    yvex_core_text_copy(engine->target_id, sizeof(engine->target_id),
                        options->target_id);
    if (options->artifact_path)
        yvex_core_text_copy(engine->artifact_path,
                            sizeof(engine->artifact_path),
                            options->artifact_path);
    if (options->runtime_binding_path)
        yvex_core_text_copy(engine->runtime_binding_path,
                            sizeof(engine->runtime_binding_path),
                            options->runtime_binding_path);
    options_rebind(engine);
    return YVEX_OK;
}

static void generation_options(const server_engine *engine,
                               yvex_runtime_generation_options *options)
{
    memset(options, 0, sizeof(*options));
    options->schema_version = YVEX_RUNTIME_GENERATION_SCHEMA_V6;
    options->backend = engine->options.backend;
    options->mode = engine->options.execution_strategy ==
                            YVEX_SERVER_EXECUTION_SPECULATIVE
                        ? YVEX_GENERATION_MODE_SPECULATIVE
                        : YVEX_GENERATION_MODE_TARGET_ONLY;
    options->workload_kind = engine->options.concurrent_sequences > 1ull
                                 ? YVEX_EXECUTION_WORKLOAD_BALANCED_SERVING
                                 : YVEX_EXECUTION_WORKLOAD_INTERACTIVE_LATENCY;
    options->context_capacity = engine->options.context_capacity;
    options->prefill_chunk_tokens = engine->options.prefill_chunk_tokens;
    options->maximum_new_tokens = engine->options.maximum_new_tokens;
    options->maximum_output_bytes = engine->options.maximum_output_bytes;
    options->maximum_host_bytes = engine->options.maximum_host_bytes;
    options->maximum_device_bytes = engine->options.maximum_device_bytes;
    options->concurrent_sequences = engine->options.concurrent_sequences;
    options->runnable_sequences = engine->runnable_sequences;
    options->compatible_operation_batching =
        engine->compatible_operation_batching;
    options->trace_policy = engine_trace_policy(engine->options.trace_level);
    options->evidence_profile = YVEX_EXECUTION_EVIDENCE_PRODUCTION;
    options->sampling_policy.schema_version = YVEX_RUNTIME_SAMPLING_SCHEMA_V1;
    options->sampling_policy.strategy = YVEX_SAMPLING_STRATEGY_STOCHASTIC;
    options->sampling_policy.temperature = 1.0;
    options->sampling_policy.top_p = 1.0;
    options->sampling_policy.typical_p = 1.0;
    options->sampling_policy.seed_present = 1;
    options->sampling_policy.seed = engine->options.sampling_seed;
}

static int engine_request_queue_open(server_engine_manager *manager,
                                     server_engine *engine, yvex_error *err)
{
    unsigned long long workers = manager->request_workers;
    if (workers > engine->options.maximum_sessions)
        workers = engine->options.maximum_sessions;
    engine->logical_runnable_capacity = workers;
    engine->runnable_sequences = workers > engine->options.concurrent_sequences
                                     ? workers
                                     : engine->options.concurrent_sequences;
    int rc = yvex_server_request_queue_open(
        &engine->request_queue, manager->request_capacity, workers,
        manager->request_execute, NULL, manager->request_context, err);
    if (rc == YVEX_OK)
        rc = yvex_server_request_queue_start(engine->request_queue, err);
    return rc;
}

static int execution_probe(server_engine *engine,
                           yvex_runtime_generation_context_summary *capacity,
                           char specialization_identity[YVEX_SHA256_HEX_CAP],
                           engine_load_progress *progress,
                           yvex_error *err)
{
    yvex_runtime_execution_session *session = NULL;
    yvex_runtime_generation_context *generation = NULL;
    yvex_runtime_session_open_request request = {0};
    yvex_runtime_generation_options options;
    yvex_runtime_session_summary session_summary = {0};
    yvex_model_engine_failure failure = {0};
    yvex_error primary = {0}, cleanup = {0};
    unsigned long long width = 1ull;
    int rc, cleanup_rc;
    request.backend = engine->options.backend;
    request.maximum_host_bytes = engine->options.maximum_host_bytes;
    request.maximum_device_bytes = engine->options.maximum_device_bytes;
    rc = yvex_model_engine_scheduler_maximum_width_copy(engine->model, &width, err);
    if (rc == YVEX_OK)
        engine->compatible_operation_batching =
            engine->options.concurrent_sequences > 1ull && width >= 2ull;
    if (rc == YVEX_OK)
        engine->physical_sequence_width =
            engine->compatible_operation_batching
                ? (width < engine->options.concurrent_sequences
                       ? width : engine->options.concurrent_sequences)
                : 1ull;
    if (rc == YVEX_OK) {
        (void)engine_load_progress_observe(
            progress, YVEX_RUNTIME_LIFECYCLE_BACKEND_OPEN, 0ull, 1ull);
        rc = yvex_runtime_session_open(&session, engine->model, &request,
                                       &failure, err);
        if (rc == YVEX_OK)
            (void)engine_load_progress_observe(
                progress, YVEX_RUNTIME_LIFECYCLE_BACKEND_OPEN, 1ull, 1ull);
    }
    generation_options(engine, &options);
    if (rc == YVEX_OK) {
        (void)engine_load_progress_observe(
            progress, YVEX_RUNTIME_LIFECYCLE_WORKSPACE_PREPARE, 0ull, 1ull);
        rc = yvex_runtime_generation_context_open(
            &generation, engine->model, session, &options, err);
        if (rc == YVEX_OK)
            (void)engine_load_progress_observe(
                progress, YVEX_RUNTIME_LIFECYCLE_WORKSPACE_PREPARE, 1ull, 1ull);
    }
    if (rc == YVEX_OK)
        rc = yvex_runtime_generation_context_summary_copy(generation, capacity, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_session_summary_copy(session, &session_summary, err);
    if (rc == YVEX_OK &&
        !yvex_sha256_hex_valid(session_summary.engine_specialization_identity)) {
        yvex_error_set(err, YVEX_ERR_STATE, "server.engine.probe",
                       "engine specialization identity is unavailable");
        rc = YVEX_ERR_STATE;
    }
    if (rc == YVEX_OK)
        yvex_runtime_identity_copy(specialization_identity,
                                   session_summary.engine_specialization_identity);
    if (err) primary = *err;
    cleanup_rc = yvex_runtime_generation_context_close(&generation, &cleanup);
    if (cleanup_rc == YVEX_OK)
        cleanup_rc = yvex_runtime_session_close(&session, &cleanup);
    if (cleanup_rc != YVEX_OK && rc == YVEX_OK) {
        rc = cleanup_rc;
        primary = cleanup;
    }
    if (err) *err = primary;
    return rc;
}

static int text_engine_open(server_engine_manager *manager,
                            server_engine *engine, yvex_error *err)
{
    const yvex_graph_execution_binding *execution;
    yvex_runtime_binding_summary binding_summary = {0};
    yvex_runtime_binding_failure binding_failure = {0};
    yvex_runtime_binding *binding = NULL;
    yvex_model_engine_open_request request = {0};
    yvex_model_engine_failure failure = {0};
    yvex_model_engine_summary model = {0};
    yvex_runtime_generation_context_summary capacity = {0};
    char specialization_identity[YVEX_SHA256_HEX_CAP] = {0};
    yvex_runtime_generation_options startup;
    server_event_scope event_scope = {0};
    const yvex_model_engine_view *view;
    yvex_paths paths;
    yvex_error path_error;
    engine_load_progress progress = {0};
    int rc;
    generation_options(engine, &startup);
    request.artifact_path = engine->artifact_path;
    request.runtime_binding_path = engine->runtime_binding_path;
    request.target_id = engine->target_id;
    execution = yvex_graph_execution_find(0ull, 0ull, engine->target_id);
    if (execution) {
        request.expected_family_adapter_id = execution->adapter_id;
        request.expected_family_adapter_version = execution->adapter_version;
        request.expected_logical_transform_identity =
            execution->logical_transform_identity;
    } else {
        /* Portable bindings have no family compiler in this process. Reopen
         * once to pin their sealed generic execution identity, then runtime
         * admission reopens and compares it before any artifact/GPU work. */
        rc = yvex_runtime_binding_open(
            &binding, engine->runtime_binding_path, &binding_summary, NULL,
            &binding_failure, err);
        if (rc != YVEX_OK) return rc;
        request.expected_family_adapter_id = binding_summary.family_adapter_id;
        request.expected_family_adapter_version = binding_summary.family_adapter_version;
        request.expected_logical_transform_identity =
            binding_summary.logical_transform_identity;
        yvex_runtime_binding_close(binding);
    }
    request.startup_generation = &startup;
    request.residency_backend = engine->options.backend;
    request.maximum_host_bytes = engine->options.maximum_host_bytes;
    request.maximum_device_bytes = engine->options.maximum_device_bytes;
    progress.manager = manager;
    progress.engine = engine;
    request.progress = engine_load_progress_observe;
    request.progress_context = &progress;
    yvex_error_clear(&path_error);
    if (yvex_paths_default(&paths, &path_error) == YVEX_OK)
        request.artifact_reopen_cache_root = paths.cache_dir;
    rc = yvex_model_engine_open(&engine->model, &request, &failure, err);
    if (rc == YVEX_OK)
        rc = yvex_model_engine_summary_copy(engine->model, &model, err);
    view = rc == YVEX_OK ? yvex_model_engine_view_get(engine->model) : NULL;
    if (rc == YVEX_OK && !view)
        rc = engine_refuse(err, YVEX_ERR_STATE,
                           "runtime model view is unavailable");
    if (rc == YVEX_OK && view->residency)
        rc = yvex_runtime_residency_snapshot(
            view->residency, &engine->model_residency, NULL, NULL, err);
    if (rc == YVEX_OK)
        rc = execution_probe(engine, &capacity, specialization_identity,
                             &progress, err);
    if (rc == YVEX_OK) {
        event_scope.engine_kind = engine->options.engine_kind;
        event_scope.execution_strategy = engine->options.execution_strategy;
        yvex_runtime_identity_copy(event_scope.runtime_model_identity,
                                   model.runtime_model_identity);
        yvex_runtime_identity_copy(event_scope.artifact_identity,
                                   model.artifact_identity);
        yvex_runtime_identity_copy(event_scope.specialization_identity,
                                   specialization_identity);
        rc = yvex_server_sessions_open(
            &engine->sessions, engine->model, &engine->options, engine->generation,
            engine->runnable_sequences, engine->compatible_operation_batching,
            &event_scope, manager->telemetry, err);
    }
    if (rc != YVEX_OK) return rc;
    engine->summary.context_capacity = engine->options.context_capacity;
    engine->summary.prefill_chunk_tokens = engine->options.prefill_chunk_tokens;
    engine->summary.maximum_new_tokens = engine->options.maximum_new_tokens;
    engine->summary.maximum_output_bytes = engine->options.maximum_output_bytes;
    engine->summary.maximum_sessions = engine->options.maximum_sessions;
    engine->summary.concurrent_sequences = engine->options.concurrent_sequences;
    engine->artifact_bytes = model.artifact_bytes;
    engine->mapped_package_bytes = model.mapped_package_bytes;
    engine->prepared_bytes = model.prepared_bytes;
    engine->model_resident_host_bytes = model.resident_host_bytes;
    engine->model_resident_device_bytes = model.resident_device_bytes;
    engine->model_component_count = 1ull;
    engine->summary.mapped_package_bytes = engine->mapped_package_bytes;
    engine->summary.prepared_bytes = engine->prepared_bytes;
    engine->summary.resident_host_bytes = engine->model_resident_host_bytes;
    engine->summary.resident_device_bytes = engine->model_resident_device_bytes;
    yvex_runtime_identity_copy(engine->summary.runtime_model_identity,
                               model.runtime_model_identity);
    yvex_runtime_identity_copy(engine->summary.runtime_binding_identity,
                               model.runtime_binding_identity);
    yvex_runtime_identity_copy(engine->summary.artifact_identity,
                               model.artifact_identity);
    yvex_runtime_identity_copy(engine->summary.specialization_identity,
                               specialization_identity);
    yvex_runtime_identity_copy(engine->summary.capacity_plan_identity,
                               capacity.capacity_plan_identity);
    {
        const yvex_tokenizer_plan_summary *tokenizer =
            yvex_tokenizer_plan_summary_get(view->tokenizer);
        engine->summary.explicit_reasoning_channel_supported =
            tokenizer && tokenizer->explicit_reasoning_supported;
    }
    yvex_server_telemetry_model_opened(
        manager->telemetry, model.mapped_package_bytes,
        model.resident_host_bytes, model.resident_device_bytes, 0ull);
    engine->telemetry_opened = 1;
    return YVEX_OK;
}

static int media_engine_open(server_engine_manager *manager,
                             server_engine *engine,
                             const yvex_server_media_options *media,
                             yvex_error *err)
{
    yvex_runtime_media_model_summary model = {0};
    server_media_summary summary = {0};
    int rc = yvex_server_media_registry_open(
        &engine->media, media, manager->telemetry, err);
    if (rc == YVEX_OK)
        rc = yvex_server_media_registry_summary(engine->media, &summary, err);
    if (rc == YVEX_OK)
        rc = yvex_server_media_registry_start(engine->media, &model, err);
    if (rc != YVEX_OK) return rc;
    yvex_core_text_copy(engine->summary.runtime_model_identity,
                        sizeof(engine->summary.runtime_model_identity),
                        model.model_identity);
    yvex_core_text_copy(engine->summary.specialization_identity,
                        sizeof(engine->summary.specialization_identity),
                        summary.specialization_identity);
    engine->artifact_bytes = model.artifact_bytes;
    engine->mapped_package_bytes = model.mapped_package_bytes;
    engine->prepared_bytes = model.prepared_bytes;
    engine->model_resident_host_bytes = model.resident_host_bytes;
    engine->model_resident_device_bytes = model.resident_device_bytes;
    engine->model_component_count = model.component_count;
    yvex_server_telemetry_media_model_opened(manager->telemetry,
                                             model.component_count);
    engine->telemetry_opened = 1;
    return YVEX_OK;
}

static int optional_summary_identity_valid(const char *identity)
{
    return identity && memchr(identity, '\0', YVEX_SHA256_HEX_CAP) &&
           (!identity[0] || yvex_sha256_hex_valid(identity));
}

int yvex_server_engine_summary_valid(const yvex_server_engine_summary *engine)
{
    if (!engine || engine->schema_version != YVEX_SERVER_ENGINE_SCHEMA_CURRENT ||
        engine->state > YVEX_SERVER_ENGINE_FAILED ||
        engine->backend > YVEX_BACKEND_KIND_CUDA ||
        engine->engine_kind == YVEX_SERVER_ENGINE_NONE ||
        engine->engine_kind > YVEX_SERVER_ENGINE_MEDIA ||
        engine->execution_strategy > YVEX_SERVER_EXECUTION_SPECULATIVE ||
        (engine->engine_kind == YVEX_SERVER_ENGINE_TEXT &&
         engine->execution_strategy == YVEX_SERVER_EXECUTION_NOT_APPLICABLE) ||
        (engine->engine_kind == YVEX_SERVER_ENGINE_MEDIA &&
         engine->execution_strategy != YVEX_SERVER_EXECUTION_NOT_APPLICABLE) ||
        !memchr(engine->alias, '\0', sizeof(engine->alias)) ||
        !memchr(engine->target_id, '\0', sizeof(engine->target_id)) ||
        !alias_valid(engine->alias) || !engine->target_id[0] ||
        !engine->generation || !engine->maximum_sessions ||
        !engine->concurrent_sequences ||
        engine->concurrent_sequences > engine->maximum_sessions ||
        (engine->execution_ready !=
         (engine->state == YVEX_SERVER_ENGINE_LOADED)) ||
        (engine->explicit_reasoning_channel_supported != 0 &&
         engine->explicit_reasoning_channel_supported != 1) ||
        (engine->continuous_batching_ready != 0 &&
         engine->continuous_batching_ready != 1) ||
        !yvex_server_execution_capacity_valid(&engine->capacity) ||
        !yvex_server_execution_resource_valid(&engine->resources) ||
        engine->continuous_batching_ready !=
            engine->capacity.continuous_batching_ready ||
        !optional_summary_identity_valid(engine->runtime_model_identity) ||
        !optional_summary_identity_valid(engine->runtime_binding_identity) ||
        !optional_summary_identity_valid(engine->artifact_identity) ||
        !optional_summary_identity_valid(engine->specialization_identity) ||
        !optional_summary_identity_valid(engine->capacity_plan_identity))
        return 0;
    return engine->state != YVEX_SERVER_ENGINE_LOADED ||
           (yvex_sha256_hex_valid(engine->runtime_model_identity) &&
            yvex_sha256_hex_valid(engine->specialization_identity));
}

static void summary_base(server_engine *engine)
{
    yvex_execution_capacity_summary *capacity = &engine->summary.capacity;
    engine->summary.schema_version = YVEX_SERVER_ENGINE_SCHEMA_CURRENT;
    engine->summary.state = engine->state;
    engine->summary.backend = engine->options.backend;
    engine->summary.engine_kind = engine->options.engine_kind;
    engine->summary.execution_strategy = engine->options.execution_strategy;
    engine->summary.generation = engine->generation;
    engine->summary.active_work = engine->active_work;
    engine->summary.context_capacity = engine->options.context_capacity;
    engine->summary.prefill_chunk_tokens = engine->options.prefill_chunk_tokens;
    engine->summary.maximum_new_tokens = engine->options.maximum_new_tokens;
    engine->summary.maximum_output_bytes = engine->options.maximum_output_bytes;
    engine->summary.maximum_sessions = engine->options.maximum_sessions;
    engine->summary.concurrent_sequences = engine->options.concurrent_sequences;
    engine->summary.execution_ready = engine->state == YVEX_SERVER_ENGINE_LOADED;
    engine->summary.continuous_batching_ready = 0;
    memset(capacity, 0, sizeof(*capacity));
    capacity->schema_version = YVEX_EXECUTION_CAPACITY_SCHEMA_V1;
    capacity->session_capacity = engine->options.maximum_sessions;
    capacity->runnable_work_capacity = engine->logical_runnable_capacity
                                           ? engine->logical_runnable_capacity
                                           : 1ull;
    if (capacity->runnable_work_capacity > capacity->session_capacity)
        capacity->runnable_work_capacity = capacity->session_capacity;
    capacity->physical_sequence_width = engine->physical_sequence_width
                                            ? engine->physical_sequence_width : 1ull;
    capacity->cooperative_scheduling_ready =
        capacity->runnable_work_capacity > 1ull;
    capacity->compatible_operation_batching_ready =
        engine->compatible_operation_batching;
    capacity->continuous_batching_ready = 0;
    if (!engine->summary.resources.schema_version)
        engine->summary.resources.schema_version =
            YVEX_EXECUTION_RESOURCE_SCHEMA_V1;
    yvex_core_text_copy(engine->summary.alias,
                        sizeof(engine->summary.alias), engine->alias);
    yvex_core_text_copy(engine->summary.target_id,
                        sizeof(engine->summary.target_id), engine->target_id);
    assert(yvex_server_engine_summary_valid(&engine->summary));
}

static void engine_lifecycle_event(
    server_engine_manager *manager, const yvex_server_engine_summary *summary,
    yvex_server_event_kind kind, yvex_server_event_severity severity,
    unsigned long long status)
{
    server_event_scope scope;
    yvex_error ignored;
    server_event_scope_from_engine(&scope, summary);
    yvex_error_clear(&ignored);
    (void)yvex_server_telemetry_emit(
        manager->telemetry, &scope, kind, severity, NULL, NULL, NULL,
        summary->target_id, summary->generation, status, summary->backend,
        0.0, 0.0, &ignored);
}

static int summary_resources(server_engine *engine, yvex_error *err)
{
    yvex_execution_resource_summary session = {0};
    server_media_summary media = {0};
    yvex_execution_resource_summary *resources = &engine->summary.resources;
    int owns_resources = engine->model || engine->sessions || engine->media;
    memset(resources, 0, sizeof(*resources));
    resources->schema_version = YVEX_EXECUTION_RESOURCE_SCHEMA_V1;
    engine->summary.mapped_package_bytes =
        owns_resources ? engine->mapped_package_bytes : 0ull;
    engine->summary.prepared_bytes =
        owns_resources ? engine->prepared_bytes : 0ull;
    engine->summary.resident_host_bytes =
        owns_resources ? engine->model_resident_host_bytes : 0ull;
    engine->summary.resident_device_bytes =
        owns_resources ? engine->model_resident_device_bytes : 0ull;
    if (!owns_resources) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    resources->available = YVEX_EXECUTION_RESOURCE_MODEL_AVAILABLE;
    resources->component_count = engine->model_component_count;
    resources->model_artifact_bytes = engine->artifact_bytes;
    resources->model_mapped_bytes = engine->mapped_package_bytes;
    resources->model_prepared_bytes = engine->prepared_bytes;
    resources->model_device_addressable_bytes =
        engine->model_residency.cuda_addressable_bytes;
    resources->logical_upload_bytes = engine->model_residency.cuda_upload_bytes;
    if (engine->media) {
        resources->placement = YVEX_EXECUTION_PLACEMENT_COMPOSITE;
        resources->model_explicit_host_bytes =
            engine->model_resident_host_bytes;
        resources->model_explicit_device_bytes =
            engine->model_resident_device_bytes;
    } else if (engine->model_residency.placement ==
               YVEX_RUNTIME_WEIGHT_PLACEMENT_CUDA_MANAGED) {
        resources->placement = YVEX_EXECUTION_PLACEMENT_MANAGED_UNIFIED;
        resources->available |= YVEX_EXECUTION_RESOURCE_UNIFIED_MEMORY;
    } else if (engine->model_residency.placement ==
                   YVEX_RUNTIME_WEIGHT_PLACEMENT_ARTIFACT_MAPPED &&
               engine->model_residency.cuda_addressable_bytes) {
        resources->placement =
            YVEX_EXECUTION_PLACEMENT_ARTIFACT_MAPPED_DEVICE_ADDRESSABLE;
        if (engine->model_residency.cuda_pageable_map_count)
            resources->available |= YVEX_EXECUTION_RESOURCE_UNIFIED_MEMORY;
    } else if (engine->model_residency.placement ==
               YVEX_RUNTIME_WEIGHT_PLACEMENT_ARTIFACT_MAPPED) {
        resources->placement = YVEX_EXECUTION_PLACEMENT_ARTIFACT_MAPPED;
    } else {
        resources->placement = YVEX_EXECUTION_PLACEMENT_EXPLICIT_HOST;
        resources->model_explicit_host_bytes =
            engine->model_resident_host_bytes;
        resources->available |=
            YVEX_EXECUTION_RESOURCE_PHYSICAL_RESIDENCY_AVAILABLE;
    }
    if (engine->sessions &&
        yvex_server_sessions_resource_summary(engine->sessions, &session,
                                              err) != YVEX_OK)
        return yvex_error_code(err);
    if (engine->media &&
        yvex_server_media_registry_summary(engine->media, &media, err) !=
            YVEX_OK)
        return yvex_error_code(err);
#define MERGE(field)                                                            \
    if (!yvex_core_u64_add(resources->field, session.field, &resources->field)) \
        return engine_refuse(err, YVEX_ERR_BOUNDS,                              \
                             "engine resource total overflowed")
    resources->available |= session.available;
    MERGE(session_attention_allocated_bytes);
    MERGE(session_attention_resident_bytes);
    MERGE(session_attention_virtual_bytes);
    MERGE(session_attention_page_table_bytes);
    MERGE(session_recurrent_state_bytes);
    MERGE(session_convolution_state_bytes);
    MERGE(session_candidate_state_bytes);
    MERGE(session_physical_state_bytes);
    MERGE(workspace_current_bytes);
    MERGE(workspace_peak_bytes);
#undef MERGE
    if (media.execution_resources_observed) {
        resources->available |= YVEX_EXECUTION_RESOURCE_WORKSPACE_AVAILABLE |
                                YVEX_EXECUTION_RESOURCE_TRANSIENT_AVAILABLE;
        if (media.execution_workspace_peak_bytes >
            resources->workspace_peak_bytes)
            resources->workspace_peak_bytes =
                media.execution_workspace_peak_bytes;
        if (media.execution_transient_peak_bytes >
            resources->transient_peak_bytes)
            resources->transient_peak_bytes =
                media.execution_transient_peak_bytes;
    }
    if (media.activation_arena_observed) {
        resources->available |= YVEX_EXECUTION_RESOURCE_ARENA_AVAILABLE;
        if (media.activation_arena_peak_bytes >
            resources->activation_arena_peak_bytes)
            resources->activation_arena_peak_bytes =
                media.activation_arena_peak_bytes;
    }
    engine->summary.resident_host_bytes =
        resources->model_explicit_host_bytes;
    engine->summary.resident_device_bytes =
        resources->model_explicit_device_bytes;
    yvex_error_clear(err);
    return YVEX_OK;
}

static void engine_cancel(server_engine *engine)
{
    if (engine->media)
        yvex_server_media_registry_cancel_all(engine->media);
    else if (engine->sessions)
        yvex_server_sessions_cancel_all(engine->sessions);
}

static int engine_close(server_engine_manager *manager, server_engine *engine,
                        yvex_error *err)
{
    yvex_error primary = {0}, cleanup = {0};
    int rc = YVEX_OK, cleanup_rc;
    if (engine->request_queue) {
        cleanup_rc = yvex_server_request_queue_finish(engine->request_queue, &cleanup);
        if (cleanup_rc != YVEX_OK) {
            rc = cleanup_rc;
            primary = cleanup;
        }
    }
    if (engine->sessions) {
        cleanup_rc = yvex_server_sessions_close(&engine->sessions, &cleanup);
        if (cleanup_rc != YVEX_OK && rc == YVEX_OK) {
            rc = cleanup_rc;
            primary = cleanup;
        }
    }
    yvex_server_media_registry_close(&engine->media);
    if (!engine->sessions && !engine->media)
        engine->summary.session_count = 0ull;
    yvex_model_engine_close(&engine->model);
    yvex_server_request_queue_close(&engine->request_queue);
    if (engine->telemetry_opened) {
        yvex_server_telemetry_model_closed(manager->telemetry);
        engine->telemetry_opened = 0;
    }
    engine->summary.execution_ready = 0;
    if (rc == YVEX_OK) {
        engine->artifact_bytes = 0ull;
        engine->mapped_package_bytes = 0ull;
        engine->prepared_bytes = 0ull;
        engine->model_resident_host_bytes = 0ull;
        engine->model_resident_device_bytes = 0ull;
        engine->model_component_count = 0ull;
        memset(&engine->model_residency, 0,
               sizeof(engine->model_residency));
        (void)summary_resources(engine, NULL);
    }
    if (err) *err = primary;
    if (rc == YVEX_OK) yvex_error_clear(err);
    return rc;
}

static server_engine *engine_find(server_engine_manager *manager,
                                  const char *alias)
{
    unsigned long long index;
    for (index = 0ull; index < manager->capacity; ++index)
        if (manager->engines[index].alias[0] &&
            strcmp(manager->engines[index].alias, alias) == 0)
            return &manager->engines[index];
    return NULL;
}

static server_engine *engine_reserve(server_engine_manager *manager,
                                     const char *alias)
{
    server_engine *empty = NULL;
    unsigned long long index;
    for (index = 0ull; index < manager->capacity; ++index) {
        server_engine *engine = &manager->engines[index];
        if (engine->alias[0] && strcmp(engine->alias, alias) == 0)
            return engine->state == YVEX_SERVER_ENGINE_UNLOADED ||
                           engine->state == YVEX_SERVER_ENGINE_FAILED
                       ? engine : NULL;
        if (!empty && (!engine->alias[0] ||
                       engine->state == YVEX_SERVER_ENGINE_UNLOADED))
            empty = engine;
    }
    return empty;
}

int yvex_server_engine_manager_open(
    server_engine_manager **out, unsigned long long capacity,
    unsigned long long request_capacity, unsigned long long request_workers,
    server_request_queue_execute request_execute, void *request_context,
    server_telemetry *telemetry, yvex_error *err)
{
    server_engine_manager *manager;
    if (out) *out = NULL;
    if (!out || !capacity ||
        capacity > YVEX_SERVER_IMPLEMENTATION_MAXIMUM_ENGINES ||
        !request_capacity || !request_workers || !request_execute || !telemetry)
        return engine_refuse(err, YVEX_ERR_INVALID_ARG,
                             "bounded engine manager inputs are required");
    manager = calloc(1u, sizeof(*manager));
    if (manager)
        manager->engines = calloc((size_t)capacity, sizeof(*manager->engines));
    if (!manager || !manager->engines) {
        free(manager ? manager->engines : NULL);
        free(manager);
        return engine_refuse(err, YVEX_ERR_NOMEM,
                             "engine manager allocation failed");
    }
    manager->capacity = capacity;
    manager->next_generation = 1ull;
    manager->request_capacity = request_capacity;
    manager->request_workers = request_workers;
    manager->request_execute = request_execute;
    manager->request_context = request_context;
    manager->telemetry = telemetry;
    if (pthread_mutex_init(&manager->mutex, NULL) != 0 ||
        (manager->mutex_ready = 1,
         pthread_cond_init(&manager->condition, NULL) != 0)) {
        if (manager->mutex_ready) (void)pthread_mutex_destroy(&manager->mutex);
        free(manager->engines);
        free(manager);
        return engine_refuse(err, YVEX_ERR_STATE,
                             "engine manager synchronization failed");
    }
    manager->condition_ready = 1;
    *out = manager;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_server_engine_manager_load(
    server_engine_manager *manager, const yvex_server_engine_options *options,
    const yvex_server_media_options *media,
    yvex_server_engine_summary *summary, yvex_error *err)
{
    server_engine candidate, *slot;
    yvex_server_engine_summary published;
    int rc;
    memset(&candidate, 0, sizeof(candidate));
    rc = options_admit(&candidate, options, media, err);
    if (rc != YVEX_OK) return rc;
    if (!manager || !summary || pthread_mutex_lock(&manager->mutex) != 0)
        return engine_refuse(err, YVEX_ERR_INVALID_ARG,
                             "open manager and summary output are required");
    slot = !manager->closing ? engine_reserve(manager, candidate.alias) : NULL;
    if (!slot) {
        (void)pthread_mutex_unlock(&manager->mutex);
        return engine_refuse(err, manager->closing ? YVEX_ERR_CANCELLED
                                                    : YVEX_ERR_BOUNDS,
                             manager->closing ? "engine manager is closing"
                                              : "engine alias is active or capacity is full");
    }
    candidate.generation = manager->next_generation++;
    candidate.state = YVEX_SERVER_ENGINE_LOADING;
    candidate.active_work = 1ull;
    summary_base(&candidate);
    *slot = candidate;
    options_rebind(slot);
    (void)pthread_mutex_unlock(&manager->mutex);
    engine_lifecycle_event(manager, &candidate.summary,
                           YVEX_SERVER_EVENT_ENGINE_LOAD_REQUESTED,
                           YVEX_SERVER_SEVERITY_INFO, candidate.state);
    candidate.active_work = 0ull;
    rc = engine_request_queue_open(manager, &candidate, err);
    if (rc == YVEX_OK)
        rc = candidate.options.engine_kind == YVEX_SERVER_ENGINE_MEDIA
                 ? media_engine_open(manager, &candidate, media, err)
                 : text_engine_open(manager, &candidate, err);
    if (rc == YVEX_OK)
        rc = summary_resources(&candidate, err);
    if (rc != YVEX_OK) {
        yvex_error primary = err ? *err : (yvex_error){0}, cleanup;
        (void)engine_close(manager, &candidate, &cleanup);
        if (err) *err = primary;
    }
    (void)pthread_mutex_lock(&manager->mutex);
    candidate.state = rc == YVEX_OK ? YVEX_SERVER_ENGINE_LOADED
                                    : YVEX_SERVER_ENGINE_FAILED;
    summary_base(&candidate);
    *slot = candidate;
    options_rebind(slot);
    published = slot->summary;
    *summary = published;
    (void)pthread_cond_broadcast(&manager->condition);
    (void)pthread_mutex_unlock(&manager->mutex);
    engine_lifecycle_event(
        manager, &published,
        rc == YVEX_OK ? YVEX_SERVER_EVENT_ENGINE_READY
                      : YVEX_SERVER_EVENT_ENGINE_LOAD_FAILED,
        rc == YVEX_OK ? YVEX_SERVER_SEVERITY_INFO
                      : YVEX_SERVER_SEVERITY_ERROR,
        rc == YVEX_OK ? published.state : (unsigned long long)rc);
    if (rc == YVEX_OK) yvex_error_clear(err);
    return rc;
}

int yvex_server_engine_manager_acquire(
    server_engine_manager *manager, const char *alias,
    unsigned long long generation, server_engine_lease *lease,
    yvex_server_engine_summary *summary, yvex_error *err)
{
    server_engine *engine = NULL;
    unsigned long long index, loaded = 0ull;
    if (lease) memset(lease, 0, sizeof(*lease));
    if (!manager || !lease || !summary ||
        pthread_mutex_lock(&manager->mutex) != 0)
        return engine_refuse(err, YVEX_ERR_INVALID_ARG,
                             "engine acquisition inputs are required");
    if (manager->closing) {
        (void)pthread_mutex_unlock(&manager->mutex);
        return engine_refuse(err, YVEX_ERR_CANCELLED,
                             "engine manager is closing");
    }
    if (alias && alias[0])
        engine = engine_find(manager, alias);
    else
        for (index = 0ull; index < manager->capacity; ++index)
            if (manager->engines[index].state == YVEX_SERVER_ENGINE_LOADED) {
                engine = &manager->engines[index];
                loaded++;
            }
    if ((!alias || !alias[0]) && loaded != 1ull) engine = NULL;
    if (!engine || engine->state != YVEX_SERVER_ENGINE_LOADED ||
        (generation && generation != engine->generation)) {
        (void)pthread_mutex_unlock(&manager->mutex);
        return engine_refuse(err, YVEX_ERR_STATE,
                             generation ? "engine generation is stale or unavailable"
                                        : "one unambiguous loaded engine is required");
    }
    engine->active_work++;
    summary_base(engine);
    lease->engine = engine;
    lease->generation = engine->generation;
    *summary = engine->summary;
    (void)pthread_mutex_unlock(&manager->mutex);
    yvex_error_clear(err);
    return YVEX_OK;
}

void yvex_server_engine_manager_release(server_engine_manager *manager,
                                        server_engine_lease *lease)
{
    server_engine *engine;
    if (!manager || !lease || !lease->engine ||
        pthread_mutex_lock(&manager->mutex) != 0)
        return;
    engine = lease->engine;
    if (engine->generation == lease->generation && engine->active_work)
        engine->active_work--;
    summary_base(engine);
    memset(lease, 0, sizeof(*lease));
    (void)pthread_cond_broadcast(&manager->condition);
    (void)pthread_mutex_unlock(&manager->mutex);
}

int yvex_server_engine_lease_submit(
    server_engine_lease *lease, void *work, const char *serialization_scope,
    unsigned long long *queued, yvex_error *err)
{
    server_engine *engine = lease ? lease->engine : NULL;
    char serialization_key[SERVER_REQUEST_QUEUE_KEY_CAP];
    int rc;
    if (!engine || engine->generation != lease->generation ||
        !engine->request_queue)
        return engine_refuse(err, YVEX_ERR_STATE,
                             "live engine request queue lease is required");
    rc = yvex_server_request_queue_key(serialization_key, lease->generation,
                                       serialization_scope, err);
    if (rc == YVEX_OK)
        rc = yvex_server_request_queue_submit(
            engine->request_queue, work, serialization_key, queued, err);
    return rc;
}

int yvex_server_engine_manager_unload(
    server_engine_manager *manager, const char *alias,
    unsigned long long generation, yvex_server_engine_summary *summary,
    yvex_error *err)
{
    server_engine *engine;
    int rc;
    if (!manager || !alias_valid(alias) || !summary ||
        pthread_mutex_lock(&manager->mutex) != 0)
        return engine_refuse(err, YVEX_ERR_INVALID_ARG,
                             "loaded engine alias and summary are required");
    engine = engine_find(manager, alias);
    if (!engine || engine->state != YVEX_SERVER_ENGINE_LOADED ||
        (generation && generation != engine->generation)) {
        (void)pthread_mutex_unlock(&manager->mutex);
        return engine_refuse(err, YVEX_ERR_STATE,
                             "exact loaded engine generation is required");
    }
    engine->state = YVEX_SERVER_ENGINE_DRAINING;
    summary_base(engine);
    engine_lifecycle_event(manager, &engine->summary,
                           YVEX_SERVER_EVENT_ENGINE_UNLOAD_STARTED,
                           YVEX_SERVER_SEVERITY_INFO, engine->state);
    engine_cancel(engine);
    while (engine->active_work)
        (void)pthread_cond_wait(&manager->condition, &manager->mutex);
    engine->state = YVEX_SERVER_ENGINE_UNLOADING;
    summary_base(engine);
    (void)pthread_mutex_unlock(&manager->mutex);
    rc = engine_close(manager, engine, err);
    (void)pthread_mutex_lock(&manager->mutex);
    engine->state = rc == YVEX_OK ? YVEX_SERVER_ENGINE_UNLOADED
                                  : YVEX_SERVER_ENGINE_FAILED;
    summary_base(engine);
    *summary = engine->summary;
    (void)pthread_cond_broadcast(&manager->condition);
    (void)pthread_mutex_unlock(&manager->mutex);
    engine_lifecycle_event(
        manager, summary,
        rc == YVEX_OK ? YVEX_SERVER_EVENT_ENGINE_UNLOADED
                      : YVEX_SERVER_EVENT_ENGINE_UNLOAD_FAILED,
        rc == YVEX_OK ? YVEX_SERVER_SEVERITY_INFO
                      : YVEX_SERVER_SEVERITY_ERROR,
        rc == YVEX_OK ? summary->state : (unsigned long long)rc);
    return rc;
}

int yvex_server_engine_manager_snapshot(
    server_engine_manager *manager, yvex_server_engine_summary *engines,
    unsigned long long capacity, unsigned long long *count, yvex_error *err)
{
    unsigned long long index, written = 0ull;
    if (count) *count = 0ull;
    if (!manager || !count || (capacity && !engines) ||
        pthread_mutex_lock(&manager->mutex) != 0)
        return engine_refuse(err, YVEX_ERR_INVALID_ARG,
                             "engine snapshot outputs are required");
    for (index = 0ull; index < manager->capacity; ++index) {
        server_engine *engine = &manager->engines[index];
        if (!engine->alias[0]) continue;
        if (written >= capacity) {
            (void)pthread_mutex_unlock(&manager->mutex);
            return engine_refuse(err, YVEX_ERR_BOUNDS,
                                 "engine snapshot capacity is insufficient");
        }
        if (engine->sessions)
            (void)yvex_server_sessions_count(
                engine->sessions, &engine->summary.session_count, NULL);
        else if (engine->media)
            (void)yvex_server_media_registry_count(
                engine->media, &engine->summary.session_count, NULL);
        if (summary_resources(engine, err) != YVEX_OK) {
            (void)pthread_mutex_unlock(&manager->mutex);
            return yvex_error_code(err);
        }
        summary_base(engine);
        engines[written++] = engine->summary;
    }
    *count = written;
    (void)pthread_mutex_unlock(&manager->mutex);
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_server_engine_lease_execute(
    server_engine_lease *lease, const yvex_client_request *request,
    const char *request_id, double queue_seconds, server_message_emit emit,
    void *emit_context, yvex_error *err)
{
    server_engine *engine = lease ? lease->engine : NULL;
    if (!engine || engine->generation != lease->generation)
        return engine_refuse(err, YVEX_ERR_STATE,
                             "live engine lease is required");
    return engine->media
               ? yvex_server_media_registry_execute(
                     engine->media, request, request_id, queue_seconds,
                     emit, emit_context, err)
               : yvex_server_sessions_execute(
                     engine->sessions, request, request_id, queue_seconds,
                     emit, emit_context, err);
}

int yvex_server_engine_lease_cancel(server_engine_lease *lease,
                                    const char *session, yvex_error *err)
{
    server_engine *engine = lease ? lease->engine : NULL;
    if (!engine || engine->generation != lease->generation)
        return engine_refuse(err, YVEX_ERR_STATE,
                             "live engine lease is required");
    return engine->media
               ? yvex_server_media_registry_cancel(engine->media, session, err)
               : yvex_server_sessions_cancel(engine->sessions, session, err);
}

int yvex_server_engine_lease_console_status(
    server_engine_lease *lease, const char *session,
    yvex_console_status *status, yvex_client_partial_turn *partial,
    yvex_error *err)
{
    server_engine *engine = lease ? lease->engine : NULL;
    if (!engine || engine->generation != lease->generation)
        return engine_refuse(err, YVEX_ERR_STATE,
                             "live engine lease is required");
    return engine->media
               ? yvex_server_media_registry_console_status(
                     engine->media, session, status, partial, err)
               : yvex_server_sessions_console_status(
                     engine->sessions, session, status, partial, err);
}

void yvex_server_engine_manager_cancel_all(server_engine_manager *manager)
{
    unsigned long long index;
    if (!manager || pthread_mutex_lock(&manager->mutex) != 0) return;
    for (index = 0ull; index < manager->capacity; ++index)
        if (manager->engines[index].state == YVEX_SERVER_ENGINE_LOADED ||
            manager->engines[index].state == YVEX_SERVER_ENGINE_DRAINING) {
            engine_cancel(&manager->engines[index]);
            yvex_server_request_queue_request_stop(
                manager->engines[index].request_queue);
        }
    (void)pthread_mutex_unlock(&manager->mutex);
}

static unsigned long long summary_add(unsigned long long left,
                                      unsigned long long right)
{
    return left > ULLONG_MAX - right ? ULLONG_MAX : left + right;
}

int yvex_server_engine_manager_request_queue_snapshot(
    server_engine_manager *manager, server_request_queue_summary *summary,
    yvex_error *err)
{
    unsigned long long index;
    if (summary) memset(summary, 0, sizeof(*summary));
    if (!manager || !summary || pthread_mutex_lock(&manager->mutex) != 0)
        return engine_refuse(err, YVEX_ERR_INVALID_ARG,
                             "engine request queue summary is required");
    for (index = 0ull; index < manager->capacity; ++index) {
        server_request_queue_summary current = {0};
        if (!manager->engines[index].request_queue) continue;
        yvex_server_request_queue_snapshot(
            manager->engines[index].request_queue, &current);
        summary->queued = summary_add(summary->queued, current.queued);
        summary->capacity = summary_add(summary->capacity, current.capacity);
        summary->active = summary_add(summary->active, current.active);
        summary->workers = summary_add(summary->workers, current.workers);
    }
    (void)pthread_mutex_unlock(&manager->mutex);
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_server_engine_manager_close(server_engine_manager **manager,
                                     yvex_error *err)
{
    server_engine_manager *owner;
    unsigned long long index;
    int rc = YVEX_OK;
    if (!manager || !*manager) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    owner = *manager;
    (void)pthread_mutex_lock(&owner->mutex);
    owner->closing = 1;
    for (index = 0ull; index < owner->capacity; ++index) {
        server_engine *engine = &owner->engines[index];
        if (engine->state == YVEX_SERVER_ENGINE_LOADED) {
            engine->state = YVEX_SERVER_ENGINE_DRAINING;
            engine_cancel(engine);
            yvex_server_request_queue_request_stop(engine->request_queue);
        }
    }
    for (;;) {
        unsigned long long active = 0ull;
        for (index = 0ull; index < owner->capacity; ++index)
            active += owner->engines[index].active_work;
        if (!active) break;
        (void)pthread_cond_wait(&owner->condition, &owner->mutex);
    }
    (void)pthread_mutex_unlock(&owner->mutex);
    for (index = 0ull; index < owner->capacity; ++index) {
        yvex_error cleanup;
        int cleanup_rc = engine_close(owner, &owner->engines[index], &cleanup);
        if (cleanup_rc != YVEX_OK && rc == YVEX_OK) {
            rc = cleanup_rc;
            if (err) *err = cleanup;
        }
    }
    if (owner->condition_ready) (void)pthread_cond_destroy(&owner->condition);
    if (owner->mutex_ready) (void)pthread_mutex_destroy(&owner->mutex);
    free(owner->engines);
    free(owner);
    *manager = NULL;
    if (rc == YVEX_OK) yvex_error_clear(err);
    return rc;
}
