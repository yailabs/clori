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
    int continuous_batching, telemetry_opened;
} server_engine;

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

static int options_admit(server_engine *engine,
                         const yvex_server_engine_options *options,
                         const yvex_server_media_options *media,
                         yvex_error *err)
{
    int text;
    if (!engine || !options ||
        options->schema_version != YVEX_SERVER_ENGINE_SCHEMA_V1 ||
        !alias_valid(options->alias) || !options->target_id ||
        strlen(options->target_id) >= sizeof(engine->target_id) ||
        (options->backend != YVEX_BACKEND_KIND_CPU &&
         options->backend != YVEX_BACKEND_KIND_CUDA) ||
        options->generation_mode > YVEX_SERVER_GENERATION_MEDIA ||
        options->trace_level > YVEX_SERVER_TRACE_FULL ||
        !options->maximum_output_bytes || !options->maximum_sessions ||
        !options->concurrent_sequences ||
        options->concurrent_sequences > options->maximum_sessions)
        return engine_refuse(err, YVEX_ERR_INVALID_ARG,
                             "complete bounded engine options are required");
    text = options->generation_mode != YVEX_SERVER_GENERATION_MEDIA;
    if ((text && (!options->artifact_path || !options->runtime_binding_path ||
                  !options->context_capacity || !options->maximum_new_tokens || media)) ||
        (!text && (!media || options->artifact_path ||
                   options->runtime_binding_path || options->context_capacity ||
                   options->prefill_chunk_tokens || options->maximum_new_tokens)))
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
    options->schema_version = YVEX_RUNTIME_GENERATION_SCHEMA_V5;
    options->backend = engine->options.backend;
    options->mode = engine->options.generation_mode == YVEX_SERVER_GENERATION_DSPARK
                        ? YVEX_GENERATION_MODE_DSPARK
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
    options->continuous_batching = engine->continuous_batching;
    options->trace_policy = engine->options.trace_level == YVEX_SERVER_TRACE_FULL
                                ? YVEX_RUNTIME_TRACE_FULL
                                : YVEX_RUNTIME_TRACE_STAGES;
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
    if (workers > engine->options.concurrent_sequences)
        workers = engine->options.concurrent_sequences;
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
        engine->continuous_batching =
            engine->options.concurrent_sequences > 1ull && width >= 2ull;
    if (rc == YVEX_OK)
        rc = yvex_runtime_session_open(&session, engine->model, &request,
                                       &failure, err);
    generation_options(engine, &options);
    if (rc == YVEX_OK)
        rc = yvex_runtime_generation_context_open(
            &generation, engine->model, session, &options, err);
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
    int rc;
    generation_options(engine, &startup);
    request.artifact_path = engine->artifact_path;
    request.runtime_binding_path = engine->runtime_binding_path;
    request.target_id = engine->target_id;
    request.startup_generation = &startup;
    request.residency_backend = engine->options.backend;
    request.maximum_host_bytes = engine->options.maximum_host_bytes;
    request.maximum_device_bytes = engine->options.maximum_device_bytes;
    yvex_error_clear(&path_error);
    if (yvex_paths_default(&paths, &path_error) == YVEX_OK)
        request.artifact_reopen_cache_root = paths.cache_dir;
    rc = yvex_model_engine_open(&engine->model, &request, &failure, err);
    if (rc == YVEX_OK)
        rc = yvex_model_engine_summary_copy(engine->model, &model, err);
    if (rc == YVEX_OK)
        rc = execution_probe(engine, &capacity, specialization_identity, err);
    if (rc == YVEX_OK) {
        event_scope.generation_mode = engine->options.generation_mode;
        yvex_runtime_identity_copy(event_scope.runtime_model_identity,
                                   model.runtime_model_identity);
        yvex_runtime_identity_copy(event_scope.artifact_identity,
                                   model.artifact_identity);
        yvex_runtime_identity_copy(event_scope.specialization_identity,
                                   specialization_identity);
        rc = yvex_server_sessions_open(
            &engine->sessions, engine->model, &engine->options, engine->generation,
            engine->continuous_batching, &event_scope, manager->telemetry, err);
    }
    view = rc == YVEX_OK ? yvex_model_engine_view_get(engine->model) : NULL;
    if (rc != YVEX_OK || !view) return rc != YVEX_OK ? rc : YVEX_ERR_STATE;
    engine->summary.context_capacity = engine->options.context_capacity;
    engine->summary.prefill_chunk_tokens = engine->options.prefill_chunk_tokens;
    engine->summary.maximum_new_tokens = engine->options.maximum_new_tokens;
    engine->summary.maximum_output_bytes = engine->options.maximum_output_bytes;
    engine->summary.maximum_sessions = engine->options.maximum_sessions;
    engine->summary.concurrent_sequences = engine->options.concurrent_sequences;
    engine->summary.mapped_package_bytes = model.mapped_package_bytes;
    engine->summary.prepared_bytes = model.prepared_bytes;
    engine->summary.resident_host_bytes = model.resident_host_bytes;
    engine->summary.resident_device_bytes = model.resident_device_bytes;
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
    if (!engine || engine->schema_version != YVEX_SERVER_ENGINE_SCHEMA_V1 ||
        engine->state > YVEX_SERVER_ENGINE_FAILED ||
        engine->backend > YVEX_BACKEND_KIND_CUDA ||
        engine->generation_mode > YVEX_SERVER_GENERATION_MEDIA ||
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
    engine->summary.schema_version = YVEX_SERVER_ENGINE_SCHEMA_V1;
    engine->summary.state = engine->state;
    engine->summary.backend = engine->options.backend;
    engine->summary.generation_mode = engine->options.generation_mode;
    engine->summary.generation = engine->generation;
    engine->summary.active_work = engine->active_work;
    engine->summary.context_capacity = engine->options.context_capacity;
    engine->summary.prefill_chunk_tokens = engine->options.prefill_chunk_tokens;
    engine->summary.maximum_new_tokens = engine->options.maximum_new_tokens;
    engine->summary.maximum_output_bytes = engine->options.maximum_output_bytes;
    engine->summary.maximum_sessions = engine->options.maximum_sessions;
    engine->summary.concurrent_sequences = engine->options.concurrent_sequences;
    engine->summary.execution_ready = engine->state == YVEX_SERVER_ENGINE_LOADED;
    engine->summary.continuous_batching_ready = engine->continuous_batching;
    yvex_core_text_copy(engine->summary.alias,
                        sizeof(engine->summary.alias), engine->alias);
    yvex_core_text_copy(engine->summary.target_id,
                        sizeof(engine->summary.target_id), engine->target_id);
    assert(yvex_server_engine_summary_valid(&engine->summary));
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
    yvex_model_engine_close(&engine->model);
    yvex_server_request_queue_close(&engine->request_queue);
    if (engine->telemetry_opened) {
        yvex_server_telemetry_model_closed(manager->telemetry);
        engine->telemetry_opened = 0;
    }
    engine->summary.execution_ready = 0;
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
    if (!out || !capacity || capacity > YVEX_SERVER_ENGINE_CAP ||
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
    candidate.active_work = 0ull;
    rc = engine_request_queue_open(manager, &candidate, err);
    if (rc == YVEX_OK)
        rc = candidate.options.generation_mode == YVEX_SERVER_GENERATION_MEDIA
                 ? media_engine_open(manager, &candidate, media, err)
                 : text_engine_open(manager, &candidate, err);
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
    if (rc == YVEX_OK) {
        server_event_scope event_scope;
        server_event_scope_from_engine(&event_scope, &published);
        (void)yvex_server_telemetry_emit(
            manager->telemetry, &event_scope, YVEX_SERVER_EVENT_RUNTIME_READY,
            YVEX_SERVER_SEVERITY_INFO, NULL, NULL, NULL, published.alias,
            published.generation, published.generation_mode,
            published.backend, 0.0, 0.0, err);
        yvex_error_clear(err);
    }
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
    server_engine_lease *lease, void *work, const char *session_name,
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
                                       session_name, err);
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
