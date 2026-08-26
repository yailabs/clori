/* Own server-session allocation, reset, fork, publication, and destruction. */
#include "src/server/private.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <yvex/internal/core.h>
#include <yvex/internal/tokenizer.h>

static int session_name_valid(const char *name)
{
    size_t index, count;
    if (!name || !name[0]) return 0;
    count = strlen(name);
    if (count >= YVEX_SERVER_SESSION_NAME_CAP) return 0;
    for (index = 0u; index < count; ++index) {
        unsigned char byte = (unsigned char)name[index];
        if (!((byte >= 'a' && byte <= 'z') ||
              (byte >= 'A' && byte <= 'Z') ||
              (byte >= '0' && byte <= '9') || byte == '-' || byte == '_' ||
              byte == '.'))
            return 0;
    }
    return 1;
}

static int session_identity(server_session_registry *registry,
                            const char *name,
                            char output[YVEX_SHA256_HEX_CAP])
{
    yvex_model_engine_summary model;
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_error err;
    if (!yvex_model_engine_view_get(registry->model) ||
        yvex_model_engine_summary_copy(registry->model, &model, &err) !=
            YVEX_OK)
        return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.server.session.v1") ||
        !yvex_sha256_update_text(&hash, model.runtime_model_identity) ||
        !yvex_sha256_update_u64(&hash, registry->engine_generation) ||
        !yvex_sha256_update_text(&hash, name) ||
        !yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

server_session *yvex_server_session_find_locked(
    server_session_registry *registry, const char *name)
{
    unsigned long long index;
    if (!registry || !name) return NULL;
    for (index = 0u; index < registry->capacity; ++index)
        if (registry->sessions[index].state != YVEX_SERVER_SESSION_CLOSED &&
            registry->sessions[index].name[0] &&
            strcmp(registry->sessions[index].name, name) == 0)
            return &registry->sessions[index];
    return NULL;
}

int yvex_server_session_execution_open(server_session_registry *registry,
                                       server_session *session,
                                       yvex_error *err)
{
    yvex_runtime_session_open_request request = {0};
    yvex_model_engine_failure failure = {0};
    request.backend = registry->options.backend;
    request.maximum_host_bytes = registry->options.maximum_host_bytes;
    request.maximum_device_bytes = registry->options.maximum_device_bytes;
    return yvex_runtime_session_open(&session->execution, registry->model,
                                     &request, &failure, err);
}

static void session_storage_release(server_session *session)
{
    free(session->transcript);
    free(session->turn_text);
    free(session->token_results);
    free(session->prompt_tokens);
    free(session->committed_tokens);
    session->transcript = session->turn_text = NULL;
    session->token_results = NULL;
    session->prompt_tokens = session->committed_tokens = NULL;
}

static int session_allocate_locked(server_session_registry *registry,
                                   const char *requested,
                                   server_session **created, yvex_error *err)
{
    server_session *session = NULL;
    char generated[YVEX_SERVER_SESSION_NAME_CAP];
    const char *name = requested;
    unsigned long long index;
    int rc;
    if (created) *created = NULL;
    if (!name || !name[0]) {
        (void)snprintf(generated, sizeof(generated), "s%06llu", registry->next_id++);
        name = generated;
    }
    if (!created || !session_name_valid(name) || yvex_server_session_find_locked(registry, name)) {
        yvex_error_set(err, YVEX_ERR_STATE, "server.session.create",
                       "session name is invalid or already exists");
        return YVEX_ERR_STATE;
    }
    for (index = 0u; index < registry->capacity; ++index)
        if (!registry->sessions[index].name[0] || registry->sessions[index].state == YVEX_SERVER_SESSION_CLOSED) {
            session = &registry->sessions[index];
            break;
        }
    if (!session) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "server.session.create",
                       "session registry capacity is exhausted");
        return YVEX_ERR_BOUNDS;
    }
    memset(session, 0, sizeof(*session));
    yvex_core_text_copy(session->name, sizeof(session->name), name);
    session->state = YVEX_SERVER_SESSION_CREATED;
    session->token_capacity = registry->options.context_capacity;
    session->text_capacity = registry->options.maximum_output_bytes;
    session->transcript_capacity = SESSION_TRANSCRIPT_BYTES;
    session->committed_tokens = calloc((size_t)session->token_capacity, sizeof(*session->committed_tokens));
    session->prompt_tokens = calloc((size_t)session->token_capacity, sizeof(*session->prompt_tokens));
    session->token_results = calloc((size_t)registry->options.maximum_new_tokens,
                                    sizeof(*session->token_results));
    session->turn_text = calloc((size_t)session->text_capacity + 1u, 1u);
    session->transcript = calloc((size_t)session->transcript_capacity, 1u);
    if (!session->committed_tokens || !session->prompt_tokens || !session->token_results ||
        !session->turn_text || !session->transcript ||
        !session_identity(registry, name, session->identity)) {
        rc = YVEX_ERR_NOMEM;
        yvex_error_set(err, rc, "server.session.create",
                       "session storage allocation or identity failed");
        goto failure;
    }
    rc = yvex_server_session_execution_open(registry, session, err);
    if (rc != YVEX_OK) goto failure;
    atomic_init(&session->cancel_requested, 0);
    atomic_init(&session->active_turn, 0);
    session->message_history_generation = 1u;
    session->transcript_generation = 1u;
    session->reasoning_policy = registry->default_reasoning_policy;
    session->state = YVEX_SERVER_SESSION_READY;
    *created = session;
    return YVEX_OK;
failure:
    session_storage_release(session);
    memset(session, 0, sizeof(*session));
    return rc;
}

static int session_publish_locked(server_session_registry *registry,
                                  server_session *session, const char *phase,
                                  unsigned long long value_a,
                                  unsigned long long value_b,
                                  yvex_error *err)
{
    int rc;
    registry->count++;
    yvex_server_telemetry_session(registry->telemetry, 1, 1);
    rc = yvex_server_telemetry_emit(
        registry->telemetry, &registry->event_scope,
        YVEX_SERVER_EVENT_SESSION_CREATED,
        YVEX_SERVER_SEVERITY_INFO, session->name, NULL, NULL, phase,
        value_a, value_b, registry->count, 0.0, 0.0, err);
    if (rc == YVEX_OK) return YVEX_OK;
    registry->count--;
    yvex_server_telemetry_session(registry->telemetry, -1, 0);
    return rc;
}

static int session_discard_unpublished(server_session *session,
                                       yvex_error *err)
{
    int rc = yvex_runtime_generation_context_close(&session->generation, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_session_close(&session->execution, err);
    if (rc != YVEX_OK) {
        session->state = YVEX_SERVER_SESSION_FAILED;
        return rc;
    }
    session_storage_release(session);
    memset(session, 0, sizeof(*session));
    return YVEX_OK;
}

int yvex_server_session_create_locked(server_session_registry *registry,
                                      const char *requested,
                                      server_session **created,
                                      yvex_error *err)
{
    int rc = session_allocate_locked(registry, requested, created, err);
    if (rc == YVEX_OK)
        rc = session_publish_locked(registry, *created, "session", 0ull,
                                    0ull, err);
    if (rc != YVEX_OK && created && *created) {
        yvex_error primary = err ? *err : (yvex_error){0}, cleanup;
        (void)session_discard_unpublished(*created, &cleanup);
        *created = NULL;
        if (err) *err = primary;
    }
    return rc;
}

int yvex_server_session_fork_locked(
    server_session_registry *registry, server_session *source,
    const char *requested, unsigned long long maximum_shared_bytes,
    server_session **created,
    yvex_runtime_session_prefix_summary *prefix_summary, yvex_error *err)
{
    const yvex_model_engine_view *view =
        registry ? yvex_model_engine_view_get(registry->model) : NULL;
    yvex_runtime_session_prefix *prefix = NULL;
    yvex_model_engine_failure failure = {0};
    server_session *child = NULL;
    int rc;
    if (created) *created = NULL;
    if (prefix_summary) memset(prefix_summary, 0, sizeof(*prefix_summary));
    if (!registry || !source || !created || !prefix_summary || !view ||
        !view->tokenizer || !requested || !maximum_shared_bytes ||
        atomic_load_explicit(&source->active_turn, memory_order_acquire) ||
        source->state == YVEX_SERVER_SESSION_PARTIAL ||
        source->state == YVEX_SERVER_SESSION_FAILED ||
        source->state == YVEX_SERVER_SESSION_CLOSING ||
        source->state == YVEX_SERVER_SESSION_CLOSED) {
        yvex_error_set(err, YVEX_ERR_STATE, "server.session.fork",
                       "idle complete source and bounded child are required");
        return YVEX_ERR_STATE;
    }
    if (source->committed_count) {
        rc = yvex_runtime_session_prefix_capture(
            source->execution, maximum_shared_bytes, &prefix, prefix_summary,
            &failure, err);
        if (rc != YVEX_OK) return rc;
        if (prefix_summary->committed_sequence_length !=
            source->committed_count) {
            yvex_runtime_session_prefix_close(&prefix);
            yvex_error_set(err, YVEX_ERR_STATE, "server.session.fork",
                           "semantic and physical source prefixes diverged");
            return YVEX_ERR_STATE;
        }
    }
    rc = session_allocate_locked(registry, requested, &child, err);
    if (rc == YVEX_OK && prefix)
        rc = yvex_runtime_session_prefix_attach(
            child->execution, prefix, prefix_summary, &failure, err);
    if (rc == YVEX_OK)
        rc = yvex_server_session_state_clone(
            source, child, yvex_tokenizer_vocab_size(view->tokenizer), err);
    if (rc == YVEX_OK &&
        (!prefix || child->committed_count ==
                        prefix_summary->committed_sequence_length))
        rc = session_publish_locked(
            registry, child, "fork", child->committed_count,
            prefix_summary->shared_bytes, err);
    else if (rc == YVEX_OK) {
        rc = YVEX_ERR_STATE;
        yvex_error_set(err, rc, "server.session.fork",
                       "forked semantic extent changed during publication");
    }
    yvex_runtime_session_prefix_close(&prefix);
    if (rc != YVEX_OK && child) {
        yvex_error primary = err ? *err : (yvex_error){0}, cleanup;
        (void)session_discard_unpublished(child, &cleanup);
        if (err) *err = primary;
        return rc;
    }
    *created = child;
    return YVEX_OK;
}

int yvex_server_session_reset_locked(server_session_registry *registry,
                                     server_session *session,
                                     yvex_error *err)
{
    unsigned long long next_messages, next_transcript;
    int rc;
    if (!yvex_core_u64_add(session->message_history_generation, 1u,
                           &next_messages) ||
        !yvex_core_u64_add(session->transcript_generation, 1u,
                           &next_transcript)) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "server.session.reset",
                       "session history generation overflowed");
        return YVEX_ERR_BOUNDS;
    }
    session->state = YVEX_SERVER_SESSION_RESETTING;
    rc = yvex_runtime_generation_context_close(&session->generation, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_session_close(&session->execution, err);
    if (rc == YVEX_OK)
        rc = yvex_server_session_execution_open(registry, session, err);
    if (rc != YVEX_OK) {
        session->state = YVEX_SERVER_SESSION_FAILED;
        return rc;
    }
    memset(session->messages, 0, sizeof(session->messages));
    memset(session->transcript, 0, (size_t)session->transcript_capacity);
    memset(session->committed_tokens, 0,
           (size_t)session->token_capacity * sizeof(*session->committed_tokens));
    memset(session->prompt_tokens, 0,
           (size_t)session->token_capacity * sizeof(*session->prompt_tokens));
    session->message_count = session->transcript_count = 0u;
    session->committed_count = session->turn_count = 0u;
    session->message_history_generation = next_messages;
    session->transcript_generation = next_transcript;
    session->policy_set = session->pending_generation_checkpoint_present = 0;
    memset(&session->policy, 0, sizeof(session->policy));
    memset(&session->pending_generation_checkpoint, 0,
           sizeof(session->pending_generation_checkpoint));
    session->reasoning_policy = registry->default_reasoning_policy;
    memset(session->last_turn_identity, 0, sizeof(session->last_turn_identity));
    memset(session->state_digest, 0, sizeof(session->state_digest));
    memset(session->generated_token_identity, 0,
           sizeof(session->generated_token_identity));
    memset(session->generated_text_digest, 0,
           sizeof(session->generated_text_digest));
    memset(&session->partial_turn, 0, sizeof(session->partial_turn));
    atomic_store_explicit(&session->cancel_requested, 0, memory_order_release);
    session->state = session->attached_clients ? YVEX_SERVER_SESSION_READY
                                               : YVEX_SERVER_SESSION_DETACHED;
    return yvex_server_telemetry_emit(
        registry->telemetry, &registry->event_scope,
        YVEX_SERVER_EVENT_SESSION_RESET,
        YVEX_SERVER_SEVERITY_INFO, session->name, NULL, NULL, "session",
        0u, 0u, 0u, 0.0, 0.0, err);
}

int yvex_server_session_close_locked(server_session_registry *registry,
                                     server_session *session,
                                     yvex_error *err)
{
    int rc;
    session->state = YVEX_SERVER_SESSION_CLOSING;
    rc = yvex_runtime_generation_context_close(&session->generation, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_session_close(&session->execution, err);
    if (rc != YVEX_OK) {
        session->state = YVEX_SERVER_SESSION_FAILED;
        return rc;
    }
    session_storage_release(session);
    registry->count--;
    yvex_server_telemetry_session(registry->telemetry, -1, 0);
    (void)yvex_server_telemetry_emit(
        registry->telemetry, &registry->event_scope,
        YVEX_SERVER_EVENT_SESSION_CLOSED,
        YVEX_SERVER_SEVERITY_INFO, session->name, NULL, NULL, "session",
        0u, registry->count, 0u, 0.0, 0.0, err);
    memset(session, 0, sizeof(*session));
    session->state = YVEX_SERVER_SESSION_CLOSED;
    return YVEX_OK;
}

int yvex_server_sessions_open(server_session_registry **out,
                              yvex_model_engine *model,
                              const yvex_server_engine_options *options,
                              unsigned long long engine_generation,
                              int continuous_batching,
                              const server_event_scope *event_scope,
                              server_telemetry *telemetry, yvex_error *err)
{
    server_session_registry *registry;
    if (out) *out = NULL;
    if (!out || !model || !options || !event_scope || !telemetry ||
        !engine_generation ||
        event_scope->generation_mode != options->generation_mode ||
        !yvex_sha256_hex_valid(event_scope->runtime_model_identity) ||
        !yvex_sha256_hex_valid(event_scope->artifact_identity) ||
        !yvex_sha256_hex_valid(event_scope->specialization_identity) ||
        !options->maximum_sessions ||
        options->maximum_sessions > SIZE_MAX / sizeof(server_session)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "server.session.registry",
                       "model, telemetry, and bounded session capacity are required");
        return YVEX_ERR_INVALID_ARG;
    }
    registry = calloc(1u, sizeof(*registry));
    if (registry) registry->sessions = calloc((size_t)options->maximum_sessions,
                                              sizeof(*registry->sessions));
    if (!registry || !registry->sessions) {
        free(registry ? registry->sessions : NULL);
        free(registry);
        yvex_error_set(err, YVEX_ERR_NOMEM, "server.session.registry",
                       "session registry allocation failed");
        return YVEX_ERR_NOMEM;
    }
    registry->model = model;
    registry->options = *options;
    registry->engine_generation = engine_generation;
    registry->event_scope = *event_scope;
    registry->continuous_batching = continuous_batching != 0;
    registry->default_reasoning_policy = server_reasoning_automatic_policy();
    registry->telemetry = telemetry;
    registry->capacity = options->maximum_sessions;
    registry->next_id = 1u;
    if (pthread_mutex_init(&registry->mutex, NULL) != 0) {
        free(registry->sessions);
        free(registry);
        yvex_error_set(err, YVEX_ERR_STATE, "server.session.registry",
                       "session registry mutex initialization failed");
        return YVEX_ERR_STATE;
    }
    registry->mutex_ready = 1;
    *out = registry;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_server_sessions_count(server_session_registry *registry,
                               unsigned long long *count, yvex_error *err)
{
    if (!registry || !count || pthread_mutex_lock(&registry->mutex) != 0) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "server.session.count",
                       "registry and count output are required");
        return YVEX_ERR_INVALID_ARG;
    }
    *count = registry->count;
    (void)pthread_mutex_unlock(&registry->mutex);
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_server_sessions_resource_bytes(
    server_session_registry *registry, unsigned long long *host_bytes,
    unsigned long long *device_bytes, yvex_error *err)
{
    unsigned long long index, host = 0ull, device = 0ull;
    if (host_bytes) *host_bytes = 0ull;
    if (device_bytes) *device_bytes = 0ull;
    if (!registry || !host_bytes || !device_bytes ||
        pthread_mutex_lock(&registry->mutex) != 0) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "server.session.resources",
                       "registry and resource outputs are required");
        return YVEX_ERR_INVALID_ARG;
    }
    for (index = 0ull; index < registry->capacity; ++index) {
        const server_session *session = &registry->sessions[index];
        yvex_runtime_session_summary summary;
        unsigned long long owned_host, owned_device;
        if (!session->name[0] || !session->execution ||
            session->state == YVEX_SERVER_SESSION_CLOSED)
            continue;
        if (yvex_runtime_session_summary_copy(session->execution, &summary,
                                              err) != YVEX_OK) {
            (void)pthread_mutex_unlock(&registry->mutex);
            return yvex_error_code(err);
        }
        owned_host = summary.peak_host_bytes >= summary.host_resident_bytes
                         ? summary.peak_host_bytes - summary.host_resident_bytes
                         : 0ull;
        owned_device =
            summary.peak_device_bytes >= summary.device_resident_bytes
                ? summary.peak_device_bytes - summary.device_resident_bytes
                : 0ull;
        if (!yvex_core_u64_add(host, owned_host, &host) ||
            !yvex_core_u64_add(device, owned_device, &device)) {
            (void)pthread_mutex_unlock(&registry->mutex);
            yvex_error_set(err, YVEX_ERR_BOUNDS, "server.session.resources",
                           "session resource total overflowed");
            return YVEX_ERR_BOUNDS;
        }
    }
    *host_bytes = host;
    *device_bytes = device;
    (void)pthread_mutex_unlock(&registry->mutex);
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_server_sessions_close(server_session_registry **registry,
                               yvex_error *err)
{
    server_session_registry *owner;
    unsigned long long index;
    int rc = YVEX_OK;
    if (!registry || !*registry) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    owner = *registry;
    if (!owner->mutex_ready || pthread_mutex_lock(&owner->mutex) != 0) {
        yvex_error_set(err, YVEX_ERR_STATE, "server.session.close",
                       "session registry close lock failed");
        return YVEX_ERR_STATE;
    }
    owner->closing = 1;
    for (index = 0u; index < owner->capacity && rc == YVEX_OK; ++index)
        if (owner->sessions[index].name[0] &&
            owner->sessions[index].state != YVEX_SERVER_SESSION_CLOSED)
            rc = yvex_server_session_close_locked(
                owner, &owner->sessions[index], err);
    (void)pthread_mutex_unlock(&owner->mutex);
    if (rc != YVEX_OK) return rc;
    (void)pthread_mutex_destroy(&owner->mutex);
    owner->mutex_ready = 0;
    free(owner->sessions);
    memset(owner, 0, sizeof(*owner));
    free(owner);
    *registry = NULL;
    yvex_error_clear(err);
    return YVEX_OK;
}
