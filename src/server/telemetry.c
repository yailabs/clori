/*
 * Make runtime/session/request transitions observable without prose scraping.
 *
 * Raw and operational consumers observe the same global event sequence and counts. Authoritative
 * server event fan-out and process metrics accumulator.
 */
#define _POSIX_C_SOURCE 200809L
#include "src/server/private.h"
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>
#include <yvex/internal/core.h>
struct server_telemetry {
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    yvex_server_event *events;
    unsigned long long capacity, next_sequence, retained_count;
    struct timespec started;
    yvex_server_metrics metrics;
    unsigned long long active_subscribers;
    int mutex_ready, condition_ready, closing;
};
static int event_identity(yvex_server_event *event);

static int event_append_locked(server_telemetry *telemetry,
                               yvex_server_event *event)
{
    unsigned long long slot;
    event->sequence = telemetry->next_sequence++;
    if (!event_identity(event)) return 0;
    slot = (event->sequence - 1u) % telemetry->capacity;
    if (telemetry->retained_count < telemetry->capacity)
        telemetry->retained_count++;
    telemetry->events[slot] = *event;
    return 1;
}

static int hash_double(yvex_sha256 *hash, double value)
{
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return yvex_sha256_update_u64(hash, bits);
}

static unsigned long long time_ns(clockid_t clock)
{
    struct timespec value;
    if (clock_gettime(clock, &value) != 0)
        return 0u;
    return (unsigned long long)value.tv_sec * 1000000000ull +
           (unsigned long long)value.tv_nsec;
}
/*
 * Derive one event identity field by field.
 *
 * Complete typed event and identity output. Writes a canonical SHA-256 identity.
 */
static int event_identity(yvex_server_event *event)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    if (!event)
        return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.server.event.v4") ||
        !yvex_sha256_update_u64(&hash, event->schema_version) ||
        !yvex_sha256_update_u64(&hash, event->sequence) ||
        !yvex_sha256_update_u64(&hash, event->kind) ||
        !yvex_sha256_update_u64(&hash, event->severity) ||
        !yvex_sha256_update_text(&hash, event->session_id) ||
        !yvex_sha256_update_text(&hash, event->request_id) ||
        !yvex_sha256_update_text(&hash, event->turn_id) ||
        !yvex_sha256_update_text(&hash, event->phase) ||
        !yvex_sha256_update_text(&hash, event->provider_adapter) ||
        !yvex_sha256_update_text(&hash, event->provider_request_identity) ||
        !yvex_sha256_update_text(&hash, event->external_correlation_id) ||
        !yvex_sha256_update_u64(&hash, event->value_a) ||
        !yvex_sha256_update_u64(&hash, event->value_b) ||
        !yvex_sha256_update_u64(&hash, event->value_c) ||
        !yvex_sha256_update_u64(&hash, event->engine_kind) ||
        !yvex_sha256_update_u64(&hash, event->execution_strategy) ||
        !yvex_sha256_update_u64(&hash, event->speculative_cycle) ||
        !yvex_sha256_update_u64(&hash, event->proposed_tokens) ||
        !yvex_sha256_update_u64(&hash,
                                event->selected_verification_tokens) ||
        !yvex_sha256_update_u64(&hash, event->accepted_tokens) ||
        !yvex_sha256_update_u64(&hash, event->rejected_tokens) ||
        !yvex_sha256_update_u64(&hash, event->discarded_tokens) ||
        !yvex_sha256_update_u64(&hash, event->verification_count) ||
        !yvex_sha256_update_u64(&hash, event->confidence_logit_count) ||
        !hash_double(&hash, event->confidence_logit_minimum) ||
        !hash_double(&hash, event->confidence_logit_maximum) ||
        !hash_double(&hash, event->confidence_logit_mean) ||
        !hash_double(&hash, event->seconds) ||
        !hash_double(&hash, event->rate) ||
        !yvex_sha256_update_text(&hash,
                                 event->speculation_policy_identity) ||
        !yvex_sha256_update_text(&hash, event->runtime_model_identity) ||
        !yvex_sha256_update_text(&hash, event->artifact_identity) ||
        !yvex_sha256_update_text(&hash, event->variant_identity) ||
        !yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, event->event_identity);
    return 1;
}

int yvex_server_telemetry_open(server_telemetry **out,
                               unsigned long long capacity, yvex_error *err)
{
    server_telemetry *telemetry;
    if (out) *out = NULL;
    if (!out || !capacity || capacity > SIZE_MAX / sizeof(yvex_server_event)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "server.telemetry.open",
                       "bounded telemetry capacity is required");
        return YVEX_ERR_INVALID_ARG;
    }
    telemetry = calloc(1u, sizeof(*telemetry));
    if (telemetry)
        telemetry->events = calloc((size_t)capacity, sizeof(*telemetry->events));
    if (!telemetry || !telemetry->events) {
        free(telemetry ? telemetry->events : NULL);
        free(telemetry);
        yvex_error_set(err, YVEX_ERR_NOMEM, "server.telemetry.open",
                       "telemetry allocation failed");
        return YVEX_ERR_NOMEM;
    }
    telemetry->capacity = capacity;
    telemetry->next_sequence = 1u;
    telemetry->metrics.schema_version = YVEX_RUNTIME_METRICS_SCHEMA_VERSION;
    (void)clock_gettime(CLOCK_MONOTONIC, &telemetry->started);
    if (pthread_mutex_init(&telemetry->mutex, NULL) != 0) {
        free(telemetry->events);
        free(telemetry);
        yvex_error_set(err, YVEX_ERR_STATE, "server.telemetry.open",
                       "telemetry mutex initialization failed");
        return YVEX_ERR_STATE;
    }
    telemetry->mutex_ready = 1;
    if (pthread_cond_init(&telemetry->condition, NULL) != 0) {
        (void)pthread_mutex_destroy(&telemetry->mutex);
        free(telemetry->events);
        free(telemetry);
        yvex_error_set(err, YVEX_ERR_STATE, "server.telemetry.open",
                       "telemetry condition initialization failed");
        return YVEX_ERR_STATE;
    }
    telemetry->condition_ready = 1;
    *out = telemetry;
    yvex_error_clear(err);
    return YVEX_OK;
}
/*
 * Publish one authoritative event into the bounded global sequence.
 *
 * Refuses invalid ownership or identity derivation.
 */
int yvex_server_telemetry_emit_provider(
    server_telemetry *telemetry, const server_event_scope *scope,
    yvex_server_event_kind kind,
    yvex_server_event_severity severity, const char *session_id,
    const char *request_id, const char *turn_id, const char *phase,
    unsigned long long value_a, unsigned long long value_b,
    unsigned long long value_c, double seconds, double rate,
    const yvex_runtime_speculation_progress *speculation,
    const yvex_provider_request *provider, yvex_server_event *emitted,
    yvex_error *err)
{
    yvex_server_event event;
    int ring_full;
    if (!telemetry || kind > YVEX_SERVER_EVENT_RUNTIME_SHUTDOWN_COMPLETE ||
        severity > YVEX_SERVER_SEVERITY_FATAL ||
        (scope &&
         (scope->engine_kind == YVEX_SERVER_ENGINE_NONE ||
          scope->engine_kind > YVEX_SERVER_ENGINE_MEDIA ||
          scope->execution_strategy > YVEX_SERVER_EXECUTION_SPECULATIVE ||
          (scope->engine_kind == YVEX_SERVER_ENGINE_TEXT &&
           scope->execution_strategy ==
               YVEX_SERVER_EXECUTION_NOT_APPLICABLE) ||
          (scope->engine_kind == YVEX_SERVER_ENGINE_MEDIA &&
           scope->execution_strategy !=
               YVEX_SERVER_EXECUTION_NOT_APPLICABLE) ||
          (scope->runtime_model_identity[0] &&
           !yvex_sha256_hex_valid(scope->runtime_model_identity)) ||
          (scope->artifact_identity[0] &&
           !yvex_sha256_hex_valid(scope->artifact_identity)) ||
          (scope->specialization_identity[0] &&
           !yvex_sha256_hex_valid(scope->specialization_identity))))) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "server.telemetry.emit",
                       "valid telemetry owner and event facts are required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (provider &&
        (!provider->adapter[0] ||
         yvex_provider_request_validate(provider, err) != YVEX_OK)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "server.telemetry.emit",
                       "sealed provider correlation facts are required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (speculation &&
        (speculation->schema_version != YVEX_RUNTIME_GENERATION_SCHEMA_V3 ||
         speculation->kind > YVEX_SPECULATION_PROGRESS_CYCLE_COMMITTED ||
         speculation->confidence_logit_count > speculation->proposed_tokens ||
         !isfinite(speculation->confidence_logit_minimum) ||
         !isfinite(speculation->confidence_logit_maximum) ||
         !isfinite(speculation->confidence_logit_mean) ||
         (speculation->confidence_logit_count &&
          (speculation->confidence_logit_minimum >
               speculation->confidence_logit_mean ||
           speculation->confidence_logit_mean >
               speculation->confidence_logit_maximum)) ||
         (!speculation->confidence_logit_count &&
          (speculation->confidence_logit_minimum ||
           speculation->confidence_logit_maximum ||
           speculation->confidence_logit_mean)) ||
         !yvex_sha256_hex_valid(speculation->policy_identity))) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "server.telemetry.emit",
                       "complete typed speculation progress is required");
        return YVEX_ERR_INVALID_ARG;
    }
    memset(&event, 0, sizeof(event));
    event.schema_version = YVEX_RUNTIME_EVENT_SCHEMA_VERSION;
    event.wall_time_ns = time_ns(CLOCK_REALTIME);
    event.monotonic_time_ns = time_ns(CLOCK_MONOTONIC);
    event.process_id = (unsigned long long)getpid();
    event.kind = kind;
    event.severity = severity;
    event.value_a = value_a;
    event.value_b = value_b;
    event.value_c = value_c;
    event.seconds = seconds;
    event.rate = rate;
    event.engine_kind = scope ? scope->engine_kind : YVEX_SERVER_ENGINE_NONE;
    event.execution_strategy = scope
                                   ? scope->execution_strategy
                                   : YVEX_SERVER_EXECUTION_NOT_APPLICABLE;
    if (speculation) {
        event.speculative_cycle = speculation->cycle;
        event.proposed_tokens = speculation->proposed_tokens;
        event.selected_verification_tokens =
            speculation->selected_verification_tokens;
        event.accepted_tokens = speculation->accepted_tokens;
        event.rejected_tokens = speculation->rejected_tokens;
        event.discarded_tokens = speculation->discarded_tokens;
        event.verification_count = speculation->verification_count;
        event.confidence_logit_count = speculation->confidence_logit_count;
        event.confidence_logit_minimum =
            speculation->confidence_logit_minimum;
        event.confidence_logit_maximum =
            speculation->confidence_logit_maximum;
        event.confidence_logit_mean = speculation->confidence_logit_mean;
        yvex_runtime_identity_copy(event.speculation_policy_identity,
                                   speculation->policy_identity);
    }
    yvex_core_text_copy(event.session_id, sizeof(event.session_id),
                        session_id ? session_id : "");
    yvex_core_text_copy(event.request_id, sizeof(event.request_id),
                        request_id ? request_id : "");
    yvex_core_text_copy(event.turn_id, sizeof(event.turn_id),
                        turn_id ? turn_id : "");
    yvex_core_text_copy(event.phase, sizeof(event.phase), phase ? phase : "");
    if (provider) {
        yvex_core_text_copy(event.provider_adapter,
                            sizeof(event.provider_adapter), provider->adapter);
        yvex_core_text_copy(event.provider_request_identity,
                            sizeof(event.provider_request_identity),
                            provider->request_identity);
        yvex_core_text_copy(event.external_correlation_id,
                            sizeof(event.external_correlation_id),
                            provider->external_correlation_id);
    }
    if (scope) {
        yvex_core_text_copy(event.runtime_model_identity,
                            sizeof(event.runtime_model_identity),
                            scope->runtime_model_identity);
        yvex_core_text_copy(event.artifact_identity,
                            sizeof(event.artifact_identity),
                            scope->artifact_identity);
        yvex_core_text_copy(event.variant_identity,
                            sizeof(event.variant_identity),
                            scope->specialization_identity);
    }
    if (pthread_mutex_lock(&telemetry->mutex) != 0) {
        yvex_error_set(err, YVEX_ERR_STATE, "server.telemetry.emit",
                       "telemetry mutex acquisition failed");
        return YVEX_ERR_STATE;
    }
    if (telemetry->closing) {
        (void)pthread_mutex_unlock(&telemetry->mutex);
        yvex_error_set(err, YVEX_ERR_STATE, "server.telemetry.emit",
                       "telemetry is closing");
        return YVEX_ERR_STATE;
    }
    ring_full = telemetry->retained_count == telemetry->capacity;
    if (ring_full &&
        kind != YVEX_SERVER_EVENT_TELEMETRY_DROPPED) {
        yvex_server_event dropped = event;
        dropped.kind = YVEX_SERVER_EVENT_TELEMETRY_DROPPED;
        dropped.severity = YVEX_SERVER_SEVERITY_WARNING;
        dropped.value_a = telemetry->metrics.telemetry_dropped + 2u;
        dropped.value_b = telemetry->capacity;
        dropped.value_c = 0u;
        dropped.speculative_cycle = 0u;
        dropped.proposed_tokens = 0u;
        dropped.selected_verification_tokens = 0u;
        dropped.accepted_tokens = 0u;
        dropped.rejected_tokens = 0u;
        dropped.discarded_tokens = 0u;
        dropped.verification_count = 0u;
        dropped.confidence_logit_count = 0u;
        dropped.confidence_logit_minimum = 0.0;
        dropped.confidence_logit_maximum = 0.0;
        dropped.confidence_logit_mean = 0.0;
        dropped.seconds = 0.0;
        dropped.rate = 0.0;
        dropped.speculation_policy_identity[0] = '\0';
        yvex_core_text_copy(dropped.phase, sizeof(dropped.phase), "telemetry");
        telemetry->metrics.telemetry_dropped += 2u;
        if (!event_append_locked(telemetry, &dropped) ||
            !event_append_locked(telemetry, &event)) {
            (void)pthread_mutex_unlock(&telemetry->mutex);
            yvex_error_set(err, YVEX_ERR_STATE, "server.telemetry.emit",
                           "event identity derivation failed");
            return YVEX_ERR_STATE;
        }
    } else if (!event_append_locked(telemetry, &event)) {
        (void)pthread_mutex_unlock(&telemetry->mutex);
        yvex_error_set(err, YVEX_ERR_STATE, "server.telemetry.emit",
                       "event identity derivation failed");
        return YVEX_ERR_STATE;
    } else if (ring_full && kind == YVEX_SERVER_EVENT_TELEMETRY_DROPPED) {
        telemetry->metrics.telemetry_dropped++;
    }
    if (emitted) *emitted = event;
    (void)pthread_cond_broadcast(&telemetry->condition);
    (void)pthread_mutex_unlock(&telemetry->mutex);
    yvex_error_clear(err);
    return YVEX_OK;
}
/*
 * Publish one native event without application-provider correlation.
 *
 * Appends one identity-sealed event.
 */
int yvex_server_telemetry_emit(server_telemetry *telemetry,
                               const server_event_scope *scope,
                               yvex_server_event_kind kind,
                               yvex_server_event_severity severity,
                               const char *session_id, const char *request_id,
                               const char *turn_id, const char *phase,
                               unsigned long long value_a,
                               unsigned long long value_b,
                               unsigned long long value_c, double seconds,
                               double rate, yvex_error *err)
{
    return yvex_server_telemetry_emit_provider(
        telemetry, scope, kind, severity, session_id, request_id, turn_id, phase,
        value_a, value_b, value_c, seconds, rate, NULL, NULL, NULL, err);
}

int yvex_server_telemetry_next(server_telemetry *telemetry,
                          unsigned long long after_sequence, int wait,
                          yvex_server_event *event, yvex_error *err)
{
    unsigned long long first, wanted;
    if (!telemetry || !event || pthread_mutex_lock(&telemetry->mutex) != 0) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "server.telemetry.next",
                       "telemetry and event output are required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (telemetry->closing) {
        (void)pthread_mutex_unlock(&telemetry->mutex);
        yvex_error_set(err, YVEX_ERR_STATE, "server.telemetry.next",
                       "telemetry is closed");
        return YVEX_ERR_STATE;
    }
    telemetry->active_subscribers++;
    while (wait && !telemetry->closing &&
           after_sequence >= telemetry->next_sequence - 1u)
        if (pthread_cond_wait(&telemetry->condition, &telemetry->mutex) != 0)
            break;
    first = telemetry->next_sequence - telemetry->retained_count;
    wanted = after_sequence + 1u;
    if (wanted < first)
        wanted = first;
    if (wanted >= telemetry->next_sequence) {
        if (telemetry->active_subscribers) telemetry->active_subscribers--;
        if (telemetry->closing)
            (void)pthread_cond_broadcast(&telemetry->condition);
        (void)pthread_mutex_unlock(&telemetry->mutex);
        yvex_error_set(err, YVEX_ERR_STATE,
                       "server.telemetry.next",
                       telemetry->closing ? "telemetry is closed" : "no later event is available");
        return YVEX_ERR_STATE;
    }
    *event = telemetry->events[(wanted - 1u) % telemetry->capacity];
    if (telemetry->active_subscribers) telemetry->active_subscribers--;
    if (telemetry->closing)
        (void)pthread_cond_broadcast(&telemetry->condition);
    (void)pthread_mutex_unlock(&telemetry->mutex);
    yvex_error_clear(err);
    return YVEX_OK;
}
/*
 * Copy current process metrics from the same authority as events.
 *
 * Refuses absent ownership.
 */
int yvex_server_telemetry_metrics_copy(server_telemetry *telemetry,
                                  yvex_server_metrics *metrics,
                                  yvex_error *err)
{
    struct rusage usage;
    FILE *status;
    unsigned long long pages = 0u, resident_pages = 0u;
    unsigned long long now;
    if (!telemetry || !metrics || pthread_mutex_lock(&telemetry->mutex) != 0) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "server.telemetry.metrics",
                       "telemetry and metrics output are required");
        return YVEX_ERR_INVALID_ARG;
    }
    *metrics = telemetry->metrics;
    now = time_ns(CLOCK_MONOTONIC);
    metrics->uptime_ns = now >= (unsigned long long)telemetry->started.tv_sec * 1000000000ull +
                                   (unsigned long long)telemetry->started.tv_nsec
                             ? now - ((unsigned long long)telemetry->started.tv_sec * 1000000000ull +
                                      (unsigned long long)telemetry->started.tv_nsec)
                             : 0u;
    status = fopen("/proc/self/statm", "r");
    if (status) {
        if (fscanf(status, "%llu %llu", &pages, &resident_pages) == 2) {
            long page_size = sysconf(_SC_PAGESIZE);
            if (page_size > 0 &&
                resident_pages <= ULLONG_MAX / (unsigned long long)page_size)
                metrics->current_rss_bytes =
                    resident_pages * (unsigned long long)page_size;
        }
        (void)fclose(status);
    }
    if (getrusage(RUSAGE_SELF, &usage) == 0 && usage.ru_maxrss > 0 &&
        (unsigned long long)usage.ru_maxrss <= ULLONG_MAX / 1024u)
        metrics->peak_rss_bytes =
            (unsigned long long)usage.ru_maxrss * 1024u;
    if (metrics->peak_rss_bytes < metrics->current_rss_bytes)
        metrics->peak_rss_bytes = metrics->current_rss_bytes;
    (void)pthread_mutex_unlock(&telemetry->mutex);
    yvex_error_clear(err);
    return YVEX_OK;
}

/*
 * Account one process-lifetime model admission.
 *
 * Telemetry, mapped/resident bytes, and elapsed startup time.
 */
void yvex_server_telemetry_model_opened(server_telemetry *telemetry,
                                   unsigned long long mapped_artifact_bytes,
                                   unsigned long long host_bytes,
                                   unsigned long long device_bytes,
                                   unsigned long long uploads)
{
    if (!telemetry || pthread_mutex_lock(&telemetry->mutex) != 0) return;
    telemetry->metrics.model_open_count++;
    telemetry->metrics.artifact_open_count++;
    telemetry->metrics.binding_open_count++;
    telemetry->metrics.materialization_count++;
    telemetry->metrics.residency_build_count++;
    telemetry->metrics.mapped_artifact_bytes = mapped_artifact_bytes;
    telemetry->metrics.resident_host_bytes = host_bytes;
    telemetry->metrics.resident_device_bytes = device_bytes;
    telemetry->metrics.output_head_upload_count = uploads;
    (void)pthread_mutex_unlock(&telemetry->mutex);
}

/* Account one admitted composite model without claiming payload materialization or residency. */
void yvex_server_telemetry_media_model_opened(
    server_telemetry *telemetry, unsigned long long artifact_count)
{
    if (!telemetry || !artifact_count || pthread_mutex_lock(&telemetry->mutex) != 0) return;
    telemetry->metrics.model_open_count++;
    telemetry->metrics.artifact_open_count += artifact_count;
    telemetry->metrics.binding_open_count++;
    (void)pthread_mutex_unlock(&telemetry->mutex);
}
/* Account the single process-lifetime model discharge. */
void yvex_server_telemetry_model_closed(server_telemetry *telemetry)
{
    if (!telemetry || pthread_mutex_lock(&telemetry->mutex) != 0) return;
    telemetry->metrics.model_close_count++;
    if (telemetry->metrics.model_close_count ==
        telemetry->metrics.model_open_count) {
        telemetry->metrics.mapped_artifact_bytes = 0ull;
        telemetry->metrics.resident_host_bytes = 0ull;
        telemetry->metrics.resident_device_bytes = 0ull;
    }
    (void)pthread_mutex_unlock(&telemetry->mutex);
}
/* Raise process-resident resource evidence from one authoritative runtime session. */
void yvex_server_telemetry_resources(server_telemetry *telemetry,
                                     unsigned long long host_bytes,
                                     unsigned long long device_bytes,
                                     unsigned long long uploads)
{
    if (!telemetry || pthread_mutex_lock(&telemetry->mutex) != 0) return;
    if (host_bytes > telemetry->metrics.resident_host_bytes)
        telemetry->metrics.resident_host_bytes = host_bytes;
    if (device_bytes > telemetry->metrics.resident_device_bytes)
        telemetry->metrics.resident_device_bytes = device_bytes;
    if (uploads > telemetry->metrics.output_head_upload_count)
        telemetry->metrics.output_head_upload_count = uploads;
    (void)pthread_mutex_unlock(&telemetry->mutex);
}

void yvex_server_telemetry_queue(server_telemetry *telemetry,
                            unsigned long long depth,
                            unsigned long long capacity)
{
    if (!telemetry || pthread_mutex_lock(&telemetry->mutex) != 0) return;
    telemetry->metrics.queue_depth = depth;
    telemetry->metrics.queue_capacity = capacity;
    (void)pthread_mutex_unlock(&telemetry->mutex);
}

void yvex_server_telemetry_session(server_telemetry *telemetry, int active_delta,
                              int created)
{
    if (!telemetry || pthread_mutex_lock(&telemetry->mutex) != 0) return;
    if (active_delta > 0) telemetry->metrics.active_sessions++;
    if (active_delta < 0 && telemetry->metrics.active_sessions)
        telemetry->metrics.active_sessions--;
    if (created) telemetry->metrics.total_sessions++;
    (void)pthread_mutex_unlock(&telemetry->mutex);
}

void yvex_server_telemetry_request(server_telemetry *telemetry, int active_delta,
                              int completed, int failed, int cancelled)
{
    if (!telemetry || pthread_mutex_lock(&telemetry->mutex) != 0) return;
    if (active_delta > 0) telemetry->metrics.active_requests++;
    if (active_delta < 0 && telemetry->metrics.active_requests)
        telemetry->metrics.active_requests--;
    telemetry->metrics.completed_requests += completed != 0;
    telemetry->metrics.failed_requests += failed != 0;
    telemetry->metrics.cancelled_requests += cancelled != 0;
    (void)pthread_mutex_unlock(&telemetry->mutex);
}

void yvex_server_telemetry_openai_request(server_telemetry *telemetry,
                                          int active_delta, int completed,
                                          int failed, int cancelled)
{
    if (!telemetry || pthread_mutex_lock(&telemetry->mutex) != 0) return;
    if (active_delta > 0) telemetry->metrics.active_http_requests++;
    if (active_delta < 0 && telemetry->metrics.active_http_requests)
        telemetry->metrics.active_http_requests--;
    telemetry->metrics.completed_http_requests += completed != 0;
    telemetry->metrics.failed_http_requests += failed != 0;
    telemetry->metrics.cancelled_http_requests += cancelled != 0;
    (void)pthread_mutex_unlock(&telemetry->mutex);
}

void yvex_server_telemetry_close(server_telemetry **telemetry)
{
    server_telemetry *owner;
    if (!telemetry || !*telemetry) return;
    owner = *telemetry;
    if (owner->mutex_ready && pthread_mutex_lock(&owner->mutex) == 0) {
        owner->closing = 1;
        if (owner->condition_ready) (void)pthread_cond_broadcast(&owner->condition);
        while (owner->active_subscribers && owner->condition_ready)
            if (pthread_cond_wait(&owner->condition, &owner->mutex) != 0)
                break;
        (void)pthread_mutex_unlock(&owner->mutex);
    }
    if (owner->condition_ready) (void)pthread_cond_destroy(&owner->condition);
    if (owner->mutex_ready) (void)pthread_mutex_destroy(&owner->mutex);
    free(owner->events);
    memset(owner, 0, sizeof(*owner));
    free(owner);
    *telemetry = NULL;
}

const char *yvex_server_event_kind_name(yvex_server_event_kind kind)
{
    static const char *const names[] = {
        "process.start", "telemetry.ready", "artifact.open.start",
        "artifact.open.complete", "binding.admitted", "materialization.start",
        "materialization.complete", "residency.ready", "runtime.ready",
        "listener.ready", "session.created", "session.attached",
        "session.detached", "session.reset", "session.closed",
        "request.received", "request.queued", "request.started",
        "tokenizer.completed", "prefill.started", "prefill.progress",
        "prefill.completed", "draft.started", "draft.completed",
        "verification.started", "verification.completed", "prefix.accepted",
        "candidate.rejected", "speculative.cycle.committed",
        "generation.first_token", "generation.fragment",
        "generation.progress", "generation.profile", "generation.completed",
        "generation.cancelled", "generation.failed", "client.disconnected", "telemetry.dropped",
        "runtime.shutdown.start", "runtime.shutdown.complete"};
    return (unsigned int)kind < sizeof(names) / sizeof(names[0])
               ? names[kind] : "unknown";
}

const char *yvex_server_session_state_name(yvex_server_session_state state)
{
    static const char *const names[] = {
        "created", "ready", "running", "partial", "detached",
        "resetting", "closing", "closed", "failed"};
    return (unsigned int)state < sizeof(names) / sizeof(names[0])
               ? names[state] : "unknown";
}
/*
 * Independently recompute one event identity before client or renderer use.
 *
 * Refuses schema, sequence, name, or identity mismatch.
 */
int yvex_server_event_validate(const yvex_server_event *event, yvex_error *err)
{
    yvex_server_event candidate;
    char supplied[YVEX_SHA256_HEX_CAP];
    int speculative;
    speculative = event && event->kind >= YVEX_SERVER_EVENT_DRAFT_STARTED &&
                  event->kind <=
                      YVEX_SERVER_EVENT_SPECULATIVE_CYCLE_COMMITTED;
    if (!event || event->schema_version != YVEX_RUNTIME_EVENT_SCHEMA_VERSION ||
        event->kind > YVEX_SERVER_EVENT_RUNTIME_SHUTDOWN_COMPLETE ||
        event->severity > YVEX_SERVER_SEVERITY_FATAL ||
        (event->provider_adapter[0] &&
         (!yvex_sha256_hex_valid(event->provider_request_identity) ||
          !event->external_correlation_id[0])) ||
        (!event->provider_adapter[0] &&
         (event->provider_request_identity[0] ||
          event->external_correlation_id[0])) ||
        event->engine_kind > YVEX_SERVER_ENGINE_MEDIA ||
        event->execution_strategy > YVEX_SERVER_EXECUTION_SPECULATIVE ||
        (event->engine_kind == YVEX_SERVER_ENGINE_TEXT &&
         event->execution_strategy ==
             YVEX_SERVER_EXECUTION_NOT_APPLICABLE) ||
        (event->engine_kind != YVEX_SERVER_ENGINE_TEXT &&
         event->execution_strategy !=
             YVEX_SERVER_EXECUTION_NOT_APPLICABLE) ||
        !isfinite(event->confidence_logit_minimum) ||
        !isfinite(event->confidence_logit_maximum) ||
        !isfinite(event->confidence_logit_mean) ||
        !isfinite(event->seconds) || !isfinite(event->rate) ||
        (speculative &&
         (event->engine_kind != YVEX_SERVER_ENGINE_TEXT ||
          event->execution_strategy != YVEX_SERVER_EXECUTION_SPECULATIVE ||
          !event->speculative_cycle ||
          event->confidence_logit_count > event->proposed_tokens ||
          (event->confidence_logit_count &&
           (event->confidence_logit_minimum >
                event->confidence_logit_mean ||
            event->confidence_logit_mean >
                event->confidence_logit_maximum)) ||
          (!event->confidence_logit_count &&
           (event->confidence_logit_minimum ||
            event->confidence_logit_maximum ||
            event->confidence_logit_mean)) ||
          !yvex_sha256_hex_valid(event->speculation_policy_identity))) ||
        (!speculative &&
         (event->speculative_cycle || event->proposed_tokens ||
          event->selected_verification_tokens || event->accepted_tokens ||
          event->rejected_tokens || event->discarded_tokens ||
          event->verification_count || event->confidence_logit_count ||
          event->confidence_logit_minimum ||
          event->confidence_logit_maximum ||
          event->confidence_logit_mean ||
          event->speculation_policy_identity[0])) ||
        !yvex_sha256_hex_valid(event->event_identity)) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "server.telemetry.validate",
                       "complete versioned event evidence is required");
        return YVEX_ERR_FORMAT;
    }
    candidate = *event;
    yvex_core_text_copy(supplied, sizeof(supplied), event->event_identity);
    if (!event_identity(&candidate) ||
        strcmp(candidate.event_identity, supplied) != 0) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "server.telemetry.validate",
                       "runtime event evidence identity does not match its fields");
        return YVEX_ERR_FORMAT;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}
/*
 * Render one typed event as a bounded privacy-preserving JSONL record.
 *
 * Refuses invalid identity or insufficient output capacity.
 */
int yvex_server_event_json(const yvex_server_event *event, char *output,
                           unsigned long long capacity, yvex_error *err)
{
    int length;
    if (!output || !capacity ||
        yvex_server_event_validate(event, err) != YVEX_OK) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "server.telemetry.json",
                       "sealed event and output capacity are required");
        return YVEX_ERR_INVALID_ARG;
    }
    length = snprintf(output, (size_t)capacity,
                      "{\"schema\":4,\"sequence\":%llu,\"process\":%llu,"
                      "\"wall_time_ns\":%llu,\"monotonic_time_ns\":%llu,\"kind\":\"%s\","
                      "\"severity\":%u,\"session\":\"%s\",\"request\":\"%s\","
                      "\"turn\":\"%s\",\"phase\":\"%s\","
                      "\"provider\":\"%s\",\"provider_request_identity\":\"%s\","
                      "\"external_correlation_id\":\"%s\",\"a\":%llu,"
                      "\"b\":%llu,\"c\":%llu,\"engine_kind\":%u,"
                      "\"execution_strategy\":%u,"
                      "\"speculative_cycle\":%llu,\"proposed_tokens\":%llu,"
                      "\"selected_verification_tokens\":%llu,"
                      "\"accepted_tokens\":%llu,\"rejected_tokens\":%llu,"
                      "\"discarded_tokens\":%llu,"
                      "\"verification_count\":%llu,"
                      "\"confidence_logit_count\":%llu,"
                      "\"confidence_logit_minimum\":%.9g,"
                      "\"confidence_logit_maximum\":%.9g,"
                      "\"confidence_logit_mean\":%.9g,"
                      "\"speculation_policy_identity\":\"%s\","
                      "\"seconds\":%.9g,\"rate\":%.9g,"
                      "\"runtime_model_identity\":\"%s\","
                      "\"artifact_identity\":\"%s\",\"variant_identity\":\"%s\","
                      "\"identity\":\"%s\"}\n",
                      event->sequence, event->process_id, event->wall_time_ns,
                      event->monotonic_time_ns,
                      yvex_server_event_kind_name(event->kind),
                      (unsigned int)event->severity, event->session_id,
                      event->request_id, event->turn_id, event->phase,
                      event->provider_adapter,
                      event->provider_request_identity,
                      event->external_correlation_id,
                      event->value_a, event->value_b, event->value_c,
                      (unsigned int)event->engine_kind,
                      (unsigned int)event->execution_strategy,
                      event->speculative_cycle, event->proposed_tokens,
                      event->selected_verification_tokens,
                      event->accepted_tokens, event->rejected_tokens,
                      event->discarded_tokens,
                      event->verification_count,
                      event->confidence_logit_count,
                      event->confidence_logit_minimum,
                      event->confidence_logit_maximum,
                      event->confidence_logit_mean,
                      event->speculation_policy_identity,
                      event->seconds, event->rate,
                      event->runtime_model_identity, event->artifact_identity,
                      event->variant_identity, event->event_identity);
    if (length < 0 || (unsigned long long)length >= capacity) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "server.telemetry.json",
                       "event JSON output capacity is insufficient");
        return YVEX_ERR_BOUNDS;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}
