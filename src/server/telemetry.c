/* Owner: server.telemetry.
 * Owns: one bounded typed event ring, canonical event identities, metrics, and JSONL projection.
 * Does not own: runtime/session decisions, client sockets, terminal layout, or trace content policy.
 * Invariants: raw and operational consumers observe the same global event sequence and counts.
 * Boundary: authoritative server event fan-out and process metrics accumulator.
 * Purpose: make runtime/session/request transitions observable without prose scraping.
 * Inputs: authoritative owner events and bounded identity/count facts.
 * Effects: appends/coalesces events, updates metrics, and wakes subscribers.
 * Failure: critical event publication refuses; overwritten low-priority history increments drops. */
#define _POSIX_C_SOURCE 200809L

#include "src/server/private.h"

#include <limits.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

#include <yvex/internal/core.h>

#define TELEMETRY_SCHEMA_V2 2u

struct server_telemetry {
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    yvex_server_event *events;
    unsigned long long capacity, next_sequence, retained_count;
    struct timespec started;
    yvex_server_metrics metrics;
    char runtime_model_identity[YVEX_SHA256_HEX_CAP];
    char artifact_identity[YVEX_SHA256_HEX_CAP];
    char variant_identity[YVEX_SHA256_HEX_CAP];
    unsigned long long active_subscribers;
    int mutex_ready, condition_ready, closing;
};

static int event_identity(yvex_server_event *event);

/* Purpose: seal and append one event while the telemetry lifecycle gate is held. */
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

/* Purpose: append one finite binary64 value through explicit canonical bits.
 * Inputs: hash state and numeric value. Effects: extends the hash with canonical bytes.
 * Failure: returns false for non-finite input or hash failure. Boundary: never hashes native padding. */
static int hash_double(yvex_sha256 *hash, double value)
{
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return yvex_sha256_update_u64(hash, bits);
}

/* Purpose: return nanoseconds for one admitted clock without overflow for realistic uptime. */
static unsigned long long time_ns(clockid_t clock)
{
    struct timespec value;
    if (clock_gettime(clock, &value) != 0)
        return 0u;
    return (unsigned long long)value.tv_sec * 1000000000ull +
           (unsigned long long)value.tv_nsec;
}

/* Purpose: derive one event identity field by field.
 * Inputs: complete typed event and identity output. Effects: writes a canonical SHA-256 identity.
 * Failure: returns false for invalid numeric or hash input. Boundary: timestamps participate as event facts. */
static int event_identity(yvex_server_event *event)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    if (!event)
        return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.server.event.v2") ||
        !yvex_sha256_update_u64(&hash, event->schema_version) ||
        !yvex_sha256_update_u64(&hash, event->sequence) ||
        !yvex_sha256_update_u64(&hash, event->wall_time_ns) ||
        !yvex_sha256_update_u64(&hash, event->monotonic_time_ns) ||
        !yvex_sha256_update_u64(&hash, event->process_id) ||
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
        !hash_double(&hash, event->seconds) ||
        !hash_double(&hash, event->rate) ||
        !yvex_sha256_update_text(&hash, event->runtime_model_identity) ||
        !yvex_sha256_update_text(&hash, event->artifact_identity) ||
        !yvex_sha256_update_text(&hash, event->variant_identity) ||
        !yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, event->event_identity);
    return 1;
}

/* Purpose: allocate the only process event and metrics authority.
 * Inputs: owner output, bounded capacity, optional clocks/context, and error output. Effects: allocates ring and locks.
 * Failure: closes partial synchronization and publishes no owner. Boundary: no renderer or runtime host is created. */
int yvex_server_telemetry_open(server_telemetry **out, unsigned long long capacity,
                          const char *runtime_model_identity,
                          const char *artifact_identity,
                          const char *variant_identity, yvex_error *err)
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
    telemetry->metrics.schema_version = TELEMETRY_SCHEMA_V2;
    (void)clock_gettime(CLOCK_MONOTONIC, &telemetry->started);
    yvex_core_text_copy(telemetry->runtime_model_identity,
                        sizeof(telemetry->runtime_model_identity),
                        runtime_model_identity ? runtime_model_identity : "");
    yvex_core_text_copy(telemetry->artifact_identity,
                        sizeof(telemetry->artifact_identity),
                        artifact_identity ? artifact_identity : "");
    yvex_core_text_copy(telemetry->variant_identity,
                        sizeof(telemetry->variant_identity),
                        variant_identity ? variant_identity : "");
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

/* Purpose: publish one authoritative event into the bounded global sequence.
 * Inputs: telemetry, typed facts, identities, counters, timing, and error output. Effects: seals and appends one event.
 * Failure: refuses invalid ownership or identity derivation. Boundary: content bytes are excluded by schema. */
int yvex_server_telemetry_emit_provider(
    server_telemetry *telemetry, yvex_server_event_kind kind,
    yvex_server_event_severity severity, const char *session_id,
    const char *request_id, const char *turn_id, const char *phase,
    unsigned long long value_a, unsigned long long value_b,
    unsigned long long value_c, double seconds, double rate,
    const yvex_provider_request *provider, yvex_error *err)
{
    yvex_server_event event;
    int ring_full;
    if (!telemetry || kind > YVEX_SERVER_EVENT_RUNTIME_SHUTDOWN_COMPLETE ||
        severity > YVEX_SERVER_SEVERITY_FATAL) {
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
    memset(&event, 0, sizeof(event));
    event.schema_version = TELEMETRY_SCHEMA_V2;
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
    yvex_core_text_copy(event.runtime_model_identity,
                        sizeof(event.runtime_model_identity),
                        telemetry->runtime_model_identity);
    yvex_core_text_copy(event.artifact_identity, sizeof(event.artifact_identity),
                        telemetry->artifact_identity);
    yvex_core_text_copy(event.variant_identity, sizeof(event.variant_identity),
                        telemetry->variant_identity);
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
        dropped.seconds = 0.0;
        dropped.rate = 0.0;
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
    (void)pthread_cond_broadcast(&telemetry->condition);
    (void)pthread_mutex_unlock(&telemetry->mutex);
    yvex_error_clear(err);
    return YVEX_OK;
}

/* Purpose: publish one native event without application-provider correlation.
 * Inputs: typed server facts and error output. Effects: appends one identity-sealed event.
 * Failure: forwards the authoritative telemetry refusal. Boundary: provider fields remain empty. */
int yvex_server_telemetry_emit(server_telemetry *telemetry,
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
        telemetry, kind, severity, session_id, request_id, turn_id, phase,
        value_a, value_b, value_c, seconds, rate, NULL, err);
}

/* Purpose: read the first retained event after a cursor, optionally waiting for publication.
 * Inputs: telemetry, sequence cursor, wait policy, event output, and error output. Effects: may wait and copies event.
 * Failure: refuses closed authority or stale cursor beyond retained history. Boundary: no subscriber-owned queue. */
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

/* Purpose: copy current process metrics from the same authority as events.
 * Inputs: telemetry, metrics output, and error output. Effects: briefly locks and writes snapshot.
 * Failure: refuses absent ownership. Boundary: operational metrics are not benchmark evidence. */
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

/* Purpose: bind admitted model identities before runtime-ready publication.
 * Inputs: telemetry and exact model/artifact/variant identities. Effects: records immutable process facts.
 * Failure: none for admitted fixed-size identities. Boundary: callers authenticate identities before binding. */
void yvex_server_telemetry_identities(server_telemetry *telemetry,
                                 const char *runtime_model_identity,
                                 const char *artifact_identity,
                                 const char *variant_identity)
{
    if (!telemetry || pthread_mutex_lock(&telemetry->mutex) != 0) return;
    yvex_core_text_copy(telemetry->runtime_model_identity,
                        sizeof(telemetry->runtime_model_identity),
                        runtime_model_identity ? runtime_model_identity : "");
    yvex_core_text_copy(telemetry->artifact_identity,
                        sizeof(telemetry->artifact_identity),
                        artifact_identity ? artifact_identity : "");
    yvex_core_text_copy(telemetry->variant_identity,
                        sizeof(telemetry->variant_identity),
                        variant_identity ? variant_identity : "");
    (void)pthread_mutex_unlock(&telemetry->mutex);
}

/* Purpose: account one process-lifetime model admission.
 * Inputs: telemetry, mapped/resident bytes, and elapsed startup time. Effects: updates exact process counters.
 * Failure: none after telemetry admission. Boundary: count does not itself establish model correctness. */
void yvex_server_telemetry_model_opened(server_telemetry *telemetry,
                                   unsigned long long artifact_bytes,
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
    telemetry->metrics.mapped_artifact_bytes = artifact_bytes;
    telemetry->metrics.resident_host_bytes = host_bytes;
    telemetry->metrics.resident_device_bytes = device_bytes;
    telemetry->metrics.output_head_upload_count = uploads;
    (void)pthread_mutex_unlock(&telemetry->mutex);
}

/* Purpose: account the single process-lifetime model discharge.
 * Inputs: telemetry owner. Effects: increments model-close count under lock.
 * Failure: none for absent telemetry. Boundary: does not close model resources. */
void yvex_server_telemetry_model_closed(server_telemetry *telemetry)
{
    if (!telemetry || pthread_mutex_lock(&telemetry->mutex) != 0) return;
    telemetry->metrics.model_close_count++;
    (void)pthread_mutex_unlock(&telemetry->mutex);
}

/* Purpose: raise process-resident resource evidence from one authoritative runtime session.
 * Inputs: host/device high-water extents and upload count. Effects: updates monotonic metrics.
 * Failure: absent telemetry is ignored. Boundary: values remain operational, not benchmark evidence. */
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

/* Purpose: replace authoritative bounded queue occupancy facts.
 * Inputs: telemetry, current depth, and capacity. Effects: updates queue gauges and peak.
 * Failure: none for absent telemetry. Boundary: queue enforcement belongs to host. */
void yvex_server_telemetry_queue(server_telemetry *telemetry,
                            unsigned long long depth,
                            unsigned long long capacity)
{
    if (!telemetry || pthread_mutex_lock(&telemetry->mutex) != 0) return;
    telemetry->metrics.queue_depth = depth;
    telemetry->metrics.queue_capacity = capacity;
    (void)pthread_mutex_unlock(&telemetry->mutex);
}

/* Purpose: account exact session registry membership transitions.
 * Inputs: telemetry, signed active delta, and created fact. Effects: updates session counters.
 * Failure: clamps impossible negative gauges. Boundary: registry remains session authority. */
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

/* Purpose: account exact request lifecycle transitions.
 * Inputs: telemetry, signed active delta, and terminal classification. Effects: updates request counters.
 * Failure: clamps impossible negative gauges. Boundary: request state remains host authority. */
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

/* Purpose: transfer and release the sole telemetry owner after waking subscribers.
 * Inputs: unique owner pointer. Effects: marks closed, wakes readers, destroys synchronization, and frees ring.
 * Failure: cleanup errors are secondary in this destructor. Boundary: caller pointer becomes NULL. */
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

/* Purpose: return one stable event spelling shared by raw and operational renderers.
 * Inputs: event enumeration. Effects: none. Failure: returns unknown for unsupported value.
 * Boundary: spelling cannot create event capability. */
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
        "prefill.completed", "generation.first_token", "generation.fragment",
        "generation.progress", "generation.completed", "generation.cancelled",
        "generation.failed", "client.disconnected", "telemetry.dropped",
        "runtime.shutdown.start", "runtime.shutdown.complete"};
    return (unsigned int)kind < sizeof(names) / sizeof(names[0])
               ? names[kind] : "unknown";
}

/* Purpose: return one stable session-state spelling.
 * Inputs: session-state enumeration. Effects: none. Failure: returns unknown for unsupported value.
 * Boundary: state transitions remain session-owned. */
const char *yvex_server_session_state_name(yvex_server_session_state state)
{
    static const char *const names[] = {
        "created", "ready", "running", "partial", "detached",
        "resetting", "closing", "closed", "failed"};
    return (unsigned int)state < sizeof(names) / sizeof(names[0])
               ? names[state] : "unknown";
}

/* Purpose: independently recompute one event identity before client or renderer use.
 * Inputs: event and error output. Effects: none beyond temporary hashing.
 * Failure: refuses schema, sequence, name, or identity mismatch. Boundary: no event repair occurs. */
int yvex_server_event_validate(const yvex_server_event *event, yvex_error *err)
{
    yvex_server_event candidate;
    char supplied[YVEX_SHA256_HEX_CAP];
    if (!event || event->schema_version != TELEMETRY_SCHEMA_V2 ||
        event->kind > YVEX_SERVER_EVENT_RUNTIME_SHUTDOWN_COMPLETE ||
        event->severity > YVEX_SERVER_SEVERITY_FATAL ||
        (event->provider_adapter[0] &&
         (!yvex_sha256_hex_valid(event->provider_request_identity) ||
          !event->external_correlation_id[0])) ||
        (!event->provider_adapter[0] &&
         (event->provider_request_identity[0] ||
          event->external_correlation_id[0])) ||
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

/* Purpose: render one typed event as a bounded privacy-preserving JSONL record.
 * Inputs: validated event, output bytes/capacity, and error output. Effects: writes one newline-terminated object.
 * Failure: refuses invalid identity or insufficient output capacity. Boundary: content is absent from schema. */
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
                      "{\"schema\":2,\"sequence\":%llu,\"process\":%llu,"
                      "\"wall_time_ns\":%llu,\"monotonic_time_ns\":%llu,\"kind\":\"%s\","
                      "\"severity\":%u,\"session\":\"%s\",\"request\":\"%s\","
                      "\"turn\":\"%s\",\"phase\":\"%s\","
                      "\"provider\":\"%s\",\"provider_request_identity\":\"%s\","
                      "\"external_correlation_id\":\"%s\",\"a\":%llu,"
                      "\"b\":%llu,\"c\":%llu,\"seconds\":%.9g,\"rate\":%.9g,"
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
