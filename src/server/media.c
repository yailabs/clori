/* Own direct hosted-media requests while the generic runtime stages component residency. */
#include "src/server/private.h"

#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <yvex/internal/core.h>
#include <yvex/internal/server_media.h>

#define MEDIA_SESSION_CAP 8u
#define MEDIA_PROMPT_CAP 16384u
#define MEDIA_PROFILE_NAME_CAP 32u

typedef struct {
    char name[YVEX_SERVER_SESSION_NAME_CAP];
    char prompt[MEDIA_PROMPT_CAP];
    yvex_server_session_state state;
    unsigned long long attached_clients, turn_count;
    atomic_int active, cancelled;
} server_media_session;

typedef struct {
    char name[MEDIA_PROFILE_NAME_CAP];
    unsigned long long width, height, maximum_frames;
    int preview_alias;
} server_media_profile_owned;

struct server_media_registry {
    pthread_mutex_t mutex;
    yvex_runtime_media_model *model;
    yvex_runtime_av_generation_request generation;
    yvex_runtime_media_execution_preset preset;
    server_media_profile_owned profiles[YVEX_SERVER_MEDIA_PROFILE_CAP];
    server_media_session sessions[MEDIA_SESSION_CAP];
    server_telemetry *telemetry;
    char output_root[YVEX_PATH_CAP], artifact_reopen_cache_root[YVEX_PATH_CAP];
    char target[128], source_identity[YVEX_SHA256_HEX_CAP];
    char text_artifact[YVEX_PATH_CAP], transformer_artifact[YVEX_PATH_CAP];
    char video_artifact[YVEX_PATH_CAP], audio_artifact[YVEX_PATH_CAP];
    char profile_identity[YVEX_SHA256_HEX_CAP];
    char runtime_model_identity[YVEX_SHA256_HEX_CAP];
    server_event_scope event_scope;
    unsigned long long profile_count, frames_per_chunk, frame_remainder;
    unsigned long long minimum_frames, maximum_frames;
    unsigned long long minimum_inference_steps, maximum_inference_steps;
    unsigned long long canvas_multiple, maximum_canvas_pixels;
    int mutex_ready, closing;
};

static int media_refuse(yvex_error *err, yvex_status status, const char *reason)
{
    yvex_error_set(err, status, "server.media", reason);
    return status;
}

static int output_root_admit(const char *path, char output[YVEX_PATH_CAP], yvex_error *err)
{
    struct stat info;
    size_t extent;
    if (!path || path[0] != '/' || (extent = strlen(path)) >= YVEX_PATH_CAP ||
        !extent || strstr(path, "/../") || !strcmp(path + extent - 1u, "/..") ||
        lstat(path, &info) != 0 || !S_ISDIR(info.st_mode) || S_ISLNK(info.st_mode) ||
        info.st_uid != getuid())
        return media_refuse(err, YVEX_ERR_IO,
                            "media output root must be an existing owned absolute directory");
    yvex_core_text_copy(output, YVEX_PATH_CAP, path);
    while (extent > 1u && output[extent - 1u] == '/') output[--extent] = '\0';
    return YVEX_OK;
}

static int registry_identity(server_media_registry *registry, yvex_error *err)
{
    const char *identities[1] = {registry->source_identity};
    unsigned long long facts[7];
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;
    yvex_sha256_init(&hash);
    facts[0] = registry->profile_count;
    facts[1] = registry->minimum_frames;
    facts[2] = registry->maximum_frames;
    facts[3] = registry->canvas_multiple;
    facts[4] = registry->maximum_canvas_pixels;
    facts[5] = registry->minimum_inference_steps;
    facts[6] = registry->maximum_inference_steps;
    if (!yvex_sha256_update_text(&hash, "yvex.server.media-profile.v1") ||
        !yvex_sha256_update_text(&hash, identities[0]) ||
        !yvex_sha256_update_text(&hash, registry->preset.identity) ||
        !yvex_sha256_update_text(
            &hash, registry->generation.video_output_specialization.physical_identity) ||
        !yvex_sha256_update_text(
            &hash, registry->generation.audio_output_specialization.physical_identity))
        return media_refuse(err, YVEX_ERR_STATE, "media profile identity could not start");
    for (index = 0ull; index < 7ull; ++index)
        if (!yvex_sha256_update_u64_be(&hash, facts[index]))
            return media_refuse(err, YVEX_ERR_STATE, "media profile identity facts failed");
    for (index = 0ull; index < registry->profile_count; ++index) {
        server_media_profile_owned *profile = registry->profiles + index;
        if (!yvex_sha256_update_text(&hash, profile->name) ||
            !yvex_sha256_update_u64_be(&hash, profile->width) ||
            !yvex_sha256_update_u64_be(&hash, profile->height) ||
            !yvex_sha256_update_u64_be(&hash, profile->maximum_frames) ||
            !yvex_sha256_update_u64_be(&hash, (unsigned long long)profile->preview_alias))
            return media_refuse(err, YVEX_ERR_STATE, "media profile identity failed");
    }
    if (!yvex_sha256_final(&hash, digest))
        return media_refuse(err, YVEX_ERR_STATE, "media profile identity could not seal");
    yvex_sha256_hex(digest, registry->profile_identity);
    return YVEX_OK;
}

static int preset_admit(server_media_registry *registry, yvex_error *err)
{
    yvex_runtime_media_host_profile host = {0};
    unsigned long long index;

    host.schema_version = YVEX_RUNTIME_MEDIA_HOST_SCHEMA_V1;
    host.profile_count = registry->profile_count;
    host.frames_per_chunk = registry->frames_per_chunk;
    host.frame_remainder = registry->frame_remainder;
    host.minimum_frames = registry->minimum_frames;
    host.maximum_frames = registry->maximum_frames;
    host.minimum_inference_steps = registry->minimum_inference_steps;
    host.maximum_inference_steps = registry->maximum_inference_steps;
    for (index = 0ull; index < registry->profile_count; ++index) {
        yvex_core_text_copy(host.profiles[index].name,
                            sizeof(host.profiles[index].name),
                            registry->profiles[index].name);
        host.profiles[index].width = registry->profiles[index].width;
        host.profiles[index].height = registry->profiles[index].height;
        host.profiles[index].maximum_frames =
            registry->profiles[index].maximum_frames;
        host.profiles[index].preview_alias = registry->profiles[index].preview_alias;
    }
    return yvex_runtime_media_execution_preset_validate(
        &host, &registry->preset, err);
}

static void request_strings_copy(server_media_registry *registry,
                                 const yvex_runtime_av_generation_request *request)
{
    yvex_core_text_copy(registry->target, sizeof(registry->target), request->target);
    yvex_core_text_copy(registry->source_identity, sizeof(registry->source_identity),
                        request->source_identity);
    yvex_core_text_copy(registry->text_artifact, sizeof(registry->text_artifact),
                        request->text_artifact_path);
    yvex_core_text_copy(registry->transformer_artifact,
                        sizeof(registry->transformer_artifact),
                        request->transformer_artifact_path);
    yvex_core_text_copy(registry->video_artifact, sizeof(registry->video_artifact),
                        request->video_artifact_path);
    yvex_core_text_copy(registry->audio_artifact, sizeof(registry->audio_artifact),
                        request->audio_artifact_path);
    registry->generation.target = registry->target;
    registry->generation.source_identity = registry->source_identity;
    registry->generation.text_artifact_path = registry->text_artifact;
    registry->generation.transformer_artifact_path = registry->transformer_artifact;
    registry->generation.video_artifact_path = registry->video_artifact;
    registry->generation.audio_artifact_path = registry->audio_artifact;
}

int yvex_server_media_registry_open(
    server_media_registry **out, const yvex_server_media_options *options,
    server_telemetry *telemetry, yvex_error *err)
{
    server_media_registry *registry;
    unsigned long long index;
    if (out) *out = NULL;
    if (!out || !options || options->schema_version != YVEX_SERVER_MEDIA_SCHEMA_V1 ||
        !telemetry || !options->profiles || !options->profile_count ||
        options->profile_count > YVEX_SERVER_MEDIA_PROFILE_CAP ||
        !options->frames_per_chunk || options->frame_remainder >= options->frames_per_chunk ||
        !options->minimum_frames || options->minimum_frames > options->maximum_frames ||
        options->minimum_inference_steps < 2ull ||
        options->minimum_inference_steps > options->maximum_inference_steps ||
        !options->canvas_multiple || !options->maximum_canvas_pixels)
        return media_refuse(err, YVEX_ERR_INVALID_ARG,
                            "complete bounded hosted media options are required");
    registry = calloc(1u, sizeof(*registry));
    if (!registry) return media_refuse(err, YVEX_ERR_NOMEM, "media registry allocation failed");
    registry->generation = options->request_template;
    registry->telemetry = telemetry;
    registry->profile_count = options->profile_count;
    registry->frames_per_chunk = options->frames_per_chunk;
    registry->frame_remainder = options->frame_remainder;
    registry->minimum_frames = options->minimum_frames;
    registry->maximum_frames = options->maximum_frames;
    registry->minimum_inference_steps = options->minimum_inference_steps;
    registry->maximum_inference_steps = options->maximum_inference_steps;
    registry->canvas_multiple = options->canvas_multiple;
    registry->maximum_canvas_pixels = options->maximum_canvas_pixels;
    registry->preset = options->execution_preset;
    if (pthread_mutex_init(&registry->mutex, NULL) != 0) {
        free(registry);
        return media_refuse(err, YVEX_ERR_STATE, "media registry mutex failed");
    }
    registry->mutex_ready = 1;
    if (output_root_admit(options->output_root, registry->output_root, err) != YVEX_OK)
        goto failed;
    if (!options->artifact_reopen_cache_root ||
        options->artifact_reopen_cache_root[0] != '/' ||
        strlen(options->artifact_reopen_cache_root) >=
            sizeof(registry->artifact_reopen_cache_root) ||
        strstr(options->artifact_reopen_cache_root, "/../") ||
        !strcmp(options->artifact_reopen_cache_root +
                    strlen(options->artifact_reopen_cache_root) - 1u,
                "/.."))
        goto invalid;
    yvex_core_text_copy(registry->artifact_reopen_cache_root,
                        sizeof(registry->artifact_reopen_cache_root),
                        options->artifact_reopen_cache_root);
    if (!registry->generation.target || !registry->generation.source_identity ||
        !registry->generation.text_artifact_path ||
        !registry->generation.transformer_artifact_path ||
        !registry->generation.video_artifact_path ||
        !registry->generation.audio_artifact_path)
        goto invalid;
    request_strings_copy(registry, &options->request_template);
    for (index = 0ull; index < options->profile_count; ++index) {
        const yvex_server_media_profile *source = options->profiles + index;
        unsigned long long pixels;
        if (!source->name[0] || strlen(source->name) >= MEDIA_PROFILE_NAME_CAP ||
            !source->width ||
            !source->height ||
            source->maximum_frames < registry->minimum_frames ||
            source->maximum_frames > registry->maximum_frames ||
            source->width % registry->canvas_multiple ||
            source->height % registry->canvas_multiple ||
            !yvex_core_u64_mul(source->width, source->height, &pixels) ||
            pixels > registry->maximum_canvas_pixels)
            goto invalid;
        yvex_core_text_copy(registry->profiles[index].name,
                            sizeof(registry->profiles[index].name), source->name);
        registry->profiles[index].width = source->width;
        registry->profiles[index].height = source->height;
        registry->profiles[index].maximum_frames = source->maximum_frames;
        registry->profiles[index].preview_alias = source->preview_alias;
    }
    if (preset_admit(registry, err) != YVEX_OK) goto failed;
    if (registry_identity(registry, err) != YVEX_OK) goto failed;
    registry->event_scope.engine_kind = YVEX_SERVER_ENGINE_MEDIA;
    registry->event_scope.execution_strategy =
        YVEX_SERVER_EXECUTION_NOT_APPLICABLE;
    yvex_runtime_identity_copy(registry->event_scope.specialization_identity,
                               registry->profile_identity);
    *out = registry;
    yvex_error_clear(err);
    return YVEX_OK;
invalid:
    media_refuse(err, YVEX_ERR_INVALID_ARG, "media profile geometry is invalid");
failed:
    yvex_server_media_registry_close(&registry);
    return yvex_error_code(err);
}

static server_media_session *session_find(server_media_registry *registry, const char *name)
{
    unsigned long long index;
    if (!name || !name[0]) return NULL;
    for (index = 0ull; index < MEDIA_SESSION_CAP; ++index)
        if (!strcmp(registry->sessions[index].name, name) &&
            registry->sessions[index].state != YVEX_SERVER_SESSION_CLOSED)
            return registry->sessions + index;
    return NULL;
}

static server_media_session *session_create(server_media_registry *registry,
                                            const char *name, yvex_error *err)
{
    unsigned long long index;
    server_media_session *session;
    if (!name || !name[0] || strlen(name) >= YVEX_SERVER_SESSION_NAME_CAP ||
        session_find(registry, name)) {
        media_refuse(err, YVEX_ERR_STATE, "media session name is invalid or already active");
        return NULL;
    }
    for (index = 0ull; index < MEDIA_SESSION_CAP; ++index)
        if (!registry->sessions[index].name[0] ||
            registry->sessions[index].state == YVEX_SERVER_SESSION_CLOSED)
            break;
    if (index == MEDIA_SESSION_CAP) {
        media_refuse(err, YVEX_ERR_BOUNDS, "media session capacity is full");
        return NULL;
    }
    session = registry->sessions + index;
    memset(session, 0, sizeof(*session));
    yvex_core_text_copy(session->name, sizeof(session->name), name);
    session->state = YVEX_SERVER_SESSION_READY;
    atomic_init(&session->active, 0);
    atomic_init(&session->cancelled, 0);
    yvex_server_telemetry_session(registry->telemetry, 1, 1);
    return session;
}

static int message_emit(server_message_emit emit, void *context,
                        yvex_client_message_kind kind, const yvex_client_request *request,
                        const server_media_session *session, const char *text,
                        yvex_error *err)
{
    yvex_client_message message = {0};
    size_t extent = text ? strlen(text) : 0u;
    if (extent > sizeof(message.bytes))
        return media_refuse(err, YVEX_ERR_BOUNDS, "media response exceeds protocol capacity");
    message.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    message.kind = kind;
    message.status = YVEX_OK;
    message.request_number = request->request_number;
    message.engine_kind = YVEX_SERVER_ENGINE_MEDIA;
    message.execution_strategy = YVEX_SERVER_EXECUTION_NOT_APPLICABLE;
    message.stream_channel = YVEX_CLIENT_STREAM_FINAL_TEXT;
    message.session_state = session ? session->state : YVEX_SERVER_SESSION_READY;
    if (session)
        yvex_core_text_copy(message.session_name, sizeof(message.session_name), session->name);
    if (extent) {
        memcpy(message.bytes, text, extent);
        message.byte_count = extent;
    }
    return emit(context, &message, err);
}

static int turn_complete(server_message_emit emit, void *context,
                         const yvex_client_request *request,
                         const server_media_session *session,
                         const server_media_registry *registry,
                         const yvex_runtime_av_generation_result *result,
                         const char *output_path, double seconds,
                         yvex_error *err)
{
    yvex_client_message message = {0};
    message.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    message.kind = YVEX_CLIENT_MESSAGE_TURN_COMPLETE;
    message.status = YVEX_OK;
    message.request_number = request->request_number;
    message.engine_kind = YVEX_SERVER_ENGINE_MEDIA;
    message.execution_strategy = YVEX_SERVER_EXECUTION_NOT_APPLICABLE;
    message.generation_phase = YVEX_CLIENT_PHASE_COMPLETE;
    message.stop_reason = YVEX_CLIENT_STOP_EOS;
    message.decode_seconds = seconds;
    message.turn_count = session->turn_count;
    message.session_state = session->state;
    message.media_result.schema_version = YVEX_CLIENT_MEDIA_RESULT_SCHEMA_V1;
    message.media_result.available = 1;
    message.media_result.width = result->width;
    message.media_result.height = result->height;
    message.media_result.frames = result->frames;
    message.media_result.fps_numerator = registry->generation.fps_numerator;
    message.media_result.fps_denominator = registry->generation.fps_denominator;
    message.media_result.audio_samples = result->audio_samples;
    message.media_result.audio_sample_rate = registry->generation.audio_sample_rate;
    message.media_result.seed = registry->preset.seed;
    message.media_result.file_bytes = result->file_bytes;
    yvex_core_text_copy(message.media_result.output_path,
                        sizeof(message.media_result.output_path), output_path);
    yvex_core_text_copy(message.media_result.preset_identity,
                        sizeof(message.media_result.preset_identity),
                        registry->preset.identity);
    yvex_core_text_copy(message.media_result.execution_identity,
                        sizeof(message.media_result.execution_identity),
                        result->execution_identity);
    yvex_core_text_copy(message.media_result.file_identity,
                        sizeof(message.media_result.file_identity),
                        result->file_identity);
    yvex_core_text_copy(message.media_result.publication_identity,
                        sizeof(message.media_result.publication_identity),
                        result->publication_identity);
    yvex_core_text_copy(message.session_name, sizeof(message.session_name), session->name);
    return emit(context, &message, err);
}

static int turn_error(server_message_emit emit, void *context,
                      const yvex_client_request *request,
                      const server_media_session *session, int status,
                      double seconds, const yvex_error *failure,
                      yvex_error *err)
{
    yvex_client_message message = {0};
    message.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    message.kind = YVEX_CLIENT_MESSAGE_ERROR;
    message.status = status;
    message.failure_class = yvex_server_failure_class_from_status(status);
    message.request_number = request->request_number;
    message.engine_kind = YVEX_SERVER_ENGINE_MEDIA;
    message.execution_strategy = YVEX_SERVER_EXECUTION_NOT_APPLICABLE;
    message.generation_phase = status == YVEX_ERR_CANCELLED
                                   ? YVEX_CLIENT_PHASE_CANCELLED
                                   : YVEX_CLIENT_PHASE_FAILED;
    message.cancellation_class = status == YVEX_ERR_CANCELLED
                                     ? YVEX_CLIENT_CANCELLATION_COMPLETED
                                     : YVEX_CLIENT_CANCELLATION_NONE;
    message.stream_channel = YVEX_CLIENT_STREAM_ERROR;
    message.stop_reason = status == YVEX_ERR_CANCELLED
                              ? YVEX_GENERATION_STOP_CANCELLED
                              : YVEX_GENERATION_STOP_OUTPUT_FAILURE;
    message.session_state = session->state;
    message.turn_count = session->turn_count;
    message.total_completion_seconds = seconds;
    yvex_core_text_copy(message.session_name, sizeof(message.session_name),
                        session->name);
    yvex_core_text_copy(message.reason, sizeof(message.reason),
                        failure ? yvex_error_message(failure)
                                : "media generation failed");
    return emit(context, &message, err);
}

static int media_cancel_requested(void *opaque)
{
    server_media_session *session = opaque;
    return atomic_load_explicit(&session->cancelled, memory_order_acquire) != 0;
}

typedef struct {
    server_media_registry *registry;
    server_media_session *session;
    const yvex_client_request *request;
    const char *request_id;
    server_message_emit emit;
    void *emit_context;
} media_progress_sink;

static int media_event_emit(
    media_progress_sink *sink, yvex_server_event_kind kind,
    const char *phase, unsigned long long value_a,
    unsigned long long value_b, unsigned long long value_c,
    double seconds, yvex_error *err)
{
    yvex_client_message message = {0};
    int rc = yvex_server_telemetry_emit_provider(
        sink->registry->telemetry, &sink->registry->event_scope, kind,
        YVEX_SERVER_SEVERITY_INFO,
        sink->session->name, sink->request_id, NULL, phase, value_a, value_b,
        value_c, seconds, 0.0, NULL, NULL, &message.event, err);
    if (rc != YVEX_OK) return rc;
    message.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    message.kind = YVEX_CLIENT_MESSAGE_EVENT;
    message.status = YVEX_OK;
    message.request_number = sink->request->request_number;
    message.engine_kind = YVEX_SERVER_ENGINE_MEDIA;
    message.execution_strategy = YVEX_SERVER_EXECUTION_NOT_APPLICABLE;
    message.stream_channel = YVEX_CLIENT_STREAM_CONTROL_EVENT;
    return sink->emit(sink->emit_context, &message, err);
}

static int media_progress_observe(
    void *opaque, const yvex_runtime_media_progress *progress, yvex_error *err)
{
    static const char *const phases[] = {
        "conditioning-start", "conditioning-complete", "latent-start",
        "latent-step", "latent-complete", "visual-vae-start",
        "visual-vae-complete", "audio-vae-start", "audio-vae-complete",
        "publication-start", "publication-complete",
    };
    media_progress_sink *sink = opaque;
    yvex_server_event_kind kind = YVEX_SERVER_EVENT_GENERATION_PROGRESS;

    if (!sink || !progress ||
        progress->schema_version != YVEX_RUNTIME_MEDIA_PROGRESS_SCHEMA_V1 ||
        progress->kind > YVEX_RUNTIME_MEDIA_PROGRESS_PUBLICATION_COMPLETE)
        return media_refuse(err, YVEX_ERR_INVALID_ARG,
                            "typed media execution progress is required");
    if (progress->kind == YVEX_RUNTIME_MEDIA_PROGRESS_CONDITIONING_START)
        kind = YVEX_SERVER_EVENT_PREFILL_STARTED;
    else if (progress->kind == YVEX_RUNTIME_MEDIA_PROGRESS_CONDITIONING_COMPLETE)
        kind = YVEX_SERVER_EVENT_PREFILL_COMPLETED;
    return media_event_emit(sink, kind, phases[progress->kind],
                            progress->completed, progress->total,
                            progress->value, 0.0, err);
}

static int output_path_build(server_media_registry *registry,
                             const server_media_session *session,
                             char path[YVEX_PATH_CAP], yvex_error *err)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    char identity[YVEX_SHA256_HEX_CAP];
    int length;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.server.media-request.v1") ||
        !yvex_sha256_update_text(&hash, registry->preset.identity) ||
        !yvex_sha256_update_text(&hash, session->prompt) ||
        !yvex_sha256_update_u64_be(&hash, session->turn_count) ||
        !yvex_sha256_final(&hash, digest))
        return media_refuse(err, YVEX_ERR_STATE, "media request identity could not seal");
    yvex_sha256_hex(digest, identity);
    length = snprintf(path, YVEX_PATH_CAP, "%s/yvex-media-%s.avi",
                      registry->output_root, identity);
    if (length < 0 || length >= YVEX_PATH_CAP)
        return media_refuse(err, YVEX_ERR_BOUNDS, "media output path exceeds capacity");
    return YVEX_OK;
}

static int generation_execute(server_media_registry *registry,
                              server_media_session *session,
                              const yvex_client_request *request,
                              const char *request_id,
                              server_message_emit emit, void *context,
                              yvex_error *err)
{
    yvex_runtime_av_generation_request generation = registry->generation;
    yvex_runtime_av_generation_result result = {0};
    media_progress_sink sink = {
        registry, session, request, request_id, emit, context,
    };
    char path[YVEX_PATH_CAP];
    unsigned long long started = yvex_core_monotonic_ns(), completed;
    double seconds;
    int rc;
    rc = output_path_build(registry, session, path, err);
    if (rc != YVEX_OK) return rc;
    generation.prompt = session->prompt;
    generation.output_path = path;
    generation.width = registry->preset.width;
    generation.height = registry->preset.height;
    generation.frames = registry->preset.frames;
    generation.inference_steps =
        (unsigned int)(registry->preset.sigma_grid_points - 1ull);
    generation.seed = registry->preset.seed;
    generation.cancel_requested = media_cancel_requested;
    generation.cancel_context = session;
    generation.observe_progress = media_progress_observe;
    generation.progress_context = &sink;
    session->state = YVEX_SERVER_SESSION_RUNNING;
    atomic_store_explicit(&session->active, 1, memory_order_release);
    atomic_store_explicit(&session->cancelled, 0, memory_order_release);
    rc = message_emit(emit, context, YVEX_CLIENT_MESSAGE_TURN_STARTED,
                      request, session, NULL, err);
    if (rc == YVEX_OK)
        rc = media_event_emit(&sink, YVEX_SERVER_EVENT_REQUEST_STARTED,
                              "media", 1ull, 0ull, 0ull, 0.0, err);
    if (rc == YVEX_OK)
        rc = media_event_emit(
            &sink, YVEX_SERVER_EVENT_GENERATION_PROFILE, "hosted-preset",
            registry->preset.width, registry->preset.height,
            registry->preset.frames, 0.0, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_media_model_generate(
            registry->model, &generation, &result, err);
    completed = yvex_core_monotonic_ns();
    seconds = completed >= started ? (double)(completed - started) / 1000000000.0 : 0.0;
    atomic_store_explicit(&session->active, 0, memory_order_release);
    session->turn_count++;
    if (rc != YVEX_OK) {
        yvex_error primary = err ? *err : (yvex_error){0};
        yvex_error telemetry_error;
        yvex_error_clear(&telemetry_error);
        session->state = rc == YVEX_ERR_CANCELLED ? YVEX_SERVER_SESSION_READY
                                                  : YVEX_SERVER_SESSION_FAILED;
        (void)yvex_server_telemetry_emit(
            registry->telemetry, &registry->event_scope,
            rc == YVEX_ERR_CANCELLED ? YVEX_SERVER_EVENT_GENERATION_CANCELLED
                                     : YVEX_SERVER_EVENT_GENERATION_FAILED,
            rc == YVEX_ERR_CANCELLED ? YVEX_SERVER_SEVERITY_INFO
                                     : YVEX_SERVER_SEVERITY_ERROR,
            session->name, request_id, NULL,
            rc == YVEX_ERR_CANCELLED ? "cancelled" : "failed", 0ull, 0ull,
            0ull, seconds, 0.0, &telemetry_error);
        yvex_error_clear(&telemetry_error);
        (void)turn_error(emit, context, request, session, rc, seconds,
                         &primary, &telemetry_error);
        if (err) *err = primary;
        return rc;
    }
    session->state = YVEX_SERVER_SESSION_READY;
    rc = yvex_server_telemetry_emit(
        registry->telemetry, &registry->event_scope,
        YVEX_SERVER_EVENT_GENERATION_COMPLETED,
        YVEX_SERVER_SEVERITY_INFO, session->name, request_id, NULL,
        "completed", result.frames, result.file_bytes, result.audio_samples,
        seconds, 0.0, err);
    if (rc != YVEX_OK) return rc;
    return turn_complete(emit, context, request, session, registry, &result,
                         path, seconds, err);
}

static int session_message(server_message_emit emit, void *context,
                           yvex_client_message_kind kind,
                           const yvex_client_request *request,
                           const server_media_session *session,
                           const char *reason, yvex_error *err)
{
    yvex_client_message message = {0};
    message.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    message.kind = kind;
    message.status = YVEX_OK;
    message.request_number = request->request_number;
    message.engine_kind = YVEX_SERVER_ENGINE_MEDIA;
    message.execution_strategy = YVEX_SERVER_EXECUTION_NOT_APPLICABLE;
    if (session) {
        yvex_core_text_copy(message.session_name, sizeof(message.session_name), session->name);
        message.session_state = session->state;
        message.turn_count = session->turn_count;
    }
    yvex_core_text_copy(message.reason, sizeof(message.reason), reason ? reason : "ok");
    return emit(context, &message, err);
}

int yvex_server_media_registry_execute(
    server_media_registry *registry, const yvex_client_request *request,
    const char *request_id, double queue_seconds, server_message_emit emit,
    void *emit_context, yvex_error *err)
{
    server_media_session *session = NULL;
    int rc = YVEX_OK;
    (void)queue_seconds;
    if (!registry || !request || !emit || pthread_mutex_lock(&registry->mutex) != 0)
        return media_refuse(err, YVEX_ERR_INVALID_ARG, "media request registry is required");
    if (registry->closing) {
        rc = media_refuse(err, YVEX_ERR_STATE, "media request registry is closing");
        goto done;
    }
    if (request->operation != YVEX_CLIENT_OP_SESSION_NEW &&
        request->operation != YVEX_CLIENT_OP_SESSION_LIST)
        session = session_find(registry, request->session_name);
    if (request->operation == YVEX_CLIENT_OP_SESSION_NEW) {
        session = session_create(registry, request->session_name, err);
        rc = session ? session_message(emit, emit_context, YVEX_CLIENT_MESSAGE_SESSION,
                                       request, session, "created", err)
                     : yvex_error_code(err);
    } else if (request->operation == YVEX_CLIENT_OP_SESSION_LIST) {
        unsigned long long index;
        for (index = 0ull; rc == YVEX_OK && index < MEDIA_SESSION_CAP; ++index)
            if (registry->sessions[index].name[0] &&
                registry->sessions[index].state != YVEX_SERVER_SESSION_CLOSED)
                rc = session_message(emit, emit_context, YVEX_CLIENT_MESSAGE_SESSION,
                                     request, registry->sessions + index, "member", err);
        if (rc == YVEX_OK)
            rc = session_message(emit, emit_context, YVEX_CLIENT_MESSAGE_ACK,
                                 request, NULL, "complete", err);
    } else if (!session) {
        rc = media_refuse(err, YVEX_ERR_STATE, "unknown media session");
    } else if (request->operation == YVEX_CLIENT_OP_SESSION_SHOW) {
        rc = session_message(emit, emit_context, YVEX_CLIENT_MESSAGE_SESSION,
                             request, session, "snapshot", err);
    } else if (request->operation == YVEX_CLIENT_OP_SESSION_ATTACH) {
        session->attached_clients++;
        rc = session_message(emit, emit_context, YVEX_CLIENT_MESSAGE_ACK,
                             request, session, "attached", err);
    } else if (request->operation == YVEX_CLIENT_OP_SESSION_DETACH) {
        if (session->attached_clients) session->attached_clients--;
        rc = session_message(emit, emit_context, YVEX_CLIENT_MESSAGE_ACK,
                             request, session, "detached", err);
    } else if (request->operation == YVEX_CLIENT_OP_SESSION_RESET) {
        if (atomic_load_explicit(&session->active, memory_order_acquire))
            rc = media_refuse(err, YVEX_ERR_STATE, "active media session cannot reset");
        else {
            char name[YVEX_SERVER_SESSION_NAME_CAP];
            unsigned long long attached = session->attached_clients;
            yvex_core_text_copy(name, sizeof(name), session->name);
            memset(session, 0, sizeof(*session));
            yvex_core_text_copy(session->name, sizeof(session->name), name);
            session->attached_clients = attached;
            session->state = YVEX_SERVER_SESSION_READY;
            atomic_init(&session->active, 0);
            atomic_init(&session->cancelled, 0);
            rc = session_message(emit, emit_context, YVEX_CLIENT_MESSAGE_ACK,
                                 request, session, "reset", err);
        }
    } else if (request->operation == YVEX_CLIENT_OP_SESSION_CLOSE) {
        if (atomic_load_explicit(&session->active, memory_order_acquire))
            rc = media_refuse(err, YVEX_ERR_STATE, "active media session cannot close");
        else {
            session->state = YVEX_SERVER_SESSION_CLOSED;
            yvex_server_telemetry_session(registry->telemetry, -1, 0);
            rc = session_message(emit, emit_context, YVEX_CLIENT_MESSAGE_ACK,
                                 request, NULL, "closed", err);
        }
    } else if (request->operation == YVEX_CLIENT_OP_GENERATION_TURN) {
        if (!request->prompt || !request->prompt_bytes ||
            request->prompt_bytes >= sizeof(session->prompt) ||
            memchr(request->prompt, '\0', (size_t)request->prompt_bytes)) {
            rc = media_refuse(err, YVEX_ERR_BOUNDS, "media prompt exceeds capacity");
            goto done;
        }
        if (atomic_load_explicit(&session->active, memory_order_acquire)) {
            rc = media_refuse(err, YVEX_ERR_STATE,
                              "media session already owns an active generation");
            goto done;
        }
        memcpy(session->prompt, request->prompt, (size_t)request->prompt_bytes);
        session->prompt[request->prompt_bytes] = '\0';
        (void)pthread_mutex_unlock(&registry->mutex);
        return generation_execute(registry, session, request, request_id,
                                  emit, emit_context, err);
    } else {
        rc = media_refuse(err, YVEX_ERR_UNSUPPORTED,
                          "operation is unavailable for hosted media sessions");
    }
done:
    (void)pthread_mutex_unlock(&registry->mutex);
    return rc;
}

int yvex_server_media_registry_console_status(
    server_media_registry *registry, const char *name, yvex_console_status *status,
    yvex_client_partial_turn *partial, yvex_error *err)
{
    server_media_session *session;
    if (!registry || !status || !partial || pthread_mutex_lock(&registry->mutex) != 0)
        return media_refuse(err, YVEX_ERR_INVALID_ARG, "media console status is required");
    session = session_find(registry, name);
    if (!session) {
        (void)pthread_mutex_unlock(&registry->mutex);
        return media_refuse(err, YVEX_ERR_STATE, "unknown media session");
    }
    memset(partial, 0, sizeof(*partial));
    status->schema_version = YVEX_CONSOLE_STATUS_SCHEMA_V1;
    status->session_available = 1;
    status->attached = session->attached_clients != 0ull;
    status->session_state = session->state;
    status->generation_phase = atomic_load_explicit(&session->active, memory_order_acquire)
                                   ? YVEX_CLIENT_PHASE_DECODE : YVEX_CLIENT_PHASE_IDLE;
    status->cancel_requested = atomic_load_explicit(&session->cancelled, memory_order_acquire);
    status->cancellation_class = status->cancel_requested
                                     ? YVEX_CLIENT_CANCELLATION_REQUESTED
                                     : YVEX_CLIENT_CANCELLATION_NONE;
    status->turn_count = session->turn_count;
    status->progress_available = 1;
    status->reasoning_policy = YVEX_REASONING_DISABLED;
    yvex_core_text_copy(status->session_name, sizeof(status->session_name), session->name);
    (void)pthread_mutex_unlock(&registry->mutex);
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_server_media_registry_cancel(server_media_registry *registry,
                                      const char *name, yvex_error *err)
{
    server_media_session *session;
    if (!registry || pthread_mutex_lock(&registry->mutex) != 0)
        return media_refuse(err, YVEX_ERR_INVALID_ARG, "media registry is required");
    session = session_find(registry, name);
    if (!session || !atomic_load_explicit(&session->active, memory_order_acquire)) {
        (void)pthread_mutex_unlock(&registry->mutex);
        return media_refuse(err, YVEX_ERR_STATE, "media session has no active generation");
    }
    atomic_store_explicit(&session->cancelled, 1, memory_order_release);
    (void)pthread_mutex_unlock(&registry->mutex);
    yvex_error_clear(err);
    return YVEX_OK;
}

void yvex_server_media_registry_cancel_all(server_media_registry *registry)
{
    unsigned long long index;
    if (!registry || pthread_mutex_lock(&registry->mutex) != 0) return;
    for (index = 0ull; index < MEDIA_SESSION_CAP; ++index)
        if (atomic_load_explicit(&registry->sessions[index].active, memory_order_acquire))
            atomic_store_explicit(&registry->sessions[index].cancelled, 1,
                                  memory_order_release);
    registry->closing = 1;
    (void)pthread_mutex_unlock(&registry->mutex);
}

int yvex_server_media_registry_count(server_media_registry *registry,
                                     unsigned long long *count, yvex_error *err)
{
    unsigned long long index, total = 0ull;
    if (!registry || !count || pthread_mutex_lock(&registry->mutex) != 0)
        return media_refuse(err, YVEX_ERR_INVALID_ARG, "media session count is required");
    for (index = 0ull; index < MEDIA_SESSION_CAP; ++index)
        if (registry->sessions[index].name[0] &&
            registry->sessions[index].state != YVEX_SERVER_SESSION_CLOSED)
            total++;
    *count = total;
    (void)pthread_mutex_unlock(&registry->mutex);
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_server_media_registry_summary(server_media_registry *registry,
                                       server_media_summary *summary, yvex_error *err)
{
    if (!registry || !summary)
        return media_refuse(err, YVEX_ERR_INVALID_ARG, "media summary is required");
    yvex_core_text_copy(summary->runtime_model_identity,
                        sizeof(summary->runtime_model_identity),
                        registry->runtime_model_identity[0]
                            ? registry->runtime_model_identity
                            : registry->profile_identity);
    yvex_core_text_copy(summary->specialization_identity,
                        sizeof(summary->specialization_identity),
                        registry->profile_identity);
    yvex_error_clear(err);
    return YVEX_OK;
}

static void component_admission_observe(
    void *opaque, const char *role,
    const yvex_artifact_admission_evidence *evidence)
{
    server_media_registry *registry = opaque;
    yvex_error event_error;
    unsigned long long receipt;
    double rate;

    if (!registry || !evidence || !evidence->complete) return;
    receipt = (unsigned long long)evidence->verification_mode |
              ((unsigned long long)evidence->reopen_state << 8u) |
              ((unsigned long long)(evidence->lease_published != 0) << 16u) |
              ((unsigned long long)(evidence->lease_repaired != 0) << 17u) |
              ((unsigned long long)(evidence->cache_failure != 0) << 18u);
    rate = evidence->seconds > 0.0
               ? (double)evidence->bytes_hashed / evidence->seconds
               : 0.0;
    yvex_error_clear(&event_error);
    (void)yvex_server_telemetry_emit(
        registry->telemetry, &registry->event_scope,
        YVEX_SERVER_EVENT_ARTIFACT_OPEN_COMPLETE,
        evidence->cache_failure ? YVEX_SERVER_SEVERITY_WARNING
                                : YVEX_SERVER_SEVERITY_INFO,
        NULL, NULL, NULL, role, evidence->bytes_hashed, evidence->file_bytes,
        receipt, evidence->seconds,
        rate, &event_error);
}

int yvex_server_media_registry_start(
    server_media_registry *registry, yvex_runtime_media_model_summary *summary,
    yvex_error *err)
{
    yvex_runtime_media_model_open_options open_options = {0};
    int rc;
    if (summary) memset(summary, 0, sizeof(*summary));
    if (!registry || !summary || pthread_mutex_lock(&registry->mutex) != 0)
        return media_refuse(err, YVEX_ERR_INVALID_ARG,
                            "media runtime model and summary are required");
    if (registry->model) {
        (void)pthread_mutex_unlock(&registry->mutex);
        return media_refuse(err, YVEX_ERR_STATE,
                            "media runtime model is already open");
    }
    open_options.schema_version = YVEX_RUNTIME_MEDIA_MODEL_OPEN_SCHEMA_V1;
    open_options.artifact_reopen_cache_root = registry->artifact_reopen_cache_root;
    open_options.observe_component = component_admission_observe;
    open_options.observer_context = registry;
    rc = yvex_runtime_media_model_open(
        &registry->model, &registry->generation, &open_options, summary, err);
    if (rc == YVEX_OK) {
        yvex_core_text_copy(registry->runtime_model_identity,
                            sizeof(registry->runtime_model_identity),
                            summary->model_identity);
        yvex_runtime_identity_copy(registry->event_scope.runtime_model_identity,
                                   summary->model_identity);
    }
    (void)pthread_mutex_unlock(&registry->mutex);
    return rc;
}

void yvex_server_media_registry_close(server_media_registry **registry)
{
    server_media_registry *owner;
    unsigned long long index;
    if (!registry || !*registry) return;
    owner = *registry;
    if (owner->mutex_ready && pthread_mutex_lock(&owner->mutex) == 0) {
        owner->closing = 1;
        for (index = 0ull; index < MEDIA_SESSION_CAP; ++index) {
            server_media_session *session = owner->sessions + index;
            if (!session->name[0] || session->state == YVEX_SERVER_SESSION_CLOSED)
                continue;
            session->state = YVEX_SERVER_SESSION_CLOSED;
            yvex_server_telemetry_session(owner->telemetry, -1, 0);
        }
        (void)pthread_mutex_unlock(&owner->mutex);
    }
    yvex_runtime_media_model_close(&owner->model);
    if (owner->mutex_ready) (void)pthread_mutex_destroy(&owner->mutex);
    memset(owner, 0, sizeof(*owner));
    free(owner);
    *registry = NULL;
}
