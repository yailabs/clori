/* Own conversational media requests while the generic runtime stages component residency. */
#include "src/server/private.h"

#include <ctype.h>
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

typedef enum {
    MEDIA_DIALOG_EMPTY = 0,
    MEDIA_DIALOG_PARAMETERS,
    MEDIA_DIALOG_RUNNING,
    MEDIA_DIALOG_COMPLETE,
    MEDIA_DIALOG_FAILED
} media_dialog_state;

typedef struct {
    char name[YVEX_SERVER_SESSION_NAME_CAP];
    char prompt[MEDIA_PROMPT_CAP];
    media_dialog_state dialog;
    yvex_server_session_state state;
    unsigned long long attached_clients, turn_count;
    unsigned long long width, height, frames, profile_maximum_frames;
    unsigned long long sigma_grid_points, seed;
    int profile_selected, duration_selected, steps_selected, format_selected;
    atomic_int active, cancelled;
} server_media_session;

typedef struct {
    char name[MEDIA_PROFILE_NAME_CAP];
    unsigned long long width, height, maximum_frames;
    int preview_alias;
} server_media_profile_owned;

struct server_media_registry {
    pthread_mutex_t mutex;
    yvex_runtime_av_generation_request generation;
    server_media_profile_owned profiles[YVEX_SERVER_MEDIA_PROFILE_CAP];
    server_media_session sessions[MEDIA_SESSION_CAP];
    server_telemetry *telemetry;
    char output_root[YVEX_PATH_CAP];
    char target[128], source_identity[YVEX_SHA256_HEX_CAP];
    char text_artifact[YVEX_PATH_CAP], transformer_artifact[YVEX_PATH_CAP];
    char video_artifact[YVEX_PATH_CAP], audio_artifact[YVEX_PATH_CAP];
    char profile_identity[YVEX_SHA256_HEX_CAP];
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

static int text_contains(const char *text, const char *needle)
{
    size_t extent, index, offset;
    if (!text || !needle || !(extent = strlen(needle))) return 0;
    for (index = 0u; text[index]; ++index) {
        for (offset = 0u; offset < extent && text[index + offset] &&
                           tolower((unsigned char)text[index + offset]) ==
                               tolower((unsigned char)needle[offset]);
             ++offset) {}
        if (offset == extent) return 1;
    }
    return 0;
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
        !yvex_sha256_update_text(&hash, identities[0]))
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
                            "complete bounded conversational media options are required");
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
    if (pthread_mutex_init(&registry->mutex, NULL) != 0) {
        free(registry);
        return media_refuse(err, YVEX_ERR_STATE, "media registry mutex failed");
    }
    registry->mutex_ready = 1;
    if (output_root_admit(options->output_root, registry->output_root, err) != YVEX_OK)
        goto failed;
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
    if (registry_identity(registry, err) != YVEX_OK) goto failed;
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
    session->dialog = MEDIA_DIALOG_EMPTY;
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
    message.generation_mode = YVEX_SERVER_GENERATION_MEDIA;
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
                         const server_media_session *session, double seconds,
                         yvex_error *err)
{
    yvex_client_message message = {0};
    message.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    message.kind = YVEX_CLIENT_MESSAGE_TURN_COMPLETE;
    message.status = YVEX_OK;
    message.request_number = request->request_number;
    message.generation_mode = YVEX_SERVER_GENERATION_MEDIA;
    message.generation_phase = YVEX_CLIENT_PHASE_COMPLETE;
    message.stop_reason = YVEX_CLIENT_STOP_EOS;
    message.decode_seconds = seconds;
    message.turn_count = session->turn_count;
    message.session_state = session->state;
    yvex_core_text_copy(message.session_name, sizeof(message.session_name), session->name);
    return emit(context, &message, err);
}

static int profile_select(server_media_registry *registry, server_media_session *session,
                          const char *text)
{
    unsigned long long index;
    int unavailable = text_contains(text, "source") || text_contains(text, "alta") ||
                      text_contains(text, "high") || text_contains(text, "hd") ||
                      text_contains(text, "768p") || text_contains(text, "1344x768") ||
                      text_contains(text, "bozza") || text_contains(text, "draft") ||
                      text_contains(text, "960x544") || text_contains(text, "fhd") ||
                      text_contains(text, "1080p") || text_contains(text, "1920x1080") ||
                      text_contains(text, "2k") || text_contains(text, "4k") ||
                      text_contains(text, "2160p") || text_contains(text, "3840x2160");
    int preview = text_contains(text, "preview") || text_contains(text, "anteprima") ||
                  text_contains(text, "192x192");
    int smoke = text_contains(text, "smoke") || text_contains(text, "test") ||
                text_contains(text, "32x32");
    if (unavailable) return -1;
    for (index = 0ull; index < registry->profile_count; ++index) {
        server_media_profile_owned *profile = registry->profiles + index;
        if (text_contains(text, profile->name) ||
            (preview && profile->preview_alias) ||
            (smoke && !strcmp(profile->name, "smoke"))) {
            session->width = profile->width;
            session->height = profile->height;
            session->profile_maximum_frames = profile->maximum_frames;
            session->profile_selected = 1;
            return 1;
        }
    }
    return 0;
}

static int text_prefix(const char *text, const char *prefix)
{
    if (!text || !prefix) return 0;
    while (*prefix && *text &&
           tolower((unsigned char)*text) == tolower((unsigned char)*prefix)) {
        text++;
        prefix++;
    }
    return *prefix == '\0';
}

static int steps_select(server_media_registry *registry, server_media_session *session,
                        const char *text)
{
    const char *cursor = text;
    while (cursor && *cursor) {
        char *end;
        unsigned long long steps;
        while (*cursor && !isdigit((unsigned char)*cursor)) cursor++;
        if (!*cursor) break;
        steps = strtoull(cursor, &end, 10);
        if (end == cursor) break;
        while (*end && (isspace((unsigned char)*end) || *end == '-')) end++;
        if (text_prefix(end, "step") || text_prefix(end, "iteraz") ||
            text_prefix(end, "punti sigma")) {
            if (steps < registry->minimum_inference_steps ||
                steps > registry->maximum_inference_steps)
                return -1;
            session->sigma_grid_points = steps;
            session->steps_selected = 1;
            return 1;
        }
        cursor = end;
    }
    return 0;
}

static int duration_select(server_media_registry *registry, server_media_session *session,
                           const char *text)
{
    const char *cursor = text;
    while (cursor && *cursor) {
        char *end, *unit;
        unsigned long long seconds, frames;
        while (*cursor && !isdigit((unsigned char)*cursor)) cursor++;
        if (!*cursor) break;
        seconds = strtoull(cursor, &end, 10);
        if (end == cursor) break;
        unit = end;
        while (*unit && isspace((unsigned char)*unit)) unit++;
        if ((*unit == 's' && (unit[1] == '\0' || isspace((unsigned char)unit[1]) ||
                             ispunct((unsigned char)unit[1]))) ||
            text_prefix(unit, "second")) {
            if (!seconds || !yvex_core_u64_mul(seconds,
                                                registry->generation.fps_numerator, &frames))
                return -1;
            frames /= registry->generation.fps_denominator;
            while (frames % registry->frames_per_chunk != registry->frame_remainder) frames++;
            if (frames < registry->minimum_frames || frames > registry->maximum_frames)
                return -1;
            session->frames = frames;
            session->duration_selected = 1;
            return 1;
        }
        cursor = end;
    }
    return 0;
}

static int format_select(server_media_session *session, const char *text)
{
    if (text_contains(text, "mp4") || text_contains(text, "mkv") ||
        text_contains(text, "webm") || text_contains(text, "mov"))
        return -1;
    if (text_contains(text, "avi")) {
        session->format_selected = 1;
        return 1;
    }
    return 0;
}

static void seed_select(server_media_session *session, const char *text)
{
    const char *cursor = text;
    while ((cursor = strstr(cursor, "seed")) != NULL) {
        char *end;
        unsigned long long value;
        cursor += 4;
        while (*cursor && (isspace((unsigned char)*cursor) || *cursor == ':' || *cursor == '='))
            cursor++;
        value = strtoull(cursor, &end, 10);
        if (end != cursor) session->seed = value;
        return;
    }
}

static int dialog_parse(server_media_registry *registry, server_media_session *session,
                        const char *text, yvex_error *err)
{
    int profile, duration, steps, format;
    profile = profile_select(registry, session, text);
    duration = duration_select(registry, session, text);
    steps = steps_select(registry, session, text);
    format = format_select(session, text);
    seed_select(session, text);
    if (profile < 0)
        return media_refuse(err, YVEX_ERR_UNSUPPORTED,
                            "this GB10 path currently admits preview 192x192 or smoke 32x32; "
                            "source, draft, HD, FHD, 2K, and 4K are not qualified");
    if (duration < 0)
        return media_refuse(err, YVEX_ERR_BOUNDS,
                            "MiniMax-H3 duration must resolve to 5 through 15 seconds");
    if (steps < 0)
        return media_refuse(err, YVEX_ERR_BOUNDS,
                            "MiniMax-H3 requires 2 through 64 explicit sigma grid points");
    if (format < 0)
        return media_refuse(err, YVEX_ERR_UNSUPPORTED,
                            "this YVEX media path currently publishes AVI only");
    if (session->profile_selected && session->duration_selected &&
        session->frames > session->profile_maximum_frames)
        return media_refuse(err, YVEX_ERR_BOUNDS,
                            "the preview profile admits 5 seconds; smoke admits 5 through 15 seconds");
    return YVEX_OK;
}

static int dialog_question(server_media_registry *registry,
                           server_message_emit emit, void *context,
                           const yvex_client_request *request,
                           server_media_session *session, yvex_error *err)
{
    char response[YVEX_SERVER_FRAGMENT_CAP];
    size_t used = 0u;
    int wrote;
    wrote = snprintf(response, sizeof(response),
                     "Prima di generare mi servono i parametri mancanti. ");
    if (wrote < 0 || (size_t)wrote >= sizeof(response)) return YVEX_ERR_BOUNDS;
    used = (size_t)wrote;
    if (!session->profile_selected) {
        unsigned long long index;
        wrote = snprintf(response + used, sizeof(response) - used, "Qualità: ");
        if (wrote < 0 || (size_t)wrote >= sizeof(response) - used) return YVEX_ERR_BOUNDS;
        used += (size_t)wrote;
        for (index = 0ull; index < registry->profile_count; ++index) {
            server_media_profile_owned *profile = registry->profiles + index;
            wrote = snprintf(response + used, sizeof(response) - used,
                             "%s%s (%llux%llu, max %llu frame)",
                             index ? ", " : "", profile->name,
                             profile->width, profile->height, profile->maximum_frames);
            if (wrote < 0 || (size_t)wrote >= sizeof(response) - used) return YVEX_ERR_BOUNDS;
            used += (size_t)wrote;
        }
        wrote = snprintf(response + used, sizeof(response) - used, ". ");
        if (wrote < 0 || (size_t)wrote >= sizeof(response) - used) return YVEX_ERR_BOUNDS;
        used += (size_t)wrote;
    }
    if (!session->duration_selected) {
        wrote = snprintf(response + used, sizeof(response) - used,
                         "Durata: da 5 a 15 secondi. ");
        if (wrote < 0 || (size_t)wrote >= sizeof(response) - used) return YVEX_ERR_BOUNDS;
        used += (size_t)wrote;
    }
    if (!session->steps_selected) {
        wrote = snprintf(response + used, sizeof(response) - used,
                         "Iterazioni: indica da 2 a 64 punti sigma; la sorgente non dichiara un default. ");
        if (wrote < 0 || (size_t)wrote >= sizeof(response) - used) return YVEX_ERR_BOUNDS;
        used += (size_t)wrote;
    }
    if (!session->format_selected) {
        wrote = snprintf(response + used, sizeof(response) - used,
                         "Formato disponibile: AVI. Puoi anche indicare un seed opzionale.\n");
        if (wrote < 0 || (size_t)wrote >= sizeof(response) - used) return YVEX_ERR_BOUNDS;
    }
    session->dialog = MEDIA_DIALOG_PARAMETERS;
    session->turn_count++;
    if (message_emit(emit, context, YVEX_CLIENT_MESSAGE_TURN_STARTED,
                     request, session, NULL, err) != YVEX_OK ||
        message_emit(emit, context, YVEX_CLIENT_MESSAGE_FRAGMENT,
                     request, session, response, err) != YVEX_OK)
        return yvex_error_code(err);
    return turn_complete(emit, context, request, session, 0.0, err);
}

static int media_cancel_requested(void *opaque)
{
    server_media_session *session = opaque;
    return atomic_load_explicit(&session->cancelled, memory_order_acquire) != 0;
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
        !yvex_sha256_update_text(&hash, registry->profile_identity) ||
        !yvex_sha256_update_text(&hash, session->prompt) ||
        !yvex_sha256_update_u64_be(&hash, session->width) ||
        !yvex_sha256_update_u64_be(&hash, session->height) ||
        !yvex_sha256_update_u64_be(&hash, session->frames) ||
        !yvex_sha256_update_u64_be(&hash, session->sigma_grid_points) ||
        !yvex_sha256_update_u64_be(&hash, session->seed) ||
        !yvex_sha256_final(&hash, digest))
        return media_refuse(err, YVEX_ERR_STATE, "media request identity could not seal");
    yvex_sha256_hex(digest, identity);
    length = snprintf(path, YVEX_PATH_CAP, "%s/minimax-h3-%s.avi",
                      registry->output_root, identity);
    if (length < 0 || length >= YVEX_PATH_CAP)
        return media_refuse(err, YVEX_ERR_BOUNDS, "media output path exceeds capacity");
    return YVEX_OK;
}

static int generation_execute(server_media_registry *registry,
                              server_media_session *session,
                              const yvex_client_request *request,
                              server_message_emit emit, void *context,
                              yvex_error *err)
{
    yvex_runtime_av_generation_request generation = registry->generation;
    yvex_runtime_av_generation_result result = {0};
    char path[YVEX_PATH_CAP], response[YVEX_SERVER_FRAGMENT_CAP];
    unsigned long long started = yvex_core_monotonic_ns(), completed;
    double seconds;
    int rc, length;
    rc = output_path_build(registry, session, path, err);
    if (rc != YVEX_OK) return rc;
    generation.prompt = session->prompt;
    generation.output_path = path;
    generation.width = session->width;
    generation.height = session->height;
    generation.frames = session->frames;
    generation.inference_steps = (unsigned int)(session->sigma_grid_points - 1ull);
    generation.seed = session->seed;
    generation.cancel_requested = media_cancel_requested;
    generation.cancel_context = session;
    session->dialog = MEDIA_DIALOG_RUNNING;
    session->state = YVEX_SERVER_SESSION_RUNNING;
    atomic_store_explicit(&session->active, 1, memory_order_release);
    atomic_store_explicit(&session->cancelled, 0, memory_order_release);
    rc = message_emit(emit, context, YVEX_CLIENT_MESSAGE_TURN_STARTED,
                      request, session, NULL, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_av_generate(&generation, &result, err);
    completed = yvex_core_monotonic_ns();
    seconds = completed >= started ? (double)(completed - started) / 1000000000.0 : 0.0;
    atomic_store_explicit(&session->active, 0, memory_order_release);
    session->turn_count++;
    if (rc != YVEX_OK) {
        session->dialog = MEDIA_DIALOG_FAILED;
        session->state = YVEX_SERVER_SESSION_FAILED;
        return rc;
    }
    session->dialog = MEDIA_DIALOG_COMPLETE;
    session->state = YVEX_SERVER_SESSION_READY;
    length = snprintf(response, sizeof(response),
                      "Video completato: %s\n%llux%llu, %llu frame, AVI, seed %llu.\n",
                      path, session->width, session->height, session->frames, session->seed);
    if (length < 0 || (size_t)length >= sizeof(response))
        return media_refuse(err, YVEX_ERR_BOUNDS, "media completion response exceeds capacity");
    rc = message_emit(emit, context, YVEX_CLIENT_MESSAGE_FRAGMENT,
                      request, session, response, err);
    if (rc == YVEX_OK) rc = turn_complete(emit, context, request, session, seconds, err);
    return rc;
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
    message.generation_mode = YVEX_SERVER_GENERATION_MEDIA;
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
    (void)request_id;
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
        char text[MEDIA_PROMPT_CAP];
        if (!request->prompt || !request->prompt_bytes ||
            request->prompt_bytes >= sizeof(text)) {
            rc = media_refuse(err, YVEX_ERR_BOUNDS, "media prompt exceeds capacity");
            goto done;
        }
        memcpy(text, request->prompt, (size_t)request->prompt_bytes);
        text[request->prompt_bytes] = '\0';
        if (session->dialog == MEDIA_DIALOG_EMPTY ||
            session->dialog == MEDIA_DIALOG_COMPLETE ||
            session->dialog == MEDIA_DIALOG_FAILED) {
            yvex_core_text_copy(session->prompt, sizeof(session->prompt), text);
            session->profile_selected = 0;
            session->duration_selected = 0;
            session->steps_selected = 0;
            session->format_selected = 0;
            session->seed = registry->generation.seed;
        }
        rc = dialog_parse(registry, session, text, err);
        if (rc == YVEX_OK && (!session->profile_selected ||
                              !session->duration_selected ||
                              !session->steps_selected ||
                              !session->format_selected))
            rc = dialog_question(registry, emit, emit_context, request, session, err);
        else if (rc == YVEX_OK) {
            (void)pthread_mutex_unlock(&registry->mutex);
            return generation_execute(registry, session, request, emit, emit_context, err);
        }
    } else {
        rc = media_refuse(err, YVEX_ERR_UNSUPPORTED,
                          "operation is unavailable for conversational media sessions");
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
    status->schema_version = 1u;
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
                                       yvex_server_summary *summary, yvex_error *err)
{
    if (!registry || !summary)
        return media_refuse(err, YVEX_ERR_INVALID_ARG, "media summary is required");
    yvex_core_text_copy(summary->runtime_model_identity,
                        sizeof(summary->runtime_model_identity), registry->profile_identity);
    yvex_core_text_copy(summary->runtime_binding_identity,
                        sizeof(summary->runtime_binding_identity), registry->profile_identity);
    yvex_core_text_copy(summary->artifact_identity, sizeof(summary->artifact_identity),
                        registry->source_identity);
    yvex_core_text_copy(summary->physical_variant_identity,
                        sizeof(summary->physical_variant_identity), registry->profile_identity);
    summary->explicit_reasoning_channel_supported = 0;
    yvex_error_clear(err);
    return YVEX_OK;
}

void yvex_server_media_registry_close(server_media_registry **registry)
{
    server_media_registry *owner;
    if (!registry || !*registry) return;
    owner = *registry;
    if (owner->mutex_ready) (void)pthread_mutex_destroy(&owner->mutex);
    memset(owner, 0, sizeof(*owner));
    free(owner);
    *registry = NULL;
}
