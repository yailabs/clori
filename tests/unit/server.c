/*
 * Exercises configured-host truth, typed event/JSON projection, model-open refusal, privacy
 * defaults, and idempotent graceful close without requiring a model artifact.
 */
#include <stdlib.h>
#include <string.h>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <yvex/server.h>
#include <yvex/internal/graph.h>
#include <yvex/internal/media.h>

#include "src/server/private.h"
#include "tests/test.h"

static void test_options(yvex_server_options *options)
{
    memset(options, 0, sizeof(*options));
    options->artifact_path = "/definitely-absent/yvex-model.gguf";
    options->runtime_binding_path = "/definitely-absent/yvex-binding";
    options->target_id = "deepseek4-v4-flash-dspark";
    options->socket_path = "/tmp/yvex-test-host/yvexd.sock";
    options->backend = YVEX_BACKEND_KIND_CPU;
    options->context_capacity = 32u;
    options->prefill_chunk_tokens = 8u;
    options->maximum_new_tokens = 4u;
    options->maximum_output_bytes = 4096u;
    options->maximum_sessions = 2u;
    options->request_queue_capacity = 2u;
}

static int test_configured_summary_and_event(void)
{
    yvex_server_options options;
    yvex_server_summary summary;
    yvex_server_event event;
    yvex_client_message wire, decoded;
    yvex_server *server = NULL;
    yvex_error err;
    char json[2048];
    unsigned char frame[8192];
    unsigned long long frame_count = 0u;
    int rc;
    test_options(&options);
    rc = yvex_server_create(&server, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && server != NULL, "configured host create");
    rc = yvex_server_get_summary(server, &summary, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "configured host summary");
    YVEX_TEST_ASSERT(summary.status == YVEX_SERVER_STATUS_CONFIGURED,
                     "configured status");
    YVEX_TEST_ASSERT(summary.metrics.model_open_count == 0u,
                     "model not opened during create");
    YVEX_TEST_ASSERT(summary.metrics.queue_depth == 0u &&
                         summary.metrics.queue_capacity == 2u,
                     "configured queue capacity is immediately observable");
    YVEX_TEST_ASSERT(!summary.runtime_ready && !summary.generation_ready,
                     "no false readiness before start");
    rc = yvex_server_event_next(server, 0u, 0, &event, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "process event available");
    YVEX_TEST_ASSERT(event.kind == YVEX_SERVER_EVENT_PROCESS_START,
                     "process event kind");
    rc = yvex_server_event_json(&event, json, sizeof(json), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "event json");
    YVEX_TEST_ASSERT(strstr(json, "process.start") != NULL, "event name rendered");
    YVEX_TEST_ASSERT(strstr(json, "\"runtime_model_identity\"") != NULL &&
                         strstr(json, "\"artifact_identity\"") != NULL &&
                         strstr(json, "\"variant_identity\"") != NULL,
                     "raw event identity domains rendered");
    YVEX_TEST_ASSERT(strstr(json, "prompt") == NULL &&
                         strstr(json, "generated_text") == NULL,
                     "default telemetry excludes content");
    memset(&wire, 0, sizeof(wire));
    wire.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    wire.kind = YVEX_CLIENT_MESSAGE_EVENT;
    wire.status = YVEX_OK;
    wire.event = event;
    rc = yvex_protocol_message_encode(&wire, frame, sizeof(frame),
                                      &frame_count, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "sealed event message encode");
    rc = yvex_protocol_message_decode(frame, frame_count, &decoded, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK &&
                         strcmp(decoded.event.event_identity,
                                event.event_identity) == 0,
                     "sealed event message roundtrip");
    event.value_a++;
    rc = yvex_server_event_validate(&event, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_FORMAT,
                     "event evidence mutation refuses");
    rc = yvex_server_stop(server, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "configured host stop");
    rc = yvex_server_finish(server, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "configured host finish");
    rc = yvex_server_finish(server, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "host finish idempotent");
    rc = yvex_server_get_summary(server, &summary, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK &&
                         summary.status == YVEX_SERVER_STATUS_STOPPED,
                     "finished status remains observable");
    {
        unsigned long long cursor = 0u, attempts;
        int shutdown_complete = 0;
        for (attempts = 0u; attempts < 16u; ++attempts) {
            rc = yvex_server_event_next(server, cursor, 0, &event, &err);
            if (rc != YVEX_OK) break;
            cursor = event.sequence;
            if (event.kind == YVEX_SERVER_EVENT_RUNTIME_SHUTDOWN_COMPLETE) {
                shutdown_complete = 1;
                break;
            }
        }
        YVEX_TEST_ASSERT(shutdown_complete,
                         "shutdown completion precedes telemetry release");
    }
    yvex_server_close(&server);
    YVEX_TEST_ASSERT(server == NULL, "host close transfers owner");
    yvex_server_close(&server);
    return 0;
}

static int test_model_open_refusal(void)
{
    yvex_server_options options;
    yvex_server_summary summary;
    yvex_server *server = NULL;
    yvex_error err;
    int rc;
    test_options(&options);
    rc = yvex_server_create(&server, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "refusal host create");
    rc = yvex_server_start(server, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK, "missing artifact refuses start");
    rc = yvex_server_get_summary(server, &summary, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "failed summary remains available");
    YVEX_TEST_ASSERT(summary.status == YVEX_SERVER_STATUS_FAILED,
                     "failed start status");
    YVEX_TEST_ASSERT(summary.metrics.model_open_count == 0u,
                     "failed start does not count model open");
    yvex_server_close(&server);
    return 0;
}

static int test_bounded_telemetry_overflow(void)
{
    server_telemetry *telemetry = NULL;
    yvex_server_event event;
    yvex_server_metrics metrics;
    yvex_error err;
    unsigned long long cursor = 0u, index;
    int rc, saw_drop = 0;
    rc = yvex_server_telemetry_open(
        &telemetry, 2u, YVEX_SERVER_GENERATION_TARGET_ONLY,
        NULL, NULL, NULL, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "bounded telemetry open");
    for (index = 0u; index < 4u; ++index) {
        rc = yvex_server_telemetry_emit(
            telemetry, YVEX_SERVER_EVENT_GENERATION_PROGRESS,
            YVEX_SERVER_SEVERITY_DEBUG, "s", "r", "t", "decode",
            index, 0u, 0u, 0.0, 0.0, &err);
        YVEX_TEST_ASSERT(rc == YVEX_OK, "bounded telemetry publish");
    }
    rc = yvex_server_telemetry_metrics_copy(telemetry, &metrics, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && metrics.telemetry_dropped > 0u,
                     "overflow count is explicit");
    yvex_server_telemetry_resources(telemetry, 1024u, 2048u, 1u);
    yvex_server_telemetry_resources(telemetry, 512u, 1024u, 0u);
    rc = yvex_server_telemetry_metrics_copy(telemetry, &metrics, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && metrics.resident_host_bytes == 1024u &&
                         metrics.resident_device_bytes == 2048u &&
                         metrics.output_head_upload_count == 1u,
                     "resource metrics retain authoritative high-water facts");
    yvex_server_telemetry_openai_request(telemetry, 1, 0, 0, 0);
    yvex_server_telemetry_openai_request(telemetry, -1, 1, 0, 0);
    yvex_server_telemetry_openai_request(telemetry, 0, 0, 1, 1);
    rc = yvex_server_telemetry_metrics_copy(telemetry, &metrics, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && metrics.active_http_requests == 0u &&
                         metrics.completed_http_requests == 1u &&
                         metrics.failed_http_requests == 1u &&
                         metrics.cancelled_http_requests == 1u,
                     "integrated HTTP counters share server metrics");
    for (index = 0u; index < 2u; ++index) {
        rc = yvex_server_telemetry_next(telemetry, cursor, 0, &event, &err);
        YVEX_TEST_ASSERT(rc == YVEX_OK, "retained overflow event");
        cursor = event.sequence;
        saw_drop |= event.kind == YVEX_SERVER_EVENT_TELEMETRY_DROPPED;
    }
    YVEX_TEST_ASSERT(saw_drop, "overflow event is not silent");
    yvex_server_telemetry_close(&telemetry);
    return 0;
}

static int loopback_reserve(unsigned short *port)
{
    struct sockaddr_in address;
    socklen_t address_count = sizeof(address);
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(*port);
    if (bind(fd, (struct sockaddr *)&address, sizeof(address)) != 0 ||
        getsockname(fd, (struct sockaddr *)&address, &address_count) != 0 ||
        listen(fd, 1) != 0) {
        (void)close(fd);
        return -1;
    }
    *port = ntohs(address.sin_port);
    return fd;
}

static int test_openai_listener_admission(void)
{
    yvex_server_options options;
    yvex_server_summary summary;
    yvex_server *server = NULL;
    yvex_error err;
    unsigned short port = 0u;
    int blocker, probe, rc;

    blocker = loopback_reserve(&port);
    YVEX_TEST_ASSERT(blocker >= 0, "loopback collision fixture");
    test_options(&options);
    options.openai_enabled = 1;
    options.openai_port = port;
    options.openai_timeout_ms = 1000u;
    rc = yvex_server_create(&server, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_IO && server == NULL,
                     "OpenAI bind collision refuses host construction");
    (void)close(blocker);

    test_options(&options);
    options.openai_enabled = 1;
    options.openai_port = port;
    options.openai_timeout_ms = 1000u;
    rc = yvex_server_create(&server, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && server != NULL,
                     "OpenAI listener reserves before model start");
    rc = yvex_server_get_summary(server, &summary, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && summary.openai_listener_enabled &&
                         !summary.openai_listener_ready &&
                         summary.openai_port == port,
                     "configured listener status is truthful");
    rc = yvex_server_start(server, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK, "model refusal precedes HTTP acceptance");
    yvex_server_close(&server);
    probe = loopback_reserve(&port);
    YVEX_TEST_ASSERT(probe >= 0, "failed startup releases HTTP listener");
    (void)close(probe);
    return 0;
}

typedef struct {
    char text[YVEX_SERVER_FRAGMENT_CAP * 2u];
    unsigned long long count, started, completed;
} media_messages;

static int media_message_collect(void *context, const yvex_client_message *message,
                                 yvex_error *err)
{
    media_messages *messages = context;
    size_t used;
    if (!messages || !message) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "test.server.media",
                       "media test response is required");
        return YVEX_ERR_INVALID_ARG;
    }
    used = strlen(messages->text);
    if (message->byte_count > sizeof(messages->text) - used - 1u) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "test.server.media",
                       "media test response exceeds capacity");
        return YVEX_ERR_BOUNDS;
    }
    messages->count++;
    messages->started += message->kind == YVEX_CLIENT_MESSAGE_TURN_STARTED;
    messages->completed += message->kind == YVEX_CLIENT_MESSAGE_TURN_COMPLETE;
    if (message->byte_count) {
        memcpy(messages->text + used, message->bytes, (size_t)message->byte_count);
        messages->text[used + message->byte_count] = '\0';
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

static void media_options(yvex_server_media_options *options, const char *output_root)
{
    static const yvex_server_media_profile profiles[] = {
        {"preview", 192ull, 192ull, 124ull, 1},
        {"smoke", 32ull, 32ull, 345ull, 0},
    };
    memset(options, 0, sizeof(*options));
    options->schema_version = YVEX_SERVER_MEDIA_SCHEMA_V1;
    options->output_root = output_root;
    options->request_template.schema_version = YVEX_RUNTIME_AV_GENERATION_SCHEMA_V1;
    options->request_template.target = "minimax-h3-base-fl2va-t2va";
    options->request_template.source_identity =
        "91972f8e4e6562562456c339b43eed1fba5f7b9d7fb13987f495b416a5109b5e";
    options->request_template.text_artifact_path = "/not-opened/text.gguf";
    options->request_template.transformer_artifact_path = "/not-opened/transformer.gguf";
    options->request_template.video_artifact_path = "/not-opened/video.gguf";
    options->request_template.audio_artifact_path = "/not-opened/audio.gguf";
    options->request_template.fps_numerator = 24ull;
    options->request_template.fps_denominator = 1ull;
    options->request_template.seed = 42ull;
    options->profiles = profiles;
    options->profile_count = sizeof(profiles) / sizeof(profiles[0]);
    options->frames_per_chunk = 17ull;
    options->frame_remainder = 5ull;
    options->minimum_frames = 124ull;
    options->maximum_frames = 345ull;
    options->minimum_inference_steps = 2ull;
    options->maximum_inference_steps = 64ull;
    options->canvas_multiple = 32ull;
    options->maximum_canvas_pixels = 192ull * 192ull;
}

static int media_registry_request(server_media_registry *registry,
                                  yvex_client_operation operation,
                                  const char *session, const char *prompt,
                                  media_messages *messages, yvex_error *err)
{
    yvex_client_request request = {0};
    request.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    request.operation = operation;
    request.request_number = messages->count + 1ull;
    strcpy(request.session_name, session);
    request.prompt = (const unsigned char *)prompt;
    request.prompt_bytes = prompt ? strlen(prompt) : 0u;
    return yvex_server_media_registry_execute(
        registry, &request, "media-test", 0.0, media_message_collect, messages, err);
}

static int test_media_dialog_and_refusals(void)
{
    char root[] = "/tmp/yvex-media-dialog-XXXXXX";
    const char *bad_prompts[] = {
        "Eclissi realistica 4K, 5 secondi, 19 punti sigma, AVI",
        "Eclissi realistica preview, 4 secondi, 19 punti sigma, AVI",
        "Eclissi realistica preview, 5 secondi, 65 punti sigma, AVI",
        "Eclissi realistica preview, 5 secondi, 19 punti sigma, MP4",
        "Eclissi realistica preview, 15 secondi, 19 punti sigma, AVI",
    };
    const int bad_status[] = {
        YVEX_ERR_UNSUPPORTED, YVEX_ERR_BOUNDS, YVEX_ERR_BOUNDS, YVEX_ERR_UNSUPPORTED,
        YVEX_ERR_BOUNDS,
    };
    yvex_server_media_options options;
    yvex_server_summary first = {0}, repeated = {0};
    server_media_registry *registry = NULL, *second = NULL;
    server_telemetry *telemetry = NULL, *second_telemetry = NULL;
    media_messages messages = {0};
    yvex_error err;
    unsigned long long index;
    int rc;
    YVEX_TEST_ASSERT(mkdtemp(root) != NULL, "media dialogue output root");
    media_options(&options, root);
    rc = yvex_server_telemetry_open(&telemetry, 16u, YVEX_SERVER_GENERATION_MEDIA,
                                    NULL, NULL, NULL, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "media dialogue telemetry");
    rc = yvex_server_media_registry_open(&registry, &options, telemetry, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "media dialogue registry");
    YVEX_TEST_ASSERT(media_registry_request(registry, YVEX_CLIENT_OP_SESSION_NEW,
                                            "eclipse", NULL, &messages, &err) == YVEX_OK,
                     "media dialogue session");
    memset(&messages, 0, sizeof(messages));
    rc = media_registry_request(
        registry, YVEX_CLIENT_OP_GENERATION_TURN, "eclipse",
        "Genera un video realistico dell eclissi del 12 agosto 2026 vista dall Italia",
        &messages, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && messages.started == 1ull &&
                         messages.completed == 1ull &&
                         strstr(messages.text, "preview (192x192, max 124 frame)") &&
                         strstr(messages.text, "5 a 15 secondi") &&
                         strstr(messages.text, "punti sigma") && strstr(messages.text, "AVI") &&
                         messages.text[strlen(messages.text) - 1u] == '\n',
                     "media dialogue requests every unresolved creative parameter");
    memset(&messages, 0, sizeof(messages));
    rc = media_registry_request(registry, YVEX_CLIENT_OP_GENERATION_TURN, "eclipse",
                                "preview, 5 secondi, 19 punti sigma, seed 42",
                                &messages, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && strstr(messages.text, "AVI") &&
                         !strstr(messages.text, "Qualità:") &&
                         !strstr(messages.text, "Durata:") &&
                         !strstr(messages.text, "Iterazioni:"),
                     "media dialogue retains selected parameters and asks only for format");
    memset(&messages, 0, sizeof(messages));
    YVEX_TEST_ASSERT(media_registry_request(registry, YVEX_CLIENT_OP_SESSION_NEW,
                                            "smoke-geometry", NULL,
                                            &messages, &err) == YVEX_OK,
                     "media geometry dialogue session");
    memset(&messages, 0, sizeof(messages));
    rc = media_registry_request(
        registry, YVEX_CLIENT_OP_GENERATION_TURN, "smoke-geometry",
        "Smoke 32x32, 5 secondi, 2 punti sigma, seed 42", &messages, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && strstr(messages.text, "AVI") &&
                         !strstr(messages.text, "Qualità:") &&
                         !strstr(messages.text, "Durata:") &&
                         !strstr(messages.text, "Iterazioni:"),
                     "media dimensions do not alias the adjacent duration");
    for (index = 0ull; index < sizeof(bad_prompts) / sizeof(bad_prompts[0]); ++index) {
        char name[32];
        (void)snprintf(name, sizeof(name), "bad-%llu", index);
        memset(&messages, 0, sizeof(messages));
        YVEX_TEST_ASSERT(media_registry_request(registry, YVEX_CLIENT_OP_SESSION_NEW,
                                                name, NULL, &messages, &err) == YVEX_OK,
                         "media refusal session");
        memset(&messages, 0, sizeof(messages));
        rc = media_registry_request(registry, YVEX_CLIENT_OP_GENERATION_TURN, name,
                                    bad_prompts[index], &messages, &err);
        YVEX_TEST_ASSERT(rc == bad_status[index] && messages.started == 0ull,
                         "media refusal occurs before generation");
    }
    YVEX_TEST_ASSERT(yvex_server_media_registry_summary(registry, &first, &err) == YVEX_OK,
                     "media first identity");
    YVEX_TEST_ASSERT(yvex_server_telemetry_open(
                         &second_telemetry, 16u, YVEX_SERVER_GENERATION_MEDIA,
                         NULL, NULL, NULL, &err) == YVEX_OK &&
                         yvex_server_media_registry_open(
                             &second, &options, second_telemetry, &err) == YVEX_OK &&
                         yvex_server_media_registry_summary(second, &repeated, &err) == YVEX_OK &&
                         !strcmp(first.runtime_model_identity,
                                 repeated.runtime_model_identity) &&
                         !strcmp(first.artifact_identity, repeated.artifact_identity),
                     "media profile and source identities are deterministic");
    yvex_server_media_registry_close(&second);
    yvex_server_telemetry_close(&second_telemetry);
    yvex_server_media_registry_close(&registry);
    yvex_server_telemetry_close(&telemetry);
    YVEX_TEST_ASSERT(rmdir(root) == 0, "media dialogue output root removed empty");
    return 0;
}

static int test_media_server_starts_without_model(void)
{
    char root[] = "/tmp/yvex-media-host-XXXXXX";
    char socket_path[YVEX_SERVER_SOCKET_PATH_CAP];
    yvex_server_media_options media;
    yvex_server_options options;
    yvex_server_summary summary;
    yvex_server *server = NULL;
    yvex_error err;
    int rc;
    YVEX_TEST_ASSERT(mkdtemp(root) != NULL, "media host output root");
    YVEX_TEST_ASSERT(snprintf(socket_path, sizeof(socket_path), "%s/yvexd.sock", root) > 0,
                     "media host socket path");
    test_options(&options);
    options.artifact_path = NULL;
    options.runtime_binding_path = NULL;
    options.target_id = "minimax-h3-base-fl2va-t2va";
    options.socket_path = socket_path;
    options.backend = YVEX_BACKEND_KIND_CUDA;
    options.generation_mode = YVEX_SERVER_GENERATION_MEDIA;
    media_options(&media, root);
    rc = yvex_server_create(&server, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && server != NULL, "media host create");
    YVEX_TEST_ASSERT(yvex_server_media_configure(server, &media, &err) == YVEX_OK,
                     "media host configure");
    YVEX_TEST_ASSERT(yvex_server_start(server, &err) == YVEX_OK,
                     "media host starts without loading components");
    YVEX_TEST_ASSERT(yvex_server_get_summary(server, &summary, &err) == YVEX_OK &&
                         summary.status == YVEX_SERVER_STATUS_READY &&
                         summary.runtime_ready && summary.generation_ready &&
                         summary.generation_mode == YVEX_SERVER_GENERATION_MEDIA &&
                         summary.metrics.model_open_count == 0ull &&
                         summary.metrics.artifact_open_count == 0ull &&
                         summary.metrics.resident_device_bytes == 0ull,
                     "media host readiness does not promote model or CUDA residency");
    YVEX_TEST_ASSERT(yvex_server_finish(server, &err) == YVEX_OK,
                     "media host finishes cleanly");
    yvex_server_close(&server);
    YVEX_TEST_ASSERT(rmdir(root) == 0, "media host output root removed empty");
    return 0;
}

static int test_media_family_profile(void)
{
    const yvex_component_variant_adapter *adapter;
    yvex_media_target_profile target;
    yvex_runtime_media_host_profile profile, repeated;
    yvex_error err;
    int rc;

    adapter = yvex_graph_component_variant_find_family("minimax-h3");
    YVEX_TEST_ASSERT(adapter && adapter->media_target_profile && adapter->media_execution,
                     "family catalog exposes one media adapter");
    rc = adapter->media_target_profile(&target, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "family builds media target facts");
    rc = yvex_runtime_media_host_profile_build(
        &profile, &target, adapter->media_execution, "/models/minimax-h3/revision",
        "/outputs/minimax-h3", &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "family catalog builds media host profile");
    YVEX_TEST_ASSERT(profile.schema_version == YVEX_RUNTIME_MEDIA_HOST_SCHEMA_V1,
                     "media host profile schema");
    YVEX_TEST_ASSERT_STREQ(profile.request_template.target,
                           "minimax-h3-fl2va", "media host target");
    YVEX_TEST_ASSERT(profile.request_template.component_backend == YVEX_BACKEND_KIND_CUDA,
                     "media host CUDA backend");
    YVEX_TEST_ASSERT(profile.request_template.condition && profile.request_template.latent &&
                         profile.request_template.video_decode &&
                         profile.request_template.audio_decode,
                     "media host execution callbacks");
    YVEX_TEST_ASSERT(profile.profile_count == 2ull &&
                         !strcmp(profile.profiles[0].name, "preview") &&
                         !strcmp(profile.profiles[1].name, "smoke"),
                     "media host user profiles");
    YVEX_TEST_ASSERT(strstr(profile.transformer_artifact,
                            "physical-v4/transformer.gguf") != NULL,
                     "media host transformer artifact path");
    rc = yvex_runtime_media_host_profile_build(
        &repeated, &target, adapter->media_execution, "/models/minimax-h3/revision",
        "/outputs/minimax-h3", &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK &&
                         !strcmp(profile.request_template.source_identity,
                                 repeated.request_template.source_identity) &&
                         !strcmp(profile.text_artifact, repeated.text_artifact) &&
                         profile.maximum_canvas_pixels == repeated.maximum_canvas_pixels,
                     "media host profile is deterministic");
    target.text_artifact = "../outside.gguf";
    rc = yvex_runtime_media_host_profile_build(
        &repeated, &target, adapter->media_execution, "/models/minimax-h3/revision",
        "/outputs/minimax-h3", &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_BOUNDS,
                     "media host refuses component-root traversal");
    YVEX_TEST_ASSERT(!yvex_graph_component_variant_find_family("unknown"),
                     "family without a media adapter refuses");
    return 0;
}

/* Prove provider correlation is identity-bound into the authoritative event stream. */
static int test_provider_telemetry(void)
{
    static const unsigned char text[] = "hello";
    yvex_provider_message message = {0};
    yvex_provider_request request = {0};
    yvex_runtime_speculation_progress progress = {0};
    server_telemetry *telemetry = NULL;
    yvex_server_event emitted, event, observation;
    yvex_error err;
    char json[4096];
    int rc;
    YVEX_TEST_ASSERT_STREQ(
        yvex_server_event_kind_name(YVEX_SERVER_EVENT_GENERATION_PROFILE),
        "generation.profile", "generation profile event spelling");
    message.role = YVEX_PROVIDER_ROLE_USER;
    message.content.bytes = text;
    message.content.count = sizeof(text) - 1u;
    request.schema_version = YVEX_PROVIDER_SCHEMA_V1;
    strcpy(request.model, "deepseek4-v4-flash-dspark");
    request.messages = &message;
    request.message_count = 1u;
    request.maximum_output_tokens = 4u;
    strcpy(request.adapter, "openai");
    strcpy(request.external_correlation_id, "chatcmpl-yvex-1");
    rc = yvex_provider_request_seal(&request, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "provider telemetry request seal");
    rc = yvex_server_telemetry_open(
        &telemetry, 4u, YVEX_SERVER_GENERATION_DSPARK,
        NULL, NULL, NULL, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "provider telemetry open");
    rc = yvex_server_telemetry_emit_provider(
        telemetry, YVEX_SERVER_EVENT_REQUEST_STARTED,
        YVEX_SERVER_SEVERITY_INFO, "session", "r1", "t1", "turn",
        1u, 0u, 4u, 0.0, 0.0, NULL, &request, &emitted, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "provider telemetry emit");
    rc = yvex_server_telemetry_next(telemetry, 0u, 0, &event, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "provider telemetry read");
    YVEX_TEST_ASSERT_STREQ(emitted.event_identity, event.event_identity,
                           "emitted event is the sealed retained event");
    YVEX_TEST_ASSERT_STREQ(event.provider_adapter, "openai",
                           "provider adapter event fact");
    YVEX_TEST_ASSERT_STREQ(event.provider_request_identity,
                           request.request_identity,
                           "provider request event identity");
    YVEX_TEST_ASSERT(event.generation_mode == YVEX_SERVER_GENERATION_DSPARK,
                     "generic telemetry preserves the configured generation mode");
    observation = event;
    observation.process_id++;
    observation.wall_time_ns++;
    observation.monotonic_time_ns++;
    rc = yvex_server_event_validate(&observation, &err);
    YVEX_TEST_ASSERT(
        rc == YVEX_OK,
        "process and clock observations do not enter semantic event identity");
    rc = yvex_server_event_json(&event, json, sizeof(json), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && strstr(json, "\"provider\":\"openai\"") != NULL,
                     "provider correlation JSON");
    event.external_correlation_id[0] = 'X';
    rc = yvex_server_event_validate(&event, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_FORMAT,
                     "provider correlation mutation refuses");
    progress.schema_version = YVEX_RUNTIME_GENERATION_SCHEMA_V3;
    progress.kind = YVEX_SPECULATION_PROGRESS_CYCLE_COMMITTED;
    progress.cycle = 2u;
    progress.proposed_tokens = 5u;
    progress.selected_verification_tokens = 5u;
    progress.accepted_tokens = 3u;
    progress.rejected_tokens = 1u;
    progress.discarded_tokens = 1u;
    progress.verification_count = 1u;
    progress.confidence_logit_count = 5u;
    progress.confidence_logit_minimum = -1.0;
    progress.confidence_logit_maximum = 2.0;
    progress.confidence_logit_mean = 0.5;
    progress.seconds = 0.25;
    strcpy(progress.policy_identity, request.request_identity);
    rc = yvex_server_telemetry_emit_provider(
        telemetry, YVEX_SERVER_EVENT_SPECULATIVE_CYCLE_COMMITTED,
        YVEX_SERVER_SEVERITY_INFO, "session", "r1", "t1", "speculation",
        0u, 0u, 0u, progress.seconds, 0.0, &progress, &request, &emitted,
        &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "typed speculation telemetry emit");
    rc = yvex_server_telemetry_next(telemetry, event.sequence, 0, &event, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && event.speculative_cycle == 2u &&
                         event.proposed_tokens == 5u &&
                         event.accepted_tokens == 3u &&
                         event.rejected_tokens == 1u &&
                         event.discarded_tokens == 1u &&
                         event.verification_count == 1u &&
                         event.confidence_logit_count == 5u &&
                         event.confidence_logit_minimum == -1.0 &&
                         event.confidence_logit_maximum == 2.0 &&
                         event.confidence_logit_mean == 0.5 &&
                         event.generation_mode == YVEX_SERVER_GENERATION_DSPARK,
                     "typed speculation telemetry facts");
    rc = yvex_server_event_json(&event, json, sizeof(json), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK &&
                         strstr(json, "\"speculative_cycle\":2") != NULL &&
                         strstr(json, "\"accepted_tokens\":3") != NULL &&
                         strstr(json, "\"discarded_tokens\":1") != NULL &&
                         strstr(json, "\"confidence_logit_count\":5") != NULL,
                     "typed speculation telemetry JSON");
    yvex_server_telemetry_close(&telemetry);
    return 0;
}

int yvex_test_server(void)
{
    if (test_configured_summary_and_event() != 0) return 1;
    if (test_model_open_refusal() != 0) return 1;
    if (test_bounded_telemetry_overflow() != 0) return 1;
    if (test_provider_telemetry() != 0) return 1;
    if (test_openai_listener_admission() != 0) return 1;
    if (test_media_dialog_and_refusals() != 0) return 1;
    if (test_media_family_profile() != 0) return 1;
    if (test_media_server_starts_without_model() != 0) return 1;
    return 0;
}
