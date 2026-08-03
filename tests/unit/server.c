/*
 * Exercises configured-host truth, typed event/JSON projection, model-open refusal, privacy
 * defaults, and idempotent graceful close without requiring a model artifact.
 */
#include <string.h>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <yvex/server.h>

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
    return 0;
}
