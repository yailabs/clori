/*
 * YVEX - runtime host and telemetry tests
 *
 * Purpose: prove configured-host truth, typed event/JSON projection, model-open refusal,
 * privacy defaults, and idempotent graceful close without requiring a model artifact.
 */
#include <string.h>

#include <yvex/server.h>

#include "src/server/private.h"
#include "tests/test.h"

static void test_options(yvex_server_options *options)
{
    memset(options, 0, sizeof(*options));
    options->artifact_path = "/definitely-absent/yvex-model.gguf";
    options->runtime_binding_path = "/definitely-absent/yvex-binding";
    options->target_id = "deepseek4-v4-flash";
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
    rc = yvex_server_telemetry_open(&telemetry, 2u, NULL, NULL, NULL, &err);
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

int yvex_test_server(void)
{
    if (test_configured_summary_and_event() != 0) return 1;
    if (test_model_open_refusal() != 0) return 1;
    if (test_bounded_telemetry_overflow() != 0) return 1;
    return 0;
}
