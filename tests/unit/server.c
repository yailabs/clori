/*
 * Exercises configured-host truth, typed event/JSON projection, model-open refusal, privacy
 * defaults, and idempotent graceful close without requiring a model artifact.
 */
#include <stdlib.h>
#include <string.h>
#include <stdlib.h>

#include <arpa/inet.h>
#include <pthread.h>
#include <sys/socket.h>
#include <unistd.h>

#include <yvex/server.h>
#include <yvex/internal/core.h>
#include <yvex/internal/graph.h>
#include <yvex/internal/media.h>

#include "src/server/private.h"
#include "tests/test.h"

static void test_options(yvex_server_options *options)
{
    memset(options, 0, sizeof(*options));
    options->schema_version = YVEX_SERVER_OPTIONS_SCHEMA_V2;
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
    options->concurrent_sequences = 1u;
}

static int test_automatic_reasoning_policy(void)
{
    YVEX_TEST_ASSERT(
        server_reasoning_automatic_policy() == YVEX_REASONING_DISABLED,
        "automatic reasoning selects ordinary chat independently of capability");
    return 0;
}

static void server_test_identity(char identity[YVEX_SHA256_HEX_CAP], char digit)
{
    memset(identity, digit, YVEX_SHA256_HEX_CAP - 1u);
    identity[YVEX_SHA256_HEX_CAP - 1u] = '\0';
}

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    server_scheduler *scheduler;
    unsigned long long active, peak, completed;
    unsigned long long ready_width_two, ready_wait_ns;
    unsigned long long ready_timeouts;
    int release, first_active, third_active, first_done, violation;
} scheduler_probe;

typedef struct {
    scheduler_probe *probe;
    int id;
} scheduler_probe_work;

static void scheduler_probe_execute(void *context, void *opaque)
{
    scheduler_probe *probe = context;
    scheduler_probe_work *work = opaque;
    unsigned long long ready_width = 0ull, wait_ns = 0ull;
    int timed_out = 0;
    yvex_error err;
    int ready_rc = yvex_server_scheduler_execution_ready(
        probe->scheduler, work->id == 3 ? "other" : "same", &ready_width,
        &wait_ns, &timed_out, &err);
    (void)pthread_mutex_lock(&probe->mutex);
    if (ready_rc != YVEX_OK) probe->violation = 1;
    probe->ready_width_two += ready_width == 2ull;
    probe->ready_wait_ns += wait_ns;
    probe->ready_timeouts += timed_out != 0;
    if (work->id == 2 && !probe->first_done) probe->violation = 1;
    probe->active++;
    if (probe->active > probe->peak) probe->peak = probe->active;
    if (work->id == 1) probe->first_active = 1;
    if (work->id == 3) probe->third_active = 1;
    (void)pthread_cond_broadcast(&probe->condition);
    while (!probe->release)
        (void)pthread_cond_wait(&probe->condition, &probe->mutex);
    probe->active--;
    if (work->id == 1) probe->first_done = 1;
    probe->completed++;
    (void)pthread_cond_broadcast(&probe->condition);
    (void)pthread_mutex_unlock(&probe->mutex);
}

static int test_scheduler_serialization(void)
{
    scheduler_probe probe = {0};
    scheduler_probe_work work[] = {
        {&probe, 1}, {&probe, 2}, {&probe, 3}
    };
    server_scheduler *scheduler = NULL;
    server_scheduler_summary summary = {0};
    yvex_error err;
    YVEX_TEST_ASSERT(pthread_mutex_init(&probe.mutex, NULL) == 0 &&
                         pthread_cond_init(&probe.condition, NULL) == 0,
                     "scheduler probe synchronization opens");
    YVEX_TEST_ASSERT(
        yvex_server_scheduler_open(
            &scheduler, 3ull, 2ull, scheduler_probe_execute, NULL,
            &probe, &err) == YVEX_OK && scheduler,
        "two-worker scheduler opens");
    probe.scheduler = scheduler;
    YVEX_TEST_ASSERT(yvex_server_scheduler_start(scheduler, &err) == YVEX_OK,
                     "two-worker scheduler starts");
    YVEX_TEST_ASSERT(
        yvex_server_scheduler_submit(
            scheduler, &work[0], "same", 1, NULL, &err) == YVEX_OK,
        "scheduler accepts the first compatible candidate");
    YVEX_TEST_ASSERT(
        yvex_server_scheduler_submit(
                scheduler, &work[1], "same", 1, NULL, &err) == YVEX_OK &&
            yvex_server_scheduler_submit(
                scheduler, &work[2], "other", 1, NULL, &err) == YVEX_OK,
        "scheduler admits independent peers without a speculative wait window");
    (void)pthread_mutex_lock(&probe.mutex);
    while (probe.active < 2ull)
        (void)pthread_cond_wait(&probe.condition, &probe.mutex);
    YVEX_TEST_ASSERT(probe.first_active && probe.third_active &&
                         probe.peak == 2ull && !probe.violation,
                     "independent keys execute while one same-key request waits");
    probe.release = 1;
    (void)pthread_cond_broadcast(&probe.condition);
    while (probe.completed < 3ull)
        (void)pthread_cond_wait(&probe.condition, &probe.mutex);
    (void)pthread_mutex_unlock(&probe.mutex);
    YVEX_TEST_ASSERT(
        yvex_server_scheduler_finish(scheduler, &err) == YVEX_OK,
        "scheduler drains and joins workers");
    yvex_server_scheduler_snapshot(scheduler, &summary);
    YVEX_TEST_ASSERT(!summary.queued && !summary.active &&
                         summary.execution_ready_limit_ns == 3000000000ull &&
                         !summary.execution_ready_timeouts &&
                         summary.maximum_execution_ready_width == 2ull &&
                         probe.ready_width_two >= 1ull &&
                         !probe.ready_timeouts &&
                         !probe.violation &&
                         probe.first_done,
                     "same-key work starts only after its predecessor completes");
    yvex_server_scheduler_close(&scheduler);
    (void)pthread_cond_destroy(&probe.condition);
    (void)pthread_mutex_destroy(&probe.mutex);
    return 0;
}

static int test_session_store(void)
{
    static const char second_message[] = {'o', 'k', '\0', '!'};
    yvex_prompt_message messages[2] = {
        {.role = YVEX_PROMPT_ROLE_USER, .content = "ciao", .content_len = 4ull},
        {.role = YVEX_PROMPT_ROLE_ASSISTANT,
         .content = second_message, .content_len = sizeof(second_message)}};
    unsigned int tokens[3] = {1u, 7u, 2u};
    server_session_store_view view = {0};
    server_session_store_state restored = {0};
    unsigned char *bytes = NULL, *corrupt = NULL, *rejected = NULL;
    unsigned long long byte_count = 0ull, rejected_count = 0ull;
    char payload_identity[YVEX_SHA256_HEX_CAP];
    yvex_error err;
    view.messages = messages;
    view.message_count = 2ull;
    view.committed_tokens = tokens;
    view.committed_count = 3ull;
    view.turn_count = 1ull;
    view.message_history_generation = 2ull;
    view.transcript_generation = 2ull;
    view.policy_set = 1;
    view.reasoning_policy = YVEX_REASONING_ENABLED;
    view.policy.schema_version = YVEX_RUNTIME_SAMPLING_SCHEMA_V1;
    view.policy.strategy = YVEX_SAMPLING_STRATEGY_STOCHASTIC;
    view.policy.temperature = 0.8;
    view.policy.top_p = 1.0;
    view.policy.typical_p = 1.0;
    view.policy.seed_present = 1;
    view.policy.seed = 42ull;
    YVEX_TEST_ASSERT(
        yvex_runtime_sampling_policy_seal(&view.policy, 8ull, &err) == YVEX_OK,
        "session checkpoint sampling policy seals");
    view.generation_checkpoint_present = 1;
    view.generation_checkpoint.schema_version =
        YVEX_RUNTIME_GENERATION_CHECKPOINT_SCHEMA_V1;
    view.generation_checkpoint.sampling.schema_version =
        YVEX_RUNTIME_SAMPLING_CHECKPOINT_SCHEMA_V1;
    view.generation_checkpoint.sampling.rng_state = 9ull;
    view.generation_checkpoint.sampling.rng_increment = 3ull;
    view.generation_checkpoint.sampling.successful_draws = 4ull;
    yvex_runtime_identity_copy(
        view.generation_checkpoint.sampling.policy_identity,
        view.policy.policy_identity);
    server_test_identity(view.generation_checkpoint.sampling.rng_state_identity, '2');
    server_test_identity(view.generation_checkpoint.sampling.checkpoint_identity, '3');
    server_test_identity(view.generation_checkpoint.generation_plan_identity, '4');
    server_test_identity(view.generation_checkpoint.checkpoint_identity, '5');
    server_test_identity(view.last_turn_identity, '6');
    server_test_identity(view.state_digest, '7');
    server_test_identity(view.generated_token_identity, '8');
    server_test_identity(view.generated_text_digest, '9');
    YVEX_TEST_ASSERT(
        yvex_server_session_store_encode(
            &view, &bytes, &byte_count, payload_identity, &err) == YVEX_OK &&
            bytes && byte_count > 0ull && yvex_sha256_hex_valid(payload_identity) &&
            yvex_server_session_store_decode(
                bytes, byte_count, 2ull, 64ull, 3ull, 8ull,
                &restored, &err) == YVEX_OK &&
            restored.message_count == 2ull && restored.committed_count == 3ull &&
            restored.turn_count == 1ull && restored.policy_set &&
            restored.reasoning_policy == YVEX_REASONING_ENABLED &&
            memcmp(restored.committed_tokens, tokens, sizeof(tokens)) == 0 &&
            restored.messages[1].content_len == sizeof(second_message) &&
            memcmp(restored.messages[1].content, second_message,
                   sizeof(second_message)) == 0 &&
            strcmp(restored.payload_identity, payload_identity) == 0,
        "session checkpoint roundtrips messages, tokens, policy, and RNG facts");
    yvex_server_session_store_close(&restored);
    view.generation_checkpoint.sampling.policy_identity[0] = '0';
    YVEX_TEST_ASSERT(
        yvex_server_session_store_encode(
            &view, &rejected, &rejected_count, payload_identity, &err) ==
            YVEX_ERR_INVALID_ARG && !rejected && !rejected_count,
        "session checkpoint rejects RNG state bound to another policy");
    yvex_runtime_identity_copy(
        view.generation_checkpoint.sampling.policy_identity,
        view.policy.policy_identity);
    corrupt = malloc((size_t)byte_count);
    YVEX_TEST_ASSERT(corrupt != NULL, "session checkpoint corruption copy allocates");
    memcpy(corrupt, bytes, (size_t)byte_count);
    corrupt[byte_count / 2ull] ^= 1u;
    YVEX_TEST_ASSERT(
        yvex_server_session_store_decode(
            corrupt, byte_count, 2ull, 64ull, 3ull, 8ull,
            &restored, &err) == YVEX_ERR_FORMAT &&
            yvex_server_session_store_decode(
                bytes, byte_count, 2ull, 64ull, 2ull, 8ull,
                &restored, &err) == YVEX_ERR_FORMAT,
        "session checkpoint corruption and capacity mismatch refuse before publication");
    free(corrupt);
    yvex_server_session_store_bytes_close(&bytes);
    return 0;
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

static int test_adaptive_prefill_policy(void)
{
    static const struct {
        unsigned long long context, concurrency, expected;
    } cases[] = {
        {4096ull, 1ull, 64ull},
        {4096ull, 2ull, 4ull},
        {4096ull, 4ull, 4ull},
        {4096ull, 8ull, 8ull},
        {32ull, 1ull, 32ull},
    };
    yvex_server_options options;
    yvex_server_summary summary;
    yvex_server *server = NULL;
    yvex_error err;
    size_t index;
    int rc;
    for (index = 0u; index < sizeof(cases) / sizeof(cases[0]); ++index) {
        test_options(&options);
        options.context_capacity = cases[index].context;
        options.maximum_sessions = cases[index].concurrency;
        options.concurrent_sequences = cases[index].concurrency;
        options.prefill_chunk_tokens = 0ull;
        rc = yvex_server_create(&server, &options, &err);
        YVEX_TEST_ASSERT(rc == YVEX_OK && server,
                         "adaptive prefill host create");
        rc = yvex_server_get_summary(server, &summary, &err);
        YVEX_TEST_ASSERT(rc == YVEX_OK &&
                             summary.prefill_chunk_tokens == cases[index].expected,
                         "adaptive prefill policy is inspectable");
        yvex_server_close(&server);
    }
    test_options(&options);
    options.concurrent_sequences = 2ull;
    options.prefill_chunk_tokens = 7ull;
    rc = yvex_server_create(&server, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && server,
                     "explicit prefill host create");
    rc = yvex_server_get_summary(server, &summary, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && summary.prefill_chunk_tokens == 7ull,
                     "explicit prefill override remains authoritative");
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
    options.schema_version = 1u;
    rc = yvex_server_create(&server, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_INVALID_ARG && !server,
                     "old server-options schema refuses");
    test_options(&options);
    options.concurrent_sequences = options.maximum_sessions + 1ull;
    rc = yvex_server_create(&server, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_INVALID_ARG && !server,
                     "execution concurrency above session capacity refuses");
    test_options(&options);
    options.socket_path = "/tmp/yvex-unsafe.sock";
    rc = yvex_server_create(&server, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && server, "unsafe socket host create");
    rc = yvex_server_start(server, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_IO, "unsafe socket refuses before model start");
    yvex_server_close(&server);
    test_options(&options);
    rc = yvex_server_create(&server, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "refusal host create");
    rc = yvex_server_start(server, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK, "missing artifact refuses start");
    YVEX_TEST_ASSERT(strstr(yvex_error_message(&err), "model admission refused:") != NULL &&
                         strstr(yvex_error_message(&err), "field=runtime-binding") != NULL,
                     "model-open refusal preserves typed failure facts");
    rc = yvex_server_get_summary(server, &summary, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "failed summary remains available");
    YVEX_TEST_ASSERT(summary.status == YVEX_SERVER_STATUS_FAILED, "failed start status");
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
    yvex_server_telemetry_model_opened(telemetry, 4096u, 0u, 0u, 0u);
    yvex_server_telemetry_resources(telemetry, 1024u, 2048u, 1u);
    yvex_server_telemetry_resources(telemetry, 512u, 1024u, 0u);
    rc = yvex_server_telemetry_metrics_copy(telemetry, &metrics, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && metrics.mapped_artifact_bytes == 4096u &&
                         metrics.resident_host_bytes == 1024u &&
                         metrics.resident_device_bytes == 2048u &&
                         metrics.output_head_upload_count == 1u,
                     "mapped backing is distinct from resident resource high-water facts");
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

static int media_fixture_admit(
    const char *component, const yvex_artifact *artifact, const yvex_gguf *gguf,
    const yvex_tensor_table *tensors, yvex_complete_artifact_admission *out,
    yvex_artifact_admission_failure *failure, yvex_error *err)
{
    if (failure) memset(failure, 0, sizeof(*failure));
    if (!component || (strcmp(component, "text_encoder") != 0 &&
                       strcmp(component, "transformer") != 0 &&
                       strcmp(component, "video_vae") != 0 &&
                       strcmp(component, "audio_vae") != 0) ||
        !artifact || !gguf || !tensors || !out) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "test.server.media-admit",
                       "four exact fixture component views are required");
        return YVEX_ERR_INVALID_ARG;
    }
    memset(out, 0, sizeof(*out));
    out->artifact_class = YVEX_ARTIFACT_CLASS_COMPLETE_YVEX;
    out->tensor_count = yvex_tensor_table_count(tensors);
    out->payload_bytes = 128ull;
    out->file_bytes = yvex_artifact_size(artifact);
    out->materialization_input_ready = 1;
    out->complete = 1;
    yvex_core_text_copy(out->artifact_path, sizeof(out->artifact_path),
                        yvex_artifact_path(artifact));
    yvex_core_text_copy(out->artifact_identity, sizeof(out->artifact_identity),
                        "2222222222222222222222222222222222222222222222222222222222222222");
    yvex_core_text_copy(out->logical_component_identity,
                        sizeof(out->logical_component_identity),
                        "3333333333333333333333333333333333333333333333333333333333333333");
    yvex_core_text_copy(out->admission_identity, sizeof(out->admission_identity),
                        "4444444444444444444444444444444444444444444444444444444444444444");
    if (yvex_artifact_snapshot_get(artifact, &out->file_snapshot, err) != YVEX_OK)
        return yvex_error_code(err);
    yvex_error_clear(err);
    return YVEX_OK;
}

static int media_options(yvex_server_media_options *options, const char *output_root)
{
    static const yvex_server_media_profile profiles[] = {
        {"preview", 192ull, 192ull, 124ull, 1},
        {"preview-256", 256ull, 256ull, 124ull, 0},
        {"preview-384", 384ull, 384ull, 124ull, 0},
        {"source-768", 768ull, 768ull, 124ull, 0},
        {"smoke", 32ull, 32ull, 345ull, 0},
    };
    const yvex_component_variant_adapter *adapter =
        yvex_graph_component_variant_find_family("minimax-h3");
    const yvex_media_execution_recipe *execution =
        adapter ? adapter->media_execution : NULL;
    yvex_media_target_profile target;
    yvex_error err;
    if (!adapter || !execution ||
        adapter->media_target_profile(&target, &err) != YVEX_OK)
        return 0;
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
    options->request_template.fps_numerator = target.fps_numerator;
    options->request_template.fps_denominator = target.fps_denominator;
    options->request_template.audio_sample_rate = target.audio_sample_rate;
    options->request_template.seed = target.seed;
    options->request_template.conditioning_layers = execution->conditioning_layers;
    options->request_template.transformer_blocks = execution->transformer_blocks;
    options->request_template.maximum_prompt_tokens = execution->maximum_prompt_tokens;
    options->request_template.maximum_packed_rows = execution->maximum_packed_rows;
    options->request_template.maximum_host_bytes = target.maximum_host_bytes;
    options->request_template.maximum_device_bytes = target.maximum_device_bytes;
    options->request_template.maximum_workspace_bytes = target.maximum_workspace_bytes;
    options->request_template.maximum_file_bytes = target.maximum_file_bytes;
    options->request_template.component_backend = execution->component_backend;
    options->request_template.video_temporal_ratio = target.video_temporal_ratio;
    options->request_template.video_clip_length = target.video_clip_length;
    options->request_template.video_token_drop = target.video_token_drop;
    options->request_template.video_spatial_ratio = target.video_spatial_ratio;
    options->request_template.video_tile_size = target.video_tile_size;
    options->request_template.video_minimum_tile_overlap = target.video_minimum_tile_overlap;
    options->request_template.video_mean = target.video_mean;
    options->request_template.video_std = target.video_std;
    options->request_template.audio_mean = target.audio_mean;
    options->request_template.audio_std = target.audio_std;
    options->request_template.pixel_mean = target.pixel_mean;
    options->request_template.pixel_std = target.pixel_std;
    options->request_template.video_channels = target.video_channels;
    options->request_template.audio_channels = target.audio_channels;
    options->request_template.pixel_channels = target.pixel_channels;
    options->request_template.audio_output_channels = target.audio_output_channels;
    options->request_template.audio_samples_per_step = target.audio_samples_per_step;
    options->request_template.plan_build = execution->plan_build;
    options->request_template.layout_build = execution->layout_build;
    options->request_template.component_admit = media_fixture_admit;
    options->request_template.condition = execution->condition;
    options->request_template.latent = execution->latent;
    options->request_template.video_decode = execution->video_decode;
    options->request_template.audio_decode = execution->audio_decode;
    options->profiles = profiles;
    options->profile_count = sizeof(profiles) / sizeof(profiles[0]);
    options->frames_per_chunk = 17ull;
    options->frame_remainder = 5ull;
    options->minimum_frames = 124ull;
    options->maximum_frames = 345ull;
    options->minimum_inference_steps = 2ull;
    options->maximum_inference_steps = 64ull;
    options->canvas_multiple = 32ull;
    options->maximum_canvas_pixels = 768ull * 768ull;
    return 1;
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
        "Eclissi realistica preview, 5 secondi, 19 punti sigma, MOV",
        "Eclissi realistica preview, 15 secondi, 19 punti sigma, AVI",
    };
    const int bad_status[] = {
        YVEX_ERR_UNSUPPORTED, YVEX_ERR_BOUNDS, YVEX_ERR_BOUNDS, YVEX_ERR_UNSUPPORTED,
        YVEX_ERR_UNSUPPORTED, YVEX_ERR_BOUNDS,
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
    YVEX_TEST_ASSERT(media_options(&options, root), "media dialogue options");
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
    memset(&messages, 0, sizeof(messages));
    YVEX_TEST_ASSERT(media_registry_request(registry, YVEX_CLIENT_OP_SESSION_NEW,
                                            "preview-256-geometry", NULL,
                                            &messages, &err) == YVEX_OK,
                     "extended preview geometry dialogue session");
    memset(&messages, 0, sizeof(messages));
    rc = media_registry_request(
        registry, YVEX_CLIENT_OP_GENERATION_TURN, "preview-256-geometry",
        "256x256, 5 secondi, 2 punti sigma, seed 42", &messages, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && strstr(messages.text, "AVI") &&
                         !strstr(messages.text, "Qualità:") &&
                         !strstr(messages.text, "Durata:") &&
                         !strstr(messages.text, "Iterazioni:"),
                     "registered media geometry selects its typed profile");
    memset(&messages, 0, sizeof(messages));
    YVEX_TEST_ASSERT(media_registry_request(registry, YVEX_CLIENT_OP_SESSION_NEW,
                                            "preview-384-geometry", NULL,
                                            &messages, &err) == YVEX_OK,
                     "larger preview geometry dialogue session");
    memset(&messages, 0, sizeof(messages));
    rc = media_registry_request(
        registry, YVEX_CLIENT_OP_GENERATION_TURN, "preview-384-geometry",
        "384x384, 5 secondi, 2 punti sigma, seed 42", &messages, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && strstr(messages.text, "AVI") &&
                         !strstr(messages.text, "Qualità:") &&
                         !strstr(messages.text, "Durata:") &&
                         !strstr(messages.text, "Iterazioni:"),
                     "larger registered media geometry selects its typed profile");
    memset(&messages, 0, sizeof(messages));
    YVEX_TEST_ASSERT(media_registry_request(registry, YVEX_CLIENT_OP_SESSION_CLOSE,
                                            "preview-384-geometry", NULL,
                                            &messages, &err) == YVEX_OK,
                     "larger preview dialogue session closes");
    memset(&messages, 0, sizeof(messages));
    YVEX_TEST_ASSERT(media_registry_request(registry, YVEX_CLIENT_OP_SESSION_NEW,
                                            "source-768-geometry", NULL,
                                            &messages, &err) == YVEX_OK,
                     "source square geometry dialogue session");
    memset(&messages, 0, sizeof(messages));
    rc = media_registry_request(
        registry, YVEX_CLIENT_OP_GENERATION_TURN, "source-768-geometry",
        "source-768 768x768, 5 secondi, 2 punti sigma, seed 42", &messages, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && strstr(messages.text, "AVI") &&
                         !strstr(messages.text, "Qualità:") &&
                         !strstr(messages.text, "Durata:") &&
                         !strstr(messages.text, "Iterazioni:"),
                     "source square geometry selects its typed profile");
    memset(&messages, 0, sizeof(messages));
    YVEX_TEST_ASSERT(media_registry_request(registry, YVEX_CLIENT_OP_SESSION_CLOSE,
                                            "source-768-geometry", NULL,
                                            &messages, &err) == YVEX_OK,
                     "source square dialogue session closes");
    memset(&messages, 0, sizeof(messages));
    YVEX_TEST_ASSERT(media_registry_request(registry, YVEX_CLIENT_OP_SESSION_NEW,
                                            "natural-language", NULL,
                                            &messages, &err) == YVEX_OK,
                     "natural-language media dialogue session");
    memset(&messages, 0, sizeof(messages));
    rc = media_registry_request(
        registry, YVEX_CLIENT_OP_GENERATION_TURN, "natural-language",
        "Genera un eclissi con nuvole in movimento, source-768 768x768, "
        "5 secondi, formato AVI, seed 42",
        &messages, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && strstr(messages.text, "Iterazioni:") &&
                         !strstr(messages.text, "Formato disponibile:") &&
                         !strstr(messages.text, "Qualità:") &&
                         !strstr(messages.text, "Durata:"),
                     "natural words containing container prefixes retain AVI selection");
    memset(&messages, 0, sizeof(messages));
    YVEX_TEST_ASSERT(media_registry_request(registry, YVEX_CLIENT_OP_SESSION_RESET,
                                            "natural-language", NULL,
                                            &messages, &err) == YVEX_OK,
                     "natural-language media dialogue resets");
    memset(&messages, 0, sizeof(messages));
    rc = media_registry_request(
        registry, YVEX_CLIENT_OP_GENERATION_TURN, "natural-language",
        "Una volpe salta e muove la testa nella neve, 5 secondi, "
        "2 punti sigma, AVI",
        &messages, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && strstr(messages.text, "Qualità:") &&
                         !strstr(messages.text, "Durata:") &&
                         !strstr(messages.text, "Iterazioni:") &&
                         !strstr(messages.text, "Formato disponibile:"),
                     "natural words containing quality aliases do not select a profile");
    memset(&messages, 0, sizeof(messages));
    YVEX_TEST_ASSERT(media_registry_request(registry, YVEX_CLIENT_OP_SESSION_CLOSE,
                                            "natural-language", NULL,
                                            &messages, &err) == YVEX_OK,
                     "natural-language media dialogue closes");
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
        memset(&messages, 0, sizeof(messages));
        YVEX_TEST_ASSERT(media_registry_request(registry, YVEX_CLIENT_OP_SESSION_CLOSE,
                                                name, NULL, &messages, &err) == YVEX_OK,
                         "refused media dialogue session closes");
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
                         !strcmp(first.physical_variant_identity,
                                 repeated.physical_variant_identity) &&
                         !first.runtime_binding_identity[0] &&
                         !first.artifact_identity[0],
                     "media runtime and profile identities are deterministic without aliases");
    yvex_server_media_registry_close(&second);
    yvex_server_telemetry_close(&second_telemetry);
    yvex_server_media_registry_close(&registry);
    yvex_server_telemetry_close(&telemetry);
    YVEX_TEST_ASSERT(rmdir(root) == 0, "media dialogue output root removed empty");
    return 0;
}

static int test_media_server_opens_model_before_ready(void)
{
    const char *fixture = "tests/fixtures/gguf/valid-tokenizer-simple.gguf";
    char root[] = "/tmp/yvex-media-host-XXXXXX";
    char socket_path[YVEX_SERVER_SOCKET_PATH_CAP];
    yvex_server_media_options media;
    yvex_server_options options;
    yvex_server_summary summary;
    yvex_client_message wire = {0}, decoded;
    unsigned char frame[16384];
    unsigned long long frame_count = 0ull;
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
    options.context_capacity = 0ull;
    options.prefill_chunk_tokens = 0ull;
    options.maximum_new_tokens = 0ull;
    YVEX_TEST_ASSERT(media_options(&media, root), "media host options");
    media.request_template.text_artifact_path = fixture;
    media.request_template.transformer_artifact_path = fixture;
    media.request_template.video_artifact_path = fixture;
    media.request_template.audio_artifact_path = fixture;
    rc = yvex_server_create(&server, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && server != NULL, "media host create");
    YVEX_TEST_ASSERT(yvex_server_media_configure(server, &media, &err) == YVEX_OK,
                     "media host configure");
    YVEX_TEST_ASSERT(yvex_server_start(server, &err) == YVEX_OK,
                     "media host opens admitted components before readiness");
    YVEX_TEST_ASSERT(yvex_server_get_summary(server, &summary, &err) == YVEX_OK &&
                         summary.status == YVEX_SERVER_STATUS_READY &&
                         summary.runtime_ready && summary.generation_ready &&
                         summary.generation_mode == YVEX_SERVER_GENERATION_MEDIA &&
                         summary.metrics.model_open_count == 1ull &&
                         summary.metrics.artifact_open_count == 4ull &&
                         summary.metrics.binding_open_count == 1ull &&
                         summary.metrics.materialization_count == 0ull &&
                         summary.metrics.residency_build_count == 0ull &&
                         summary.metrics.resident_device_bytes == 0ull &&
                         summary.context_capacity == 0ull &&
                         summary.prefill_chunk_tokens == 0ull &&
                         summary.maximum_new_tokens == 0ull &&
                         !summary.runtime_binding_identity[0] &&
                         !summary.artifact_identity[0] &&
                         !summary.capacity_plan_identity[0] &&
                         summary.capacity_required_bytes == 0ull,
                     "media host readiness admits the model without false payload residency");
    wire.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    wire.kind = YVEX_CLIENT_MESSAGE_STATUS;
    wire.status = YVEX_OK;
    wire.runtime = summary;
    YVEX_TEST_ASSERT(
        yvex_protocol_message_encode(&wire, frame, sizeof(frame), &frame_count,
                                     &err) == YVEX_OK &&
            yvex_protocol_message_decode(frame, frame_count, &decoded, &err) == YVEX_OK &&
            decoded.runtime.generation_mode == YVEX_SERVER_GENERATION_MEDIA &&
            decoded.runtime.runtime_ready,
        "ready media host crosses the capability-aware local protocol");
    YVEX_TEST_ASSERT(yvex_server_finish(server, &err) == YVEX_OK,
                     "media host finishes cleanly");
    YVEX_TEST_ASSERT(yvex_server_get_summary(server, &summary, &err) == YVEX_OK &&
                         summary.metrics.model_close_count == 1ull,
                     "media host closes the process-lifetime model once");
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
    YVEX_TEST_ASSERT(profile.profile_count == 5ull &&
                         !strcmp(profile.profiles[0].name, "preview") &&
                         !strcmp(profile.profiles[1].name, "preview-256") &&
                         !strcmp(profile.profiles[2].name, "preview-384") &&
                         !strcmp(profile.profiles[3].name, "source-768") &&
                         !strcmp(profile.profiles[4].name, "smoke"),
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
    if (test_automatic_reasoning_policy() != 0) return 1;
    if (test_scheduler_serialization() != 0) return 1;
    if (test_session_store() != 0) return 1;
    if (test_configured_summary_and_event() != 0) return 1;
    if (test_adaptive_prefill_policy() != 0) return 1;
    if (test_model_open_refusal() != 0) return 1;
    if (test_bounded_telemetry_overflow() != 0) return 1;
    if (test_provider_telemetry() != 0) return 1;
    if (test_openai_listener_admission() != 0) return 1;
    if (test_media_dialog_and_refusals() != 0) return 1;
    if (test_media_family_profile() != 0) return 1;
    if (test_media_server_opens_model_before_ready() != 0) return 1;
    return 0;
}
