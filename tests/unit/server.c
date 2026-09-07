/*
 * Exercises persistent-host truth, independent engine lifecycle, typed event/JSON projection,
 * privacy defaults, and idempotent graceful close.
 */
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <arpa/inet.h>
#include <pthread.h>
#include <sys/stat.h>
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
    options->schema_version = YVEX_SERVER_OPTIONS_SCHEMA_CURRENT;
    options->socket_path = "/tmp/yvex-test-host/yvexd.sock";
    options->request_queue_capacity = 2u;
    options->worker_count = 1u;
}

static void test_engine_options(yvex_server_engine_options *options,
                                const char *alias)
{
    memset(options, 0, sizeof(*options));
    options->schema_version = YVEX_SERVER_ENGINE_SCHEMA_CURRENT;
    options->alias = alias;
    options->artifact_path = "/definitely-absent/yvex-model.gguf";
    options->runtime_binding_path = "/definitely-absent/yvex-binding";
    options->target_id = "deepseek4-v4-flash-dspark";
    options->backend = YVEX_BACKEND_KIND_CPU;
    options->engine_kind = YVEX_SERVER_ENGINE_TEXT;
    options->execution_strategy = YVEX_SERVER_EXECUTION_TARGET_ONLY;
    options->context_capacity = 32u;
    options->prefill_chunk_tokens = 8u;
    options->maximum_new_tokens = 4u;
    options->maximum_output_bytes = 4096u;
    options->maximum_sessions = 2u;
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
    unsigned long long active, peak, completed;
    int release, first_active, third_active, first_done, violation;
} request_queue_probe;

typedef struct {
    request_queue_probe *probe;
    int id;
} request_queue_probe_work;

static void request_queue_probe_execute(void *context, void *opaque)
{
    request_queue_probe *probe = context;
    request_queue_probe_work *work = opaque;
    (void)pthread_mutex_lock(&probe->mutex);
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

static int test_request_queue_serialization(void)
{
    request_queue_probe probe = {0};
    request_queue_probe_work work[] = {
        {&probe, 1}, {&probe, 2}, {&probe, 3}
    };
    server_request_queue *request_queue = NULL;
    server_request_queue_summary summary = {0};
    char serialization_key[SERVER_REQUEST_QUEUE_KEY_CAP];
    yvex_error err;
    YVEX_TEST_ASSERT(
        yvex_server_request_queue_key(serialization_key, 7ull,
                                      "@engine.sessions", &err) == YVEX_OK &&
            !strcmp(serialization_key, "7:@engine.sessions"),
        "engine-scoped work receives a deterministic non-session queue key");
    YVEX_TEST_ASSERT(pthread_mutex_init(&probe.mutex, NULL) == 0 &&
                         pthread_cond_init(&probe.condition, NULL) == 0,
                     "request_queue probe synchronization opens");
    YVEX_TEST_ASSERT(
        yvex_server_request_queue_open(
            &request_queue, 3ull, 2ull, request_queue_probe_execute, NULL,
            &probe, &err) == YVEX_OK && request_queue,
        "two-worker request_queue opens");
    YVEX_TEST_ASSERT(yvex_server_request_queue_start(request_queue, &err) == YVEX_OK,
                     "two-worker request_queue starts");
    YVEX_TEST_ASSERT(
        yvex_server_request_queue_submit(
            request_queue, &work[0], "same", NULL, &err) == YVEX_OK,
        "request executor accepts the first serialized session");
    YVEX_TEST_ASSERT(
        yvex_server_request_queue_submit(
                request_queue, &work[1], "same", NULL, &err) == YVEX_OK &&
            yvex_server_request_queue_submit(
                request_queue, &work[2], "other", NULL, &err) == YVEX_OK,
        "request executor admits independent sessions without inference policy");
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
        yvex_server_request_queue_finish(request_queue, &err) == YVEX_OK,
        "request_queue drains and joins workers");
    yvex_server_request_queue_snapshot(request_queue, &summary);
    YVEX_TEST_ASSERT(!summary.queued && !summary.active &&
                         !probe.violation &&
                         probe.first_done,
                     "same-key work starts only after its predecessor completes");
    yvex_server_request_queue_close(&request_queue);
    (void)pthread_cond_destroy(&probe.condition);
    (void)pthread_mutex_destroy(&probe.mutex);
    return 0;
}

static int test_session_store(void)
{
    static const char second_message[] = {'o', 'k', '\0', '!'};
    static const char second_reasoning[] = "typed reasoning";
    yvex_prompt_message messages[2] = {
        {.schema_version = YVEX_PROMPT_MESSAGE_SCHEMA_V1,
         .role = YVEX_PROMPT_ROLE_USER, .content = "ciao", .content_len = 4ull},
        {.schema_version = YVEX_PROMPT_MESSAGE_SCHEMA_V1,
         .role = YVEX_PROMPT_ROLE_ASSISTANT,
         .content = second_message, .content_len = sizeof(second_message),
         .reasoning_content = second_reasoning,
         .reasoning_content_len = sizeof(second_reasoning) - 1u}};
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
            restored.messages[1].reasoning_content_len ==
                sizeof(second_reasoning) - 1u &&
            memcmp(restored.messages[1].reasoning_content, second_reasoning,
                   sizeof(second_reasoning) - 1u) == 0 &&
            strcmp(restored.payload_identity, payload_identity) == 0,
        "session checkpoint roundtrips typed messages, policy, and RNG facts");
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
    yvex_server *server = NULL, *contender = NULL;
    yvex_error err;
    char json[2048];
    unsigned char frame[8192];
    unsigned long long frame_count = 0u;
    int rc;
    test_options(&options);
    options.maximum_engines =
        YVEX_SERVER_IMPLEMENTATION_MAXIMUM_ENGINES + 1ull;
    rc = yvex_server_create(&server, &options, &err);
    YVEX_TEST_ASSERT(
        rc == YVEX_ERR_INVALID_ARG && server == NULL,
        "engine-slot configuration refuses beyond the implementation safety maximum");
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
                         summary.metrics.queue_capacity == 0u,
                     "empty host publishes no nonexistent execution queue");
    YVEX_TEST_ASSERT(!summary.host_ready && !summary.engine_count &&
                         !summary.loaded_engine_count &&
                         summary.maximum_engines ==
                             YVEX_SERVER_DEFAULT_MAXIMUM_ENGINES,
                     "configured host owns no implicit model engine");
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
    rc = yvex_server_start(server, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "zero-engine host starts");
    rc = yvex_server_get_summary(server, &summary, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && summary.host_ready &&
                         summary.status == YVEX_SERVER_STATUS_READY &&
                         !summary.engine_count && !summary.loaded_engine_count &&
                         summary.metrics.model_open_count == 0ull,
                     "persistent host becomes ready without loading a model");
    rc = yvex_server_create(&contender, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && contender,
                     "singleton contender configures independently");
    rc = yvex_server_start(contender, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_STATE,
                     "singleton contender refuses the live host lock");
    yvex_server_close(&contender);
    {
        struct stat endpoint;
        YVEX_TEST_ASSERT(lstat(options.socket_path, &endpoint) == 0 &&
                             S_ISSOCK(endpoint.st_mode),
                         "refused contender cleanup preserves the live host endpoint");
    }
    rc = yvex_server_stop(server, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "zero-engine host stop");
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
    yvex_server_engine_options engine_options;
    yvex_server_engine_summary engine;
    yvex_server *server = NULL;
    yvex_error err;
    size_t index;
    int rc;
    for (index = 0u; index < sizeof(cases) / sizeof(cases[0]); ++index) {
        test_options(&options);
        rc = yvex_server_create(&server, &options, &err);
        YVEX_TEST_ASSERT(rc == YVEX_OK && server &&
                             yvex_server_start(server, &err) == YVEX_OK,
                         "adaptive prefill host starts independently");
        test_engine_options(&engine_options, "adaptive");
        engine_options.context_capacity = cases[index].context;
        engine_options.maximum_sessions = cases[index].concurrency;
        engine_options.concurrent_sequences = cases[index].concurrency;
        engine_options.prefill_chunk_tokens = 0ull;
        memset(&engine, 0, sizeof(engine));
        rc = yvex_server_engine_load(server, &engine_options, &engine, &err);
        YVEX_TEST_ASSERT(rc != YVEX_OK &&
                             engine.state == YVEX_SERVER_ENGINE_FAILED &&
                             engine.prefill_chunk_tokens == cases[index].expected,
                         "engine specialization publishes adaptive prefill before open refusal");
        yvex_server_close(&server);
    }
    test_options(&options);
    rc = yvex_server_create(&server, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && server &&
                         yvex_server_start(server, &err) == YVEX_OK,
                     "explicit prefill host starts independently");
    test_engine_options(&engine_options, "explicit");
    engine_options.maximum_sessions = 2ull;
    engine_options.concurrent_sequences = 2ull;
    engine_options.prefill_chunk_tokens = 7ull;
    memset(&engine, 0, sizeof(engine));
    rc = yvex_server_engine_load(server, &engine_options, &engine, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK && engine.prefill_chunk_tokens == 7ull,
                     "explicit prefill override remains authoritative");
    yvex_server_close(&server);
    return 0;
}

static int test_model_open_refusal(void)
{
    yvex_server_options options;
    yvex_server_engine_options engine_options;
    yvex_server_engine_summary engine;
    yvex_server_summary summary;
    yvex_server *server = NULL;
    yvex_error err;
    int rc;
    test_options(&options);
    options.schema_version = YVEX_SERVER_OPTIONS_SCHEMA_V3;
    rc = yvex_server_create(&server, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_INVALID_ARG && !server &&
                         strstr(yvex_error_message(&err),
                                "unsupported server-options schema") != NULL,
                     "legacy v3 server-options layout refuses before reinterpretation");
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
    YVEX_TEST_ASSERT(rc == YVEX_OK, "host start is independent from model admission");
    test_engine_options(&engine_options, "legacy");
    engine_options.schema_version = YVEX_SERVER_ENGINE_SCHEMA_V1;
    engine_options.engine_kind = YVEX_SERVER_ENGINE_MEDIA;
    engine_options.execution_strategy = YVEX_SERVER_EXECUTION_SPECULATIVE;
    memset(&engine, 0, sizeof(engine));
    rc = yvex_server_engine_load(server, &engine_options, &engine, &err);
    YVEX_TEST_ASSERT(
        rc == YVEX_ERR_UNSUPPORTED &&
            strstr(yvex_error_message(&err),
                   "server engine options schema is unsupported") != NULL,
        "legacy v1 engine layout refuses before new fields are interpreted");
    test_engine_options(&engine_options, "strategy");
    engine_options.execution_strategy = YVEX_SERVER_EXECUTION_NOT_APPLICABLE;
    rc = yvex_server_engine_load(server, &engine_options, &engine, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_INVALID_ARG,
                     "text engine refuses an inapplicable execution strategy");
    test_engine_options(&engine_options, "invalid");
    engine_options.concurrent_sequences = engine_options.maximum_sessions + 1ull;
    rc = yvex_server_engine_load(server, &engine_options, &engine, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_INVALID_ARG,
                     "engine concurrency above session capacity refuses locally");
    test_engine_options(&engine_options, "missing");
    memset(&engine, 0, sizeof(engine));
    rc = yvex_server_engine_load(server, &engine_options, &engine, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK &&
                         engine.state == YVEX_SERVER_ENGINE_FAILED &&
                         strstr(yvex_error_message(&err), "binding") != NULL,
                     "missing package binding refuses only the engine load");
    rc = yvex_server_get_summary(server, &summary, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && summary.host_ready &&
                         summary.status == YVEX_SERVER_STATUS_READY &&
                         summary.engine_count == 1ull &&
                         !summary.loaded_engine_count &&
                         summary.metrics.model_open_count == 0u,
                     "failed engine remains inspectable without failing the host");
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
    int rc, saw_drop = 0, saw_terminal = 0;
    rc = yvex_server_telemetry_open(&telemetry, 2u, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "bounded telemetry open");
    for (index = 0u; index < 4u; ++index) {
        rc = yvex_server_telemetry_emit(
            telemetry, NULL, YVEX_SERVER_EVENT_GENERATION_PROGRESS,
            YVEX_SERVER_SEVERITY_DEBUG, "s", "r", "t", "decode",
            index, 0u, 0u, 0.0, 0.0, &err);
        YVEX_TEST_ASSERT(rc == YVEX_OK, "bounded telemetry publish");
    }
    rc = yvex_server_telemetry_metrics_copy(telemetry, &metrics, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && metrics.telemetry_dropped == 2u,
                     "overflow count names exactly the overwritten events");
    rc = yvex_server_telemetry_emit(
        telemetry, NULL, YVEX_SERVER_EVENT_GENERATION_COMPLETED,
        YVEX_SERVER_SEVERITY_INFO, "s", "r", "t", "turn",
        4u, 9u, YVEX_GENERATION_STOP_EOS, 1.0, 4.0, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "terminal telemetry retained under pressure");
    for (index = 0u; index < 1024u; ++index) {
        rc = yvex_server_telemetry_emit(
            telemetry, NULL, YVEX_SERVER_EVENT_GENERATION_PROGRESS,
            YVEX_SERVER_SEVERITY_DEBUG, "s", "r", "t", "decode",
            index + 4u, index + 9u, 0u, 1.0, 4.0, &err);
        YVEX_TEST_ASSERT(rc == YVEX_OK, "pressure progress coalescing");
    }
    rc = yvex_server_telemetry_metrics_copy(telemetry, &metrics, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && metrics.telemetry_dropped == 1027u,
                     "pressure accounting is exact and non-amplifying");
    yvex_server_telemetry_model_opened(telemetry, 4096u, 0u, 0u, 0u);
    yvex_server_telemetry_resources(telemetry, 1024u, 2048u, 1u);
    yvex_server_telemetry_resources(telemetry, 512u, 1024u, 0u);
    rc = yvex_server_telemetry_metrics_copy(telemetry, &metrics, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && metrics.mapped_artifact_bytes == 4096u &&
                         metrics.resident_host_bytes == 1024u &&
                         metrics.resident_device_bytes == 2048u &&
                         metrics.output_head_upload_count == 1u,
                     "mapped backing is distinct from resident resource high-water facts");
    yvex_server_telemetry_model_closed(telemetry);
    rc = yvex_server_telemetry_metrics_copy(telemetry, &metrics, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && !metrics.mapped_artifact_bytes &&
                         !metrics.resident_host_bytes &&
                         !metrics.resident_device_bytes &&
                         metrics.model_close_count == metrics.model_open_count,
                     "closing the final model clears current resource facts");
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
        if (event.kind == YVEX_SERVER_EVENT_TELEMETRY_DROPPED) {
            saw_drop = event.value_a == 1027u && event.value_c == 1026u;
        }
        saw_terminal |= event.kind == YVEX_SERVER_EVENT_GENERATION_COMPLETED;
    }
    YVEX_TEST_ASSERT(saw_drop, "overflow pressure has one exact aggregate warning");
    YVEX_TEST_ASSERT(saw_terminal,
                     "replaceable progress cannot evict a terminal event");
    yvex_server_telemetry_close(&telemetry);
    return 0;
}

static unsigned long long telemetry_measurement_clock(void)
{
    struct timespec value = {0};
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0u;
    return (unsigned long long)value.tv_sec * 1000000000ull +
           (unsigned long long)value.tv_nsec;
}

static int telemetry_measure_progress(unsigned long long tokens,
                                      unsigned long long stride,
                                      unsigned long long *elapsed_ns,
                                      unsigned long long *event_count)
{
    server_telemetry *telemetry = NULL;
    yvex_error err;
    volatile unsigned long long checksum = 0u;
    unsigned long long begin, end, index, events = 0u;
    int rc;
    if (!elapsed_ns || !event_count) return 0;
    rc = yvex_server_telemetry_open(&telemetry, 64u, &err);
    if (rc != YVEX_OK) return 0;
    begin = telemetry_measurement_clock();
    for (index = 0u; index < tokens; ++index) {
        checksum ^= index + 1u;
        if (stride && (index + 1u) % stride == 0u) {
            rc = yvex_server_telemetry_emit(
                telemetry, NULL, YVEX_SERVER_EVENT_GENERATION_PROGRESS,
                YVEX_SERVER_SEVERITY_DEBUG, "s", "r", "t", "decode",
                index + 1u, index + 9u, 0u, 1.0, 4.0, &err);
            if (rc != YVEX_OK) {
                yvex_server_telemetry_close(&telemetry);
                return 0;
            }
            events++;
        }
    }
    end = telemetry_measurement_clock();
    yvex_server_telemetry_close(&telemetry);
    if (!begin || end <= begin || checksum == ULLONG_MAX) return 0;
    *elapsed_ns = end - begin;
    *event_count = events;
    return 1;
}

static int test_telemetry_observability_economics(void)
{
    const unsigned long long tokens = 65536u;
    unsigned long long off_ns, normal_ns, detailed_ns;
    unsigned long long off_events, normal_events, detailed_events;
    YVEX_TEST_ASSERT(
        telemetry_measure_progress(tokens, 0u, &off_ns, &off_events),
        "observability disabled measurement");
    YVEX_TEST_ASSERT(
        telemetry_measure_progress(tokens, 64u, &normal_ns, &normal_events),
        "observability normal measurement");
    YVEX_TEST_ASSERT(
        telemetry_measure_progress(tokens, 1u, &detailed_ns, &detailed_events),
        "observability detailed measurement");
    YVEX_TEST_ASSERT(!off_events && normal_events == tokens / 64u &&
                         detailed_events == tokens,
                     "observability cadence economics");
    (void)fprintf(stderr,
                  "telemetry economics: tokens=%llu off_ns=%llu "
                  "normal_events=%llu normal_ns=%llu detailed_events=%llu "
                  "detailed_ns=%llu\n",
                  tokens, off_ns, normal_events, normal_ns, detailed_events,
                  detailed_ns);
    return 0;
}

static int test_decode_rate_scopes(void)
{
    unsigned long long commits[YVEX_SERVER_DECODE_RATE_WINDOW_TOKENS + 1ull];
    yvex_execution_measurement measurement;
    unsigned int index;
    commits[0] = 1800000000ull;
    for (index = 1u;
         index < YVEX_SERVER_DECODE_RATE_WINDOW_TOKENS + 1ull; ++index)
        commits[index] = commits[index - 1u] + 1000000000ull;
    yvex_server_decode_measurement(
        1000000000ull, 41ull, commits,
        YVEX_SERVER_DECODE_RATE_WINDOW_TOKENS + 1u,
        commits[YVEX_SERVER_DECODE_RATE_WINDOW_TOKENS], &measurement);
    YVEX_TEST_ASSERT(
        yvex_server_execution_measurement_valid(&measurement) &&
            measurement.scope == YVEX_EXECUTION_SCOPE_SUBSEQUENT_DECODE &&
            measurement.completed_units == 40ull &&
            measurement.duration_ns == 32800000000ull &&
            measurement.rolling_units == 32ull &&
            measurement.rolling_duration_ns == 32000000000ull &&
            measurement.rolling_window_units == 32ull &&
            measurement.cumulative_rate > 1.21 &&
            measurement.cumulative_rate < 1.23 &&
            measurement.rolling_rate == 1.0 &&
            measurement.cumulative_rate > measurement.rolling_rate,
        "bounded rolling decode reveals a local slowdown hidden by the cumulative rate");
    return 0;
}

static int test_profile_wall_reconciliation(void)
{
    yvex_runtime_profile_record profile = {0};
    unsigned long long attributed = 0ull, unattributed = 0ull;
    profile.phase_ns[YVEX_RUNTIME_PROFILE_TOKENIZER] = 100ull;
    profile.phase_ns[YVEX_RUNTIME_PROFILE_TOTAL_PREFILL] = 200ull;
    profile.phase_ns[YVEX_RUNTIME_PROFILE_FIRST_DECODE] = 300ull;
    profile.phase_ns[YVEX_RUNTIME_PROFILE_SUBSEQUENT_DECODE] = 300ull;
    profile.phase_ns[YVEX_RUNTIME_PROFILE_OUTPUT_HEAD] = 50ull;
    profile.phase_ns[YVEX_RUNTIME_PROFILE_PROVIDER_PUBLICATION] = 20ull;
    profile.phase_ns[YVEX_RUNTIME_PROFILE_TOTAL_GENERATION] = 1000ull;
    YVEX_TEST_ASSERT(
        yvex_server_profile_reconcile(
            &profile, YVEX_GENERATION_MODE_TARGET_ONLY,
            &attributed, &unattributed) &&
            attributed == 970ull && unattributed == 30ull,
        "ordinary decode wall adds disjoint output and publication children");
    YVEX_TEST_ASSERT(
        yvex_server_profile_reconcile(
            &profile, YVEX_GENERATION_MODE_SPECULATIVE,
            &attributed, &unattributed) &&
            attributed == 900ull && unattributed == 100ull,
        "speculative enclosing decode does not double-count nested children");
    profile.phase_ns[YVEX_RUNTIME_PROFILE_TOTAL_GENERATION] = 800ull;
    YVEX_TEST_ASSERT(
        !yvex_server_profile_reconcile(
            &profile, YVEX_GENERATION_MODE_SPECULATIVE,
            &attributed, &unattributed) && !attributed && !unattributed,
        "overlapping wall presented as disjoint fails reconciliation honestly");
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
    YVEX_TEST_ASSERT(rc == YVEX_OK, "OpenAI host starts without an implicit model");
    rc = yvex_server_get_summary(server, &summary, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && summary.host_ready &&
                         summary.openai_listener_ready &&
                         !summary.loaded_engine_count,
                     "OpenAI listener is healthy with an empty engine catalog");
    yvex_server_close(&server);
    probe = loopback_reserve(&port);
    YVEX_TEST_ASSERT(probe >= 0, "host shutdown releases HTTP listener");
    (void)close(probe);
    return 0;
}

typedef struct {
    char text[YVEX_SERVER_FRAGMENT_CAP * 2u];
    unsigned long long count, started, completed, errors, events, media_results;
    int status;
    yvex_client_generation_phase phase;
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
    messages->errors += message->kind == YVEX_CLIENT_MESSAGE_ERROR;
    messages->events += message->kind == YVEX_CLIENT_MESSAGE_EVENT;
    messages->media_results += message->media_result.available != 0;
    if (message->kind == YVEX_CLIENT_MESSAGE_ERROR) {
        messages->status = message->status;
        messages->phase = message->generation_phase;
    }
    if (message->byte_count) {
        memcpy(messages->text + used, message->bytes, (size_t)message->byte_count);
        messages->text[used + message->byte_count] = '\0';
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

static int media_fixture_admit(
    const char *component, const yvex_artifact *artifact, const yvex_gguf *gguf,
    const yvex_tensor_table *tensors, const yvex_artifact_admission_options *options,
    yvex_complete_artifact_admission *out, yvex_artifact_admission_evidence *evidence,
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
    if (options && options->progress &&
        (!options->progress(options->progress_context, 0ull,
                            yvex_artifact_size(artifact)) ||
         !options->progress(options->progress_context,
                            yvex_artifact_size(artifact),
                            yvex_artifact_size(artifact)))) {
        yvex_error_set(err, YVEX_ERR_CANCELLED, "test.server.media-admit",
                       "fixture component admission was cancelled");
        return YVEX_ERR_CANCELLED;
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
    if (evidence) {
        memset(evidence, 0, sizeof(*evidence));
        evidence->schema_version = YVEX_ARTIFACT_ADMISSION_OPTIONS_SCHEMA_V1;
        evidence->verification_mode = YVEX_ARTIFACT_VERIFICATION_FULL_HASH;
        evidence->file_bytes = out->file_bytes;
        evidence->bytes_hashed = out->file_bytes;
        evidence->complete = 1;
    }
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
    yvex_runtime_media_host_profile host;
    yvex_media_target_profile target;
    yvex_error err;
    if (!adapter || !execution ||
        adapter->media_target_profile(&target, &err) != YVEX_OK ||
        yvex_runtime_media_host_profile_build(
            &host, &target, execution, "/not-opened", output_root,
            &err) != YVEX_OK)
        return 0;
    memset(options, 0, sizeof(*options));
    options->schema_version = YVEX_SERVER_MEDIA_SCHEMA_V2;
    options->output_root = output_root;
    options->artifact_reopen_cache_root = output_root;
    options->request_template.schema_version = YVEX_RUNTIME_AV_GENERATION_SCHEMA_V2;
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
    options->request_template.keyframe_encode_seed = target.keyframe_encode_seed;
    options->request_template.conditioning_layers = execution->conditioning_layers;
    options->request_template.transformer_blocks = execution->transformer_blocks;
    options->request_template.maximum_prompt_tokens = execution->maximum_prompt_tokens;
    options->request_template.maximum_packed_rows = execution->maximum_packed_rows;
    options->request_template.maximum_host_bytes = target.maximum_host_bytes;
    options->request_template.maximum_device_bytes = target.maximum_device_bytes;
    options->request_template.maximum_workspace_bytes = target.maximum_workspace_bytes;
    options->request_template.maximum_file_bytes = target.maximum_file_bytes;
    options->request_template.component_backend = execution->component_backend;
    options->request_template.output_semantic_domain =
        host.request_template.output_semantic_domain;
    options->request_template.video_output_requirement =
        host.request_template.video_output_requirement;
    options->request_template.audio_output_requirement =
        host.request_template.audio_output_requirement;
    options->request_template.video_output_specialization =
        host.request_template.video_output_specialization;
    options->request_template.audio_output_specialization =
        host.request_template.audio_output_specialization;
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
    options->request_template.keyframe_encode = execution->keyframe_encode;
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
    options->released_sigma_grid_points = target.released_sigma_grid_points;
    options->default_seed = target.seed;
    options->canvas_multiple = target.canvas_multiple;
    options->canvas_short_edge = target.canvas_short_edge;
    options->minimum_canvas_pixels = target.minimum_canvas_pixels;
    options->maximum_canvas_pixels = target.maximum_canvas_pixels;
    options->released_width = target.released_width;
    options->released_height = target.released_height;
    options->minimum_duration_milliseconds = target.minimum_duration_milliseconds;
    options->maximum_duration_milliseconds = target.maximum_duration_milliseconds;
    options->minimum_aspect_numerator = target.minimum_aspect_numerator;
    options->minimum_aspect_denominator = target.minimum_aspect_denominator;
    options->maximum_aspect_numerator = target.maximum_aspect_numerator;
    options->maximum_aspect_denominator = target.maximum_aspect_denominator;
    if (yvex_runtime_media_execution_preset_build(
            &host, &options->execution_preset, &err) != YVEX_OK)
        return 0;
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

static int test_media_direct_prompt_routing(void)
{
    static const char *const prompts[] = {
        "high desert",
        "HD stars",
        "a seed falling in the sand",
        "20 people",
        "5 seconds later in the story",
        "draft horses",
        "MOVimento della camera",
    };
    char root[] = "/tmp/yvex-media-direct-XXXXXX";
    yvex_server_media_options options;
    server_media_summary first = {0}, repeated = {0};
    server_media_registry *registry = NULL, *second = NULL;
    server_telemetry *telemetry = NULL, *second_telemetry = NULL;
    yvex_server_metrics metrics = {0};
    media_messages messages = {0};
    yvex_error err;
    unsigned long long index;
    int rc;
    YVEX_TEST_ASSERT(mkdtemp(root) != NULL, "direct media output root");
    YVEX_TEST_ASSERT(media_options(&options, root), "direct media options");
    rc = yvex_server_telemetry_open(&telemetry, 16u, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "direct media telemetry");
    rc = yvex_server_media_registry_open(&registry, &options, telemetry, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "direct media registry");
    for (index = 0ull; index < sizeof(prompts) / sizeof(prompts[0]); ++index) {
        char name[32];
        (void)snprintf(name, sizeof(name), "opaque-%llu", index);
        memset(&messages, 0, sizeof(messages));
        YVEX_TEST_ASSERT(media_registry_request(registry, YVEX_CLIENT_OP_SESSION_NEW,
                                                name, NULL, &messages, &err) == YVEX_OK,
                         "opaque media session");
        memset(&messages, 0, sizeof(messages));
        rc = media_registry_request(registry, YVEX_CLIENT_OP_GENERATION_TURN, name,
                                    prompts[index], &messages, &err);
        YVEX_TEST_ASSERT(rc == YVEX_ERR_FORMAT && messages.started == 1ull &&
                             messages.events == 2ull && messages.errors == 1ull &&
                             messages.status == YVEX_ERR_FORMAT &&
                             messages.phase == YVEX_CLIENT_PHASE_FAILED &&
                             !messages.completed &&
                             !messages.media_results && !messages.text[0],
                         "media execution failure emits one terminal response");
        memset(&messages, 0, sizeof(messages));
        YVEX_TEST_ASSERT(media_registry_request(registry, YVEX_CLIENT_OP_SESSION_CLOSE,
                                                name, NULL, &messages, &err) == YVEX_OK,
                         "failed direct media session closes");
    }
    YVEX_TEST_ASSERT(yvex_server_media_registry_summary(registry, &first, &err) == YVEX_OK,
                     "media first identity");
    YVEX_TEST_ASSERT(yvex_server_telemetry_open(
                         &second_telemetry, 16u, &err) == YVEX_OK &&
                         yvex_server_media_registry_open(
                             &second, &options, second_telemetry, &err) == YVEX_OK &&
                         yvex_server_media_registry_summary(second, &repeated, &err) == YVEX_OK &&
                         !strcmp(first.runtime_model_identity,
                                 repeated.runtime_model_identity) &&
                         !strcmp(first.specialization_identity,
                                 repeated.specialization_identity),
                     "media runtime and profile identities are deterministic without aliases");
    memset(&messages, 0, sizeof(messages));
    YVEX_TEST_ASSERT(media_registry_request(second, YVEX_CLIENT_OP_SESSION_NEW,
                                            "unload-owned", NULL, &messages,
                                            &err) == YVEX_OK &&
                         yvex_server_telemetry_metrics_copy(
                             second_telemetry, &metrics, &err) == YVEX_OK &&
                         metrics.active_sessions == 1ull &&
                         metrics.total_sessions == 1ull,
                     "media registry owns one active session");
    yvex_server_media_registry_close(&second);
    memset(&metrics, 0, sizeof(metrics));
    YVEX_TEST_ASSERT(yvex_server_telemetry_metrics_copy(
                         second_telemetry, &metrics, &err) == YVEX_OK &&
                         metrics.active_sessions == 0ull &&
                         metrics.total_sessions == 1ull,
                     "media registry close releases active session telemetry");
    yvex_server_telemetry_close(&second_telemetry);
    yvex_server_media_registry_close(&registry);
    yvex_server_telemetry_close(&telemetry);
    YVEX_TEST_ASSERT(rmdir(root) == 0, "direct media output root removed empty");
    return 0;
}

static int test_media_engine_lifecycle(void)
{
    const char *fixture = "tests/fixtures/gguf/valid-tokenizer-simple.gguf";
    char root[] = "/tmp/yvex-media-host-XXXXXX";
    char socket_path[YVEX_SERVER_SOCKET_PATH_CAP];
    yvex_server_media_options media;
    yvex_server_options options;
    yvex_server_engine_options engine_options;
    yvex_server_engine_summary first, second, reloaded, unloaded;
    yvex_server_engine_summary engines[YVEX_SERVER_IMPLEMENTATION_MAXIMUM_ENGINES];
    yvex_server_summary summary;
    yvex_server_event lifecycle_event;
    yvex_client_message wire = {0}, decoded;
    unsigned char frame[16384];
    unsigned long long frame_count = 0ull, engine_count = 0ull;
    unsigned long long event_cursor = 0ull;
    yvex_server *server = NULL;
    yvex_error err;
    int rc, saw_load_requested = 0, saw_load_progress = 0, saw_ready = 0;
    int saw_unload_started = 0, saw_unloaded = 0;
    YVEX_TEST_ASSERT(mkdtemp(root) != NULL, "media host output root");
    YVEX_TEST_ASSERT(snprintf(socket_path, sizeof(socket_path), "%s/yvexd.sock", root) > 0,
                     "media host socket path");
    test_options(&options);
    options.socket_path = socket_path;
    options.maximum_engines = 2ull;
    options.worker_count = 2ull;
    YVEX_TEST_ASSERT(media_options(&media, root), "media host options");
    media.request_template.text_artifact_path = fixture;
    media.request_template.transformer_artifact_path = fixture;
    media.request_template.video_artifact_path = fixture;
    media.request_template.audio_artifact_path = fixture;
    test_engine_options(&engine_options, "minimax-a");
    engine_options.artifact_path = NULL;
    engine_options.runtime_binding_path = NULL;
    engine_options.target_id = "minimax-h3-base-fl2va-t2va";
    engine_options.backend = YVEX_BACKEND_KIND_CUDA;
    engine_options.engine_kind = YVEX_SERVER_ENGINE_MEDIA;
    engine_options.execution_strategy =
        YVEX_SERVER_EXECUTION_NOT_APPLICABLE;
    engine_options.context_capacity = 0ull;
    engine_options.prefill_chunk_tokens = 0ull;
    engine_options.maximum_new_tokens = 0ull;
    rc = yvex_server_create(&server, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && server != NULL, "media host create");
    YVEX_TEST_ASSERT(yvex_server_start(server, &err) == YVEX_OK,
                     "media host starts with no implicit model");
    YVEX_TEST_ASSERT(yvex_server_get_summary(server, &summary, &err) == YVEX_OK &&
                         summary.status == YVEX_SERVER_STATUS_READY &&
                         summary.host_ready && !summary.engine_count &&
                         !summary.metrics.model_open_count &&
                         summary.maximum_engines == 2ull,
                     "empty media-capable host is independently ready");
    engine_options.execution_strategy = YVEX_SERVER_EXECUTION_TARGET_ONLY;
    YVEX_TEST_ASSERT(
        yvex_server_media_engine_load(
            server, &engine_options, &media, &first, &err) ==
            YVEX_ERR_INVALID_ARG,
        "media engine refuses a text execution strategy");
    engine_options.execution_strategy =
        YVEX_SERVER_EXECUTION_NOT_APPLICABLE;
    memset(&first, 0, sizeof(first));
    YVEX_TEST_ASSERT(yvex_server_media_engine_load(
                         server, &engine_options, &media, &first, &err) == YVEX_OK &&
                         first.state == YVEX_SERVER_ENGINE_LOADED &&
                         first.execution_ready && first.generation != 0ull &&
                         first.engine_kind == YVEX_SERVER_ENGINE_MEDIA &&
                         first.execution_strategy ==
                             YVEX_SERVER_EXECUTION_NOT_APPLICABLE &&
                         first.capacity.runnable_work_capacity == 2ull &&
                         first.capacity.physical_sequence_width == 1ull &&
                         first.capacity.cooperative_scheduling_ready &&
                         !first.capacity.continuous_batching_ready &&
                         first.resources.placement ==
                             YVEX_EXECUTION_PLACEMENT_COMPOSITE &&
                         first.resources.component_count == 4ull &&
                         !(first.resources.available &
                           YVEX_EXECUTION_RESOURCE_PHYSICAL_RESIDENCY_AVAILABLE),
                     "first composite engine loads into the running host");
    YVEX_TEST_ASSERT(yvex_server_media_engine_load(server, &engine_options, &media, &unloaded, &err) ==
        YVEX_ERR_STATE && unloaded.generation == first.generation,
                     "second load reports the existing generation without creating another engine");
    engine_options.alias = "minimax-b";
    memset(&second, 0, sizeof(second));
    YVEX_TEST_ASSERT(yvex_server_media_engine_load(
                         server, &engine_options, &media, &second, &err) == YVEX_OK &&
                         second.state == YVEX_SERVER_ENGINE_LOADED &&
                         second.generation != first.generation,
                     "host can own a second fitting engine generation");
    engine_options.alias = "minimax-c";
    YVEX_TEST_ASSERT(
        yvex_server_media_engine_load(
            server, &engine_options, &media, &unloaded, &err) ==
            YVEX_ERR_BOUNDS,
        "configured engine-slot capacity refuses a third engine independently of resources");
    engine_options.alias = "minimax-b";
    YVEX_TEST_ASSERT(yvex_server_engine_snapshot(
                         server, engines, YVEX_SERVER_IMPLEMENTATION_MAXIMUM_ENGINES,
                         &engine_count, &err) == YVEX_OK &&
                         engine_count == 2ull &&
                         engines[0].state == YVEX_SERVER_ENGINE_LOADED &&
                         engines[1].state == YVEX_SERVER_ENGINE_LOADED,
                     "engine manager publishes both real loaded engines");
    YVEX_TEST_ASSERT(yvex_server_get_summary(server, &summary, &err) == YVEX_OK &&
                         summary.host_ready && summary.loaded_engine_count == 2ull &&
                         summary.metrics.queue_depth == 0ull &&
                         summary.metrics.queue_capacity == 4ull &&
                         summary.metrics.model_open_count == 2ull &&
                         summary.metrics.artifact_open_count == 8ull &&
                         summary.metrics.binding_open_count == 2ull &&
                         summary.metrics.materialization_count == 0ull &&
                         summary.metrics.residency_build_count == 0ull &&
                         summary.metrics.resident_device_bytes == 0ull &&
                         summary.metrics.resources.component_count == 8ull &&
                         !(summary.metrics.resources.available &
                           YVEX_EXECUTION_RESOURCE_PHYSICAL_RESIDENCY_AVAILABLE) &&
                         !first.runtime_binding_identity[0] &&
                         !first.artifact_identity[0],
                     "each engine owns a distinct bounded request_queue without false payload residency");
    wire.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    wire.kind = YVEX_CLIENT_MESSAGE_ENGINE;
    wire.status = YVEX_OK;
    wire.engine = first;
    YVEX_TEST_ASSERT(
        yvex_protocol_message_encode(&wire, frame, sizeof(frame), &frame_count,
                                     &err) == YVEX_OK &&
            yvex_protocol_message_decode(frame, frame_count, &decoded, &err) == YVEX_OK &&
            decoded.engine.engine_kind == YVEX_SERVER_ENGINE_MEDIA &&
            decoded.engine.execution_strategy ==
                YVEX_SERVER_EXECUTION_NOT_APPLICABLE &&
            decoded.engine.generation == first.generation &&
            !strcmp(decoded.engine.alias, "minimax-a"),
        "exact media engine generation crosses the local protocol");
    YVEX_TEST_ASSERT(yvex_server_engine_unload(
                         server, first.alias, first.generation,
                         &unloaded, &err) == YVEX_OK &&
                         unloaded.state == YVEX_SERVER_ENGINE_UNLOADED &&
                         yvex_server_engine_unload(
                             server, first.alias, first.generation,
                             &unloaded, &err) == YVEX_OK,
                     "unload of the same retired generation is a successful no-op");
    engine_options.alias = "minimax-a";
    memset(&reloaded, 0, sizeof(reloaded));
    YVEX_TEST_ASSERT(yvex_server_media_engine_load(
                         server, &engine_options, &media, &reloaded, &err) == YVEX_OK &&
                         reloaded.generation > first.generation,
                     "reloading one alias creates a distinct engine generation");
    YVEX_TEST_ASSERT(yvex_server_engine_unload(server, first.alias, first.generation, &unloaded, &err) ==
        YVEX_ERR_STATE, "idempotent unload cannot retire a newer engine with a stale generation");
    YVEX_TEST_ASSERT(yvex_server_engine_unload(
                         server, second.alias, second.generation,
                         &unloaded, &err) == YVEX_OK &&
                         yvex_server_engine_unload(
                             server, reloaded.alias, reloaded.generation,
                             &unloaded, &err) == YVEX_OK,
                     "independent engines unload without stopping the host");
    YVEX_TEST_ASSERT(yvex_server_get_summary(server, &summary, &err) == YVEX_OK &&
                         summary.host_ready && !summary.loaded_engine_count &&
                         summary.metrics.model_open_count == 3ull &&
                         summary.metrics.model_close_count == 3ull &&
                         !summary.metrics.mapped_artifact_bytes &&
                         !summary.metrics.resident_host_bytes &&
                         !summary.metrics.resident_device_bytes,
                     "all engine resources close while the host remains ready");
    while (yvex_server_event_next(server, event_cursor, 0,
                                  &lifecycle_event, &err) == YVEX_OK) {
        event_cursor = lifecycle_event.sequence;
        saw_load_requested |= lifecycle_event.kind ==
                              YVEX_SERVER_EVENT_ENGINE_LOAD_REQUESTED;
        if (lifecycle_event.kind == YVEX_SERVER_EVENT_ENGINE_LOAD_PROGRESS) {
            YVEX_TEST_ASSERT(
                lifecycle_event.engine_kind == YVEX_SERVER_ENGINE_MEDIA &&
                    lifecycle_event.measurement.scope ==
                        YVEX_EXECUTION_SCOPE_MODEL_LIFECYCLE &&
                    lifecycle_event.measurement.work_unit ==
                        YVEX_EXECUTION_WORK_BYTES &&
                    (lifecycle_event.measurement.available &
                     YVEX_EXECUTION_MEASUREMENT_DENOMINATOR_AVAILABLE) &&
                    lifecycle_event.measurement.total_units > 0ull &&
                    lifecycle_event.measurement.completed_units <=
                        lifecycle_event.measurement.total_units,
                "composite load progress carries real byte denominator");
            saw_load_progress = 1;
        }
        saw_ready |= lifecycle_event.kind == YVEX_SERVER_EVENT_ENGINE_READY;
        saw_unload_started |= lifecycle_event.kind ==
                              YVEX_SERVER_EVENT_ENGINE_UNLOAD_STARTED;
        saw_unloaded |= lifecycle_event.kind == YVEX_SERVER_EVENT_ENGINE_UNLOADED;
    }
    YVEX_TEST_ASSERT(saw_load_requested && saw_load_progress && saw_ready &&
                         saw_unload_started && saw_unloaded,
                     "engine load and unload chronology is retained");
    YVEX_TEST_ASSERT(yvex_server_finish(server, &err) == YVEX_OK,
                     "empty persistent host finishes separately");
    yvex_server_close(&server);
    YVEX_TEST_ASSERT(rmdir(root) == 0, "media host output root removed empty");
    return 0;
}

static int test_media_family_profile(void)
{
    const yvex_component_variant_adapter *adapter;
    yvex_runtime_media_execution_preset preset, released, mutated;
    yvex_runtime_media_execution_request request = {0};
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
    YVEX_TEST_ASSERT(profile.schema_version == YVEX_RUNTIME_MEDIA_HOST_SCHEMA_V2,
                     "media host profile schema");
    YVEX_TEST_ASSERT_STREQ(profile.request_template.target,
                           "minimax-h3-fl2va", "media host target");
    YVEX_TEST_ASSERT(profile.request_template.component_backend == YVEX_BACKEND_KIND_CUDA,
                     "media host CUDA backend");
    YVEX_TEST_ASSERT(profile.request_template.condition &&
                         profile.request_template.keyframe_encode &&
                         profile.request_template.latent &&
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
    YVEX_TEST_ASSERT(profile.canvas_short_edge == 768ull &&
                         profile.minimum_canvas_pixels == 768ull * 768ull &&
                         profile.maximum_canvas_pixels == 768ull * 1344ull &&
                         profile.released_width == 1344ull &&
                         profile.released_height == 768ull &&
                         profile.released_sigma_grid_points == 50ull &&
                         profile.request_template.maximum_device_bytes == 64ull << 30u &&
                         profile.request_template.maximum_workspace_bytes == 16ull << 30u &&
                         profile.request_template.maximum_packed_rows == 106238ull,
                     "released FL2VA canvas, trajectory, row, and memory envelope");
    rc = yvex_runtime_media_execution_preset_build(&profile, &preset, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && preset.complete &&
                         !strcmp(preset.name, "interactive-preview-v1") &&
                         !strcmp(preset.profile, "preview") &&
                         !strcmp(preset.format, "avi") &&
                         preset.width == 192ull && preset.height == 192ull &&
                         preset.frames == 124ull && preset.sigma_grid_points == 2ull &&
                         preset.seed == 42ull && yvex_sha256_hex_valid(preset.identity),
                     "hosted media preset is one explicit identity-bearing YVEX policy");
    request.schema_version = YVEX_RUNTIME_MEDIA_EXECUTION_SCHEMA_V1;
    request.kind = YVEX_RUNTIME_MEDIA_EXECUTION_DEFAULT;
    rc = yvex_runtime_media_execution_resolve(&profile, &request, &released, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && released.complete &&
                         !strcmp(released.name, "released-fl2va-v1") &&
                         !strcmp(released.profile, "released") &&
                         released.width == 1344ull && released.height == 768ull &&
                         released.frames == 124ull &&
                         released.sigma_grid_points == 50ull &&
                         released.seed == 42ull,
                     "normal media execution resolves to released FL2VA independently from preview");
    request.present = YVEX_RUNTIME_MEDIA_EXECUTION_WIDTH |
                      YVEX_RUNTIME_MEDIA_EXECUTION_HEIGHT |
                      YVEX_RUNTIME_MEDIA_EXECUTION_DURATION |
                      YVEX_RUNTIME_MEDIA_EXECUTION_SEED;
    request.width = 768ull;
    request.height = 1344ull;
    request.duration_milliseconds = 10000ull;
    request.seed = 7ull;
    rc = yvex_runtime_media_execution_resolve(&profile, &request, &released, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && released.width == 768ull &&
                         released.height == 1344ull && released.frames == 243ull &&
                         released.seed == 7ull &&
                         yvex_sha256_hex_valid(released.identity),
                     "portrait duration aligns upward under the released contract");
    request.duration_milliseconds = 15000ull;
    YVEX_TEST_ASSERT(yvex_runtime_media_execution_resolve(
                         &profile, &request, &released, &err) == YVEX_ERR_BOUNDS,
                     "duration that aligns beyond 15 seconds refuses instead of truncating");
    request.duration_milliseconds = 5000ull;
    request.width = 192ull;
    request.height = 192ull;
    YVEX_TEST_ASSERT(yvex_runtime_media_execution_resolve(
                         &profile, &request, &released, &err) == YVEX_ERR_BOUNDS,
                     "released trajectory refuses the bounded preview canvas");
    request.width = 4096ull;
    request.height = 768ull;
    YVEX_TEST_ASSERT(yvex_runtime_media_execution_resolve(
                         &profile, &request, &released, &err) == YVEX_ERR_BOUNDS,
                     "released canvas outside aspect and area bounds refuses");
    mutated = preset;
    mutated.seed++;
    YVEX_TEST_ASSERT(yvex_runtime_media_execution_preset_validate(
                         &profile, &mutated, &err) == YVEX_ERR_FORMAT,
                     "hosted media preset settings cannot mutate without identity change");
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
    server_event_scope scope = {0};
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
    scope.engine_kind = YVEX_SERVER_ENGINE_TEXT;
    scope.execution_strategy = YVEX_SERVER_EXECUTION_SPECULATIVE;
    strcpy(scope.runtime_model_identity, request.request_identity);
    strcpy(scope.artifact_identity, request.request_identity);
    strcpy(scope.specialization_identity, request.request_identity);
    rc = yvex_server_telemetry_open(&telemetry, 4u, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "provider telemetry open");
    rc = yvex_server_telemetry_emit_provider(
        telemetry, &scope, YVEX_SERVER_EVENT_REQUEST_STARTED,
        YVEX_SERVER_SEVERITY_INFO, "session", "r1", "t1", "turn",
        1u, 0u, 4u, 0.0, 0.0, NULL, &request, NULL, &emitted, &err);
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
    YVEX_TEST_ASSERT(
        event.engine_kind == YVEX_SERVER_ENGINE_TEXT &&
            event.execution_strategy == YVEX_SERVER_EXECUTION_SPECULATIVE,
        "event scope preserves engine kind and execution strategy");
    YVEX_TEST_ASSERT_STREQ(event.runtime_model_identity,
                           scope.runtime_model_identity,
                           "event scope owns model lineage");
    YVEX_TEST_ASSERT_STREQ(event.artifact_identity, scope.artifact_identity,
                           "event scope owns artifact lineage");
    YVEX_TEST_ASSERT_STREQ(event.variant_identity,
                           scope.specialization_identity,
                           "event scope owns specialization lineage");
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
        telemetry, &scope, YVEX_SERVER_EVENT_SPECULATIVE_CYCLE_COMMITTED,
        YVEX_SERVER_SEVERITY_INFO, "session", "r1", "t1", "speculation",
        0u, 0u, 0u, progress.seconds, 0.0, &progress, &request, NULL, &emitted,
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
                         event.engine_kind == YVEX_SERVER_ENGINE_TEXT &&
                         event.execution_strategy ==
                             YVEX_SERVER_EXECUTION_SPECULATIVE,
                     "typed speculation telemetry facts");
    rc = yvex_server_event_json(&event, json, sizeof(json), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK &&
                         strstr(json, "\"speculative_cycle\":2") != NULL &&
                         strstr(json, "\"accepted_tokens\":3") != NULL &&
                         strstr(json, "\"discarded_tokens\":1") != NULL &&
                         strstr(json, "\"confidence_logit_count\":5") != NULL,
                     "typed speculation telemetry JSON");
    scope.runtime_model_identity[0] = 'z';
    rc = yvex_server_telemetry_emit_provider(
        telemetry, &scope, YVEX_SERVER_EVENT_REQUEST_STARTED,
        YVEX_SERVER_SEVERITY_INFO, "session", "r2", "t2", "turn",
        0u, 0u, 0u, 0.0, 0.0, NULL, &request, NULL, NULL, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_INVALID_ARG,
                     "invalid engine event lineage refuses before publication");
    yvex_server_telemetry_close(&telemetry);
    return 0;
}

int yvex_test_server(void)
{
    if (test_automatic_reasoning_policy() != 0) return 1;
    if (test_request_queue_serialization() != 0) return 1;
    if (test_session_store() != 0) return 1;
    if (test_configured_summary_and_event() != 0) return 1;
    if (test_adaptive_prefill_policy() != 0) return 1;
    if (test_model_open_refusal() != 0) return 1;
    if (test_bounded_telemetry_overflow() != 0) return 1;
    if (test_telemetry_observability_economics() != 0) return 1;
    if (test_decode_rate_scopes() != 0) return 1;
    if (test_profile_wall_reconciliation() != 0) return 1;
    if (test_provider_telemetry() != 0) return 1;
    if (test_openai_listener_admission() != 0) return 1;
    if (test_media_direct_prompt_routing() != 0) return 1;
    if (test_media_family_profile() != 0) return 1;
    if (test_media_engine_lifecycle() != 0) return 1;
    return 0;
}
