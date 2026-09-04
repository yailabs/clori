#define _POSIX_C_SOURCE 200809L

/* Test-only finite native-protocol consumer. The product CLI intentionally has no one-shot
 * generation command; integration tests use this client to exercise the same typed host plane. */
#include <yvex/server.h>

#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    char alias[YVEX_SERVER_MODEL_ALIAS_CAP];
    unsigned long long generation;
} test_engine;

typedef struct {
    test_engine engine;
    char session[YVEX_SERVER_SESSION_NAME_CAP];
    atomic_int done;
    atomic_int interrupted;
    pthread_t thread;
    sigset_t previous_mask;
    int active;
} test_signals;

typedef struct {
    const char *model;
    const char *session;
    const unsigned char *prompt;
    unsigned long long prompt_bytes;
    unsigned long long maximum_new_tokens;
    yvex_reasoning_policy reasoning;
    const char *attachment;
} test_options;

static unsigned long long next_request = 1ull;

static void request_init(yvex_client_request *request,
                         yvex_client_operation operation)
{
    yvex_provider_request defaults;
    yvex_provider_request_default(&defaults);
    memset(request, 0, sizeof(*request));
    request->schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    request->operation = operation;
    request->request_number = next_request++;
    request->maximum_new_tokens = defaults.maximum_output_tokens;
    request->stochastic = defaults.sampling.stochastic;
    request->seed_present = defaults.sampling.seed_present;
    request->seed = defaults.sampling.seed;
    request->temperature = defaults.sampling.temperature;
    request->top_k = defaults.sampling.top_k;
    request->top_p = defaults.sampling.top_p;
    request->min_p = defaults.sampling.min_p;
    request->typical_p = defaults.sampling.typical_p;
    request->reasoning_policy = YVEX_REASONING_DISABLED;
}

static void request_bind(yvex_client_request *request,
                         const test_engine *engine)
{
    if (!engine) return;
    (void)snprintf(request->model_alias, sizeof(request->model_alias), "%s",
                   engine->alias);
    request->engine_generation = engine->generation;
}

static int request_open(yvex_client **client,
                        const yvex_client_request *request, yvex_error *err)
{
    int rc = yvex_client_connect(client, NULL, err);
    if (rc == YVEX_OK) rc = yvex_client_send(*client, request, err);
    if (rc != YVEX_OK) yvex_client_close(client);
    return rc;
}

static int message_error(const yvex_client_message *message, yvex_error *err,
                         const char *where)
{
    yvex_error_set(err, (yvex_status)message->status, where, message->reason);
    return message->status;
}

static int model_lease_action(yvex_client_operation operation,
                              const char *value, yvex_error *err)
{
    yvex_client_request request;
    yvex_client_message message;
    yvex_client *client = NULL;
    int rc;
    request_init(&request, operation);
    if (operation == YVEX_CLIENT_OP_ENGINE_ENSURE_ACTIVE)
        (void)snprintf(request.model_alias, sizeof(request.model_alias), "%s",
                       value);
    else
        (void)snprintf(request.model_lease_identity,
                       sizeof(request.model_lease_identity), "%s", value);
    rc = request_open(&client, &request, err);
    if (rc == YVEX_OK) rc = yvex_client_receive(client, &message, err);
    if (rc == YVEX_OK && message.kind == YVEX_CLIENT_MESSAGE_ERROR)
        rc = message_error(&message, err, "test.native-turn.model-lease");
    else if (rc == YVEX_OK &&
             (message.kind != YVEX_CLIENT_MESSAGE_ENGINE ||
              !message.model_lease_identity[0])) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "test.native-turn.model-lease",
                       "host returned an invalid model lease response");
        rc = YVEX_ERR_FORMAT;
    }
    if (rc == YVEX_OK)
        printf("%s %llu %s\n", message.engine.alias,
               message.engine.generation, message.model_lease_identity);
    yvex_client_close(&client);
    return rc;
}

static int engine_resolve(const char *requested, test_engine *engine,
                          yvex_error *err)
{
    yvex_client_request request;
    yvex_client_message message;
    yvex_client *client = NULL;
    unsigned long long matches = 0ull;
    int rc;
    memset(engine, 0, sizeof(*engine));
    request_init(&request, YVEX_CLIENT_OP_ENGINE_LIST);
    rc = request_open(&client, &request, err);
    while (rc == YVEX_OK) {
        rc = yvex_client_receive(client, &message, err);
        if (rc != YVEX_OK || message.kind == YVEX_CLIENT_MESSAGE_ACK) break;
        if (message.kind == YVEX_CLIENT_MESSAGE_ERROR) {
            rc = message_error(&message, err, "test.native-turn.engine-list");
            break;
        }
        if (message.kind != YVEX_CLIENT_MESSAGE_ENGINE) {
            yvex_error_set(err, YVEX_ERR_FORMAT,
                           "test.native-turn.engine-list",
                           "host returned an invalid engine catalog");
            rc = YVEX_ERR_FORMAT;
            break;
        }
        if (message.engine.state != YVEX_SERVER_ENGINE_LOADED ||
            (requested && strcmp(requested, message.engine.alias)))
            continue;
        matches++;
        (void)snprintf(engine->alias, sizeof(engine->alias), "%s",
                       message.engine.alias);
        engine->generation = message.engine.generation;
    }
    yvex_client_close(&client);
    if (rc != YVEX_OK) return rc;
    if (!matches) {
        yvex_error_set(err, YVEX_ERR_STATE, "test.native-turn.engine",
                       requested ? "requested engine is not loaded"
                                 : "one unambiguous loaded engine is required");
        return YVEX_ERR_STATE;
    }
    if (!requested && matches != 1ull) {
        yvex_error_set(err, YVEX_ERR_STATE, "test.native-turn.engine",
                       "one unambiguous loaded engine is required");
        return YVEX_ERR_STATE;
    }
    return YVEX_OK;
}

static int session_action(yvex_client_operation operation,
                          const test_engine *engine, const char *session,
                          yvex_error *err)
{
    yvex_client_request request;
    yvex_client_message message;
    yvex_client *client = NULL;
    int rc;
    request_init(&request, operation);
    request_bind(&request, engine);
    (void)snprintf(request.session_name, sizeof(request.session_name), "%s",
                   session);
    rc = request_open(&client, &request, err);
    if (rc == YVEX_OK) rc = yvex_client_receive(client, &message, err);
    if (rc == YVEX_OK && message.kind == YVEX_CLIENT_MESSAGE_ERROR)
        rc = message_error(&message, err, "test.native-turn.session");
    else if (rc == YVEX_OK &&
             message.kind != YVEX_CLIENT_MESSAGE_SESSION &&
             message.kind != YVEX_CLIENT_MESSAGE_ACK) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "test.native-turn.session",
                       "host returned an invalid session response");
        rc = YVEX_ERR_FORMAT;
    }
    yvex_client_close(&client);
    return rc;
}

static int cancellation_request(const test_engine *engine,
                                const char *session)
{
    yvex_client_request request;
    yvex_client_message message;
    yvex_client *client = NULL;
    yvex_error err;
    int rc;
    request_init(&request, YVEX_CLIENT_OP_GENERATION_CANCEL);
    request_bind(&request, engine);
    (void)snprintf(request.session_name, sizeof(request.session_name), "%s",
                   session);
    rc = request_open(&client, &request, &err);
    if (rc == YVEX_OK) rc = yvex_client_receive(client, &message, &err);
    yvex_client_close(&client);
    return rc == YVEX_OK && message.kind == YVEX_CLIENT_MESSAGE_ACK;
}

static void *signal_main(void *opaque)
{
    test_signals *state = opaque;
    sigset_t signals;
    int number;
    (void)sigemptyset(&signals);
    (void)sigaddset(&signals, SIGINT);
    (void)sigaddset(&signals, SIGUSR1);
    while (sigwait(&signals, &number) == 0) {
        if (number == SIGUSR1) break;
        atomic_store_explicit(&state->interrupted, 1, memory_order_release);
        while (!atomic_load_explicit(&state->done, memory_order_acquire)) {
            struct timespec delay = {0, 10000000L};
            if (cancellation_request(&state->engine, state->session)) break;
            (void)nanosleep(&delay, NULL);
        }
    }
    return NULL;
}

static int signals_open(test_signals *state, const test_engine *engine,
                        const char *session)
{
    sigset_t signals;
    memset(state, 0, sizeof(*state));
    state->engine = *engine;
    (void)snprintf(state->session, sizeof(state->session), "%s", session);
    atomic_init(&state->done, 0);
    atomic_init(&state->interrupted, 0);
    (void)sigemptyset(&signals);
    (void)sigaddset(&signals, SIGINT);
    (void)sigaddset(&signals, SIGUSR1);
    if (pthread_sigmask(SIG_BLOCK, &signals, &state->previous_mask) != 0)
        return 0;
    if (pthread_create(&state->thread, NULL, signal_main, state) != 0) {
        (void)pthread_sigmask(SIG_SETMASK, &state->previous_mask, NULL);
        return 0;
    }
    state->active = 1;
    return 1;
}

static int signals_close(test_signals *state)
{
    int interrupted;
    if (!state->active) return 0;
    atomic_store_explicit(&state->done, 1, memory_order_release);
    (void)pthread_kill(state->thread, SIGUSR1);
    (void)pthread_join(state->thread, NULL);
    (void)pthread_sigmask(SIG_SETMASK, &state->previous_mask, NULL);
    interrupted = atomic_load_explicit(&state->interrupted,
                                       memory_order_acquire);
    state->active = 0;
    return interrupted;
}

static const char *stop_name(unsigned int reason)
{
    switch ((yvex_client_stop_reason)reason) {
    case YVEX_CLIENT_STOP_EOS: return "eos";
    case YVEX_CLIENT_STOP_TOKENIZER_TOKEN: return "tokenizer token";
    case YVEX_CLIENT_STOP_MAXIMUM_TOKENS: return "maximum tokens";
    case YVEX_CLIENT_STOP_CONTEXT_CAPACITY: return "context capacity";
    case YVEX_CLIENT_STOP_CANCELLED: return "cancelled";
    case YVEX_CLIENT_STOP_MODEL_FAILURE: return "model failure";
    case YVEX_CLIENT_STOP_TOKENIZER_FAILURE: return "tokenizer failure";
    case YVEX_CLIENT_STOP_OUTPUT_FAILURE: return "output failure";
    case YVEX_CLIENT_STOP_NONE: break;
    }
    return "none";
}

static void metrics_print(const yvex_client_message *message)
{
    fprintf(stderr,
            "generation %llu tokens · %llu prompt/%llu reused · stop %s · session %s\n",
            message->generated_tokens, message->prompt_tokens,
            message->reused_tokens, stop_name(message->stop_reason),
            message->session_name);
    if (message->reasoning_tokens || message->first_reasoning_seconds > 0.0)
        fprintf(stderr,
                "reasoning %llu tokens · %.2f s · final %llu tokens · %.2f s\n",
                message->reasoning_tokens, message->reasoning_seconds,
                message->final_tokens, message->final_seconds);
}

static int turn_execute(const test_engine *engine, const char *session,
                        const test_options *options, yvex_error *err)
{
    yvex_content_part content[2] = {0};
    yvex_client_request request;
    yvex_client_message message;
    yvex_client *client = NULL;
    test_signals signals;
    int rc, complete = 0;
    request_init(&request, YVEX_CLIENT_OP_GENERATION_TURN);
    request_bind(&request, engine);
    (void)snprintf(request.session_name, sizeof(request.session_name), "%s",
                   session);
    if (options->attachment) {
        struct stat facts;
        if (stat(options->attachment, &facts) != 0 ||
            !S_ISREG(facts.st_mode) || facts.st_size <= 0) {
            yvex_error_set(err, YVEX_ERR_INVALID_ARG,
                           "test.native-turn.content",
                           "regular attachment file is required");
            return YVEX_ERR_INVALID_ARG;
        }
        content[0].schema_version = YVEX_CONTENT_PART_SCHEMA_V1;
        content[0].kind = YVEX_CONTENT_AUDIO;
        content[0].storage = YVEX_CONTENT_LOCAL_FILE;
        content[0].byte_count = (unsigned long long)facts.st_size;
        (void)snprintf(content[0].media_type, sizeof(content[0].media_type),
                       "audio/wav");
        (void)snprintf(content[0].reference, sizeof(content[0].reference),
                       "%s", options->attachment);
        if (yvex_content_part_seal(content, err) != YVEX_OK)
            return yvex_error_code(err);
        content[1].schema_version = YVEX_CONTENT_PART_SCHEMA_V1;
        content[1].kind = YVEX_CONTENT_TEXT;
        content[1].storage = YVEX_CONTENT_INLINE;
        content[1].bytes = options->prompt;
        content[1].byte_count = options->prompt_bytes;
        (void)snprintf(content[1].media_type, sizeof(content[1].media_type),
                       "text/plain;charset=utf-8");
        if (yvex_content_part_seal(content + 1u, err) != YVEX_OK)
            return yvex_error_code(err);
        request.content_parts = content;
        request.content_part_count = 2u;
    } else {
        request.prompt = options->prompt;
        request.prompt_bytes = options->prompt_bytes;
    }
    request.maximum_new_tokens = options->maximum_new_tokens;
    request.stochastic = 0;
    request.seed_present = 0;
    request.temperature = 1.0;
    request.top_k = 0ull;
    request.top_p = 1.0;
    request.min_p = 0.0;
    request.typical_p = 1.0;
    request.reasoning_policy = options->reasoning;
    if (!signals_open(&signals, engine, session)) {
        yvex_error_set(err, YVEX_ERR_STATE, "test.native-turn.signals",
                       "could not establish cancellation ownership");
        return YVEX_ERR_STATE;
    }
    rc = request_open(&client, &request, err);
    while (rc == YVEX_OK && !complete) {
        rc = yvex_client_receive(client, &message, err);
        if (rc != YVEX_OK) break;
        if (message.kind == YVEX_CLIENT_MESSAGE_FRAGMENT) {
            if (message.stream_channel == YVEX_CLIENT_STREAM_FINAL_TEXT ||
                message.stream_channel == YVEX_CLIENT_STREAM_UNSPECIFIED) {
                if (fwrite(message.bytes, 1u, (size_t)message.byte_count,
                           stdout) != (size_t)message.byte_count) {
                    yvex_error_set(err, YVEX_ERR_IO, "test.native-turn.output",
                                   "could not publish generation bytes");
                    rc = YVEX_ERR_IO;
                }
            }
        } else if (message.kind == YVEX_CLIENT_MESSAGE_TURN_COMPLETE) {
            fputc('\n', stdout);
            metrics_print(&message);
            complete = 1;
        } else if (message.kind == YVEX_CLIENT_MESSAGE_ERROR) {
            if (message.content_part_count)
                fprintf(stderr, "content %llu identity %s\n",
                        message.content_part_count,
                        message.input_content_identity);
            if (message.partial_turn.available)
                fprintf(stderr,
                        "partial · %llu committed token%s · position %llu · %s\n",
                        message.partial_turn.committed_token_count,
                        message.partial_turn.committed_token_count == 1ull ? "" : "s",
                        message.partial_turn.final_committed_position,
                        message.partial_turn.reset_required
                            ? "reset required"
                            : "recovery unavailable");
            rc = message_error(&message, err, "test.native-turn.generation");
        }
    }
    yvex_client_close(&client);
    if (signals_close(&signals)) return 130;
    if (rc == YVEX_OK && !complete) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "test.native-turn.generation",
                       "host ended the stream without terminal completion");
        return YVEX_ERR_FORMAT;
    }
    return rc;
}

static int u64_parse(const char *text, unsigned long long *value)
{
    char *end = NULL;
    unsigned long long parsed;
    if (!text || !text[0] || text[0] == '-') return 0;
    parsed = strtoull(text, &end, 10);
    if (!end || *end || !parsed) return 0;
    *value = parsed;
    return 1;
}

static int options_parse(int argc, char **argv, test_options *options)
{
    int index;
    yvex_provider_request defaults;
    memset(options, 0, sizeof(*options));
    yvex_provider_request_default(&defaults);
    options->maximum_new_tokens = defaults.maximum_output_tokens;
    options->reasoning = YVEX_REASONING_DISABLED;
    for (index = 1; index < argc; ++index) {
        const char *arg = argv[index];
        if (!strcmp(arg, "--model") || !strcmp(arg, "--session") ||
            !strcmp(arg, "--max-new-tokens") || !strcmp(arg, "--strategy") ||
            !strcmp(arg, "--reasoning") || !strcmp(arg, "--attach")) {
            const char *value;
            if (++index == argc) return 0;
            value = argv[index];
            if (!strcmp(arg, "--model")) options->model = value;
            else if (!strcmp(arg, "--session")) options->session = value;
            else if (!strcmp(arg, "--attach")) options->attachment = value;
            else if (!strcmp(arg, "--max-new-tokens")) {
                if (!u64_parse(value, &options->maximum_new_tokens)) return 0;
            } else if (!strcmp(arg, "--strategy")) {
                if (strcmp(value, "greedy")) return 0;
            } else if (!strcmp(value, "none"))
                options->reasoning = YVEX_REASONING_DISABLED;
            else if (!strcmp(value, "high"))
                options->reasoning = YVEX_REASONING_ENABLED;
            else if (!strcmp(value, "max"))
                options->reasoning = YVEX_REASONING_MAXIMUM;
            else return 0;
            continue;
        }
        if (arg[0] == '-' || options->prompt) return 0;
        options->prompt = (const unsigned char *)arg;
        options->prompt_bytes = (unsigned long long)strlen(arg);
    }
    return options->prompt && options->prompt_bytes;
}

int main(int argc, char **argv)
{
    test_options options;
    test_engine engine;
    char ephemeral[YVEX_SERVER_SESSION_NAME_CAP];
    const char *session;
    yvex_error err;
    int rc, created = 0;
    if (argc == 3 &&
        (!strcmp(argv[1], "--ensure-active") ||
         !strcmp(argv[1], "--release-lease"))) {
        yvex_client_operation operation = !strcmp(argv[1], "--ensure-active")
            ? YVEX_CLIENT_OP_ENGINE_ENSURE_ACTIVE
            : YVEX_CLIENT_OP_ENGINE_LEASE_RELEASE;
        rc = model_lease_action(operation, argv[2], &err);
        if (rc == YVEX_OK) return 0;
        goto failed;
    }
    if (!options_parse(argc, argv, &options)) {
        fputs("usage: native_turn [--model ENGINE] [--session SESSION] "
              "[--max-new-tokens N] [--strategy greedy] "
              "[--reasoning none|high|max] [--attach ABSOLUTE-PATH] TEXT\n"
              "       native_turn --ensure-active MODEL\n"
              "       native_turn --release-lease LEASE-ID\n", stderr);
        return 2;
    }
    rc = engine_resolve(options.model, &engine, &err);
    if (rc != YVEX_OK) goto failed;
    session = options.session;
    if (!session) {
        (void)snprintf(ephemeral, sizeof(ephemeral), "native-turn-%ld",
                       (long)getpid());
        session = ephemeral;
        rc = session_action(YVEX_CLIENT_OP_SESSION_NEW, &engine, session, &err);
        if (rc != YVEX_OK) goto failed;
        created = 1;
    }
    rc = turn_execute(&engine, session, &options, &err);
    if (created) {
        yvex_error close_error;
        int close_rc = session_action(YVEX_CLIENT_OP_SESSION_CLOSE, &engine,
                                      session, &close_error);
        if (rc == YVEX_OK && close_rc != YVEX_OK) {
            err = close_error;
            rc = close_rc;
        }
    }
    if (rc == 130) return 130;
    if (rc == YVEX_OK) return 0;
failed:
    fprintf(stderr, "yvex-test-native-turn: %s\n", yvex_error_message(&err));
    return 1;
}
