/*
 * Convert client turns into reusable generation turns without reopening the model.
 *
 * One registry row owns one execution session and exact committed token prefix. The sole
 * server-side conversation/session authority used by the model worker.
 */
#define _POSIX_C_SOURCE 200809L
#include "src/server/private.h"
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <yvex/internal/core.h>
#define SESSION_SCHEMA_V1 1u
#define SESSION_MAX_MESSAGES 128u
#define SESSION_TRANSCRIPT_BYTES 1048576u
typedef struct {
    char name[YVEX_SERVER_SESSION_NAME_CAP];
    char identity[YVEX_SHA256_HEX_CAP];
    yvex_server_session_state state;
    yvex_runtime_execution_session *execution;
    yvex_runtime_generation_context *generation;
    yvex_prompt_message messages[SESSION_MAX_MESSAGES];
    unsigned long long message_count;
    unsigned char *transcript;
    unsigned long long transcript_count, transcript_capacity;
    unsigned int *committed_tokens, *prompt_tokens;
    unsigned long long committed_count, token_capacity;
    yvex_runtime_generation_token_result *token_results;
    unsigned char *turn_text;
    unsigned long long text_capacity, turn_count, attached_clients;
    unsigned long long message_history_generation, transcript_generation;
    char last_turn_identity[YVEX_SHA256_HEX_CAP];
    char state_digest[YVEX_SHA256_HEX_CAP];
    char generated_token_identity[YVEX_SHA256_HEX_CAP];
    char generated_text_digest[YVEX_SHA256_HEX_CAP];
    yvex_client_partial_turn partial_turn;
    yvex_runtime_sampling_policy policy;
    yvex_reasoning_policy reasoning_policy;
    int policy_set;
    atomic_int cancel_requested;
    atomic_int active_turn;
} server_session;
struct server_session_registry {
    pthread_mutex_t mutex;
    yvex_runtime_model *model;
    yvex_server_options options;
    server_telemetry *telemetry;
    server_session *sessions;
    unsigned long long capacity, count, next_id;
    int mutex_ready, closing;
};
typedef struct {
    server_session_registry *registry;
    server_session *session;
    const yvex_client_request *request;
    server_message_emit emit;
    void *emit_context;
    char request_id[YVEX_SERVER_ID_CAP];
    char turn_id[YVEX_SERVER_ID_CAP];
    unsigned long long started_ns, prefill_started_ns, prefill_completed_ns;
    unsigned long long first_fragment_ns;
    double queue_seconds;
    yvex_tokenizer_reasoning_stream *reasoning_stream;
} turn_sink;

static int provider_text_stream_direct(const yvex_provider_request *request)
{
    return request && request->response_format == YVEX_PROVIDER_RESPONSE_TEXT &&
           request->stop_count == 0u && request->tool_count == 0u &&
           request->tool_choice.kind == YVEX_PROVIDER_TOOL_CHOICE_NONE;
}

static int provider_output_emit(turn_sink *sink,
                                yvex_provider_output_kind kind,
                                const unsigned char *bytes,
                                unsigned long long count,
                                const yvex_provider_tool_call *call,
                                yvex_error *err);

static unsigned long long monotonic_ns(void)
{
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0u;
    return (unsigned long long)value.tv_sec * 1000000000ull +
           (unsigned long long)value.tv_nsec;
}

static yvex_client_failure_class turn_failure_class(int status)
{
    switch (status) {
    case YVEX_ERR_FORMAT:
    case YVEX_ERR_INVALID_ARG: return YVEX_CLIENT_FAILURE_INVALID_REQUEST;
    case YVEX_ERR_UNSUPPORTED:
        return YVEX_CLIENT_FAILURE_UNSUPPORTED_PARAMETER;
    case YVEX_ERR_BOUNDS: return YVEX_CLIENT_FAILURE_REQUEST_TOO_LARGE;
    case YVEX_ERR_STATE: return YVEX_CLIENT_FAILURE_INCOMPATIBLE_STATE;
    case YVEX_ERR_CANCELLED: return YVEX_CLIENT_FAILURE_CLIENT_CANCELLED;
    case YVEX_ERR_IO:
    case YVEX_ERR_BACKEND: return YVEX_CLIENT_FAILURE_RUNTIME_UNAVAILABLE;
    case YVEX_ERR_TIMEOUT: return YVEX_CLIENT_FAILURE_GATEWAY_TIMEOUT;
    default: return YVEX_CLIENT_FAILURE_INTERNAL;
    }
}

static void session_partial_turn_set(
    server_session *session, const yvex_runtime_generation_result *result,
    int status)
{
    const yvex_runtime_partial_turn *runtime = &result->partial_turn;
    yvex_client_partial_turn *partial = &session->partial_turn;
    memset(partial, 0, sizeof(*partial));
    partial->schema_version = YVEX_CLIENT_PARTIAL_TURN_SCHEMA_V1;
    partial->available = 1;
    partial->committed_progress = result->final_position > result->initial_position ||
                                  result->model_committed_token_count ||
                                  result->generated_text_bytes;
    partial->reset_required = 1;
    partial->failure_status = status;
    partial->failure_class = turn_failure_class(status);
    partial->stop_reason = result->stop_reason;
    partial->initial_position = result->initial_position;
    partial->final_committed_position = result->final_position;
    partial->committed_token_count = result->model_committed_token_count;
    partial->published_text_bytes = result->generated_text_bytes;
    partial->target_state_generation = result->final_persistent_generation;
    partial->rng_generation = result->final_rng_generation;
    partial->token_ledger_generation = result->final_token_ledger_generation;
    partial->message_history_generation = session->message_history_generation;
    partial->transcript_generation = session->transcript_generation;
    yvex_runtime_identity_copy(partial->target_state_identity,
                               result->final_persistent_state_digest);
    yvex_runtime_identity_copy(partial->rng_state_identity,
                               result->final_rng_identity);
    yvex_runtime_identity_copy(partial->token_ledger_identity,
                               result->final_token_ledger_identity);
    yvex_runtime_identity_copy(partial->published_text_identity,
                               result->generated_text_digest);
    if (runtime->available) {
        partial->draft_state_generation_available =
            runtime->draft_state_generation_available;
        partial->detokenizer_generation_available =
            runtime->detokenizer_generation_available;
        partial->draft_state_generation = runtime->draft_state_generation;
        partial->detokenizer_generation = runtime->detokenizer_generation;
    }
}

static int provider_usage(const yvex_runtime_generation_result *result,
                          yvex_client_message *completed, yvex_error *err)
{
    completed->prompt_tokens = result->prompt_token_count;
    completed->reused_tokens = result->reusable_prefix_token_count;
    completed->prefill_tokens = result->new_prefill_token_count;
    completed->generated_tokens = result->model_committed_token_count;
    completed->completion_tokens = result->model_committed_token_count;
    if (!yvex_core_u64_add(completed->prompt_tokens,
                           completed->completion_tokens,
                           &completed->total_tokens)) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "server.provider.usage",
                       "provider usage count overflowed");
        return YVEX_ERR_BOUNDS;
    }
    return YVEX_OK;
}

static double elapsed_seconds(unsigned long long start,
                              unsigned long long finish)
{
    return finish >= start ? (double)(finish - start) / 1000000000.0 : 0.0;
}

static int session_name_valid(const char *name)
{
    size_t index, count;
    if (!name || !name[0]) return 0;
    count = strlen(name);
    if (count >= YVEX_SERVER_SESSION_NAME_CAP) return 0;
    for (index = 0u; index < count; ++index) {
        unsigned char byte = (unsigned char)name[index];
        if (!((byte >= 'a' && byte <= 'z') ||
              (byte >= 'A' && byte <= 'Z') ||
              (byte >= '0' && byte <= '9') || byte == '-' || byte == '_' ||
              byte == '.'))
            return 0;
    }
    return 1;
}

static int session_identity(server_session_registry *registry,
                            const char *name,
                            char output[YVEX_SHA256_HEX_CAP])
{
    const yvex_runtime_model_view *view = yvex_runtime_model_view_get(registry->model);
    yvex_runtime_model_summary model;
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_error err;
    if (!view || yvex_runtime_model_summary_copy(registry->model, &model, &err) != YVEX_OK)
        return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.server.session.v1") ||
        !yvex_sha256_update_text(&hash, model.runtime_model_identity) ||
        !yvex_sha256_update_text(&hash, name) ||
        !yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

static server_session *session_find_locked(server_session_registry *registry,
                                           const char *name)
{
    unsigned long long index;
    for (index = 0u; index < registry->capacity; ++index)
        if (registry->sessions[index].state != YVEX_SERVER_SESSION_CLOSED &&
            registry->sessions[index].name[0] &&
            strcmp(registry->sessions[index].name, name) == 0)
            return &registry->sessions[index];
    return NULL;
}

static int session_message_append(server_session *session, yvex_prompt_role role,
                                  const unsigned char *bytes,
                                  unsigned long long count, yvex_error *err)
{
    yvex_prompt_message *message;
    unsigned long long next;
    if (!session || (!bytes && count) ||
        session->message_count >= SESSION_MAX_MESSAGES ||
        !yvex_core_u64_add(session->transcript_count, count + 1u, &next) ||
        next > session->transcript_capacity) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "server.session.transcript",
                       "session transcript capacity is exhausted");
        return YVEX_ERR_BOUNDS;
    }
    message = &session->messages[session->message_count];
    if (count)
        memcpy(session->transcript + session->transcript_count, bytes,
               (size_t)count);
    session->transcript[session->transcript_count + count] = '\0';
    message->role = role;
    message->content = (const char *)session->transcript + session->transcript_count;
    message->content_len = count;
    session->transcript_count = next;
    session->message_count++;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int session_cancelled(void *opaque)
{
    server_session *session = opaque;
    return session && atomic_load_explicit(&session->cancel_requested,
                                           memory_order_acquire);
}

static int session_policy(const yvex_client_request *request,
                          yvex_runtime_sampling_policy *policy,
                          unsigned long long vocabulary_size, yvex_error *err)
{
    const yvex_provider_sampling *provider = request->provider_request
                                                 ? &request->provider_request->sampling
                                                 : NULL;
    int stochastic = provider ? provider->stochastic : request->stochastic;
    memset(policy, 0, sizeof(*policy));
    policy->schema_version = YVEX_RUNTIME_SAMPLING_SCHEMA_V1;
    policy->strategy = stochastic ? YVEX_SAMPLING_STRATEGY_STOCHASTIC
                                  : YVEX_SAMPLING_STRATEGY_GREEDY;
    policy->temperature = stochastic
                              ? (provider ? provider->temperature
                                          : request->temperature)
                              : 1.0;
    policy->top_k = stochastic ? (provider ? provider->top_k : request->top_k)
                               : 0u;
    policy->top_p = stochastic ? (provider ? provider->top_p : request->top_p)
                               : 1.0;
    policy->min_p = stochastic ? (provider ? provider->min_p : request->min_p)
                               : 0.0;
    policy->typical_p = stochastic
                            ? (provider ? provider->typical_p
                                        : request->typical_p)
                            : 1.0;
    policy->seed_present = stochastic
                               ? (provider ? provider->seed_present
                                           : request->seed_present)
                               : 0;
    policy->seed = stochastic ? (provider ? provider->seed : request->seed) : 0u;
    policy->rng_algorithm = YVEX_SAMPLING_RNG_PCG_XSH_RR_64_32;
    policy->rng_version = YVEX_SAMPLING_RNG_VERSION_V1;
    policy->filter_order_version = YVEX_SAMPLING_FILTER_ORDER_V2;
    return yvex_runtime_sampling_policy_seal(policy, vocabulary_size, err);
}

static int session_generation_open(server_session_registry *registry,
                                   server_session *session,
                                   const yvex_client_request *request,
                                   yvex_error *err)
{
    const yvex_runtime_model_view *view = yvex_runtime_model_view_get(registry->model);
    yvex_runtime_generation_options options;
    yvex_runtime_sampling_policy policy;
    int rc;
    if (!view || !view->tokenizer)
        return YVEX_ERR_STATE;
    {
        const yvex_tokenizer_plan_summary *tokenizer =
            yvex_tokenizer_plan_summary_get(view->tokenizer);
        if (request->reasoning_policy > YVEX_REASONING_MAXIMUM ||
            (request->provider_request &&
             request->reasoning_policy != YVEX_REASONING_DISABLED) ||
            (request->reasoning_policy != YVEX_REASONING_DISABLED &&
             (!tokenizer || !tokenizer->explicit_reasoning_supported)) ||
            (request->reasoning_policy == YVEX_REASONING_MAXIMUM &&
             !tokenizer->maximum_reasoning_supported)) {
            yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "server.session.reasoning",
                           "requested reasoning policy is unavailable");
            return YVEX_ERR_UNSUPPORTED;
        }
    }
    rc = session_policy(request, &policy,
                        yvex_tokenizer_vocab_size(view->tokenizer), err);
    if (rc != YVEX_OK) return rc;
    if (session->generation) {
        if (strcmp(policy.policy_identity, session->policy.policy_identity) != 0) {
            yvex_error_set(err, YVEX_ERR_STATE, "server.session.policy",
                           "sampling policy is immutable until session reset");
            return YVEX_ERR_STATE;
        }
        if (request->reasoning_policy != session->reasoning_policy) {
            yvex_error_set(err, YVEX_ERR_STATE, "server.session.reasoning",
                           "reasoning policy is immutable until session reset");
            return YVEX_ERR_STATE;
        }
        return YVEX_OK;
    }
    memset(&options, 0, sizeof(options));
    options.schema_version = YVEX_RUNTIME_GENERATION_SCHEMA_V4;
    options.backend = registry->options.backend;
    options.mode = registry->options.generation_mode ==
                           YVEX_SERVER_GENERATION_DSPARK
                       ? YVEX_GENERATION_MODE_DSPARK
                       : YVEX_GENERATION_MODE_TARGET_ONLY;
    options.context_capacity = registry->options.context_capacity;
    options.prefill_chunk_tokens = registry->options.prefill_chunk_tokens;
    options.maximum_new_tokens = registry->options.maximum_new_tokens;
    options.maximum_output_bytes = registry->options.maximum_output_bytes;
    options.maximum_host_bytes = registry->options.maximum_host_bytes;
    options.maximum_device_bytes = registry->options.maximum_device_bytes;
    options.trace_policy = registry->options.trace_level == YVEX_SERVER_TRACE_FULL
        ? YVEX_RUNTIME_TRACE_FULL : (registry->options.trace_level >= YVEX_SERVER_TRACE_STAGES
                                         ? YVEX_RUNTIME_TRACE_STAGES : YVEX_RUNTIME_TRACE_SUMMARY);
    options.evidence_profile = YVEX_EXECUTION_EVIDENCE_PRODUCTION;
    options.sampling_policy = policy;
    options.cancel_requested = session_cancelled;
    options.cancel_context = session;
    rc = yvex_runtime_generation_context_open(
        &session->generation, registry->model, session->execution,
        &options, err);
    if (rc == YVEX_OK) {
        session->policy = policy;
        session->policy_set = 1;
        session->reasoning_policy = request->reasoning_policy;
    }
    return rc;
}

static int session_execution_open(server_session_registry *registry,
                                  server_session *session, yvex_error *err)
{
    yvex_runtime_session_open_request request;
    yvex_runtime_model_failure failure;
    memset(&request, 0, sizeof(request));
    memset(&failure, 0, sizeof(failure));
    request.backend = registry->options.backend;
    request.maximum_host_bytes = registry->options.maximum_host_bytes;
    request.maximum_device_bytes = registry->options.maximum_device_bytes;
    return yvex_runtime_session_open(&session->execution, registry->model,
                                     &request, &failure, err);
}
/*
 * Initialize one registry slot and its unique runtime execution session.
 *
 * Closes all partial slot ownership and publishes no member.
 */
static int session_create_locked(server_session_registry *registry,
                                 const char *requested,
                                 server_session **created, yvex_error *err)
{
    server_session *session = NULL;
    char generated[YVEX_SERVER_SESSION_NAME_CAP];
    const char *name = requested;
    unsigned long long index;
    int rc;
    if (!name || !name[0]) {
        (void)snprintf(generated, sizeof(generated), "s%06llu", registry->next_id++);
        name = generated;
    }
    if (!session_name_valid(name) || session_find_locked(registry, name)) {
        yvex_error_set(err, YVEX_ERR_STATE, "server.session.create",
                       "session name is invalid or already exists");
        return YVEX_ERR_STATE;
    }
    for (index = 0u; index < registry->capacity; ++index)
        if (!registry->sessions[index].name[0] ||
            registry->sessions[index].state == YVEX_SERVER_SESSION_CLOSED) {
            session = &registry->sessions[index];
            break;
        }
    if (!session) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "server.session.create",
                       "session registry capacity is exhausted");
        return YVEX_ERR_BOUNDS;
    }
    memset(session, 0, sizeof(*session));
    yvex_core_text_copy(session->name, sizeof(session->name), name);
    session->state = YVEX_SERVER_SESSION_CREATED;
    session->token_capacity = registry->options.context_capacity;
    session->text_capacity = registry->options.maximum_output_bytes;
    session->transcript_capacity = SESSION_TRANSCRIPT_BYTES;
    session->committed_tokens = calloc((size_t)session->token_capacity,
                                       sizeof(*session->committed_tokens));
    session->prompt_tokens = calloc((size_t)session->token_capacity,
                                    sizeof(*session->prompt_tokens));
    session->token_results = calloc(
        (size_t)registry->options.maximum_new_tokens,
        sizeof(*session->token_results));
    session->turn_text = calloc((size_t)session->text_capacity + 1u, 1u);
    session->transcript = calloc((size_t)session->transcript_capacity, 1u);
    if (!session->committed_tokens || !session->prompt_tokens ||
        !session->token_results || !session->turn_text || !session->transcript ||
        !session_identity(registry, name, session->identity)) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "server.session.create",
                       "session storage allocation or identity failed");
        rc = YVEX_ERR_NOMEM;
        goto failure;
    }
    rc = session_execution_open(registry, session, err);
    if (rc != YVEX_OK) goto failure;
    atomic_init(&session->cancel_requested, 0);
    atomic_init(&session->active_turn, 0);
    session->message_history_generation = 1u;
    session->transcript_generation = 1u;
    session->state = YVEX_SERVER_SESSION_READY;
    registry->count++;
    yvex_server_telemetry_session(registry->telemetry, 1, 1);
    *created = session;
    return yvex_server_telemetry_emit(
        registry->telemetry, YVEX_SERVER_EVENT_SESSION_CREATED,
        YVEX_SERVER_SEVERITY_INFO, session->name, NULL, NULL, "session",
        0u, registry->count, 0u, 0.0, 0.0, err);
failure:
    free(session->transcript);
    free(session->turn_text);
    free(session->token_results);
    free(session->prompt_tokens);
    free(session->committed_tokens);
    memset(session, 0, sizeof(*session));
    return rc;
}
static int turn_classified_fragment(void *opaque,
                                    yvex_reasoning_segment segment,
                                    const unsigned char *bytes,
                                    unsigned long long byte_count,
                                    yvex_error *err)
{
    turn_sink *sink = opaque;
    unsigned long long offset = 0u;
    int rc = YVEX_OK;
    while (rc == YVEX_OK && offset < byte_count) {
        yvex_client_message message;
        unsigned long long count = byte_count - offset;
        if (count > YVEX_SERVER_FRAGMENT_CAP) count = YVEX_SERVER_FRAGMENT_CAP;
        memset(&message, 0, sizeof(message));
        message.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
        message.kind = YVEX_CLIENT_MESSAGE_FRAGMENT;
        message.status = YVEX_OK;
        message.generation_phase = YVEX_CLIENT_PHASE_DECODE;
        message.stream_channel = segment == YVEX_REASONING_SEGMENT_EXPLICIT
                                     ? YVEX_CLIENT_STREAM_EXPLICIT_REASONING
                                     : YVEX_CLIENT_STREAM_FINAL_TEXT;
        message.request_number = sink->request->request_number;
        yvex_core_text_copy(message.session_name, sizeof(message.session_name),
                            sink->session->name);
        message.byte_count = count;
        memcpy(message.bytes, bytes + offset, (size_t)count);
        rc = sink->emit(sink->emit_context, &message, err);
        offset += count;
    }
    return rc;
}

/* Emit one generation fragment only after its internal model and text commit. */
static int turn_fragment(void *opaque,
                         const yvex_runtime_generation_token_result *token,
                         const unsigned char *bytes,
                         unsigned long long byte_count, yvex_error *err)
{
    turn_sink *sink = opaque;
    int rc = YVEX_OK;
    if (!sink->first_fragment_ns) {
        sink->first_fragment_ns = monotonic_ns();
        rc = yvex_server_telemetry_emit_provider(
            sink->registry->telemetry, YVEX_SERVER_EVENT_GENERATION_FIRST_TOKEN,
            YVEX_SERVER_SEVERITY_INFO, sink->session->name,
            sink->request_id, sink->turn_id, "decode", token->ordinal,
            token->sampled_token_id, 0u,
            elapsed_seconds(sink->started_ns, sink->first_fragment_ns), 0.0,
            NULL, sink->request->provider_request, NULL, err);
    }
    if (sink->request->provider_request) {
        if (rc == YVEX_OK && provider_text_stream_direct(
                                 sink->request->provider_request))
            rc = provider_output_emit(
                sink, YVEX_PROVIDER_OUTPUT_ASSISTANT_TEXT, bytes,
                byte_count, NULL, err);
        if (rc == YVEX_OK)
            rc = yvex_server_telemetry_emit_provider(
                sink->registry->telemetry,
                YVEX_SERVER_EVENT_GENERATION_FRAGMENT,
                YVEX_SERVER_SEVERITY_DEBUG, sink->session->name,
                sink->request_id, sink->turn_id, "decode", token->ordinal,
                byte_count, token->sampled_token_id, 0.0, 0.0,
                NULL, sink->request->provider_request, NULL, err);
        return rc;
    }
    if (rc == YVEX_OK)
        rc = yvex_tokenizer_reasoning_stream_push(
            sink->reasoning_stream, bytes, byte_count, err);
    if (rc == YVEX_OK)
        rc = yvex_server_telemetry_emit_provider(
            sink->registry->telemetry, YVEX_SERVER_EVENT_GENERATION_FRAGMENT,
            YVEX_SERVER_SEVERITY_DEBUG, sink->session->name,
            sink->request_id, sink->turn_id, "decode", token->ordinal,
            byte_count, token->sampled_token_id, 0.0, 0.0,
            NULL, sink->request->provider_request, NULL, err);
    return rc;
}

static int turn_progress(void *opaque,
                         yvex_runtime_generation_progress_kind kind,
                         unsigned long long value_a,
                         unsigned long long value_b, yvex_error *err)
{
    turn_sink *sink = opaque;
    yvex_client_message message;
    yvex_server_event event;
    yvex_server_event_kind event_kind;
    const char *phase;
    int rc;
    if (kind == YVEX_GENERATION_PROGRESS_PROMPT_ACCEPTED) {
        event_kind = YVEX_SERVER_EVENT_TOKENIZER_COMPLETED;
        phase = "tokenizer";
    } else if (kind == YVEX_GENERATION_PROGRESS_PREFILL_STARTED) {
        sink->prefill_started_ns = monotonic_ns();
        event_kind = YVEX_SERVER_EVENT_PREFILL_STARTED;
        phase = "prefill";
    } else if (kind == YVEX_GENERATION_PROGRESS_PREFILL_PROGRESS) {
        event_kind = YVEX_SERVER_EVENT_PREFILL_PROGRESS;
        phase = "prefill";
    } else {
        sink->prefill_completed_ns = monotonic_ns();
        event_kind = YVEX_SERVER_EVENT_PREFILL_COMPLETED;
        phase = "prefill";
    }
    {
        unsigned long long now = kind == YVEX_GENERATION_PROGRESS_PREFILL_PROGRESS
                                     ? monotonic_ns() : sink->prefill_completed_ns;
        double elapsed = kind >= YVEX_GENERATION_PROGRESS_PREFILL_PROGRESS
                             ? elapsed_seconds(sink->prefill_started_ns, now) : 0.0;
        rc = yvex_server_telemetry_emit_provider(
            sink->registry->telemetry, event_kind,
            YVEX_SERVER_SEVERITY_INFO, sink->session->name,
            sink->request_id, sink->turn_id, phase, value_a, value_b, 0u,
            elapsed, elapsed > 0.0 ? (double)value_a / elapsed : 0.0,
            NULL, sink->request->provider_request, &event, err);
        if (rc != YVEX_OK || sink->request->provider_request) return rc;
        memset(&message, 0, sizeof(message));
        message.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
        message.kind = YVEX_CLIENT_MESSAGE_EVENT;
        message.status = YVEX_OK;
        message.request_number = sink->request->request_number;
        message.stream_channel = YVEX_CLIENT_STREAM_CONTROL_EVENT;
        message.event = event;
        return sink->emit(sink->emit_context, &message, err);
    }
}

static int turn_speculation_progress(
    void *opaque, const yvex_runtime_speculation_progress *progress,
    yvex_error *err)
{
    static const yvex_server_event_kind kinds[] = {
        YVEX_SERVER_EVENT_DRAFT_STARTED,
        YVEX_SERVER_EVENT_DRAFT_COMPLETED,
        YVEX_SERVER_EVENT_VERIFICATION_STARTED,
        YVEX_SERVER_EVENT_VERIFICATION_COMPLETED,
        YVEX_SERVER_EVENT_PREFIX_ACCEPTED,
        YVEX_SERVER_EVENT_CANDIDATE_REJECTED,
        YVEX_SERVER_EVENT_SPECULATIVE_CYCLE_COMMITTED};
    turn_sink *sink = opaque;
    yvex_client_message message;
    yvex_server_event event;
    int rc;
    if (!progress || progress->kind >
                         YVEX_SPECULATION_PROGRESS_CYCLE_COMMITTED) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "server.turn.speculation",
                       "typed speculation progress is required");
        return YVEX_ERR_INVALID_ARG;
    }
    rc = yvex_server_telemetry_emit_provider(
        sink->registry->telemetry, kinds[progress->kind],
        YVEX_SERVER_SEVERITY_INFO, sink->session->name, sink->request_id,
        sink->turn_id, "speculation", 0u, 0u, 0u, progress->seconds, 0.0,
        progress, sink->request->provider_request, &event, err);
    if (rc != YVEX_OK || sink->request->provider_request) return rc;
    memset(&message, 0, sizeof(message));
    message.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    message.kind = YVEX_CLIENT_MESSAGE_EVENT;
    message.status = YVEX_OK;
    message.request_number = sink->request->request_number;
    message.stream_channel = YVEX_CLIENT_STREAM_CONTROL_EVENT;
    message.event = event;
    return sink->emit(sink->emit_context, &message, err);
}

static int session_message(server_message_emit emit, void *emit_context,
                           yvex_client_message_kind kind, int status,
                           const yvex_client_request *request,
                           const server_session *session, const char *reason,
                           yvex_error *err)
{
    yvex_client_message message;
    memset(&message, 0, sizeof(message));
    message.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    message.kind = kind;
    message.status = status;
    message.request_number = request->request_number;
    if (session) {
        yvex_core_text_copy(message.session_name, sizeof(message.session_name),
                            session->name);
        message.session_state = session->state;
        message.final_position = session->committed_count;
        message.turn_count = session->turn_count;
        message.context_used = session->committed_count;
        yvex_runtime_identity_copy(message.session_identity,
                                   session->identity);
        yvex_runtime_identity_copy(message.turn_identity,
                                   session->last_turn_identity);
        yvex_runtime_identity_copy(message.state_digest,
                                   session->state_digest);
        yvex_runtime_identity_copy(message.generated_token_identity,
                                   session->generated_token_identity);
        yvex_runtime_identity_copy(message.generated_text_digest,
                                   session->generated_text_digest);
        message.partial_turn = session->partial_turn;
    }
    yvex_core_text_copy(message.reason, sizeof(message.reason),
                        reason ? reason : "");
    return emit(emit_context, &message, err);
}

static int session_turn_commit(server_session *session,
                               const yvex_client_request *request,
                               const yvex_runtime_generation_result *result,
                               unsigned long long prior_messages,
                               unsigned long long prior_transcript,
                               int status, yvex_error *err)
{
    unsigned long long index;
    unsigned long long next_message_generation, next_transcript_generation;
    unsigned long long committed =
        result->final_position >= result->model_committed_token_count
            ? result->final_position - result->model_committed_token_count
            : 0u;
    if (committed > result->prompt_token_count ||
        committed > session->token_capacity ||
        result->model_committed_token_count >
            session->token_capacity - committed) {
        if (status == YVEX_OK) {
            yvex_error_set(err, YVEX_ERR_BOUNDS, "server.session.turn",
                           "committed token ledger exceeds session capacity");
            status = YVEX_ERR_BOUNDS;
        }
        return status;
    }
    for (index = 0u; index < result->sampled_token_count; ++index)
        if (session->token_results[index].model_committed)
            session->prompt_tokens[committed++] =
                session->token_results[index].sampled_token_id;
    if (result->final_position > result->initial_position ||
        result->model_committed_token_count) {
        memcpy(session->committed_tokens, session->prompt_tokens,
               (size_t)committed * sizeof(*session->committed_tokens));
        session->committed_count = committed;
    }
    if (status == YVEX_OK && result->completed && !request->provider_request) {
        yvex_error transcript_error;
        int transcript_rc;
        yvex_error_clear(&transcript_error);
        if (!yvex_core_u64_add(session->message_history_generation, 1u,
                               &next_message_generation) ||
            !yvex_core_u64_add(session->transcript_generation, 1u,
                               &next_transcript_generation)) {
            yvex_error_set(&transcript_error, YVEX_ERR_BOUNDS,
                           "server.session.transcript",
                           "session transcript generation overflowed");
            transcript_rc = YVEX_ERR_BOUNDS;
        } else {
            transcript_rc = session_message_append(
                session, YVEX_PROMPT_ROLE_USER, request->prompt,
                request->prompt_bytes, &transcript_error);
        }
        if (transcript_rc == YVEX_OK)
            transcript_rc = session_message_append(
                session, YVEX_PROMPT_ROLE_ASSISTANT, session->turn_text,
                result->generated_text_bytes, &transcript_error);
        if (transcript_rc != YVEX_OK) {
            session->message_count = prior_messages;
            session->transcript_count = prior_transcript;
            status = transcript_rc;
            if (err) *err = transcript_error;
        } else {
            session->message_history_generation = next_message_generation;
            session->transcript_generation = next_transcript_generation;
        }
    }
    if (status == YVEX_OK && result->completed) {
        session->turn_count++;
        session->state = session->attached_clients
                             ? YVEX_SERVER_SESSION_READY
                             : YVEX_SERVER_SESSION_DETACHED;
        memset(&session->partial_turn, 0, sizeof(session->partial_turn));
    } else if (status == YVEX_ERR_CANCELLED || result->partial ||
               result->sampled_token_count ||
               result->final_position > result->initial_position) {
        /* A cancelled generation context deliberately refuses continuation,
         * even when its persistent-state candidate never became visible. Keep
         * the server state aligned so the next turn cannot bypass reset. */
        session->state = YVEX_SERVER_SESSION_PARTIAL;
        session_partial_turn_set(session, result, status);
    } else {
        session->state = YVEX_SERVER_SESSION_FAILED;
    }
    return status;
}

static unsigned long long provider_visible_bytes(
    const yvex_provider_request *request, const unsigned char *bytes,
    unsigned long long count, int *matched)
{
    unsigned long long stop, offset, visible = count;
    *matched = 0;
    for (stop = 0u; stop < request->stop_count; ++stop) {
        yvex_provider_span pattern = request->stop_strings[stop];
        if (pattern.count > count) continue;
        for (offset = 0u; offset <= count - pattern.count; ++offset)
            if (memcmp(bytes + offset, pattern.bytes,
                       (size_t)pattern.count) == 0) {
                if (offset < visible) visible = offset;
                *matched = 1;
                break;
            }
    }
    return visible;
}

static int provider_output_emit(turn_sink *sink,
                                yvex_provider_output_kind kind,
                                const unsigned char *bytes,
                                unsigned long long count,
                                const yvex_provider_tool_call *call,
                                yvex_error *err)
{
    unsigned long long offset = 0u;
    int rc = YVEX_OK;
    while (rc == YVEX_OK && (offset < count || (!count && !offset))) {
        yvex_client_message message;
        unsigned long long extent = count - offset;
        if (extent > YVEX_SERVER_FRAGMENT_CAP) extent = YVEX_SERVER_FRAGMENT_CAP;
        memset(&message, 0, sizeof(message));
        message.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
        message.kind = YVEX_CLIENT_MESSAGE_FRAGMENT;
        message.status = YVEX_OK;
        message.request_number = sink->request->request_number;
        message.provider_output_kind = kind;
        message.generation_phase = YVEX_CLIENT_PHASE_DECODE;
        if (kind == YVEX_PROVIDER_OUTPUT_FUNCTION_CALL)
            message.stream_channel = YVEX_CLIENT_STREAM_TOOL_CALL;
        else
            message.stream_channel = YVEX_CLIENT_STREAM_FINAL_TEXT;
        yvex_core_text_copy(message.session_name, sizeof(message.session_name),
                            sink->session->name);
        if (sink->request->provider_request) {
            yvex_runtime_identity_copy(
                message.provider_request_identity,
                sink->request->provider_request->request_identity);
            yvex_core_text_copy(
                message.external_correlation_id,
                sizeof(message.external_correlation_id),
                sink->request->provider_request->external_correlation_id);
        }
        if (call) {
            yvex_core_text_copy(message.tool_call_id,
                                sizeof(message.tool_call_id), call->call_id);
            yvex_core_text_copy(message.tool_name,
                                sizeof(message.tool_name), call->name);
        }
        message.byte_count = extent;
        if (extent) memcpy(message.bytes, bytes + offset, (size_t)extent);
        rc = sink->emit(sink->emit_context, &message, err);
        offset += extent;
        if (!count) offset = 1u;
    }
    return rc;
}

static void session_speculation_result_project(
    yvex_client_message *message,
    const yvex_runtime_generation_result *result)
{
    message->generation_mode =
        result->execution_mode == YVEX_GENERATION_MODE_DSPARK
            ? YVEX_SERVER_GENERATION_DSPARK
            : YVEX_SERVER_GENERATION_TARGET_ONLY;
    message->draft_cycle_count = result->draft_cycle_count;
    message->draft_forward_count = result->draft_forward_count;
    message->proposed_tokens = result->proposed_token_count;
    message->selected_verification_tokens =
        result->selected_verification_token_count;
    message->target_verification_count = result->target_verification_count;
    message->accepted_draft_tokens = result->accepted_draft_token_count;
    message->rejected_draft_tokens = result->rejected_draft_token_count;
    message->discarded_draft_tokens = result->discarded_draft_token_count;
    message->target_correction_or_bonus_tokens =
        result->target_correction_or_bonus_token_count;
    message->maximum_accepted_prefix = result->maximum_accepted_prefix;
    message->confidence_logit_count = result->confidence_logit_count;
    message->draft_seconds = (double)result->draft_ns / 1000000000.0;
    message->verification_seconds =
        (double)result->verification_ns / 1000000000.0;
    message->speculative_commit_seconds =
        (double)result->speculative_commit_ns / 1000000000.0;
    message->mean_accepted_prefix = result->mean_accepted_prefix;
    message->effective_committed_rate =
        result->effective_committed_tokens_per_second;
    message->confidence_logit_minimum = result->confidence_logit_minimum;
    message->confidence_logit_maximum = result->confidence_logit_maximum;
    message->confidence_logit_mean = result->confidence_logit_mean;
    message->stream_channel = YVEX_CLIENT_STREAM_CONTROL_EVENT;
    yvex_runtime_identity_copy(message->speculation_policy_identity,
                               result->speculation_policy_identity);
}

static int session_provider_result_prepare(
    server_session_registry *registry, server_session *session,
    const yvex_client_request *request, turn_sink *sink,
    const yvex_runtime_generation_result *result,
    yvex_tokenizer_provider_result *provider_result, int *stop_matched,
    yvex_error *err)
{
    const yvex_runtime_model_view *view = yvex_runtime_model_view_get(registry->model);
    unsigned long long visible_count;
    int status;
    if (!request->provider_request) return YVEX_OK;
    visible_count = provider_visible_bytes(
        request->provider_request, session->turn_text,
        result->generated_text_bytes, stop_matched);
    if (!view || !view->tokenizer) return YVEX_ERR_STATE;
    status = yvex_tokenizer_parse_provider_completion(
        view->tokenizer, request->provider_request, session->turn_text,
        visible_count, provider_result, err);
    if (status == YVEX_OK &&
        request->provider_request->response_format ==
            YVEX_PROVIDER_RESPONSE_JSON_OBJECT &&
        provider_result->kind == YVEX_PROVIDER_OUTPUT_ASSISTANT_TEXT)
        status = yvex_provider_json_value_validate(
            provider_result->content, provider_result->content_count, 0, err);
    if (status == YVEX_OK &&
        (request->provider_request->tool_choice.kind ==
             YVEX_PROVIDER_TOOL_CHOICE_REQUIRED ||
         request->provider_request->tool_choice.kind ==
             YVEX_PROVIDER_TOOL_CHOICE_FUNCTION) &&
        provider_result->kind != YVEX_PROVIDER_OUTPUT_FUNCTION_CALL) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "server.provider.tool",
                       "required function tool call was not produced");
        status = YVEX_ERR_FORMAT;
    }
    if (status == YVEX_OK &&
        request->provider_request->tool_choice.kind ==
            YVEX_PROVIDER_TOOL_CHOICE_NONE &&
        provider_result->kind == YVEX_PROVIDER_OUTPUT_FUNCTION_CALL) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "server.provider.tool",
                       "function tool call was produced while tools are disabled");
        status = YVEX_ERR_FORMAT;
    }
    if (status == YVEX_OK &&
        request->provider_request->tool_choice.kind ==
            YVEX_PROVIDER_TOOL_CHOICE_FUNCTION &&
        strcmp(provider_result->tool_call.name,
               request->provider_request->tool_choice.function_name) != 0) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "server.provider.tool",
                       "model selected a function other than the required function");
        status = YVEX_ERR_FORMAT;
    }
    if (status == YVEX_OK && provider_result->content_count &&
        !provider_text_stream_direct(request->provider_request))
        status = provider_output_emit(
            sink, YVEX_PROVIDER_OUTPUT_ASSISTANT_TEXT, provider_result->content,
            provider_result->content_count, NULL, err);
    if (status == YVEX_OK &&
        provider_result->kind == YVEX_PROVIDER_OUTPUT_FUNCTION_CALL)
        status = provider_output_emit(
            sink, YVEX_PROVIDER_OUTPUT_FUNCTION_CALL,
            provider_result->tool_call.arguments_json.bytes,
            provider_result->tool_call.arguments_json.count,
            &provider_result->tool_call, err);
    return status;
}

static int session_turn_publish(server_session_registry *registry, server_session *session,
                                const yvex_client_request *request, turn_sink *sink,
                                const yvex_runtime_generation_result *result, int status,
                                yvex_error *err)
{
    yvex_client_message completed;
    yvex_tokenizer_provider_result provider_result;
    yvex_runtime_session_summary runtime_summary;
    unsigned long long now = monotonic_ns();
    yvex_error secondary;
    int stop_matched = 0, send_rc;
    memset(&provider_result, 0, sizeof(provider_result));
    if (status == YVEX_OK) {
        status = session_provider_result_prepare(
            registry, session, request, sink, result, &provider_result,
            &stop_matched, err);
        if (status != YVEX_OK && request->provider_request)
            session->state = YVEX_SERVER_SESSION_PARTIAL;
    }
    if (status != YVEX_OK && session->state == YVEX_SERVER_SESSION_PARTIAL)
        session_partial_turn_set(session, result, status);
    memset(&completed, 0, sizeof(completed));
    completed.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    completed.kind = status == YVEX_OK ? YVEX_CLIENT_MESSAGE_TURN_COMPLETE
                                       : YVEX_CLIENT_MESSAGE_ERROR;
    completed.status = status;
    completed.request_number = request->request_number;
    completed.session_state = session->state;
    completed.partial_turn = session->partial_turn;
    if (provider_usage(result, &completed, err) != YVEX_OK)
        status = YVEX_ERR_BOUNDS;
    if (status != YVEX_OK)
        completed.failure_class = turn_failure_class(status);
    completed.final_position = result->final_position;
    completed.context_used = result->final_position;
    completed.turn_count = session->turn_count;
    completed.stop_reason = result->stop_reason;
    session_speculation_result_project(&completed, result);
    completed.generation_phase = status == YVEX_OK
                                     ? YVEX_CLIENT_PHASE_COMPLETE
                                     : status == YVEX_ERR_CANCELLED
                                           ? YVEX_CLIENT_PHASE_CANCELLED
                                           : YVEX_CLIENT_PHASE_FAILED;
    completed.cancellation_class =
        status == YVEX_ERR_CANCELLED ? YVEX_CLIENT_CANCELLATION_COMPLETED
                                     : YVEX_CLIENT_CANCELLATION_NONE;
    if (request->provider_request) {
        completed.provider_output_kind = YVEX_PROVIDER_OUTPUT_TERMINAL;
        if (status != YVEX_OK)
            completed.provider_finish = YVEX_PROVIDER_FINISH_FAILED;
        else if (provider_result.kind == YVEX_PROVIDER_OUTPUT_FUNCTION_CALL)
            completed.provider_finish = YVEX_PROVIDER_FINISH_TOOL_CALLS;
        else if (result->stop_reason == YVEX_GENERATION_STOP_MAX_NEW_TOKENS)
            completed.provider_finish = YVEX_PROVIDER_FINISH_LENGTH;
        else if (result->stop_reason == YVEX_GENERATION_STOP_CANCELLED)
            completed.provider_finish = YVEX_PROVIDER_FINISH_CANCELLED;
        else
            completed.provider_finish = YVEX_PROVIDER_FINISH_STOP;
        if (stop_matched) completed.provider_finish = YVEX_PROVIDER_FINISH_STOP;
        yvex_runtime_identity_copy(
            completed.provider_request_identity,
            request->provider_request->request_identity);
        yvex_core_text_copy(
            completed.external_correlation_id,
            sizeof(completed.external_correlation_id),
            request->provider_request->external_correlation_id);
        if (provider_result.kind == YVEX_PROVIDER_OUTPUT_FUNCTION_CALL) {
            yvex_core_text_copy(completed.tool_call_id,
                                sizeof(completed.tool_call_id),
                                provider_result.tool_call.call_id);
            yvex_core_text_copy(completed.tool_name,
                                sizeof(completed.tool_name),
                                provider_result.tool_call.name);
        }
    }
    completed.queue_seconds = sink->queue_seconds;
    completed.prefill_seconds = elapsed_seconds(
        sink->prefill_started_ns, sink->prefill_completed_ns);
    completed.first_token_seconds = sink->first_fragment_ns
                                        ? elapsed_seconds(sink->started_ns,
                                                          sink->first_fragment_ns)
                                        : 0.0;
    completed.decode_seconds = sink->prefill_completed_ns
                                   ? elapsed_seconds(sink->prefill_completed_ns,
                                                     now)
                                   : 0.0;
    completed.prefill_rate = completed.prefill_seconds > 0.0
                                 ? (double)completed.prefill_tokens /
                                       completed.prefill_seconds
                                 : 0.0;
    completed.decode_rate = completed.decode_seconds > 0.0
                                ? (double)completed.generated_tokens /
                                      completed.decode_seconds
                                : 0.0;
    yvex_core_text_copy(completed.session_name, sizeof(completed.session_name),
                        session->name);
    yvex_runtime_identity_copy(completed.session_identity, session->identity);
    yvex_runtime_identity_copy(completed.turn_identity,
                               result->generation_execution_identity);
    yvex_runtime_identity_copy(completed.state_digest,
                               result->final_persistent_state_digest);
    yvex_runtime_identity_copy(completed.generated_token_identity,
                               result->generated_token_identity);
    yvex_runtime_identity_copy(completed.generated_text_digest,
                               result->generated_text_digest);
    if (yvex_sha256_hex_valid(result->generation_execution_identity))
        yvex_runtime_identity_copy(session->last_turn_identity,
                                   result->generation_execution_identity);
    if (yvex_sha256_hex_valid(result->final_persistent_state_digest))
        yvex_runtime_identity_copy(session->state_digest,
                                   result->final_persistent_state_digest);
    if (yvex_sha256_hex_valid(result->generated_token_identity))
        yvex_runtime_identity_copy(session->generated_token_identity,
                                   result->generated_token_identity);
    if (yvex_sha256_hex_valid(result->generated_text_digest))
        yvex_runtime_identity_copy(session->generated_text_digest,
                                   result->generated_text_digest);
    if (status != YVEX_OK)
        yvex_core_text_copy(completed.reason, sizeof(completed.reason),
                            yvex_error_message(err));
    memset(&runtime_summary, 0, sizeof(runtime_summary));
    yvex_error_clear(&secondary);
    if (yvex_runtime_session_summary_copy(session->execution, &runtime_summary,
                                          &secondary) == YVEX_OK)
        yvex_server_telemetry_resources(
            registry->telemetry, runtime_summary.peak_host_bytes,
            runtime_summary.peak_device_bytes, runtime_summary.upload_count);
    yvex_error_clear(&secondary);
    send_rc = sink->emit(sink->emit_context, &completed, &secondary);
    if (status == YVEX_OK && send_rc != YVEX_OK) {
        status = send_rc;
        if (err) *err = secondary;
        session->state = YVEX_SERVER_SESSION_PARTIAL;
        session_partial_turn_set(session, result, status);
    }
    (void)yvex_server_telemetry_emit_provider(
        registry->telemetry,
        status == YVEX_OK
            ? YVEX_SERVER_EVENT_GENERATION_COMPLETED
            : (status == YVEX_ERR_CANCELLED
                   ? YVEX_SERVER_EVENT_GENERATION_CANCELLED
                   : YVEX_SERVER_EVENT_GENERATION_FAILED),
        status == YVEX_OK ? YVEX_SERVER_SEVERITY_INFO
                          : YVEX_SERVER_SEVERITY_ERROR,
        session->name, sink->request_id, sink->turn_id, "turn",
        result->model_committed_token_count, result->final_position,
        result->stop_reason, elapsed_seconds(sink->started_ns, now),
        completed.decode_rate, NULL, request->provider_request, NULL,
        &secondary);
    yvex_tokenizer_provider_result_clear(&provider_result);
    return status;
}

static int session_profile_publish(server_session_registry *registry,
                                   const server_session *session,
                                   const yvex_client_request *request,
                                   const turn_sink *sink,
                                   const yvex_runtime_generation_result *result,
                                   yvex_error *err)
{
    const yvex_runtime_profile_record *profile = result ? &result->profile : NULL;
    int rc;
#define PROFILE_EVENT(phase_, a_, b_, c_, nanoseconds_)                                   \
    yvex_server_telemetry_emit_provider(                                                   \
        registry->telemetry, YVEX_SERVER_EVENT_GENERATION_PROFILE,                        \
        YVEX_SERVER_SEVERITY_DEBUG, session->name, sink->request_id, sink->turn_id,       \
        (phase_), (a_), (b_), (c_), (double)(nanoseconds_) / 1000000000.0, 0.0,           \
        NULL, request->provider_request, NULL, err)
    if (!profile || profile->mode == YVEX_RUNTIME_PROFILE_OFF) return YVEX_OK;
    rc = PROFILE_EVENT("movement",
        profile->counters[YVEX_RUNTIME_PROFILE_H2D_BYTES],
        profile->counters[YVEX_RUNTIME_PROFILE_D2H_BYTES],
        profile->counters[YVEX_RUNTIME_PROFILE_D2D_BYTES], 0ull);
    if (rc == YVEX_OK)
        rc = PROFILE_EVENT("transfers",
            profile->counters[YVEX_RUNTIME_PROFILE_UPLOADS],
            profile->counters[YVEX_RUNTIME_PROFILE_DOWNLOADS],
            profile->counters[YVEX_RUNTIME_PROFILE_EXPERT_SUBVIEWS], 0ull);
    if (rc == YVEX_OK)
        rc = PROFILE_EVENT("launches",
            profile->counters[YVEX_RUNTIME_PROFILE_KERNEL_LAUNCHES],
            profile->counters[YVEX_RUNTIME_PROFILE_STREAM_SYNCHRONIZATIONS],
            profile->counters[YVEX_RUNTIME_PROFILE_DEVICE_SYNCHRONIZATIONS],
            profile->phase_ns[YVEX_RUNTIME_PROFILE_SYNCHRONIZATION_WAIT]);
    if (rc == YVEX_OK)
        rc = PROFILE_EVENT("attention",
            profile->phase_calls[YVEX_RUNTIME_PROFILE_ATTENTION],
            profile->counters[YVEX_RUNTIME_PROFILE_CACHE_HITS],
            profile->counters[YVEX_RUNTIME_PROFILE_CACHE_MISSES],
            profile->phase_ns[YVEX_RUNTIME_PROFILE_ATTENTION]);
    if (rc == YVEX_OK)
        rc = PROFILE_EVENT("moe",
            profile->counters[YVEX_RUNTIME_PROFILE_ROW_EXPERT_PAIRS],
            profile->counters[YVEX_RUNTIME_PROFILE_EXPERT_SUBVIEWS],
            profile->counters[YVEX_RUNTIME_PROFILE_EXPERT_BYTES],
            profile->phase_ns[YVEX_RUNTIME_PROFILE_MOE_TOTAL]);
    if (rc == YVEX_OK)
        rc = PROFILE_EVENT("output",
            profile->counters[YVEX_RUNTIME_PROFILE_OUTPUT_HEAD_ROWS],
            profile->counters[YVEX_RUNTIME_PROFILE_LOGITS_D2H_BYTES],
            profile->counters[YVEX_RUNTIME_PROFILE_GENERATED_TOKENS],
            profile->phase_ns[YVEX_RUNTIME_PROFILE_OUTPUT_HEAD] +
                profile->phase_ns[YVEX_RUNTIME_PROFILE_SAMPLING]);
    if (rc == YVEX_OK)
        rc = PROFILE_EVENT("target",
            profile->counters[YVEX_RUNTIME_PROFILE_TARGET_FORWARDS],
            profile->counters[YVEX_RUNTIME_PROFILE_TARGET_ROWS],
            profile->counters[YVEX_RUNTIME_PROFILE_REPLAYED_ACCEPTED_TARGET_ROWS],
            profile->phase_ns[YVEX_RUNTIME_PROFILE_TOTAL_GENERATION]);
    if (rc == YVEX_OK)
        rc = PROFILE_EVENT("speculation",
            profile->counters[YVEX_RUNTIME_PROFILE_DRAFT_FORWARDS],
            profile->counters[YVEX_RUNTIME_PROFILE_VERIFIED_ROWS],
            profile->counters[YVEX_RUNTIME_PROFILE_PROMOTED_TARGET_ROWS], 0ull);
    if (rc == YVEX_OK)
        rc = PROFILE_EVENT("candidate",
            profile->counters[YVEX_RUNTIME_PROFILE_ACCEPTED_DRAFT_TOKENS],
            profile->counters[YVEX_RUNTIME_PROFILE_DISCARDED_CANDIDATE_ROWS],
            profile->counters[YVEX_RUNTIME_PROFILE_TARGET_EXTENSIONS], 0ull);
    if (rc == YVEX_OK)
        rc = PROFILE_EVENT("shape",
            profile->counters[YVEX_RUNTIME_PROFILE_SHAPE_REGISTRY_HITS],
            profile->counters[YVEX_RUNTIME_PROFILE_SHAPE_REGISTRY_MISSES],
            profile->counters[YVEX_RUNTIME_PROFILE_FULL_ARRAY_HOST_SCAN_BYTES], 0ull);
    if (rc == YVEX_OK)
        rc = PROFILE_EVENT("prefill",
            profile->counters[YVEX_RUNTIME_PROFILE_PROMPT_TOKENS],
            profile->counters[YVEX_RUNTIME_PROFILE_REUSED_TOKENS],
            profile->counters[YVEX_RUNTIME_PROFILE_NEW_PREFILL_TOKENS],
            profile->phase_ns[YVEX_RUNTIME_PROFILE_TOTAL_PREFILL]);
    if (rc == YVEX_OK)
        rc = PROFILE_EVENT("decode",
            profile->phase_calls[YVEX_RUNTIME_PROFILE_FIRST_DECODE],
            profile->phase_calls[YVEX_RUNTIME_PROFILE_SUBSEQUENT_DECODE],
            profile->counters[YVEX_RUNTIME_PROFILE_GENERATED_TOKENS],
            profile->phase_ns[YVEX_RUNTIME_PROFILE_FIRST_DECODE] +
                profile->phase_ns[YVEX_RUNTIME_PROFILE_SUBSEQUENT_DECODE]);
#undef PROFILE_EVENT
    return rc;
}

static int session_turn(server_session_registry *registry,
                        server_session *session,
                        const yvex_client_request *request,
                        const char *request_id,
                        double queue_seconds,
                        server_message_emit emit, void *emit_context,
                        yvex_error *err)
{
    const yvex_runtime_model_view *model_view =
        yvex_runtime_model_view_get(registry->model);
    yvex_prompt_message prompt_messages[SESSION_MAX_MESSAGES + 1u];
    yvex_runtime_generation_request prompt;
    yvex_runtime_generation_turn_request turn;
    yvex_runtime_generation_result result;
    yvex_client_message started;
    turn_sink sink;
    unsigned long long prior_messages = session->message_count;
    unsigned long long prior_transcript = session->transcript_count;
    yvex_error primary_error;
    int generation_rc;
    int rc;
    {
        unsigned long long maximum = request->provider_request
                                         ? request->provider_request->maximum_output_tokens
                                         : request->maximum_new_tokens;
        int provider_valid = request->provider_request &&
            yvex_provider_request_validate(request->provider_request, err) == YVEX_OK;
        int native_valid = request->prompt && request->prompt_bytes &&
            request->prompt_bytes < SESSION_TRANSCRIPT_BYTES;
        if ((!provider_valid && !native_valid) || !maximum ||
            maximum > registry->options.maximum_new_tokens) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "server.session.turn",
                       "nonempty bounded prompt and token limit are required");
        return YVEX_ERR_INVALID_ARG;
        }
    }
    if (session->state != YVEX_SERVER_SESSION_READY &&
        session->state != YVEX_SERVER_SESSION_DETACHED) {
        yvex_error_set(err, YVEX_ERR_STATE, "server.session.turn",
                       "session requires READY or DETACHED state for a new turn");
        return YVEX_ERR_STATE;
    }
    rc = session_generation_open(registry, session, request, err);
    if (rc != YVEX_OK) return rc;
    memset(&prompt, 0, sizeof(prompt));
    prompt.schema_version = YVEX_RUNTIME_GENERATION_SCHEMA_V3;
    if (request->provider_request) {
        prompt.kind = YVEX_GENERATION_INPUT_PROVIDER;
        prompt.provider_request = request->provider_request;
    } else {
        memcpy(prompt_messages, session->messages,
               (size_t)session->message_count * sizeof(*prompt_messages));
        prompt_messages[session->message_count].role = YVEX_PROMPT_ROLE_USER;
        prompt_messages[session->message_count].content =
            (const char *)request->prompt;
        prompt_messages[session->message_count].content_len =
            request->prompt_bytes;
        prompt.kind = YVEX_GENERATION_INPUT_MESSAGES;
        prompt.messages = prompt_messages;
        prompt.message_count = session->message_count + 1u;
        prompt.prompt_options.add_bos = 1;
        prompt.prompt_options.add_generation_prompt = 1;
        prompt.prompt_options.drop_thinking =
            request->reasoning_policy == YVEX_REASONING_DISABLED;
        prompt.prompt_options.mode =
            request->reasoning_policy == YVEX_REASONING_DISABLED
                ? YVEX_PROMPT_MODE_CHAT : YVEX_PROMPT_MODE_THINKING;
        prompt.prompt_options.reasoning_policy = request->reasoning_policy;
    }
    prompt.encode_options.maximum_tokens = registry->options.context_capacity;
    memset(&sink, 0, sizeof(sink));
    sink.registry = registry;
    sink.session = session;
    sink.request = request;
    sink.emit = emit;
    sink.emit_context = emit_context;
    sink.started_ns = monotonic_ns();
    sink.queue_seconds = queue_seconds;
    yvex_core_text_copy(sink.request_id, sizeof(sink.request_id), request_id);
    (void)snprintf(sink.turn_id, sizeof(sink.turn_id), "t%llu",
                   session->turn_count + 1u);
    if (!model_view || !model_view->tokenizer)
        return YVEX_ERR_STATE;
    rc = yvex_tokenizer_reasoning_stream_open(
        &sink.reasoning_stream, model_view->tokenizer,
        request->reasoning_policy, turn_classified_fragment, &sink, err);
    if (rc != YVEX_OK) return rc;
    memset(&turn, 0, sizeof(turn));
    turn.schema_version = YVEX_RUNTIME_GENERATION_TURN_SCHEMA_V1;
    turn.prompt = &prompt;
    turn.committed_prefix_token_ids = session->committed_tokens;
    turn.committed_prefix_token_count = session->committed_count;
    turn.maximum_new_tokens = request->provider_request
                                  ? request->provider_request->maximum_output_tokens
                                  : request->maximum_new_tokens;
    turn.prompt_token_ids = session->prompt_tokens;
    turn.prompt_token_capacity = session->token_capacity;
    turn.fragment_sink = turn_fragment;
    turn.fragment_context = &sink;
    turn.progress_sink = turn_progress;
    turn.progress_context = &sink;
    turn.speculation_progress_sink = turn_speculation_progress;
    turn.speculation_progress_context = &sink;
    atomic_store_explicit(&session->cancel_requested, 0, memory_order_release);
    atomic_store_explicit(&session->active_turn, 1, memory_order_release);
    session->state = YVEX_SERVER_SESSION_RUNNING;
    memset(&started, 0, sizeof(started));
    started.schema_version = YVEX_LOCAL_PROTOCOL_VERSION;
    started.kind = YVEX_CLIENT_MESSAGE_TURN_STARTED;
    started.status = YVEX_OK;
    started.generation_phase = YVEX_CLIENT_PHASE_TOKENIZING;
    started.stream_channel = YVEX_CLIENT_STREAM_CONTROL_EVENT;
    started.request_number = request->request_number;
    yvex_core_text_copy(started.session_name, sizeof(started.session_name),
                        session->name);
    rc = emit(emit_context, &started, err);
    if (rc == YVEX_OK)
        rc = yvex_server_telemetry_emit_provider(
            registry->telemetry, YVEX_SERVER_EVENT_REQUEST_STARTED,
            YVEX_SERVER_SEVERITY_INFO, session->name, sink.request_id,
            sink.turn_id, "turn",
            request->provider_request
                ? request->provider_request->message_count
                : request->prompt_bytes,
            session->committed_count, turn.maximum_new_tokens,
            0.0, 0.0, NULL, request->provider_request, NULL, err);
    memset(&result, 0, sizeof(result));
    if (rc == YVEX_OK)
        rc = yvex_runtime_generation_turn_execute(
            session->generation, &turn, session->token_results,
            registry->options.maximum_new_tokens, session->turn_text,
            session->text_capacity, &result, err);
    {
        yvex_error stream_error;
        int stream_rc;
        yvex_error_clear(&stream_error);
        stream_rc = yvex_tokenizer_reasoning_stream_finish(
            sink.reasoning_stream, &stream_error);
        yvex_tokenizer_reasoning_stream_close(&sink.reasoning_stream);
        if (rc == YVEX_OK && stream_rc != YVEX_OK) {
            rc = stream_rc;
            if (err) *err = stream_error;
        }
    }
    generation_rc = rc;
    yvex_error_clear(&primary_error);
    if (generation_rc != YVEX_OK && err) primary_error = *err;
    rc = session_turn_commit(session, request, &result, prior_messages,
                             prior_transcript, rc, err);
    if (generation_rc != YVEX_OK && err) *err = primary_error;
    if (rc == YVEX_OK)
        rc = session_profile_publish(registry, session, request, &sink, &result, err);
    rc = session_turn_publish(registry, session, request, &sink, &result,
                              rc, err);
    atomic_store_explicit(&session->active_turn, 0, memory_order_release);
    return rc;
}
/*
 * Clear exact mutable session state while keeping model residency process-owned.
 * Reopening the execution session also drops a workspace recipe sealed by an
 * interrupted turn; reusing that recipe for a differently sized first chunk
 * would otherwise violate runtime workspace identity admission.
 */
static int session_reset(server_session_registry *registry,
                         server_session *session, yvex_error *err)
{
    unsigned long long next_message_generation, next_transcript_generation;
    int rc;
    if (!yvex_core_u64_add(session->message_history_generation, 1u,
                           &next_message_generation) ||
        !yvex_core_u64_add(session->transcript_generation, 1u,
                           &next_transcript_generation)) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "server.session.reset",
                       "session history generation overflowed");
        return YVEX_ERR_BOUNDS;
    }
    session->state = YVEX_SERVER_SESSION_RESETTING;
    rc = yvex_runtime_generation_context_close(&session->generation, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_session_close(&session->execution, err);
    if (rc == YVEX_OK)
        rc = session_execution_open(registry, session, err);
    if (rc != YVEX_OK) {
        session->state = YVEX_SERVER_SESSION_FAILED;
        return rc;
    }
    memset(session->messages, 0, sizeof(session->messages));
    memset(session->transcript, 0, (size_t)session->transcript_capacity);
    memset(session->committed_tokens, 0,
           (size_t)session->token_capacity * sizeof(*session->committed_tokens));
    session->message_count = 0u;
    session->transcript_count = 0u;
    session->committed_count = 0u;
    session->turn_count = 0u;
    session->message_history_generation = next_message_generation;
    session->transcript_generation = next_transcript_generation;
    session->policy_set = 0;
    memset(&session->policy, 0, sizeof(session->policy));
    session->reasoning_policy = YVEX_REASONING_DISABLED;
    memset(session->last_turn_identity, 0, sizeof(session->last_turn_identity));
    memset(session->state_digest, 0, sizeof(session->state_digest));
    memset(session->generated_token_identity, 0,
           sizeof(session->generated_token_identity));
    memset(session->generated_text_digest, 0,
           sizeof(session->generated_text_digest));
    memset(&session->partial_turn, 0, sizeof(session->partial_turn));
    atomic_store_explicit(&session->cancel_requested, 0, memory_order_release);
    session->state = session->attached_clients ? YVEX_SERVER_SESSION_READY
                                               : YVEX_SERVER_SESSION_DETACHED;
    return yvex_server_telemetry_emit(
        registry->telemetry, YVEX_SERVER_EVENT_SESSION_RESET,
        YVEX_SERVER_SEVERITY_INFO, session->name, NULL, NULL, "session",
        0u, 0u, 0u, 0.0, 0.0, err);
}

static int session_close_locked(server_session_registry *registry,
                                server_session *session, yvex_error *err)
{
    int rc;
    session->state = YVEX_SERVER_SESSION_CLOSING;
    rc = yvex_runtime_generation_context_close(&session->generation, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_session_close(&session->execution, err);
    if (rc != YVEX_OK) {
        session->state = YVEX_SERVER_SESSION_FAILED;
        return rc;
    }
    free(session->transcript);
    free(session->turn_text);
    free(session->token_results);
    free(session->prompt_tokens);
    free(session->committed_tokens);
    session->state = YVEX_SERVER_SESSION_CLOSED;
    registry->count--;
    yvex_server_telemetry_session(registry->telemetry, -1, 0);
    (void)yvex_server_telemetry_emit(
        registry->telemetry, YVEX_SERVER_EVENT_SESSION_CLOSED,
        YVEX_SERVER_SEVERITY_INFO, session->name, NULL, NULL, "session",
        0u, registry->count, 0u, 0.0, 0.0, err);
    memset(session, 0, sizeof(*session));
    session->state = YVEX_SERVER_SESSION_CLOSED;
    return YVEX_OK;
}
/*
 * Allocate a bounded registry over one borrowed process-resident model.
 *
 * Model ownership stays with host.
 */
int yvex_server_sessions_open(server_session_registry **out,
                                 yvex_runtime_model *model,
                                 const yvex_server_options *options,
                                 server_telemetry *telemetry,
                                 yvex_error *err)
{
    server_session_registry *registry;
    if (out) *out = NULL;
    if (!out || !model || !options || !telemetry ||
        !options->maximum_sessions ||
        options->maximum_sessions > SIZE_MAX / sizeof(server_session)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "server.session.registry",
                       "model, telemetry, and bounded session capacity are required");
        return YVEX_ERR_INVALID_ARG;
    }
    registry = calloc(1u, sizeof(*registry));
    if (registry)
        registry->sessions = calloc((size_t)options->maximum_sessions,
                                    sizeof(*registry->sessions));
    if (!registry || !registry->sessions) {
        free(registry ? registry->sessions : NULL);
        free(registry);
        yvex_error_set(err, YVEX_ERR_NOMEM, "server.session.registry",
                       "session registry allocation failed");
        return YVEX_ERR_NOMEM;
    }
    registry->model = model;
    registry->options = *options;
    registry->telemetry = telemetry;
    registry->capacity = options->maximum_sessions;
    registry->next_id = 1u;
    if (pthread_mutex_init(&registry->mutex, NULL) != 0) {
        free(registry->sessions);
        free(registry);
        yvex_error_set(err, YVEX_ERR_STATE, "server.session.registry",
                       "session registry mutex initialization failed");
        return YVEX_ERR_STATE;
    }
    registry->mutex_ready = 1;
    *out = registry;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_server_sessions_execute(server_session_registry *registry,
                                    const yvex_client_request *request,
                                    const char *request_id,
                                    double queue_seconds,
                                    server_message_emit emit,
                                    void *emit_context, yvex_error *err)
{
    server_session *session = NULL;
    int rc = YVEX_OK;
    if (!registry || !request || !request_id || !request_id[0] || !emit ||
        pthread_mutex_lock(&registry->mutex) != 0) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "server.session.execute",
                       "registry, request, and response sink are required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (registry->closing) {
        rc = YVEX_ERR_STATE;
        yvex_error_set(err, YVEX_ERR_STATE, "server.session.execute",
                       "session registry is closing");
        goto done;
    }
    if (request->operation != YVEX_CLIENT_OP_SESSION_NEW &&
        request->operation != YVEX_CLIENT_OP_SESSION_LIST)
        session = session_find_locked(registry, request->session_name);
    if (request->operation == YVEX_CLIENT_OP_SESSION_NEW) {
        rc = session_create_locked(registry, request->session_name,
                                   &session, err);
        if (rc == YVEX_OK)
            rc = session_message(emit, emit_context,
                                 YVEX_CLIENT_MESSAGE_SESSION, YVEX_OK,
                                 request, session, "created", err);
    } else if (request->operation == YVEX_CLIENT_OP_SESSION_LIST) {
        unsigned long long index;
        for (index = 0u; rc == YVEX_OK && index < registry->capacity; ++index)
            if (registry->sessions[index].name[0] &&
                registry->sessions[index].state != YVEX_SERVER_SESSION_CLOSED)
                rc = session_message(emit, emit_context,
                                     YVEX_CLIENT_MESSAGE_SESSION, YVEX_OK,
                                     request, &registry->sessions[index],
                                     "member", err);
        if (rc == YVEX_OK)
            rc = session_message(emit, emit_context, YVEX_CLIENT_MESSAGE_ACK,
                                 YVEX_OK, request, NULL, "complete", err);
    } else if (!session) {
        rc = YVEX_ERR_STATE;
        yvex_error_set(err, YVEX_ERR_STATE, "server.session.lookup",
                       "unknown session");
    } else if (request->operation == YVEX_CLIENT_OP_SESSION_SHOW) {
        rc = session_message(emit, emit_context,
                             YVEX_CLIENT_MESSAGE_SESSION, YVEX_OK,
                             request, session, "snapshot", err);
    } else if (request->operation == YVEX_CLIENT_OP_SESSION_ATTACH) {
        session->attached_clients++;
        if (session->state == YVEX_SERVER_SESSION_DETACHED)
            session->state = YVEX_SERVER_SESSION_READY;
        rc = yvex_server_telemetry_emit(
            registry->telemetry, YVEX_SERVER_EVENT_SESSION_ATTACHED,
            YVEX_SERVER_SEVERITY_INFO, session->name, NULL, NULL, "session",
            session->attached_clients, 0u, 0u, 0.0, 0.0, err);
        if (rc == YVEX_OK)
            rc = session_message(emit, emit_context, YVEX_CLIENT_MESSAGE_ACK,
                                 YVEX_OK, request, session, "attached", err);
    } else if (request->operation == YVEX_CLIENT_OP_SESSION_DETACH) {
        if (session->attached_clients) session->attached_clients--;
        if (!session->attached_clients && session->state == YVEX_SERVER_SESSION_READY)
            session->state = YVEX_SERVER_SESSION_DETACHED;
        rc = yvex_server_telemetry_emit(
            registry->telemetry, YVEX_SERVER_EVENT_SESSION_DETACHED,
            YVEX_SERVER_SEVERITY_INFO, session->name, NULL, NULL, "session",
            session->attached_clients, 0u, 0u, 0.0, 0.0, err);
        if (rc == YVEX_OK)
            rc = session_message(emit, emit_context, YVEX_CLIENT_MESSAGE_ACK,
                                 YVEX_OK, request, session, "detached", err);
    } else if (request->operation == YVEX_CLIENT_OP_SESSION_RESET) {
        rc = session_reset(registry, session, err);
        if (rc == YVEX_OK)
            rc = session_message(emit, emit_context, YVEX_CLIENT_MESSAGE_ACK,
                                 YVEX_OK, request, session, "reset", err);
    } else if (request->operation == YVEX_CLIENT_OP_SESSION_CLOSE) {
        rc = session_close_locked(registry, session, err);
        if (rc == YVEX_OK)
            rc = session_message(emit, emit_context, YVEX_CLIENT_MESSAGE_ACK,
                                 YVEX_OK, request, NULL, "closed", err);
    } else if (request->operation == YVEX_CLIENT_OP_GENERATION_TURN) {
        (void)pthread_mutex_unlock(&registry->mutex);
        rc = session_turn(registry, session, request, request_id, queue_seconds,
                          emit, emit_context, err);
        return rc;
    } else {
        rc = YVEX_ERR_UNSUPPORTED;
        yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "server.session.execute",
                       "operation is not owned by the session registry");
    }
done:
    (void)pthread_mutex_unlock(&registry->mutex);
    return rc;
}

/*
 * Capture one composed-console session slice under the sole registry authority.
 *
 * Unknown sessions or synchronization failure publish no partial session slice.
 */
int yvex_server_sessions_console_status(server_session_registry *registry,
                                        const char *session_name,
                                        yvex_console_status *status,
                                        yvex_client_partial_turn *partial_turn,
                                        yvex_error *err)
{
    server_session *session;
    if (!registry || !session_name || !status || !partial_turn ||
        pthread_mutex_lock(&registry->mutex) != 0) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "server.session.console-status",
                       "registry, session name, and status output are required");
        return YVEX_ERR_INVALID_ARG;
    }
    session = session_find_locked(registry, session_name);
    if (!session) {
        (void)pthread_mutex_unlock(&registry->mutex);
        yvex_error_set(err, YVEX_ERR_STATE, "server.session.console-status",
                       "unknown session");
        return YVEX_ERR_STATE;
    }
    status->schema_version = 1u;
    status->session_available = 1;
    status->attached = session->attached_clients != 0u;
    status->cancel_requested = atomic_load_explicit(&session->cancel_requested,
                                                    memory_order_acquire) != 0;
    status->session_state = session->state;
    status->position = session->committed_count;
    status->turn_count = session->turn_count;
    status->context_capacity = registry->options.context_capacity;
    status->context_used = session->committed_count;
    status->kv_used_available = 0;
    status->progress_available = 0;
    status->reasoning_policy = session->reasoning_policy;
    status->generation_phase = atomic_load_explicit(&session->active_turn,
                                                     memory_order_acquire)
                                   ? YVEX_CLIENT_PHASE_UNAVAILABLE
                                   : YVEX_CLIENT_PHASE_IDLE;
    status->cancellation_class = status->cancel_requested
                                     ? YVEX_CLIENT_CANCELLATION_REQUESTED
                                     : YVEX_CLIENT_CANCELLATION_NONE;
    yvex_core_text_copy(status->session_name, sizeof(status->session_name),
                        session->name);
    *partial_turn = session->partial_turn;
    (void)pthread_mutex_unlock(&registry->mutex);
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_server_sessions_cancel(server_session_registry *registry,
                                   const char *session_name,
                                   yvex_error *err)
{
    server_session *session;
    if (!registry || !session_name ||
        pthread_mutex_lock(&registry->mutex) != 0) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "server.session.cancel",
                       "registry and session name are required");
        return YVEX_ERR_INVALID_ARG;
    }
    session = session_find_locked(registry, session_name);
    if (!session || !atomic_load_explicit(&session->active_turn,
                                          memory_order_acquire)) {
        (void)pthread_mutex_unlock(&registry->mutex);
        yvex_error_set(err, YVEX_ERR_STATE, "server.session.cancel",
                       "session has no active turn");
        return YVEX_ERR_STATE;
    }
    atomic_store_explicit(&session->cancel_requested, 1, memory_order_release);
    (void)pthread_mutex_unlock(&registry->mutex);
    yvex_error_clear(err);
    return YVEX_OK;
}
/*
 * Request cancellation for every active turn during daemon shutdown.
 *
 * Returns if lock ownership is unavailable.
 */
void yvex_server_sessions_cancel_all(server_session_registry *registry)
{
    unsigned long long index;
    if (!registry || pthread_mutex_lock(&registry->mutex) != 0) return;
    for (index = 0u; index < registry->capacity; ++index)
        if (registry->sessions[index].name[0] &&
            atomic_load_explicit(&registry->sessions[index].active_turn,
                                 memory_order_acquire))
            atomic_store_explicit(&registry->sessions[index].cancel_requested,
                                  1, memory_order_release);
    (void)pthread_mutex_unlock(&registry->mutex);
}

int yvex_server_sessions_count(server_session_registry *registry,
                                  unsigned long long *count, yvex_error *err)
{
    if (!registry || !count || pthread_mutex_lock(&registry->mutex) != 0) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "server.session.count",
                       "registry and count output are required");
        return YVEX_ERR_INVALID_ARG;
    }
    *count = registry->count;
    (void)pthread_mutex_unlock(&registry->mutex);
    yvex_error_clear(err);
    return YVEX_OK;
}
/* Drain every session and release registry ownership before model close. */
int yvex_server_sessions_close(server_session_registry **registry,
                                  yvex_error *err)
{
    server_session_registry *owner;
    unsigned long long index;
    int rc = YVEX_OK;
    if (!registry || !*registry) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    owner = *registry;
    if (!owner->mutex_ready || pthread_mutex_lock(&owner->mutex) != 0) {
        yvex_error_set(err, YVEX_ERR_STATE, "server.session.close",
                       "session registry close lock failed");
        return YVEX_ERR_STATE;
    }
    owner->closing = 1;
    for (index = 0u; index < owner->capacity && rc == YVEX_OK; ++index)
        if (owner->sessions[index].name[0] &&
            owner->sessions[index].state != YVEX_SERVER_SESSION_CLOSED)
            rc = session_close_locked(owner, &owner->sessions[index], err);
    (void)pthread_mutex_unlock(&owner->mutex);
    if (rc != YVEX_OK) return rc;
    (void)pthread_mutex_destroy(&owner->mutex);
    owner->mutex_ready = 0;
    free(owner->sessions);
    memset(owner, 0, sizeof(*owner));
    free(owner);
    *registry = NULL;
    yvex_error_clear(err);
    return YVEX_OK;
}
