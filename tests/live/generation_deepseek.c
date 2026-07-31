/*
 * Exercises text-to-text autoregressive composition and exact partial/publication evidence.
 * Every ordinary sampled ID is the exact next decode input and no teacher-forced tail exists.
 * Test-only consumer of production lower-owner APIs over isolated CPU/CUDA sessions.
 */
#include <yvex/internal/decode.h>
#include <yvex/internal/core.h>
#include <yvex/internal/generation.h>
#include <yvex/internal/logits.h>
#include <yvex/internal/runtime.h>
#include <yvex/internal/sampling.h>
#include <yvex/internal/transformer.h>
#include <yvex/tokenizer.h>

#include <stddef.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LIVE_GENERATION_MAX_TOKENS 4ull
#define LIVE_GENERATION_TEXT_BYTES 256ull

typedef struct {
    yvex_runtime_generation_plan_summary plan;
    yvex_runtime_generation_result result;
    yvex_runtime_generation_token_result tokens[LIVE_GENERATION_MAX_TOKENS];
    unsigned char text[LIVE_GENERATION_TEXT_BYTES];
} live_generation;

typedef struct {
    unsigned int tokens[LIVE_GENERATION_MAX_TOKENS];
    unsigned long long sampled, committed, text_bytes, final_position;
    unsigned char text[LIVE_GENERATION_TEXT_BYTES];
    char state_digest[YVEX_SHA256_HEX_CAP], rng_identity[YVEX_SHA256_HEX_CAP];
} live_manual;

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    int entered, release;
} live_cancel_gate;

typedef struct {
    yvex_runtime_generation_context *context;
    live_cancel_gate *gate;
    int rc;
    yvex_error err;
} live_execute_thread;

typedef struct {
    yvex_runtime_generation_context *context;
    atomic_int started;
    int rc;
    yvex_error err;
} live_close_thread;

typedef struct {
    yvex_runtime_execution_session *session;
    unsigned long long cancel_at_position;
    atomic_int cancel, stop;
} live_position_cancel;

static void live_failure(const char *step, int rc, const yvex_error *err)
{
    fprintf(stderr, "generation_live step=%s status=%d where=%s reason=%s\n",
            step, rc, err ? yvex_error_where(err) : "",
            err ? yvex_error_message(err) : "");
}

static int live_state(const yvex_runtime_execution_session *session,
                      yvex_graph_attention_state_summary *summary,
                      yvex_error *err)
{
    const yvex_runtime_session_view *view = yvex_runtime_session_view_get(session);
    if (!view || !view->attention_state_provider ||
        !view->attention_state_provider->summary)
        return YVEX_ERR_STATE;
    return view->attention_state_provider->summary(
        view->attention_state_provider->context, summary, err);
}

static int live_cancel_at_position(void *opaque)
{
    live_position_cancel *cancel = opaque;
    return !cancel || atomic_load_explicit(&cancel->cancel, memory_order_acquire);
}

static void *live_position_monitor_main(void *opaque)
{
    live_position_cancel *cancel = opaque;
    while (!atomic_load_explicit(&cancel->stop, memory_order_acquire)) {
        yvex_graph_attention_state_summary summary;
        yvex_error err;
        if (live_state(cancel->session, &summary, &err) != YVEX_OK ||
            summary.next_position >= cancel->cancel_at_position) {
            atomic_store_explicit(&cancel->cancel, 1, memory_order_release);
            break;
        }
        sched_yield();
    }
    return NULL;
}

static int live_cancel_block(void *opaque)
{
    live_cancel_gate *gate = opaque;
    (void)pthread_mutex_lock(&gate->mutex);
    gate->entered = 1;
    (void)pthread_cond_broadcast(&gate->condition);
    while (!gate->release)
        (void)pthread_cond_wait(&gate->condition, &gate->mutex);
    (void)pthread_mutex_unlock(&gate->mutex);
    return 1;
}

static void *live_execute_main(void *opaque)
{
    static const unsigned char prompt[] = "Hi";
    live_execute_thread *thread = opaque;
    yvex_runtime_generation_token_result token;
    yvex_runtime_generation_result result;
    unsigned char text[LIVE_GENERATION_TEXT_BYTES];
    yvex_runtime_generation_request request = {
        .schema_version = YVEX_RUNTIME_GENERATION_SCHEMA_V2,
        .kind = YVEX_GENERATION_INPUT_TEXT, .text = prompt,
        .text_bytes = sizeof(prompt) - 1ull,
        .encode_options = {.maximum_tokens = 8ull}};
    thread->rc = yvex_runtime_generation_execute(
        thread->context, &request, &token, 1ull, text, sizeof(text),
        &result, &thread->err);
    return NULL;
}

static void *live_close_main(void *opaque)
{
    live_close_thread *thread = opaque;
    atomic_store_explicit(&thread->started, 1, memory_order_release);
    thread->rc = yvex_runtime_generation_context_close(
        &thread->context, &thread->err);
    return NULL;
}

static int live_input_open(yvex_transformer_input **input,
                           const yvex_transformer_plan_summary *plan,
                           const yvex_tokenizer_encode_result *encoded,
                           yvex_error *err)
{
    yvex_transformer_input_summary summary;
    int rc;
    memset(&summary, 0, sizeof(summary));
    summary.schema_version = YVEX_TRANSFORMER_INPUT_SCHEMA_V1;
    summary.token_count = encoded->tokens.len;
    summary.vocabulary_size = plan->vocabulary_size;
    yvex_runtime_identity_copy(summary.logical_model_identity,
                               plan->logical_model_identity);
    yvex_runtime_identity_copy(summary.runtime_numeric_identity,
                               plan->runtime_numeric_identity);
    yvex_runtime_identity_copy(summary.runtime_descriptor_identity,
                               plan->runtime_descriptor_identity);
    yvex_runtime_identity_copy(summary.transformer_plan_identity,
                               plan->transformer_plan_identity);
    rc = yvex_transformer_input_seal(&summary, encoded->tokens.ids, err);
    if (rc == YVEX_OK)
        rc = yvex_transformer_input_open_memory(
            input, &summary, encoded->tokens.ids, err);
    return rc;
}

static int live_production(yvex_runtime_model *model, yvex_backend_kind backend,
                           yvex_runtime_sampling_policy policy,
                           unsigned long long maximum_tokens,
                           live_generation *out, yvex_error *err)
{
    static const unsigned char prompt[] = "Hi";
    yvex_runtime_session_open_request session_options = {.backend = backend};
    yvex_runtime_generation_options options = {
        .schema_version = YVEX_RUNTIME_GENERATION_SCHEMA_V2,
        .context_capacity = 8ull, .prefill_chunk_tokens = 8ull,
        .maximum_output_bytes = LIVE_GENERATION_TEXT_BYTES - 1ull,
        .trace_policy = YVEX_RUNTIME_TRACE_SUMMARY};
    yvex_runtime_generation_request request = {
        .schema_version = YVEX_RUNTIME_GENERATION_SCHEMA_V2,
        .kind = YVEX_GENERATION_INPUT_TEXT, .text = prompt,
        .text_bytes = sizeof(prompt) - 1ull,
        .encode_options = {.maximum_tokens = 8ull}};
    yvex_runtime_model_failure failure = {0};
    yvex_runtime_execution_session *session = NULL;
    yvex_runtime_generation_context *context = NULL;
    yvex_error cleanup;
    int rc, close_rc;
    memset(out, 0, sizeof(*out));
    options.backend = backend;
    options.maximum_new_tokens = maximum_tokens;
    options.sampling_policy = policy;
    rc = yvex_runtime_session_open(&session, model, &session_options,
                                   &failure, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_generation_context_open(
            &context, model, session, &options, err);
    if (rc == YVEX_OK)
        out->plan = *yvex_runtime_generation_plan_summary_get(context);
    if (rc == YVEX_OK)
        rc = yvex_runtime_generation_execute(
            context, &request, out->tokens, LIVE_GENERATION_MAX_TOKENS,
            out->text, sizeof(out->text), &out->result, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_generation_result_validate(
            &out->plan, out->tokens, LIVE_GENERATION_MAX_TOKENS,
            out->text, sizeof(out->text), &out->result, err);
    yvex_error_clear(&cleanup);
    close_rc = yvex_runtime_generation_context_close(&context, &cleanup);
    if (rc == YVEX_OK && close_rc != YVEX_OK) { rc = close_rc; *err = cleanup; }
    yvex_error_clear(&cleanup);
    close_rc = yvex_runtime_session_close(&session, &cleanup);
    if (rc == YVEX_OK && close_rc != YVEX_OK) { rc = close_rc; *err = cleanup; }
    return rc;
}

static int live_boundary_execute(
    yvex_runtime_model *model, yvex_backend_kind backend,
    yvex_runtime_sampling_policy policy, unsigned long long context_capacity,
    const unsigned int *stop_token, live_generation *out, yvex_error *err)
{
    static const unsigned char prompt[] = "Hi";
    yvex_runtime_session_open_request session_options = {.backend = backend};
    yvex_runtime_generation_options options = {
        .schema_version = YVEX_RUNTIME_GENERATION_SCHEMA_V2,
        .context_capacity = context_capacity, .prefill_chunk_tokens = 1ull,
        .maximum_new_tokens = 1ull,
        .maximum_output_bytes = LIVE_GENERATION_TEXT_BYTES - 1ull,
        .trace_policy = YVEX_RUNTIME_TRACE_SUMMARY};
    yvex_runtime_generation_request request = {
        .schema_version = YVEX_RUNTIME_GENERATION_SCHEMA_V2,
        .kind = YVEX_GENERATION_INPUT_TEXT, .text = prompt,
        .text_bytes = sizeof(prompt) - 1ull,
        .encode_options = {.maximum_tokens = 8ull}};
    yvex_runtime_model_failure failure = {0};
    yvex_runtime_execution_session *session = NULL;
    yvex_runtime_generation_context *context = NULL;
    yvex_error cleanup;
    int rc, close_rc;
    memset(out, 0, sizeof(*out));
    options.backend = backend;
    options.sampling_policy = policy;
    options.additional_stop_token_ids = stop_token;
    options.additional_stop_token_count = stop_token ? 1ull : 0ull;
    rc = yvex_runtime_session_open(&session, model, &session_options,
                                   &failure, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_generation_context_open(
            &context, model, session, &options, err);
    if (rc == YVEX_OK)
        out->plan = *yvex_runtime_generation_plan_summary_get(context);
    if (rc == YVEX_OK)
        rc = yvex_runtime_generation_execute(
            context, &request, out->tokens, LIVE_GENERATION_MAX_TOKENS,
            out->text, sizeof(out->text), &out->result, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_generation_result_validate(
            &out->plan, out->tokens, LIVE_GENERATION_MAX_TOKENS,
            out->text, sizeof(out->text), &out->result, err);
    yvex_error_clear(&cleanup);
    close_rc = yvex_runtime_generation_context_close(&context, &cleanup);
    if (rc == YVEX_OK && close_rc != YVEX_OK) { rc = close_rc; *err = cleanup; }
    yvex_error_clear(&cleanup);
    close_rc = yvex_runtime_session_close(&session, &cleanup);
    if (rc == YVEX_OK && close_rc != YVEX_OK) { rc = close_rc; *err = cleanup; }
    return rc;
}

static int live_stop_capacity_proof(
    yvex_runtime_model *model, yvex_backend_kind backend,
    yvex_runtime_sampling_policy policy, unsigned int first_token,
    yvex_error *err)
{
    live_generation stop, capacity;
    int rc = live_boundary_execute(model, backend, policy, 8ull,
                                   &first_token, &stop, err);
    if (rc == YVEX_OK &&
        (!stop.result.completed ||
         stop.result.stop_reason != YVEX_GENERATION_STOP_TOKENIZER_TOKEN ||
         stop.result.prompt_token_count != 1ull ||
         stop.result.sampled_token_count != 1ull ||
         stop.result.terminal_token_count != 1ull ||
         stop.result.model_committed_token_count ||
         stop.result.decode_step_count || stop.result.final_position != 1ull ||
         stop.tokens[0].sampled_token_id != first_token ||
         !stop.tokens[0].terminal || stop.tokens[0].decode_submitted ||
         stop.tokens[0].model_committed))
        rc = YVEX_ERR_FORMAT;
    if (rc == YVEX_OK)
        rc = live_boundary_execute(model, backend, policy, 1ull, NULL,
                                   &capacity, err);
    if (rc == YVEX_OK &&
        (!capacity.result.completed ||
         capacity.result.stop_reason != YVEX_GENERATION_STOP_CONTEXT_CAPACITY ||
         capacity.result.prompt_token_count != 1ull ||
         capacity.result.sampled_token_count ||
         capacity.result.model_committed_token_count ||
         capacity.result.logits_projection_count ||
         capacity.result.decode_step_count || capacity.result.final_position != 1ull))
        rc = YVEX_ERR_FORMAT;
    if (rc != YVEX_OK && !yvex_error_message(err)[0])
        yvex_error_set(err, rc, "generation_live",
                       "terminal/capacity publication proof failed");
    return rc;
}

/*
 * Execute one expected post-model-commit cancellation or output-budget failure.
 *
 * Immutable model/policy, failure kind, and isolated session ownership.
 */
static int live_partial_execute(
    yvex_runtime_model *model, yvex_backend_kind backend,
    yvex_runtime_sampling_policy policy, int cancel_after_commit,
    live_generation *out, int *execution_status, yvex_error *err)
{
    static const unsigned char prompt[] = "Hi";
    yvex_runtime_session_open_request session_options = {.backend = backend};
    yvex_runtime_generation_options options = {
        .schema_version = YVEX_RUNTIME_GENERATION_SCHEMA_V2,
        .context_capacity = 8ull, .prefill_chunk_tokens = 1ull,
        .maximum_new_tokens = 2ull,
        .trace_policy = YVEX_RUNTIME_TRACE_SUMMARY};
    yvex_runtime_generation_request request = {
        .schema_version = YVEX_RUNTIME_GENERATION_SCHEMA_V2,
        .kind = YVEX_GENERATION_INPUT_TEXT, .text = prompt,
        .text_bytes = sizeof(prompt) - 1ull,
        .encode_options = {.maximum_tokens = 8ull}};
    yvex_runtime_model_failure failure = {0};
    yvex_runtime_execution_session *session = NULL;
    yvex_runtime_generation_context *context = NULL;
    live_position_cancel cancel = {0};
    pthread_t monitor;
    yvex_error primary, secondary;
    int rc, close_rc, monitor_created = 0;
    memset(out, 0, sizeof(*out));
    yvex_error_clear(&primary);
    options.backend = backend;
    options.sampling_policy = policy;
    options.maximum_output_bytes = cancel_after_commit
                                       ? LIVE_GENERATION_TEXT_BYTES - 1ull
                                       : 1ull;
    rc = yvex_runtime_session_open(&session, model, &session_options,
                                   &failure, err);
    if (rc == YVEX_OK && cancel_after_commit) {
        cancel.session = session;
        cancel.cancel_at_position = 2ull;
        atomic_init(&cancel.cancel, 0);
        atomic_init(&cancel.stop, 0);
        options.cancel_requested = live_cancel_at_position;
        options.cancel_context = &cancel;
    }
    if (rc == YVEX_OK)
        rc = yvex_runtime_generation_context_open(
            &context, model, session, &options, err);
    if (rc == YVEX_OK)
        out->plan = *yvex_runtime_generation_plan_summary_get(context);
    if (rc == YVEX_OK && cancel_after_commit) {
        if (pthread_create(&monitor, NULL, live_position_monitor_main,
                           &cancel) != 0)
            rc = YVEX_ERR_STATE;
        else
            monitor_created = 1;
    }
    if (rc == YVEX_OK) {
        rc = yvex_runtime_generation_execute(
            context, &request, out->tokens, LIVE_GENERATION_MAX_TOKENS,
            out->text, sizeof(out->text), &out->result, err);
        primary = *err;
    }
    if (monitor_created) {
        atomic_store_explicit(&cancel.stop, 1, memory_order_release);
        (void)pthread_join(monitor, NULL);
    }
    *execution_status = rc;
    yvex_error_clear(&secondary);
    if ((rc == YVEX_ERR_CANCELLED || rc == YVEX_ERR_NOMEM) &&
        yvex_runtime_generation_result_validate(
            &out->plan, out->tokens, LIVE_GENERATION_MAX_TOKENS,
            out->text, sizeof(out->text), &out->result, &secondary) == YVEX_OK)
        rc = YVEX_OK;
    else if (rc == YVEX_ERR_CANCELLED || rc == YVEX_ERR_NOMEM)
        *err = secondary;
    yvex_error_clear(&secondary);
    close_rc = yvex_runtime_generation_context_close(&context, &secondary);
    if (rc == YVEX_OK && close_rc != YVEX_OK) { rc = close_rc; *err = secondary; }
    yvex_error_clear(&secondary);
    close_rc = yvex_runtime_session_close(&session, &secondary);
    if (rc == YVEX_OK && close_rc != YVEX_OK) { rc = close_rc; *err = secondary; }
    if (rc == YVEX_OK) yvex_error_clear(err);
    else if (!yvex_error_message(err)[0]) *err = primary;
    return rc;
}

static int live_partial_progress_proof(
    yvex_runtime_model *model, yvex_backend_kind backend,
    yvex_runtime_sampling_policy policy, yvex_error *err)
{
    live_generation cancelled, output_failed;
    int status = YVEX_OK;
    int rc = live_partial_execute(model, backend, policy, 1,
                                  &cancelled, &status, err);
    if (rc == YVEX_OK &&
        (status != YVEX_ERR_CANCELLED || !cancelled.result.cancelled ||
         !cancelled.result.partial ||
         cancelled.result.stop_reason != YVEX_GENERATION_STOP_CANCELLED ||
         cancelled.result.sampled_token_count != 1ull ||
         cancelled.result.model_committed_token_count != 1ull ||
         cancelled.result.final_position != 2ull ||
         !cancelled.tokens[0].model_committed ||
         cancelled.result.text_published_token_count !=
             (unsigned long long)cancelled.tokens[0].text_published ||
         cancelled.tokens[0].detokenized !=
             cancelled.tokens[0].text_published ||
         cancelled.result.generated_text_bytes !=
             cancelled.tokens[0].text_byte_count))
        rc = YVEX_ERR_FORMAT;
    if (rc == YVEX_OK)
        rc = live_partial_execute(model, backend, policy, 0,
                                  &output_failed, &status, err);
    if (rc == YVEX_OK &&
        (status != YVEX_ERR_NOMEM || !output_failed.result.failed ||
         !output_failed.result.partial ||
         output_failed.result.stop_reason != YVEX_GENERATION_STOP_OUTPUT_FAILURE ||
         output_failed.result.sampled_token_count != 1ull ||
         output_failed.result.model_committed_token_count != 1ull ||
         output_failed.result.text_published_token_count ||
         output_failed.result.generated_text_bytes ||
         output_failed.result.final_position != 2ull ||
         !output_failed.tokens[0].model_committed ||
         !output_failed.tokens[0].detokenized ||
         output_failed.tokens[0].text_published))
        rc = YVEX_ERR_FORMAT;
    if (rc != YVEX_OK && !yvex_error_message(err)[0])
        yvex_error_set(err, rc, "generation_live",
                       "post-commit partial-progress proof failed");
    return rc;
}

static int live_lifecycle_proof(yvex_runtime_model *model,
                                yvex_backend_kind backend,
                                yvex_runtime_sampling_policy policy,
                                yvex_error *err)
{
    static const unsigned char prompt[] = "Hi";
    yvex_runtime_session_open_request session_options = {.backend = backend};
    yvex_runtime_generation_options options = {
        .schema_version = YVEX_RUNTIME_GENERATION_SCHEMA_V2,
        .context_capacity = 8ull, .prefill_chunk_tokens = 8ull,
        .maximum_new_tokens = 1ull,
        .maximum_output_bytes = LIVE_GENERATION_TEXT_BYTES - 1ull,
        .trace_policy = YVEX_RUNTIME_TRACE_SUMMARY};
    yvex_runtime_generation_request request = {
        .schema_version = YVEX_RUNTIME_GENERATION_SCHEMA_V2,
        .kind = YVEX_GENERATION_INPUT_TEXT, .text = prompt,
        .text_bytes = sizeof(prompt) - 1ull,
        .encode_options = {.maximum_tokens = 8ull}};
    yvex_runtime_generation_token_result token;
    yvex_runtime_generation_result result;
    yvex_runtime_model_failure failure = {0};
    yvex_runtime_execution_session *session = NULL;
    yvex_runtime_generation_context *context = NULL;
    yvex_graph_attention_state_summary state;
    live_cancel_gate gate;
    live_execute_thread execute = {0};
    live_close_thread close = {0};
    pthread_t execute_id, close_id;
    unsigned char text[LIVE_GENERATION_TEXT_BYTES];
    unsigned int attempts;
    int execute_created = 0, close_created = 0, closing_refused = 0;
    int rc, close_rc;
    memset(&gate, 0, sizeof(gate));
    if (pthread_mutex_init(&gate.mutex, NULL) != 0 ||
        pthread_cond_init(&gate.condition, NULL) != 0)
        return YVEX_ERR_STATE;
    options.backend = backend;
    options.sampling_policy = policy;
    options.cancel_requested = live_cancel_block;
    options.cancel_context = &gate;
    rc = yvex_runtime_session_open(&session, model, &session_options,
                                   &failure, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_generation_context_open(
            &context, model, session, &options, err);
    if (rc == YVEX_OK) {
        execute.context = context;
        execute.gate = &gate;
        if (pthread_create(&execute_id, NULL, live_execute_main, &execute) != 0)
            rc = YVEX_ERR_STATE;
        else
            execute_created = 1;
    }
    if (rc == YVEX_OK) {
        (void)pthread_mutex_lock(&gate.mutex);
        while (!gate.entered)
            (void)pthread_cond_wait(&gate.condition, &gate.mutex);
        (void)pthread_mutex_unlock(&gate.mutex);
        close.context = context;
        atomic_init(&close.started, 0);
        if (pthread_create(&close_id, NULL, live_close_main, &close) != 0)
            rc = YVEX_ERR_STATE;
        else
            close_created = 1;
    }
    for (attempts = 0u; rc == YVEX_OK && attempts < 100000u; ++attempts) {
        yvex_error contender;
        int contender_rc;
        if (!atomic_load_explicit(&close.started, memory_order_acquire)) {
            sched_yield();
            continue;
        }
        contender_rc = yvex_runtime_generation_execute(
            context, &request, &token, 1ull, text, sizeof(text),
            &result, &contender);
        if (contender_rc == YVEX_ERR_STATE &&
            strstr(yvex_error_message(&contender), "closing")) {
            closing_refused = 1;
            break;
        }
        sched_yield();
    }
    (void)pthread_mutex_lock(&gate.mutex);
    gate.release = 1;
    (void)pthread_cond_broadcast(&gate.condition);
    (void)pthread_mutex_unlock(&gate.mutex);
    if (execute_created) (void)pthread_join(execute_id, NULL);
    if (close_created) (void)pthread_join(close_id, NULL);
    if (rc == YVEX_OK &&
        (execute.rc != YVEX_ERR_CANCELLED || close.rc != YVEX_OK ||
         close.context || !closing_refused))
        rc = YVEX_ERR_FORMAT;
    if (rc == YVEX_OK) rc = live_state(session, &state, err);
    if (rc == YVEX_OK &&
        (state.next_position || state.committed_sequence_length ||
         state.transaction_active))
        rc = YVEX_ERR_FORMAT;
    if (context && (!close_created || close.rc != YVEX_OK)) {
        yvex_error cleanup;
        yvex_error_clear(&cleanup);
        close_rc = yvex_runtime_generation_context_close(&context, &cleanup);
        if (rc == YVEX_OK && close_rc != YVEX_OK) { rc = close_rc; *err = cleanup; }
    }
    {
        yvex_error cleanup;
        yvex_error_clear(&cleanup);
        close_rc = yvex_runtime_session_close(&session, &cleanup);
        if (rc == YVEX_OK && close_rc != YVEX_OK) { rc = close_rc; *err = cleanup; }
    }
    (void)pthread_cond_destroy(&gate.condition);
    (void)pthread_mutex_destroy(&gate.mutex);
    if (rc == YVEX_OK) yvex_error_clear(err);
    return rc;
}

static int live_fragment_append(yvex_tokenizer_decoder *decoder,
                                unsigned int token, live_manual *out,
                                yvex_error *err)
{
    yvex_tokenizer_fragment fragment = {0};
    unsigned long long next;
    int rc = yvex_tokenizer_decoder_push(decoder, token, &fragment, err);
    if (rc == YVEX_OK &&
        (!yvex_core_u64_add(out->text_bytes, fragment.byte_count, &next) ||
         next >= LIVE_GENERATION_TEXT_BYTES))
        rc = YVEX_ERR_NOMEM;
    if (rc == YVEX_OK && fragment.byte_count)
        memcpy(out->text + out->text_bytes, fragment.bytes,
               (size_t)fragment.byte_count);
    if (rc == YVEX_OK) out->text_bytes = next;
    yvex_tokenizer_fragment_clear(&fragment);
    return rc;
}

static int live_manual_execute(yvex_runtime_model *model,
                               yvex_backend_kind backend,
                               yvex_runtime_sampling_policy policy,
                               unsigned long long maximum_tokens,
                               live_manual *out, yvex_error *err)
{
    static const unsigned char prompt[] = "Hi";
    const yvex_runtime_model_view *view = yvex_runtime_model_view_get(model);
    yvex_runtime_session_open_request session_options = {.backend = backend};
    yvex_runtime_transformer_options transformer_options = {.context_capacity = 8ull};
    yvex_runtime_decode_options decode_options = {.maximum_steps = LIVE_GENERATION_MAX_TOKENS};
    yvex_runtime_logits_options logits_options = {.maximum_rows = 1ull};
    yvex_runtime_sampling_options sampling_options = {
        .maximum_vocabulary_size = 129280ull, .maximum_rows = 1ull};
    yvex_tokenizer_encode_options encode_options = {.maximum_tokens = 8ull};
    yvex_tokenizer_decode_options decoder_options = {
        .skip_special_tokens = 1, .require_complete_utf8 = 1};
    yvex_runtime_execution_session *session = NULL;
    yvex_runtime_transformer_context *transformer = NULL;
    yvex_runtime_decode_context *decode = NULL;
    yvex_runtime_logits_context *logits = NULL;
    yvex_runtime_sampling_context *sampling = NULL;
    yvex_tokenizer_decoder *decoder = NULL;
    yvex_transformer_input *input = NULL;
    yvex_tokenizer_encode_result encoded = {0};
    yvex_runtime_transformer_result prefill = {0};
    yvex_runtime_decode_step_result decoded = {0};
    yvex_runtime_sampling_context_summary sampling_summary;
    yvex_graph_attention_state_summary state;
    yvex_runtime_model_failure failure = {0};
    const yvex_transformer_plan_summary *plan = NULL;
    const yvex_runtime_logits_plan_summary *logits_plan = NULL;
    float *prefill_hidden = NULL, *decode_hidden = NULL, *raw_logits = NULL;
    unsigned long long prefill_values = 0ull;
    int rc, first = 1;
    memset(out, 0, sizeof(*out));
    if (!view || !view->tokenizer) return YVEX_ERR_STATE;
    rc = yvex_runtime_session_open(&session, model, &session_options, &failure, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_transformer_context_open(
            &transformer, model, session, &transformer_options, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_decode_context_open(
            &decode, transformer, session, &decode_options, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_logits_context_open(
            &logits, model, session,
            yvex_runtime_transformer_context_plan(transformer),
            &logits_options, err);
    plan = transformer ? yvex_transformer_plan_summary_get(
                             yvex_runtime_transformer_context_plan(transformer)) : NULL;
    logits_plan = logits ? yvex_runtime_logits_plan_summary_get(logits) : NULL;
    if (rc == YVEX_OK)
        rc = yvex_runtime_sampling_policy_seal(
            &policy, logits_plan->vocabulary_size, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_sampling_context_open(
            &sampling, logits_plan, &policy, &sampling_options, err);
    if (rc == YVEX_OK)
        rc = yvex_tokenizer_decoder_open(
            &decoder, view->tokenizer, &decoder_options, err);
    if (rc == YVEX_OK)
        rc = yvex_tokenizer_encode(view->tokenizer, prompt, sizeof(prompt) - 1ull,
                                   &encode_options, &encoded, err);
    if (rc == YVEX_OK) rc = live_input_open(&input, plan, &encoded, err);
    if (rc == YVEX_OK &&
        (!yvex_core_u64_mul(encoded.tokens.len, plan->hidden_width,
                            &prefill_values) ||
         prefill_values > SIZE_MAX / sizeof(float))) rc = YVEX_ERR_BOUNDS;
    if (rc == YVEX_OK) {
        yvex_runtime_transformer_request request = {
            .chunk_tokens = encoded.tokens.len, .backend = backend,
            .phase = YVEX_TRANSFORMER_PHASE_PREFILL};
        yvex_runtime_transformer_output output;
        prefill_hidden = calloc((size_t)prefill_values, sizeof(float));
        decode_hidden = calloc((size_t)plan->hidden_width, sizeof(float));
        raw_logits = calloc((size_t)logits_plan->vocabulary_size, sizeof(float));
        if (!prefill_hidden || !decode_hidden || !raw_logits) rc = YVEX_ERR_NOMEM;
        output.normalized_hidden = prefill_hidden;
        output.capacity = prefill_values;
        if (rc == YVEX_OK)
            rc = yvex_runtime_transformer_execute(
                transformer, input, &request, &output, &prefill, err);
    }
    while (rc == YVEX_OK && out->committed < maximum_tokens) {
        yvex_runtime_logits_source logits_source;
        yvex_runtime_logits_row_result logits_result;
        yvex_runtime_sampling_source sample_source;
        yvex_runtime_sampling_result sample;
        yvex_tokenizer_token_classification classification;
        if (first)
            rc = yvex_runtime_logits_source_from_transformer(
                logits, &logits_source, &prefill, prefill_hidden,
                prefill_values, encoded.tokens.len - 1ull, err);
        else
            rc = yvex_runtime_logits_source_from_decode(
                logits, &logits_source, &decoded, decode_hidden,
                plan->hidden_width, err);
        if (rc == YVEX_OK)
            rc = yvex_runtime_logits_project(
                logits, &logits_source, backend, raw_logits,
                logits_plan->vocabulary_size, &logits_result, err);
        if (rc == YVEX_OK)
            rc = yvex_runtime_sampling_source_from_logits(
                sampling, &sample_source, raw_logits,
                logits_plan->vocabulary_size, &logits_result, err);
        if (rc == YVEX_OK)
            rc = yvex_runtime_sampling_select(
                sampling, &sample_source, &sample, err);
        if (rc == YVEX_OK) {
            out->tokens[out->sampled++] = sample.selected_token_id;
            rc = yvex_tokenizer_token_classify(
                view->tokenizer, sample.selected_token_id,
                &classification, err);
        }
        if (rc != YVEX_OK || classification.eos || classification.stop) break;
        if (rc == YVEX_OK) rc = live_state(session, &state, err);
        if (rc == YVEX_OK)
            rc = yvex_runtime_decode_step(
                decode, out->committed, state.next_position,
                sample.selected_token_id, backend, decode_hidden,
                plan->hidden_width, &decoded, err);
        if (rc == YVEX_OK) out->committed++;
        if (rc == YVEX_OK)
            rc = live_fragment_append(
                decoder, sample.selected_token_id, out, err);
        first = 0;
    }
    if (rc == YVEX_OK) {
        yvex_tokenizer_fragment finish = {0};
        rc = yvex_tokenizer_decoder_finish(decoder, &finish, err);
        yvex_tokenizer_fragment_clear(&finish);
    }
    if (rc == YVEX_OK) rc = live_state(session, &state, err);
    if (rc == YVEX_OK) {
        out->final_position = state.next_position;
        yvex_runtime_identity_copy(out->state_digest, state.state_content_identity);
        rc = yvex_runtime_sampling_context_snapshot(
            sampling, &sampling_summary, err);
    }
    if (rc == YVEX_OK)
        yvex_runtime_identity_copy(out->rng_identity,
                                   sampling_summary.rng_state_identity);
    free(raw_logits); free(decode_hidden); free(prefill_hidden);
    yvex_transformer_input_close(&input);
    yvex_tokenizer_encode_result_clear(&encoded);
    yvex_tokenizer_decoder_close(&decoder);
    {
        yvex_error cleanup;
        int close_rc;
        yvex_error_clear(&cleanup);
        close_rc = yvex_runtime_sampling_context_close(&sampling, &cleanup);
        if (rc == YVEX_OK && close_rc != YVEX_OK) { rc = close_rc; *err = cleanup; }
        close_rc = yvex_runtime_logits_context_close(&logits, &cleanup);
        if (rc == YVEX_OK && close_rc != YVEX_OK) { rc = close_rc; *err = cleanup; }
        close_rc = yvex_runtime_decode_context_close(&decode, &cleanup);
        if (rc == YVEX_OK && close_rc != YVEX_OK) { rc = close_rc; *err = cleanup; }
        close_rc = yvex_runtime_transformer_context_close(&transformer, &cleanup);
        if (rc == YVEX_OK && close_rc != YVEX_OK) { rc = close_rc; *err = cleanup; }
        close_rc = yvex_runtime_session_close(&session, &cleanup);
        if (rc == YVEX_OK && close_rc != YVEX_OK) { rc = close_rc; *err = cleanup; }
    }
    return rc;
}

static int live_mutation_proof(const live_generation *source, yvex_error *err)
{
    static const size_t aggregate_offsets[] = {
        offsetof(yvex_runtime_generation_result, status),
        offsetof(yvex_runtime_generation_result, stop_reason),
        offsetof(yvex_runtime_generation_result, prompt_token_count),
        offsetof(yvex_runtime_generation_result, sampled_token_count),
        offsetof(yvex_runtime_generation_result, model_committed_token_count),
        offsetof(yvex_runtime_generation_result, generated_text_bytes),
        offsetof(yvex_runtime_generation_result, final_position)};
    static const size_t token_offsets[] = {
        offsetof(yvex_runtime_generation_token_result, sampled_token_id),
        offsetof(yvex_runtime_generation_token_result, decode_input_token_id),
        offsetof(yvex_runtime_generation_token_result, decode_submitted),
        offsetof(yvex_runtime_generation_token_result, model_committed),
        offsetof(yvex_runtime_generation_token_result, position_after),
        offsetof(yvex_runtime_generation_token_result, text_byte_count)};
    yvex_runtime_generation_result result;
    yvex_runtime_generation_token_result tokens[LIVE_GENERATION_MAX_TOKENS];
    yvex_runtime_generation_plan_summary plan;
    unsigned char text[LIVE_GENERATION_TEXT_BYTES];
    unsigned long long index;
    int rc;
    for (index = 0ull; index < sizeof(aggregate_offsets) / sizeof(aggregate_offsets[0]); ++index) {
        result = source->result;
        *((unsigned char *)&result + aggregate_offsets[index]) ^= 1u;
        rc = yvex_runtime_generation_result_validate(
            &source->plan, source->tokens, LIVE_GENERATION_MAX_TOKENS,
            source->text, sizeof(source->text), &result, err);
        if (rc == YVEX_OK) return YVEX_ERR_FORMAT;
    }
    if (source->result.sampled_token_count) {
        for (index = 0ull; index < sizeof(token_offsets) / sizeof(token_offsets[0]); ++index) {
            memcpy(tokens, source->tokens, sizeof(tokens));
            *((unsigned char *)&tokens[0] + token_offsets[index]) ^= 1u;
            rc = yvex_runtime_generation_result_validate(
                &source->plan, tokens, LIVE_GENERATION_MAX_TOKENS,
                source->text, sizeof(source->text), &source->result, err);
            if (rc == YVEX_OK) return YVEX_ERR_FORMAT;
        }
    }
    plan = source->plan;
    plan.backend = plan.backend == YVEX_BACKEND_KIND_CPU
                       ? YVEX_BACKEND_KIND_CUDA : YVEX_BACKEND_KIND_CPU;
    rc = yvex_runtime_generation_result_validate(
        &plan, source->tokens, LIVE_GENERATION_MAX_TOKENS,
        source->text, sizeof(source->text), &source->result, err);
    if (rc == YVEX_OK) return YVEX_ERR_FORMAT;
    if (source->result.generated_text_bytes) {
        memcpy(text, source->text, sizeof(text));
        text[0] ^= 1u;
        rc = yvex_runtime_generation_result_validate(
            &source->plan, source->tokens, LIVE_GENERATION_MAX_TOKENS,
            text, sizeof(text), &source->result, err);
        if (rc == YVEX_OK) return YVEX_ERR_FORMAT;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

static int live_compare(const live_generation *production,
                        const live_manual *manual, yvex_error *err)
{
    unsigned long long index;
    if (!production->result.completed ||
        production->result.sampled_token_count != manual->sampled ||
        production->result.model_committed_token_count != manual->committed ||
        production->result.generated_text_bytes != manual->text_bytes ||
        production->result.final_position != manual->final_position ||
        strcmp(production->result.final_persistent_state_digest,
               manual->state_digest) != 0 ||
        strcmp(production->result.final_rng_identity,
               manual->rng_identity) != 0 ||
        memcmp(production->text, manual->text,
               (size_t)manual->text_bytes) != 0)
        return YVEX_ERR_FORMAT;
    for (index = 0ull; index < manual->sampled; ++index)
        if (production->tokens[index].sampled_token_id != manual->tokens[index] ||
            (production->tokens[index].model_committed &&
             (!production->tokens[index].decode_submitted ||
              production->tokens[index].decode_input_token_id !=
                  production->tokens[index].sampled_token_id)))
            return YVEX_ERR_FORMAT;
    yvex_error_clear(err);
    return YVEX_OK;
}

int main(int argc, char **argv)
{
    yvex_runtime_model_open_request request = {0};
    yvex_runtime_model_failure failure = {0};
    yvex_runtime_model *model = NULL;
    yvex_runtime_sampling_policy policy = {
        .schema_version = YVEX_RUNTIME_SAMPLING_SCHEMA_V1,
        .strategy = YVEX_SAMPLING_STRATEGY_GREEDY,
        .temperature = 1.0, .top_p = 1.0, .typical_p = 1.0};
    live_generation production;
    live_manual manual;
    yvex_backend_kind backend;
    yvex_error err;
    unsigned long long maximum_tokens, seed;
    const char *step = "arguments";
    int rc;
    if (argc != 7 ||
        (strcmp(argv[3], "cpu") != 0 && strcmp(argv[3], "cuda") != 0) ||
        (strcmp(argv[4], "greedy") != 0 && strcmp(argv[4], "stochastic") != 0) ||
        sscanf(argv[5], "%llu", &seed) != 1 ||
        sscanf(argv[6], "%llu", &maximum_tokens) != 1 ||
        !maximum_tokens || maximum_tokens > LIVE_GENERATION_MAX_TOKENS) {
        fprintf(stderr, "usage: %s ARTIFACT BINDING cpu|cuda greedy|stochastic SEED MAX_TOKENS\n", argv[0]);
        return 2;
    }
    backend = strcmp(argv[3], "cuda") == 0
                  ? YVEX_BACKEND_KIND_CUDA : YVEX_BACKEND_KIND_CPU;
    if (strcmp(argv[4], "stochastic") == 0) {
        policy.strategy = YVEX_SAMPLING_STRATEGY_STOCHASTIC;
        policy.temperature = 0.8;
        policy.top_k = 50ull;
        policy.top_p = 0.95;
        policy.min_p = 0.05;
        policy.typical_p = 0.9;
        policy.seed_present = 1;
        policy.seed = seed;
    }
    request.artifact_path = argv[1];
    request.runtime_binding_path = argv[2];
    request.target_id = "deepseek4-v4-flash";
    rc = yvex_runtime_model_open(&model, &request, &failure, &err);
    if (rc == YVEX_OK) { step = "production"; rc = live_production(
        model, backend, policy, maximum_tokens, &production, &err); }
    if (rc == YVEX_OK) { step = "manual"; rc = live_manual_execute(
        model, backend, policy, maximum_tokens, &manual, &err); }
    if (rc == YVEX_OK) { step = "composition-compare"; rc = live_compare(
        &production, &manual, &err); }
    if (rc == YVEX_OK) { step = "mutation"; rc = live_mutation_proof(
        &production, &err); }
    if (rc == YVEX_OK) { step = "lifecycle"; rc = live_lifecycle_proof(
        model, backend, policy, &err); }
    if (rc == YVEX_OK && backend == YVEX_BACKEND_KIND_CUDA &&
        policy.strategy == YVEX_SAMPLING_STRATEGY_GREEDY &&
        maximum_tokens >= 3ull) {
        step = "stop-capacity";
        rc = live_stop_capacity_proof(
            model, backend, policy, production.tokens[0].sampled_token_id,
            &err);
    }
    if (rc == YVEX_OK && backend == YVEX_BACKEND_KIND_CUDA &&
        policy.strategy == YVEX_SAMPLING_STRATEGY_GREEDY &&
        maximum_tokens >= 3ull) {
        step = "partial-progress";
        rc = live_partial_progress_proof(model, backend, policy, &err);
    }
    if (rc != YVEX_OK) live_failure(step, rc, &err);
    else {
        printf("generation_backend=%s strategy=%s prompt_tokens=%llu "
               "sampled=%llu committed=%llu decode_inputs_match=pass "
               "teacher_forced_tail=0 text_bytes=%llu text_digest=%s "
               "final_position=%llu state_digest=%s stop=%s "
               "manual_composition=pass mutation_matrix=pass "
               "cancellation_close=pass stop_capacity=%s partial_progress=%s tokens=",
               argv[3], argv[4], production.result.prompt_token_count,
               production.result.sampled_token_count,
               production.result.model_committed_token_count,
               production.result.generated_text_bytes,
               production.result.generated_text_digest,
               production.result.final_position,
               production.result.final_persistent_state_digest,
               yvex_runtime_generation_stop_reason_name(
                   production.result.stop_reason),
               backend == YVEX_BACKEND_KIND_CUDA &&
                       policy.strategy == YVEX_SAMPLING_STRATEGY_GREEDY &&
                       maximum_tokens >= 3ull
                   ? "pass" : "not-run",
               backend == YVEX_BACKEND_KIND_CUDA &&
                       policy.strategy == YVEX_SAMPLING_STRATEGY_GREEDY &&
                       maximum_tokens >= 3ull
                   ? "pass" : "not-run");
        for (unsigned long long index = 0ull;
             index < production.result.sampled_token_count; ++index)
            printf("%s%u", index ? "," : "", production.tokens[index].sampled_token_id);
        printf("\n");
    }
    yvex_runtime_model_close(&model);
    return rc == YVEX_OK ? 0 : 1;
}
