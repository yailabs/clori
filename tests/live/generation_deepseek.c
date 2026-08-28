/*
 * Exercises text-to-text autoregressive composition and exact partial/publication evidence.
 * Every ordinary sampled ID is the exact next decode input and no teacher-forced tail exists.
 * Test-only consumer of production lower-owner APIs over isolated CPU/CUDA sessions.
 */
#include <yvex/internal/backend.h>
#include <yvex/internal/decode.h>
#include <yvex/internal/core.h>
#include <yvex/internal/deployment.h>
#include <yvex/internal/engine_scheduler.h>
#include <yvex/internal/execution_batch.h>
#include <yvex/internal/generation.h>
#include <yvex/internal/logits.h>
#include <yvex/internal/moe.h>
#include <yvex/internal/runtime.h>
#include <yvex/internal/sampling.h>
#include <yvex/internal/transformer.h>
#include <yvex/tokenizer.h>

#include <build_commit.h>

#include <stddef.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LIVE_GENERATION_MAX_TOKENS 16ull
#define LIVE_GENERATION_TEXT_BYTES 1024ull

typedef struct {
    yvex_runtime_generation_plan_summary plan;
    yvex_runtime_generation_result result;
    yvex_runtime_generation_evidence evidence;
    yvex_runtime_state_residency_summary state_residency;
    yvex_engine_scheduler_summary scheduler;
    yvex_runtime_generation_token_result tokens[LIVE_GENERATION_MAX_TOKENS];
    unsigned char text[LIVE_GENERATION_TEXT_BYTES];
    char semantic_state_digest[YVEX_SHA256_HEX_CAP];
} live_generation;

typedef struct {
    unsigned int tokens[LIVE_GENERATION_MAX_TOKENS];
    unsigned long long token_text_bytes[LIVE_GENERATION_MAX_TOKENS];
    unsigned long long sampled, committed, text_bytes, final_position;
    unsigned char text[LIVE_GENERATION_TEXT_BYTES];
    char semantic_state_digest[YVEX_SHA256_HEX_CAP];
    char rng_identity[YVEX_SHA256_HEX_CAP];
} live_manual;

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    int entered, release;
} live_cancel_gate;

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    unsigned int participants, ready;
    int released, aborted;
} live_start_gate;

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

typedef struct {
    atomic_int cancel;
    unsigned long long phase_counts[7];
} live_speculation_cancel;

typedef struct {
    unsigned long long prompt_count, proposed, verifications;
    unsigned long long accepted, rejected, maximum_accepted_prefix;
} live_acceptance_corpus;

typedef struct {
    yvex_model_engine *model;
    yvex_runtime_sampling_policy policy;
    live_start_gate *gate;
    live_generation *result;
    unsigned long long maximum_tokens;
    int rc;
    yvex_error err;
} live_scheduled_thread;

static int live_start_gate_wait(live_start_gate *gate, yvex_error *err)
{
    int aborted;
    if (!gate || pthread_mutex_lock(&gate->mutex) != 0) {
        yvex_error_set(err, YVEX_ERR_STATE, "generation_live.scheduler",
                       "scheduled generation start gate is unavailable");
        return YVEX_ERR_STATE;
    }
    gate->ready++;
    if (gate->ready == gate->participants) {
        gate->released = 1;
        (void)pthread_cond_broadcast(&gate->condition);
    }
    while (!gate->released && !gate->aborted)
        (void)pthread_cond_wait(&gate->condition, &gate->mutex);
    aborted = gate->aborted;
    (void)pthread_mutex_unlock(&gate->mutex);
    if (aborted) {
        yvex_error_set(err, YVEX_ERR_STATE, "generation_live.scheduler",
                       "one scheduled peer failed before execution admission");
        return YVEX_ERR_STATE;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

static void live_start_gate_abort(live_start_gate *gate)
{
    if (!gate || pthread_mutex_lock(&gate->mutex) != 0) return;
    gate->aborted = 1;
    (void)pthread_cond_broadcast(&gate->condition);
    (void)pthread_mutex_unlock(&gate->mutex);
}

static int live_generation_turn_run(
    yvex_runtime_generation_context *context,
    const yvex_runtime_generation_turn_request *turn,
    yvex_runtime_generation_token_result *tokens, unsigned long long token_capacity,
    unsigned char *text, unsigned long long text_capacity,
    yvex_runtime_generation_result *result, yvex_error *err)
{
    int active = 0, complete = 0, rc = yvex_runtime_generation_turn_begin(
        context, turn, tokens, token_capacity, text, text_capacity, result, err);
    active = rc == YVEX_OK;
    while (rc == YVEX_OK && !complete)
        rc = yvex_runtime_generation_turn_advance(context, 1ull, &complete, err);
    if (active) rc = yvex_runtime_generation_turn_finish(context, err);
    return rc;
}

static void live_failure(const char *step, int rc, const yvex_error *err)
{
    fprintf(stderr, "generation_live step=%s status=%d where=%s reason=%s\n",
            step, rc, err ? yvex_error_where(err) : "",
            err ? yvex_error_message(err) : "");
}

static void live_failure_primary(const live_generation *primary)
{
    const yvex_expert_worklist_observation *worklists;
    unsigned long long index;
    int width_present = 0;
    if (!primary || !primary->result.completed) return;
    worklists = &primary->evidence.expert_worklists;
    fprintf(stderr,
            "generation_live primary mode=%u sampled=%llu committed=%llu cycles=%llu "
            "proposed=%llu verified=%llu accepted=%llu rejected=%llu tokens=",
            (unsigned int)primary->result.execution_mode,
            primary->result.sampled_token_count,
            primary->result.model_committed_token_count,
            primary->result.draft_cycle_count,
            primary->result.proposed_token_count,
            primary->result.target_verification_count,
            primary->result.accepted_draft_token_count,
            primary->result.rejected_draft_token_count);
    for (index = 0ull; index < primary->result.sampled_token_count; ++index)
        fprintf(stderr, "%s%u", index ? "," : "",
                primary->tokens[index].sampled_token_id);
    fprintf(stderr,
            " worklists=%llu pairs=%llu buckets=%llu max_bucket=%llu tc=%llu/%llu "
            "narrow=%llu widths=",
            worklists->worklist_count, worklists->pair_count,
            worklists->bucket_count, worklists->maximum_bucket_population,
            worklists->tensor_core_executed_pairs,
            worklists->tensor_core_eligible_pairs, worklists->narrow_pairs);
    for (index = 1ull; index < YVEX_EXPERT_WORKLIST_HISTOGRAM_CAP; ++index)
        if (worklists->width_histogram[index]) {
            fprintf(stderr, "%s%llu:%llu", width_present ? "," : "",
                    index, worklists->width_histogram[index]);
            width_present = 1;
        }
    fputc('\n', stderr);
}

static int live_scope_state(const yvex_runtime_execution_session *session,
                            int draft,
                            yvex_graph_attention_state_summary *summary,
                            yvex_error *err)
{
    const yvex_runtime_session_view *view = yvex_runtime_session_view_get(session);
    const yvex_attention_state_provider *provider =
        !view ? NULL : draft ? view->draft_attention_state_provider
                            : view->attention_state_provider;
    if (!provider || !provider->summary)
        return YVEX_ERR_STATE;
    return provider->summary(provider->context, summary, err);
}

static int live_state(const yvex_runtime_execution_session *session,
                      yvex_graph_attention_state_summary *summary,
                      yvex_error *err)
{
    return live_scope_state(session, 0, summary, err);
}

static int live_semantic_state_digest(
    const yvex_runtime_execution_session *session,
    char digest[YVEX_SHA256_HEX_CAP], yvex_error *err)
{
    const yvex_runtime_session_view *view = yvex_runtime_session_view_get(session);
    const yvex_attention_state_provider *provider =
        view ? view->attention_state_provider : NULL;
    yvex_graph_attention_state_summary summary;
    yvex_sha256 hash;
    unsigned char bytes[YVEX_SHA256_DIGEST_BYTES];
    char layer_identity[YVEX_SHA256_HEX_CAP];
    unsigned long long layer;
    int rc;

    if (!provider || !provider->summary || !provider->identity || !digest) {
        yvex_error_set(err, YVEX_ERR_STATE, "generation_live.state",
                       "persistent state identity provider is unavailable");
        return YVEX_ERR_STATE;
    }
    rc = provider->summary(provider->context, &summary, err);
    if (rc != YVEX_OK) return rc;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.live.generation.state-semantics.v1") ||
        !yvex_sha256_update_u64(&hash, summary.layer_count)) {
        yvex_error_set(err, YVEX_ERR_STATE, "generation_live.state",
                       "persistent state digest initialization failed");
        return YVEX_ERR_STATE;
    }
    for (layer = 0ull; layer < summary.layer_count; ++layer) {
        rc = provider->identity(provider->context, layer, layer_identity, err);
        if (rc != YVEX_OK) return rc;
        if (!yvex_sha256_update_u64(&hash, layer) ||
            !yvex_sha256_update_text(&hash, layer_identity)) {
            yvex_error_set(err, YVEX_ERR_STATE, "generation_live.state",
                           "persistent layer identity could not be aggregated");
            return YVEX_ERR_STATE;
        }
    }
    if (!yvex_sha256_final(&hash, bytes)) {
        yvex_error_set(err, YVEX_ERR_STATE, "generation_live.state",
                       "persistent state digest finalization failed");
        return YVEX_ERR_STATE;
    }
    yvex_sha256_hex(bytes, digest);
    yvex_error_clear(err);
    return YVEX_OK;
}

static int live_cancel_at_position(void *opaque)
{
    live_position_cancel *cancel = opaque;
    return !cancel || atomic_load_explicit(&cancel->cancel, memory_order_acquire);
}

static int live_cancel_flag(void *opaque)
{
    live_speculation_cancel *cancel = opaque;
    return !cancel || atomic_load_explicit(&cancel->cancel, memory_order_acquire);
}

static int live_cancel_verification(
    void *opaque, const yvex_runtime_speculation_progress *progress,
    yvex_error *err)
{
    live_speculation_cancel *cancel = opaque;
    if (!cancel || !progress ||
        progress->schema_version != YVEX_RUNTIME_GENERATION_SCHEMA_V3 ||
        progress->kind > YVEX_SPECULATION_PROGRESS_CYCLE_COMMITTED)
        return YVEX_ERR_INVALID_ARG;
    cancel->phase_counts[progress->kind]++;
    if (progress->kind == YVEX_SPECULATION_PROGRESS_VERIFICATION_STARTED)
        atomic_store_explicit(&cancel->cancel, 1, memory_order_release);
    yvex_error_clear(err);
    return YVEX_OK;
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
        .schema_version = YVEX_RUNTIME_GENERATION_SCHEMA_V3,
        .kind = YVEX_GENERATION_INPUT_TEXT, .text = prompt,
        .text_bytes = sizeof(prompt) - 1ull,
        .encode_options = {.maximum_tokens = 8ull}};
    thread->rc = yvex_runtime_generation_execute(
        thread->context, &request, &token, 1ull, text, sizeof(text),
        &result, NULL, &thread->err);
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

static int live_production_request(
    yvex_model_engine *model, yvex_backend_kind backend,
    yvex_runtime_generation_mode mode, yvex_runtime_sampling_policy policy,
    const yvex_runtime_generation_request *request,
    unsigned long long context_capacity,
    unsigned long long maximum_tokens, unsigned long long concurrent_sequences,
    live_start_gate *start_gate, live_generation *out, yvex_error *err)
{
    yvex_runtime_session_open_request session_options = {.backend = backend};
    yvex_runtime_generation_options options = {
        .schema_version = YVEX_RUNTIME_GENERATION_SCHEMA_V5,
        .context_capacity = context_capacity, .prefill_chunk_tokens = 4ull,
        .maximum_output_bytes = LIVE_GENERATION_TEXT_BYTES - 1ull,
        .trace_policy = YVEX_RUNTIME_TRACE_SUMMARY};
    yvex_model_engine_failure failure = {0};
    yvex_runtime_execution_session *session = NULL;
    yvex_runtime_generation_context *context = NULL;
    yvex_error cleanup;
    int rc, close_rc;
    memset(out, 0, sizeof(*out));
    options.backend = backend;
    options.mode = mode;
    options.maximum_new_tokens = maximum_tokens;
    options.sampling_policy = policy;
    options.concurrent_sequences = concurrent_sequences;
    options.continuous_batching = concurrent_sequences > 1ull;
    rc = yvex_runtime_session_open(&session, model, &session_options,
                                   &failure, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_generation_context_open(
            &context, model, session, &options, err);
    if (rc == YVEX_OK)
        out->plan = *yvex_runtime_generation_plan_summary_get(context);
    if (rc == YVEX_OK && start_gate)
        rc = live_start_gate_wait(start_gate, err);
    else if (rc != YVEX_OK && start_gate)
        live_start_gate_abort(start_gate);
    if (rc == YVEX_OK)
        rc = yvex_runtime_generation_execute(
            context, request, out->tokens, LIVE_GENERATION_MAX_TOKENS,
            out->text, sizeof(out->text), &out->result, &out->evidence, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_generation_result_validate(
            &out->plan, out->tokens, LIVE_GENERATION_MAX_TOKENS,
            out->text, sizeof(out->text), &out->result, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_generation_evidence_validate(
            &out->plan, &out->evidence, err);
    if (rc == YVEX_OK) {
        const yvex_runtime_session_view *view = yvex_runtime_session_view_get(session);
        rc = view && view->state_residency
                 ? yvex_runtime_state_residency_summary_copy(
                       view->state_residency, &out->state_residency, err)
                 : YVEX_ERR_STATE;
        if (rc != YVEX_OK && !yvex_error_is_set(err))
            yvex_error_set(err, rc, "generation_live",
                           "generation state residency evidence is unavailable");
    }
    if (rc == YVEX_OK)
        rc = live_semantic_state_digest(
            session, out->semantic_state_digest, err);
    if (rc == YVEX_OK)
        rc = yvex_model_engine_scheduler_summary_copy(
            model, &out->scheduler, err);
    if (rc == YVEX_OK && !options.continuous_batching &&
        out->scheduler.physical_batches) {
        rc = YVEX_ERR_STATE;
        yvex_error_set(
            err, rc, "generation_live.scheduler",
            "single-sequence execution entered the compatible physical scheduler");
    }
    if (rc == YVEX_OK && backend == YVEX_BACKEND_KIND_CUDA &&
        out->evidence.profile.counters[
            YVEX_RUNTIME_PROFILE_FULL_ARRAY_HOST_SCAN_BYTES]) {
        rc = YVEX_ERR_FORMAT;
        yvex_error_set(
            err, rc, "generation_live",
            "production CUDA generation performed a full-array host scan");
    }
    if (rc == YVEX_OK && backend == YVEX_BACKEND_KIND_CUDA &&
        (!out->state_residency.cuda_ready ||
         !out->state_residency.device_stage_bytes ||
         !out->state_residency.device_stage_count ||
         (out->result.decode_step_count &&
          (!out->state_residency.copy_bytes || !out->state_residency.copy_count)))) {
        rc = YVEX_ERR_STATE;
        yvex_error_set(err, rc, "generation_live",
                       "production CUDA generation did not retain device-native state banks");
    }
    if (rc == YVEX_OK && backend == YVEX_BACKEND_KIND_CUDA &&
        mode == YVEX_GENERATION_MODE_TARGET_ONLY &&
        ((out->state_residency.paged &&
          (!out->state_residency.page_granularity ||
           out->state_residency.host_bytes ||
           !out->state_residency.virtual_device_bytes ||
           out->state_residency.device_bytes !=
               out->state_residency.page_commit_count *
                   out->state_residency.page_granularity)) ||
         (!out->state_residency.paged &&
          (out->state_residency.upload_bytes != out->state_residency.device_bytes ||
           out->state_residency.upload_count !=
               2ull * out->state_residency.layer_count)))) {
        rc = YVEX_ERR_STATE;
        yvex_error_setf(
            err, rc, "generation_live",
            "target-only CUDA state residency accounting is inconsistent "
            "(paged=%d uploads=%llu bytes=%llu resident=%llu virtual=%llu "
            "pages=%llu granularity=%llu)",
            out->state_residency.paged,
            out->state_residency.upload_count,
            out->state_residency.upload_bytes,
            out->state_residency.device_bytes,
            out->state_residency.virtual_device_bytes,
            out->state_residency.page_commit_count,
            out->state_residency.page_granularity);
    }
    yvex_error_clear(&cleanup);
    close_rc = yvex_runtime_generation_context_close(&context, &cleanup);
    if (rc == YVEX_OK && close_rc != YVEX_OK) { rc = close_rc; *err = cleanup; }
    yvex_error_clear(&cleanup);
    close_rc = yvex_runtime_session_close(&session, &cleanup);
    if (rc == YVEX_OK && close_rc != YVEX_OK) { rc = close_rc; *err = cleanup; }
    return rc;
}

static int live_production_prompt(
    yvex_model_engine *model, yvex_backend_kind backend,
    yvex_runtime_generation_mode mode, yvex_runtime_sampling_policy policy,
    const unsigned char *prompt, unsigned long long prompt_bytes,
    unsigned long long maximum_tokens, live_generation *out, yvex_error *err)
{
    yvex_runtime_generation_request request = {
        .schema_version = YVEX_RUNTIME_GENERATION_SCHEMA_V3,
        .kind = YVEX_GENERATION_INPUT_TEXT, .text = prompt,
        .text_bytes = prompt_bytes,
        .encode_options = {.maximum_tokens = 16ull}};
    return live_production_request(model, backend, mode, policy, &request,
                                   64u, maximum_tokens, 1ull, NULL, out, err);
}

static int live_production(yvex_model_engine *model,
                           yvex_backend_kind backend,
                           yvex_runtime_generation_mode mode,
                           yvex_runtime_sampling_policy policy,
                           unsigned long long maximum_tokens,
                           live_generation *out, yvex_error *err)
{
    static const unsigned char prompt[] = "Hi";
    return live_production_prompt(
        model, backend, mode, policy, prompt, sizeof(prompt) - 1ull,
        maximum_tokens, out, err);
}

static int live_boundary_execute(
    yvex_model_engine *model, yvex_backend_kind backend,
    yvex_runtime_sampling_policy policy, unsigned long long context_capacity,
    const unsigned int *stop_token, live_generation *out, yvex_error *err)
{
    static const unsigned char prompt[] = "Hi";
    yvex_runtime_session_open_request session_options = {.backend = backend};
    yvex_runtime_generation_options options = {
        .schema_version = YVEX_RUNTIME_GENERATION_SCHEMA_V5,
        .context_capacity = context_capacity, .prefill_chunk_tokens = 1ull,
        .maximum_new_tokens = 1ull,
        .maximum_output_bytes = LIVE_GENERATION_TEXT_BYTES - 1ull,
        .trace_policy = YVEX_RUNTIME_TRACE_SUMMARY};
    yvex_runtime_generation_request request = {
        .schema_version = YVEX_RUNTIME_GENERATION_SCHEMA_V3,
        .kind = YVEX_GENERATION_INPUT_TEXT, .text = prompt,
        .text_bytes = sizeof(prompt) - 1ull,
        .encode_options = {.maximum_tokens = 8ull}};
    yvex_model_engine_failure failure = {0};
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
            out->text, sizeof(out->text), &out->result, &out->evidence, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_generation_result_validate(
            &out->plan, out->tokens, LIVE_GENERATION_MAX_TOKENS,
            out->text, sizeof(out->text), &out->result, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_generation_evidence_validate(
            &out->plan, &out->evidence, err);
    yvex_error_clear(&cleanup);
    close_rc = yvex_runtime_generation_context_close(&context, &cleanup);
    if (rc == YVEX_OK && close_rc != YVEX_OK) { rc = close_rc; *err = cleanup; }
    yvex_error_clear(&cleanup);
    close_rc = yvex_runtime_session_close(&session, &cleanup);
    if (rc == YVEX_OK && close_rc != YVEX_OK) { rc = close_rc; *err = cleanup; }
    return rc;
}

static int live_stop_capacity_proof(
    yvex_model_engine *model, yvex_backend_kind backend,
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
    yvex_model_engine *model, yvex_backend_kind backend,
    yvex_runtime_sampling_policy policy, int cancel_after_commit,
    live_generation *out, int *execution_status, yvex_error *err)
{
    static const unsigned char prompt[] = "Hi";
    yvex_runtime_session_open_request session_options = {.backend = backend};
    yvex_runtime_generation_options options = {
        .schema_version = YVEX_RUNTIME_GENERATION_SCHEMA_V5,
        .context_capacity = 8ull, .prefill_chunk_tokens = 1ull,
        .maximum_new_tokens = 2ull,
        .trace_policy = YVEX_RUNTIME_TRACE_SUMMARY};
    yvex_runtime_generation_request request = {
        .schema_version = YVEX_RUNTIME_GENERATION_SCHEMA_V3,
        .kind = YVEX_GENERATION_INPUT_TEXT, .text = prompt,
        .text_bytes = sizeof(prompt) - 1ull,
        .encode_options = {.maximum_tokens = 8ull}};
    yvex_model_engine_failure failure = {0};
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
            out->text, sizeof(out->text), &out->result, &out->evidence, err);
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
    yvex_model_engine *model, yvex_backend_kind backend,
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

static int live_lifecycle_proof(yvex_model_engine *model,
                                yvex_backend_kind backend,
                                yvex_runtime_sampling_policy policy,
                                yvex_error *err)
{
    static const unsigned char prompt[] = "Hi";
    yvex_runtime_session_open_request session_options = {.backend = backend};
    yvex_runtime_generation_options options = {
        .schema_version = YVEX_RUNTIME_GENERATION_SCHEMA_V5,
        .context_capacity = 8ull, .prefill_chunk_tokens = 8ull,
        .maximum_new_tokens = 1ull,
        .maximum_output_bytes = LIVE_GENERATION_TEXT_BYTES - 1ull,
        .trace_policy = YVEX_RUNTIME_TRACE_SUMMARY};
    yvex_runtime_generation_request request = {
        .schema_version = YVEX_RUNTIME_GENERATION_SCHEMA_V3,
        .kind = YVEX_GENERATION_INPUT_TEXT, .text = prompt,
        .text_bytes = sizeof(prompt) - 1ull,
        .encode_options = {.maximum_tokens = 8ull}};
    yvex_runtime_generation_token_result token;
    yvex_runtime_generation_result result;
    yvex_model_engine_failure failure = {0};
    yvex_runtime_execution_session *session = NULL;
    yvex_runtime_generation_context *context = NULL;
    yvex_graph_attention_state_summary state;
    live_cancel_gate gate;
    live_execute_thread execute = {0};
    live_close_thread close = {0};
    pthread_t execute_id, close_id;
    unsigned char text[LIVE_GENERATION_TEXT_BYTES];
    unsigned long long refusal_deadline = 0ull;
    yvex_error contender = {0};
    int contender_rc = YVEX_OK;
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
        else {
            close_created = 1;
            refusal_deadline = yvex_core_monotonic_ns() + 5000000000ull;
        }
    }
    while (rc == YVEX_OK && close_created && !closing_refused &&
           yvex_core_monotonic_ns() < refusal_deadline) {
        if (!atomic_load_explicit(&close.started, memory_order_acquire)) {
            sched_yield();
            continue;
        }
        contender_rc = yvex_runtime_generation_execute(
            context, &request, &token, 1ull, text, sizeof(text),
            &result, NULL, &contender);
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
         close.context || !closing_refused)) {
        yvex_error_setf(
            err, YVEX_ERR_FORMAT, "generation_live.lifecycle",
            "close drain mismatch execute=%d close=%d owner=%s refusal=%d "
            "contender=%d execute_reason=%s close_reason=%s contender_reason=%s",
            execute.rc, close.rc, close.context ? "retained" : "released",
            closing_refused, contender_rc, yvex_error_message(&execute.err),
            yvex_error_message(&close.err), yvex_error_message(&contender));
        rc = YVEX_ERR_FORMAT;
    }
    if (rc == YVEX_OK) rc = live_state(session, &state, err);
    if (rc == YVEX_OK &&
        (state.next_position || state.committed_sequence_length ||
         state.transaction_active)) {
        yvex_error_setf(
            err, YVEX_ERR_FORMAT, "generation_live.lifecycle",
            "cancelled close published state position=%llu length=%llu transaction=%d",
            state.next_position, state.committed_sequence_length,
            state.transaction_active);
        rc = YVEX_ERR_FORMAT;
    }
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

static int live_dspark_cancellation_proof(
    yvex_model_engine *model, yvex_backend_kind backend,
    yvex_runtime_sampling_policy policy, yvex_error *err)
{
    static const unsigned char prompt[] = "Hi";
    yvex_runtime_session_open_request session_options = {.backend = backend};
    yvex_runtime_generation_options options = {
        .schema_version = YVEX_RUNTIME_GENERATION_SCHEMA_V5,
        .backend = YVEX_BACKEND_KIND_CPU,
        .mode = YVEX_GENERATION_MODE_SPECULATIVE,
        .context_capacity = 32ull,
        .prefill_chunk_tokens = 8ull,
        .maximum_new_tokens = 8ull,
        .maximum_output_bytes = LIVE_GENERATION_TEXT_BYTES - 1ull,
        .trace_policy = YVEX_RUNTIME_TRACE_SUMMARY};
    yvex_runtime_generation_request request = {
        .schema_version = YVEX_RUNTIME_GENERATION_SCHEMA_V3,
        .kind = YVEX_GENERATION_INPUT_TEXT,
        .text = prompt,
        .text_bytes = sizeof(prompt) - 1ull,
        .encode_options = {.maximum_tokens = 8ull}};
    yvex_runtime_generation_turn_request turn = {
        .schema_version = YVEX_RUNTIME_GENERATION_TURN_SCHEMA_V1,
        .prompt = &request,
        .maximum_new_tokens = 8ull};
    yvex_model_engine_failure failure = {0};
    yvex_runtime_execution_session *session = NULL;
    yvex_runtime_generation_context *context = NULL;
    yvex_runtime_generation_plan_summary plan = {0};
    yvex_runtime_generation_token_result tokens[LIVE_GENERATION_MAX_TOKENS];
    unsigned int prompt_tokens[32];
    yvex_runtime_generation_result result;
    yvex_graph_attention_state_summary target_state, draft_state;
    live_speculation_cancel cancel = {0};
    unsigned char text[LIVE_GENERATION_TEXT_BYTES];
    yvex_error primary, cleanup;
    int rc, execute_rc = YVEX_OK, close_rc;

    yvex_error_clear(&primary);
    options.backend = backend;
    options.sampling_policy = policy;
    options.cancel_requested = live_cancel_flag;
    options.cancel_context = &cancel;
    atomic_init(&cancel.cancel, 0);
    turn.prompt_token_ids = prompt_tokens;
    turn.prompt_token_capacity = sizeof(prompt_tokens) / sizeof(prompt_tokens[0]);
    turn.speculation_progress_sink = live_cancel_verification;
    turn.speculation_progress_context = &cancel;
    rc = yvex_runtime_session_open(&session, model, &session_options,
                                   &failure, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_generation_context_open(
            &context, model, session, &options, err);
    if (rc == YVEX_OK)
        plan = *yvex_runtime_generation_plan_summary_get(context);
    if (rc == YVEX_OK) {
        execute_rc = live_generation_turn_run(
            context, &turn, tokens, LIVE_GENERATION_MAX_TOKENS,
            text, sizeof(text), &result, err);
        primary = *err;
        rc = execute_rc == YVEX_ERR_CANCELLED ? YVEX_OK : YVEX_ERR_FORMAT;
    }
    if (rc == YVEX_OK)
        rc = yvex_runtime_generation_result_validate(
            &plan, tokens, LIVE_GENERATION_MAX_TOKENS, text, sizeof(text),
            &result, err);
    if (rc == YVEX_OK &&
        (!result.cancelled || result.completed ||
         result.stop_reason != YVEX_GENERATION_STOP_CANCELLED ||
         result.model_committed_token_count || result.sampled_token_count ||
         result.generated_text_bytes || result.draft_cycle_count != 1ull ||
         result.draft_forward_count != 1ull ||
         result.target_verification_count ||
         result.proposed_token_count != 5ull ||
         result.selected_verification_token_count == 0ull ||
         result.accepted_draft_token_count ||
         result.rejected_draft_token_count ||
         result.discarded_draft_token_count != 5ull ||
         result.confidence_logit_count != 5ull ||
         cancel.phase_counts[YVEX_SPECULATION_PROGRESS_DRAFT_STARTED] != 1ull ||
         cancel.phase_counts[YVEX_SPECULATION_PROGRESS_DRAFT_COMPLETED] != 1ull ||
         cancel.phase_counts[
             YVEX_SPECULATION_PROGRESS_VERIFICATION_STARTED] != 1ull ||
         cancel.phase_counts[
             YVEX_SPECULATION_PROGRESS_VERIFICATION_COMPLETED] ||
         cancel.phase_counts[YVEX_SPECULATION_PROGRESS_PREFIX_ACCEPTED] ||
         cancel.phase_counts[YVEX_SPECULATION_PROGRESS_CYCLE_COMMITTED]))
        rc = YVEX_ERR_FORMAT;
    if (rc == YVEX_OK) rc = live_scope_state(session, 0, &target_state, err);
    if (rc == YVEX_OK) rc = live_scope_state(session, 1, &draft_state, err);
    if (rc == YVEX_OK &&
        (target_state.next_position != result.prompt_token_count ||
         draft_state.next_position != result.prompt_token_count ||
         target_state.transaction_active || draft_state.transaction_active))
        rc = YVEX_ERR_STATE;
    yvex_error_clear(&cleanup);
    close_rc = yvex_runtime_generation_context_close(&context, &cleanup);
    if (rc == YVEX_OK && close_rc != YVEX_OK) {
        rc = close_rc;
        *err = cleanup;
    }
    if (rc == YVEX_OK)
        rc = yvex_runtime_session_reset_persistent_state(
            session, &failure, err);
    if (rc == YVEX_OK) rc = live_scope_state(session, 0, &target_state, err);
    if (rc == YVEX_OK) rc = live_scope_state(session, 1, &draft_state, err);
    if (rc == YVEX_OK &&
        (target_state.next_position || draft_state.next_position ||
         target_state.transaction_active || draft_state.transaction_active))
        rc = YVEX_ERR_STATE;
    options.cancel_requested = NULL;
    options.cancel_context = NULL;
    turn.maximum_new_tokens = 4ull;
    turn.speculation_progress_sink = NULL;
    turn.speculation_progress_context = NULL;
    if (rc == YVEX_OK)
        rc = yvex_runtime_generation_context_open(
            &context, model, session, &options, err);
    if (rc == YVEX_OK)
        plan = *yvex_runtime_generation_plan_summary_get(context);
    if (rc == YVEX_OK)
        rc = live_generation_turn_run(
            context, &turn, tokens, LIVE_GENERATION_MAX_TOKENS,
            text, sizeof(text), &result, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_generation_result_validate(
            &plan, tokens, LIVE_GENERATION_MAX_TOKENS, text, sizeof(text),
            &result, err);
    if (rc == YVEX_OK &&
        (!result.completed || result.execution_mode != YVEX_GENERATION_MODE_SPECULATIVE ||
         !result.model_committed_token_count || !result.draft_cycle_count ||
         !result.target_verification_count))
        rc = YVEX_ERR_FORMAT;
    yvex_error_clear(&cleanup);
    close_rc = yvex_runtime_generation_context_close(&context, &cleanup);
    if (rc == YVEX_OK && close_rc != YVEX_OK) {
        rc = close_rc;
        *err = cleanup;
    }
    yvex_error_clear(&cleanup);
    close_rc = yvex_runtime_session_close(&session, &cleanup);
    if (rc == YVEX_OK && close_rc != YVEX_OK) {
        rc = close_rc;
        *err = cleanup;
    }
    if (rc == YVEX_OK) yvex_error_clear(err);
    else if (!yvex_error_message(err)[0]) *err = primary;
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
    if (rc == YVEX_OK) {
        if (!out->sampled || out->sampled > LIVE_GENERATION_MAX_TOKENS)
            rc = YVEX_ERR_BOUNDS;
        else {
            out->token_text_bytes[out->sampled - 1ull] = fragment.byte_count;
            out->text_bytes = next;
        }
    }
    yvex_tokenizer_fragment_clear(&fragment);
    return rc;
}

/* The independent composition must execute the production-admitted physical profile.  A
 * profile-less transformer is an intentionally degraded reference regime and is not an exact
 * oracle for the selected CUDA generation path. */
static int live_execution_profile(
    yvex_model_engine *model, yvex_runtime_execution_session *session,
    yvex_backend_kind backend, yvex_sampling_strategy strategy,
    yvex_runtime_execution_profile *profile, yvex_error *err)
{
    const yvex_model_engine_view *model_view = yvex_model_engine_view_get(model);
    const yvex_runtime_session_view *session_view = yvex_runtime_session_view_get(session);
    const yvex_runtime_binding_summary *binding = model_view ? model_view->binding : NULL;
    yvex_backend_cuda_attention_graph_summary cuda = {0};
    yvex_backend_cuda_graph_capability graph = {0};
    yvex_runtime_session_summary summary;
    yvex_runtime_execution_profile_request request = {0};
    yvex_execution_workload_profile workload = {0};
    const char *kernel_bundle = YVEX_BUILD_IDENTITY;
    int rc;

    if (!model_view || !session_view || !session_view->backend || !binding || !profile ||
        yvex_runtime_session_summary_copy(session, &summary, err) != YVEX_OK ||
        !summary.engine_generation ||
        !yvex_sha256_hex_valid(summary.engine_specialization_identity)) {
        if (!yvex_error_is_set(err))
            yvex_error_set(err, YVEX_ERR_STATE, "generation_live.profile",
                           "runtime workload profile owners are unavailable");
        return yvex_error_is_set(err) ? yvex_error_code(err) : YVEX_ERR_STATE;
    }
    workload.schema_version = YVEX_EXECUTION_WORKLOAD_PROFILE_SCHEMA_V1;
    workload.kind = YVEX_EXECUTION_WORKLOAD_INTERACTIVE_LATENCY;
    workload.minimum_session_context = workload.requested_session_context = 64ull;
    workload.concurrent_sequences = 1ull;
    workload.logical_batch_tokens = workload.prefill_chunk_tokens = 8ull;
    workload.attention_microbatch_rows = workload.moe_row_tile = 8ull;
    workload.output_head_rows = 1ull;
    workload.system_reserve_bytes = YVEX_EXECUTION_MINIMUM_SYSTEM_RESERVE;
    workload.latency_priority = 1;
    yvex_core_text_copy(workload.name, sizeof(workload.name), "manual-generation");
    if (yvex_execution_workload_profile_seal(&workload, err) != YVEX_OK)
        return yvex_error_code(err);
    if (backend == YVEX_BACKEND_KIND_CUDA) {
        rc = yvex_backend_cuda_attention_graph_summary_get(
            session_view->backend, &cuda, err);
        if (rc != YVEX_OK || !yvex_sha256_hex_valid(cuda.cuda_build_identity))
            return rc != YVEX_OK ? rc : YVEX_ERR_STATE;
        rc = yvex_backend_cuda_graph_query(session_view->backend, &graph, err);
        if (rc != YVEX_OK) return rc;
        kernel_bundle = cuda.cuda_build_identity;
    }
    request.schema_version = YVEX_RUNTIME_EXECUTION_PROFILE_SCHEMA_V1;
    request.engine_generation = summary.engine_generation;
    request.engine_specialization_identity = summary.engine_specialization_identity;
    request.kernel_bundle_identity = kernel_bundle;
    request.workload_profile_identity = workload.identity;
    request.generation_mode = YVEX_EXECUTION_GENERATION_TARGET_ONLY;
    request.evidence = YVEX_EXECUTION_EVIDENCE_PRODUCTION;
    request.execution_class =
        backend == YVEX_BACKEND_KIND_CUDA && cuda.kernel_bundle_native
            ? YVEX_EXECUTION_CLASS_DEVICE_NATIVE
            : YVEX_EXECUTION_CLASS_PORTABLE_REFERENCE;
    request.sampling_resolution =
        strategy == YVEX_SAMPLING_STRATEGY_GREEDY ||
                (backend == YVEX_BACKEND_KIND_CUDA &&
                 yvex_backend_sampling_operations_get(session_view->backend) != NULL)
            ? YVEX_EXECUTION_RESOLUTION_EXACT
            : YVEX_EXECUTION_RESOLUTION_COMPATIBLE_DEGRADED;
    request.moe_resolution =
        backend == YVEX_BACKEND_KIND_CUDA && cuda.kernel_bundle_native &&
                yvex_backend_moe_operations_get(session_view->backend) != NULL
            ? YVEX_EXECUTION_RESOLUTION_EXACT
            : YVEX_EXECUTION_RESOLUTION_COMPATIBLE_DEGRADED;
    request.attention_resolution =
        backend == YVEX_BACKEND_KIND_CUDA &&
                binding->capabilities.cuda_full_graph_implemented &&
                graph.state == YVEX_BACKEND_CUDA_GRAPH_OPEN &&
                graph.edge_inventory_available && graph.async_memory_available &&
                graph.async_copy_available && graph.pinned_host_memory_available
            ? YVEX_EXECUTION_RESOLUTION_EXACT
            : YVEX_EXECUTION_RESOLUTION_COMPATIBLE_DEGRADED;
    return yvex_runtime_execution_profile_seal(&request, profile, err);
}

static int live_manual_execute(yvex_model_engine *model,
                               yvex_backend_kind backend,
                               yvex_runtime_sampling_policy policy,
                               unsigned long long maximum_tokens,
                               live_manual *out, yvex_error *err)
{
    static const unsigned char prompt[] = "Hi";
    const yvex_model_engine_view *view = yvex_model_engine_view_get(model);
    yvex_runtime_session_open_request session_options = {.backend = backend};
    yvex_runtime_transformer_options transformer_options = {
        .context_capacity = 64ull, .workspace_token_capacity = 8ull};
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
    yvex_model_engine_failure failure = {0};
    yvex_runtime_execution_profile execution_profile = {0};
    const yvex_transformer_plan_summary *plan = NULL;
    const yvex_runtime_logits_plan_summary *logits_plan = NULL;
    float *prefill_hidden = NULL, *decode_hidden = NULL, *raw_logits = NULL;
    unsigned long long prefill_values = 0ull;
    int rc, first = 1;
    memset(out, 0, sizeof(*out));
    if (!view || !view->tokenizer) return YVEX_ERR_STATE;
    rc = yvex_runtime_session_open(&session, model, &session_options, &failure, err);
    if (rc == YVEX_OK)
        rc = live_execution_profile(
            model, session, backend, policy.strategy, &execution_profile, err);
    transformer_options.execution_profile = &execution_profile;
    if (rc == YVEX_OK)
        rc = yvex_runtime_transformer_context_open(
            &transformer, model, session, &transformer_options, NULL, err);
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
        yvex_runtime_transformer_output output = {0};
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
                sampling, NULL, &sample_source, &sample, err);
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
        rc = live_semantic_state_digest(
            session, out->semantic_state_digest, err);
        if (rc == YVEX_OK)
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
        production->result.final_position != manual->final_position) {
        yvex_error_setf(err, YVEX_ERR_FORMAT, "generation_live.compare",
                        "composition facts differ completed=%d sampled=%llu/%llu committed=%llu/%llu "
                        "position=%llu/%llu",
                        production->result.completed,
                        production->result.sampled_token_count, manual->sampled,
                        production->result.model_committed_token_count, manual->committed,
                        production->result.final_position, manual->final_position);
        return YVEX_ERR_FORMAT;
    }
    for (index = 0ull; index < manual->sampled; ++index)
        if (production->tokens[index].sampled_token_id != manual->tokens[index] ||
            (production->tokens[index].model_committed &&
             (!production->tokens[index].decode_submitted ||
              production->tokens[index].decode_input_token_id !=
                  production->tokens[index].sampled_token_id))) {
            yvex_error_setf(err, YVEX_ERR_FORMAT, "generation_live.compare",
                            "composition token %llu differs production=%u manual=%u",
                            index, production->tokens[index].sampled_token_id,
                            manual->tokens[index]);
            return YVEX_ERR_FORMAT;
        }
    if (production->result.generated_text_bytes != manual->text_bytes) {
        yvex_error_setf(
            err, YVEX_ERR_FORMAT, "generation_live.compare",
            "composition text extent differs text=%llu/%llu "
            "token_text=%llu,%llu,%llu/%llu,%llu,%llu",
            production->result.generated_text_bytes, manual->text_bytes,
            production->tokens[0].text_byte_count,
            production->tokens[1].text_byte_count,
            production->tokens[2].text_byte_count,
            manual->token_text_bytes[0], manual->token_text_bytes[1],
            manual->token_text_bytes[2]);
        return YVEX_ERR_FORMAT;
    }
    if (strcmp(production->semantic_state_digest,
               manual->semantic_state_digest) != 0) {
        yvex_error_setf(err, YVEX_ERR_FORMAT, "generation_live.compare",
                        "persistent state differs production=%.16s manual=%.16s",
                        production->semantic_state_digest,
                        manual->semantic_state_digest);
        return YVEX_ERR_FORMAT;
    }
    if (strcmp(production->result.final_rng_identity, manual->rng_identity) != 0 ||
        memcmp(production->text, manual->text, (size_t)manual->text_bytes) != 0) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "generation_live.compare",
                       "composition RNG or rendered text differs");
        return YVEX_ERR_FORMAT;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

static int live_greedy_equivalence(const live_generation *target,
                                   const live_generation *dspark,
                                   yvex_error *err)
{
    unsigned long long index;
    if (!target || !dspark) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "generation_live.equivalence",
                       "target and DSpark results are required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (target->result.execution_mode != YVEX_GENERATION_MODE_TARGET_ONLY ||
        dspark->result.execution_mode != YVEX_GENERATION_MODE_SPECULATIVE ||
        !target->result.completed || !dspark->result.completed ||
        target->result.sampled_token_count != dspark->result.sampled_token_count ||
        target->result.model_committed_token_count !=
            dspark->result.model_committed_token_count ||
        target->result.final_position != dspark->result.final_position) {
        yvex_error_setf(
            err, YVEX_ERR_FORMAT, "generation_live.equivalence",
            "result facts differ completed=%d/%d sampled=%llu/%llu committed=%llu/%llu "
            "text=%llu/%llu position=%llu/%llu",
            target->result.completed, dspark->result.completed,
            target->result.sampled_token_count, dspark->result.sampled_token_count,
            target->result.model_committed_token_count,
            dspark->result.model_committed_token_count,
            target->result.generated_text_bytes, dspark->result.generated_text_bytes,
            target->result.final_position, dspark->result.final_position);
        return YVEX_ERR_FORMAT;
    }
    /* Execution modes may admit different physical state layouts; equivalence is the exact
     * per-layer semantic timeline, while each layout-bound identity must remain valid. */
    if (!yvex_sha256_hex_valid(target->result.final_persistent_state_digest) ||
        !yvex_sha256_hex_valid(dspark->result.final_persistent_state_digest) ||
        strcmp(target->semantic_state_digest,
               dspark->semantic_state_digest) != 0) {
        yvex_error_setf(err, YVEX_ERR_FORMAT, "generation_live.equivalence",
                        "semantic state differs target=%.16s dspark=%.16s "
                        "physical=%.16s/%.16s cycles=%llu proposed=%llu "
                        "accepted=%llu rejected=%llu verified=%llu",
                        target->semantic_state_digest,
                        dspark->semantic_state_digest,
                        target->result.final_persistent_state_digest,
                        dspark->result.final_persistent_state_digest,
                        dspark->result.draft_cycle_count,
                        dspark->result.proposed_token_count,
                        dspark->result.accepted_draft_token_count,
                        dspark->result.rejected_draft_token_count,
                        dspark->result.target_verification_count);
        return YVEX_ERR_FORMAT;
    }
    for (index = 0ull; index < target->result.sampled_token_count; ++index)
        if (target->tokens[index].sampled_token_id !=
            dspark->tokens[index].sampled_token_id) {
            yvex_error_setf(err, YVEX_ERR_FORMAT, "generation_live.equivalence",
                            "token %llu differs target=%u dspark=%u", index,
                            target->tokens[index].sampled_token_id,
                            dspark->tokens[index].sampled_token_id);
            return YVEX_ERR_FORMAT;
        }
    if (target->result.generated_text_bytes !=
            dspark->result.generated_text_bytes ||
        memcmp(target->text, dspark->text,
               (size_t)target->result.generated_text_bytes) != 0) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "generation_live.equivalence",
                       "rendered text differs between target-only and DSpark");
        return YVEX_ERR_FORMAT;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

static int live_target_equivalence(const live_generation *expected,
                                   const live_generation *actual,
                                   yvex_error *err)
{
    unsigned long long index;
    if (!expected || !actual ||
        expected->result.execution_mode != YVEX_GENERATION_MODE_TARGET_ONLY ||
        actual->result.execution_mode != YVEX_GENERATION_MODE_TARGET_ONLY ||
        !expected->result.completed || !actual->result.completed ||
        expected->result.sampled_token_count != actual->result.sampled_token_count ||
        expected->result.model_committed_token_count !=
            actual->result.model_committed_token_count ||
        expected->result.generated_text_bytes != actual->result.generated_text_bytes ||
        expected->result.final_position != actual->result.final_position) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "generation_live.scheduler",
                       "scheduled target result differs from its serial reference");
        return YVEX_ERR_FORMAT;
    }
    for (index = 0ull; index < expected->result.sampled_token_count; ++index)
        if (expected->tokens[index].sampled_token_id !=
            actual->tokens[index].sampled_token_id) {
            yvex_error_setf(err, YVEX_ERR_FORMAT, "generation_live.scheduler",
                            "scheduled target token %llu differs reference=%u actual=%u",
                            index, expected->tokens[index].sampled_token_id,
                            actual->tokens[index].sampled_token_id);
            return YVEX_ERR_FORMAT;
        }
    if (strcmp(expected->semantic_state_digest,
               actual->semantic_state_digest) != 0 ||
        strcmp(expected->result.final_rng_identity,
               actual->result.final_rng_identity) != 0 ||
        memcmp(expected->text, actual->text,
               (size_t)expected->result.generated_text_bytes) != 0) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "generation_live.scheduler",
                       "scheduled target state, RNG, or rendered text differs");
        return YVEX_ERR_FORMAT;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

static void *live_scheduled_generation_main(void *opaque)
{
    static const unsigned char prompt[] = "Hi";
    live_scheduled_thread *thread = opaque;
    yvex_runtime_generation_request request = {
        .schema_version = YVEX_RUNTIME_GENERATION_SCHEMA_V3,
        .kind = YVEX_GENERATION_INPUT_TEXT,
        .text = prompt,
        .text_bytes = sizeof(prompt) - 1ull,
        .encode_options = {.maximum_tokens = 16ull}};
    yvex_error_clear(&thread->err);
    thread->rc = live_production_request(
        thread->model, YVEX_BACKEND_KIND_CUDA,
        YVEX_GENERATION_MODE_TARGET_ONLY, thread->policy, &request, 64ull,
        thread->maximum_tokens, 2ull, thread->gate, thread->result,
        &thread->err);
    if (thread->rc != YVEX_OK) live_start_gate_abort(thread->gate);
    return NULL;
}

static int live_continuous_batching_proof(
    yvex_model_engine *model, yvex_runtime_sampling_policy policy,
    unsigned long long maximum_tokens, const live_generation *reference,
    yvex_engine_scheduler_summary *out, yvex_error *err)
{
    live_start_gate gate = {.participants = 2u};
    live_generation results[2];
    live_scheduled_thread jobs[2];
    pthread_t threads[2];
    unsigned int created = 0u, index;
    int rc = YVEX_OK;

    memset(results, 0, sizeof(results));
    memset(jobs, 0, sizeof(jobs));
    memset(out, 0, sizeof(*out));
    if (pthread_mutex_init(&gate.mutex, NULL) != 0) {
        yvex_error_set(err, YVEX_ERR_STATE, "generation_live.scheduler",
                       "scheduled generation mutex could not initialize");
        return YVEX_ERR_STATE;
    }
    if (pthread_cond_init(&gate.condition, NULL) != 0) {
        (void)pthread_mutex_destroy(&gate.mutex);
        yvex_error_set(err, YVEX_ERR_STATE, "generation_live.scheduler",
                       "scheduled generation condition could not initialize");
        return YVEX_ERR_STATE;
    }
    for (index = 0u; index < 2u; ++index) {
        jobs[index].model = model;
        jobs[index].policy = policy;
        jobs[index].gate = &gate;
        jobs[index].result = &results[index];
        jobs[index].maximum_tokens = maximum_tokens;
        if (pthread_create(&threads[index], NULL,
                           live_scheduled_generation_main, &jobs[index]) != 0) {
            rc = YVEX_ERR_STATE;
            yvex_error_set(err, rc, "generation_live.scheduler",
                           "scheduled generation thread could not start");
            live_start_gate_abort(&gate);
            break;
        }
        created++;
    }
    for (index = 0u; index < created; ++index)
        (void)pthread_join(threads[index], NULL);
    for (index = 0u; rc == YVEX_OK && index < 2u; ++index) {
        if (jobs[index].rc != YVEX_OK) {
            rc = jobs[index].rc;
            *err = jobs[index].err;
        } else {
            rc = live_target_equivalence(reference, &results[index], err);
        }
    }
    if (rc == YVEX_OK) {
        *out = results[0].scheduler;
        if (results[1].scheduler.rendezvous_steps > out->rendezvous_steps ||
            (results[1].scheduler.rendezvous_steps == out->rendezvous_steps &&
             results[1].scheduler.physical_batches > out->physical_batches))
            *out = results[1].scheduler;
        if (!out->enabled || out->admitted_maximum_width < 2ull ||
            !out->multi_source_rendezvous ||
            out->maximum_rendezvous_width < 2ull ||
            !out->multi_source_batches ||
            out->maximum_multi_source_width < 2ull ||
            out->maximum_source_count < 2ull ||
            !out->multi_source_worklists ||
            !out->rendezvous_steps_by_phase[YVEX_EXECUTION_PHASE_PREFILL] ||
            !out->rendezvous_steps_by_phase[YVEX_EXECUTION_PHASE_DECODE]) {
            rc = YVEX_ERR_STATE;
            yvex_error_setf(
                err, rc, "generation_live.scheduler",
                "real continuous batching was not observed "
                "(rendezvous=%llu/%llu batches=%llu/%llu sources=%llu worklists=%llu)",
                out->multi_source_rendezvous, out->maximum_rendezvous_width,
                out->multi_source_batches, out->maximum_multi_source_width,
                out->maximum_source_count, out->multi_source_worklists);
        }
    }
    (void)pthread_cond_destroy(&gate.condition);
    (void)pthread_mutex_destroy(&gate.mutex);
    if (rc == YVEX_OK) yvex_error_clear(err);
    return rc;
}

static int live_reasoning_mode_equivalence(
    yvex_model_engine *model, yvex_backend_kind backend,
    yvex_runtime_sampling_policy policy, yvex_error *err)
{
    static const yvex_reasoning_policy reasoning_modes[] = {
        YVEX_REASONING_DISABLED, YVEX_REASONING_ENABLED,
        YVEX_REASONING_MAXIMUM};
    static const unsigned char prompt[] =
        "mi spieghi come funziona un llm";
    yvex_provider_message message = {
        .role = YVEX_PROVIDER_ROLE_USER,
        .content = {prompt, sizeof(prompt) - 1u}};
    unsigned long long index;
    int rc = YVEX_OK;

    for (index = 0u;
         rc == YVEX_OK &&
         index < sizeof(reasoning_modes) / sizeof(reasoning_modes[0]);
         ++index) {
        yvex_provider_request provider;
        yvex_runtime_generation_request request = {
            .schema_version = YVEX_RUNTIME_GENERATION_SCHEMA_V3,
            .kind = YVEX_GENERATION_INPUT_PROVIDER,
            .encode_options = {.maximum_tokens = 512u}};
        live_generation target, dspark;
        yvex_provider_request_default(&provider);
        strcpy(provider.model, "deepseek4-v4-flash-dspark");
        provider.messages = &message;
        provider.message_count = 1u;
        provider.maximum_output_tokens = 1u;
        provider.reasoning_policy = reasoning_modes[index];
        rc = yvex_provider_request_seal(&provider, err);
        request.provider_request = &provider;
        if (rc == YVEX_OK)
            rc = live_production_request(
                model, backend, YVEX_GENERATION_MODE_TARGET_ONLY, policy,
                &request, 512u, 1u, 1u, NULL, &target, err);
        if (rc == YVEX_OK)
            rc = live_production_request(
                model, backend, YVEX_GENERATION_MODE_SPECULATIVE, policy,
                &request, 512u, 1u, 1u, NULL, &dspark, err);
        if (rc == YVEX_OK)
            rc = live_greedy_equivalence(&target, &dspark, err);
    }
    if (rc != YVEX_OK && !yvex_error_message(err)[0])
        yvex_error_set(err, rc, "generation_live",
                       "reasoning-mode target/DSpark equivalence failed");
    return rc;
}

static void live_acceptance_corpus_add(
    live_acceptance_corpus *summary,
    const yvex_runtime_generation_result *result)
{
    summary->prompt_count++;
    summary->proposed += result->proposed_token_count;
    summary->verifications += result->target_verification_count;
    summary->accepted += result->accepted_draft_token_count;
    summary->rejected += result->rejected_draft_token_count;
    if (result->maximum_accepted_prefix > summary->maximum_accepted_prefix)
        summary->maximum_accepted_prefix = result->maximum_accepted_prefix;
}

static int live_acceptance_corpus_proof(
    yvex_model_engine *model, yvex_backend_kind backend,
    yvex_runtime_sampling_policy policy, unsigned long long maximum_tokens,
    const live_generation *initial_target,
    const live_generation *initial_dspark,
    live_acceptance_corpus *summary, yvex_error *err)
{
    static const struct {
        const unsigned char *text;
        unsigned long long bytes;
    } prompts[] = {
        {(const unsigned char *)"The capital of France is", 24ull},
        {(const unsigned char *)"One two three four", 18ull},
    };
    unsigned long long index;
    int rc = live_greedy_equivalence(initial_target, initial_dspark, err);

    memset(summary, 0, sizeof(*summary));
    if (rc == YVEX_OK)
        live_acceptance_corpus_add(summary, &initial_dspark->result);
    for (index = 0ull;
         rc == YVEX_OK && index < sizeof(prompts) / sizeof(prompts[0]);
         ++index) {
        live_generation target, dspark;
        rc = live_production_prompt(
            model, backend, YVEX_GENERATION_MODE_TARGET_ONLY, policy,
            prompts[index].text, prompts[index].bytes, maximum_tokens,
            &target, err);
        if (rc == YVEX_OK)
            rc = live_production_prompt(
                model, backend, YVEX_GENERATION_MODE_SPECULATIVE, policy,
                prompts[index].text, prompts[index].bytes, maximum_tokens,
                &dspark, err);
        if (rc == YVEX_OK) rc = live_greedy_equivalence(&target, &dspark, err);
        if (rc == YVEX_OK) live_acceptance_corpus_add(summary, &dspark.result);
    }
    if (rc == YVEX_OK &&
        (summary->prompt_count != 3ull || !summary->proposed ||
         !summary->verifications || !summary->accepted ||
         summary->maximum_accepted_prefix < 2ull))
        rc = YVEX_ERR_UNSUPPORTED;
    if (rc != YVEX_OK && !yvex_error_message(err)[0])
        yvex_error_set(
            err, rc, "generation_live",
            "the fixed DSpark acceptance corpus produced no multi-token accepted prefix");
    return rc;
}

int main(int argc, char **argv)
{
    yvex_model_engine_open_request request = {0};
    yvex_model_engine_failure failure = {0};
    yvex_model_engine *model = NULL;
    yvex_runtime_sampling_policy policy = {
        .schema_version = YVEX_RUNTIME_SAMPLING_SCHEMA_V1,
        .strategy = YVEX_SAMPLING_STRATEGY_GREEDY,
        .temperature = 1.0, .top_p = 1.0, .typical_p = 1.0};
    live_generation production;
    live_generation reference;
    live_manual manual;
    live_acceptance_corpus corpus = {0};
    yvex_engine_scheduler_summary continuous = {0};
    yvex_paths paths;
    yvex_backend_kind backend;
    yvex_runtime_generation_mode mode;
    yvex_error err;
    yvex_error path_error;
    unsigned long long maximum_tokens, seed;
    const char *step = "arguments";
    int rc;
    if (argc != 8 ||
        (strcmp(argv[3], "cpu") != 0 && strcmp(argv[3], "cuda") != 0) ||
        (strcmp(argv[4], "target-only") != 0 && strcmp(argv[4], "dspark") != 0) ||
        (strcmp(argv[5], "greedy") != 0 && strcmp(argv[5], "stochastic") != 0) ||
        sscanf(argv[6], "%llu", &seed) != 1 ||
        sscanf(argv[7], "%llu", &maximum_tokens) != 1 ||
        !maximum_tokens || maximum_tokens > LIVE_GENERATION_MAX_TOKENS) {
        fprintf(stderr,
                "usage: %s ARTIFACT BINDING cpu|cuda target-only|dspark "
                "greedy|stochastic SEED MAX_TOKENS\n",
                argv[0]);
        return 2;
    }
    backend = strcmp(argv[3], "cuda") == 0
                  ? YVEX_BACKEND_KIND_CUDA : YVEX_BACKEND_KIND_CPU;
    mode = strcmp(argv[4], "dspark") == 0
               ? YVEX_GENERATION_MODE_SPECULATIVE
               : YVEX_GENERATION_MODE_TARGET_ONLY;
    if (strcmp(argv[5], "stochastic") == 0) {
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
    request.target_id = "deepseek4-v4-flash-dspark";
    yvex_error_clear(&path_error);
    if (yvex_paths_default(&paths, &path_error) == YVEX_OK)
        request.artifact_reopen_cache_root = paths.cache_dir;
    rc = yvex_model_engine_open(&model, &request, &failure, &err);
    if (rc == YVEX_OK) { step = "production"; rc = live_production(
        model, backend, mode, policy, maximum_tokens, &production, &err); }
    if (rc == YVEX_OK && mode == YVEX_GENERATION_MODE_TARGET_ONLY) {
        step = "manual";
        rc = live_manual_execute(model, backend, policy, maximum_tokens,
                                 &manual, &err);
    }
    if (rc == YVEX_OK && mode == YVEX_GENERATION_MODE_TARGET_ONLY) {
        step = "composition-compare";
        rc = live_compare(&production, &manual, &err);
    }
    if (rc == YVEX_OK && mode == YVEX_GENERATION_MODE_SPECULATIVE &&
        policy.strategy == YVEX_SAMPLING_STRATEGY_GREEDY) {
        step = "target-only-reference";
        rc = live_production(model, backend, YVEX_GENERATION_MODE_TARGET_ONLY,
                             policy, maximum_tokens, &reference, &err);
    }
    if (rc == YVEX_OK && mode == YVEX_GENERATION_MODE_SPECULATIVE &&
        policy.strategy == YVEX_SAMPLING_STRATEGY_GREEDY) {
        step = "greedy-equivalence";
        rc = live_greedy_equivalence(&reference, &production, &err);
    }
    if (rc == YVEX_OK && backend == YVEX_BACKEND_KIND_CUDA &&
        mode == YVEX_GENERATION_MODE_SPECULATIVE &&
        policy.strategy == YVEX_SAMPLING_STRATEGY_GREEDY) {
        step = "reasoning-mode-equivalence";
        rc = live_reasoning_mode_equivalence(model, backend, policy, &err);
    }
    if (rc == YVEX_OK && backend == YVEX_BACKEND_KIND_CUDA &&
        mode == YVEX_GENERATION_MODE_SPECULATIVE &&
        policy.strategy == YVEX_SAMPLING_STRATEGY_GREEDY &&
        maximum_tokens >= 8ull) {
        step = "acceptance-corpus";
        rc = live_acceptance_corpus_proof(
            model, backend, policy, maximum_tokens, &reference, &production,
            &corpus, &err);
    }
    if (rc == YVEX_OK && backend == YVEX_BACKEND_KIND_CUDA &&
        mode == YVEX_GENERATION_MODE_SPECULATIVE &&
        policy.strategy == YVEX_SAMPLING_STRATEGY_GREEDY &&
        maximum_tokens >= 8ull) {
        step = "speculation-cancellation";
        rc = live_dspark_cancellation_proof(model, backend, policy, &err);
    }
    if (rc == YVEX_OK) { step = "mutation"; rc = live_mutation_proof(
        &production, &err); }
    if (rc == YVEX_OK) { step = "lifecycle"; rc = live_lifecycle_proof(
        model, backend, policy, &err); }
    if (rc == YVEX_OK && backend == YVEX_BACKEND_KIND_CUDA &&
        mode == YVEX_GENERATION_MODE_TARGET_ONLY &&
        policy.strategy == YVEX_SAMPLING_STRATEGY_GREEDY &&
        maximum_tokens >= 3ull) {
        step = "continuous-batching";
        rc = live_continuous_batching_proof(
            model, policy, maximum_tokens, &production, &continuous, &err);
    }
    if (rc == YVEX_OK && backend == YVEX_BACKEND_KIND_CUDA &&
        mode == YVEX_GENERATION_MODE_TARGET_ONLY &&
        policy.strategy == YVEX_SAMPLING_STRATEGY_GREEDY &&
        maximum_tokens >= 3ull) {
        step = "stop-capacity";
        rc = live_stop_capacity_proof(
            model, backend, policy, production.tokens[0].sampled_token_id,
            &err);
    }
    if (rc == YVEX_OK && backend == YVEX_BACKEND_KIND_CUDA &&
        mode == YVEX_GENERATION_MODE_TARGET_ONLY &&
        policy.strategy == YVEX_SAMPLING_STRATEGY_GREEDY &&
        maximum_tokens >= 3ull) {
        step = "partial-progress";
        rc = live_partial_progress_proof(model, backend, policy, &err);
    }
    if (rc != YVEX_OK) {
        live_failure_primary(&production);
        live_failure(step, rc, &err);
    }
    else {
        printf("generation_backend=%s mode=%s strategy=%s prompt_tokens=%llu "
               "sampled=%llu committed=%llu decode_inputs_match=pass "
               "teacher_forced_tail=0 text_bytes=%llu text_digest=%s "
               "final_position=%llu state_digest=%s stop=%s "
               "manual_composition=%s mutation_matrix=pass "
               "cancellation_close=pass stop_capacity=%s partial_progress=%s "
               "draft_cycles=%llu proposed=%llu verified=%llu accepted=%llu "
               "rejected=%llu max_accepted_prefix=%llu greedy_equivalence=%s "
               "acceptance_corpus=%s corpus_prompts=%llu corpus_proposed=%llu "
               "corpus_verified=%llu corpus_accepted=%llu corpus_rejected=%llu "
               "corpus_max_accepted_prefix=%llu speculation_cancellation=%s "
               "reasoning_mode_equivalence=%s state_device_stages=%llu "
               "state_device_stage_bytes=%llu state_device_copies=%llu "
               "state_device_copy_bytes=%llu tokens=",
               argv[3], argv[4], argv[5], production.result.prompt_token_count,
               production.result.sampled_token_count,
               production.result.model_committed_token_count,
               production.result.generated_text_bytes,
               production.result.generated_text_digest,
               production.result.final_position,
               production.semantic_state_digest,
               yvex_runtime_generation_stop_reason_name(
                   production.result.stop_reason),
               mode == YVEX_GENERATION_MODE_TARGET_ONLY ? "pass" : "not-applicable",
               backend == YVEX_BACKEND_KIND_CUDA &&
                       mode == YVEX_GENERATION_MODE_TARGET_ONLY &&
                       policy.strategy == YVEX_SAMPLING_STRATEGY_GREEDY &&
                       maximum_tokens >= 3ull
                   ? "pass" : "not-run",
               backend == YVEX_BACKEND_KIND_CUDA &&
                       mode == YVEX_GENERATION_MODE_TARGET_ONLY &&
                       policy.strategy == YVEX_SAMPLING_STRATEGY_GREEDY &&
                       maximum_tokens >= 3ull
                   ? "pass" : "not-run",
               production.result.draft_cycle_count,
               production.result.proposed_token_count,
               production.result.target_verification_count,
               production.result.accepted_draft_token_count,
               production.result.rejected_draft_token_count,
               production.result.maximum_accepted_prefix,
               mode == YVEX_GENERATION_MODE_SPECULATIVE &&
                       policy.strategy == YVEX_SAMPLING_STRATEGY_GREEDY
                   ? "pass" : "not-applicable",
               corpus.prompt_count ? "pass" : "not-run",
               corpus.prompt_count, corpus.proposed, corpus.verifications,
               corpus.accepted, corpus.rejected,
               corpus.maximum_accepted_prefix,
               corpus.prompt_count ? "pass" : "not-run",
               backend == YVEX_BACKEND_KIND_CUDA &&
                       mode == YVEX_GENERATION_MODE_SPECULATIVE &&
                       policy.strategy == YVEX_SAMPLING_STRATEGY_GREEDY
                   ? "pass" : "not-run",
               production.state_residency.device_stage_count,
               production.state_residency.device_stage_bytes,
               production.state_residency.copy_count,
               production.state_residency.copy_bytes);
        for (unsigned long long index = 0ull;
             index < production.result.sampled_token_count; ++index)
            printf("%s%u", index ? "," : "", production.tokens[index].sampled_token_id);
        printf("\n");
        printf(
            "generation_profile identity=%s kernel_launches=%llu graph_launches=%llu "
            "graph_captures=%llu graph_replays=%llu stream_syncs=%llu event_syncs=%llu "
            "device_syncs=%llu h2d_bytes=%llu d2h_bytes=%llu d2d_bytes=%llu "
            "target_forwards=%llu target_rows=%llu row_expert_pairs=%llu "
            "unique_experts=%llu expert_bytes=%llu output_head_rows=%llu "
            "logits_d2h_bytes=%llu replayed_accepted_rows=%llu "
            "attention_ns=%llu moe_ns=%llu output_sampling_ns=%llu sync_wait_ns=%llu "
            "prefill_ns=%llu first_decode_ns=%llu subsequent_decode_ns=%llu "
            "generation_ns=%llu\n",
            production.evidence.profile.profile_identity,
            production.evidence.profile.counters[YVEX_RUNTIME_PROFILE_KERNEL_LAUNCHES],
            production.evidence.profile.counters[YVEX_RUNTIME_PROFILE_GRAPH_LAUNCHES],
            production.evidence.profile.counters[YVEX_RUNTIME_PROFILE_GRAPH_CAPTURES],
            production.evidence.profile.counters[YVEX_RUNTIME_PROFILE_GRAPH_REPLAYS],
            production.evidence.profile.counters[YVEX_RUNTIME_PROFILE_STREAM_SYNCHRONIZATIONS],
            production.evidence.profile.counters[YVEX_RUNTIME_PROFILE_EVENT_SYNCHRONIZATIONS],
            production.evidence.profile.counters[YVEX_RUNTIME_PROFILE_DEVICE_SYNCHRONIZATIONS],
            production.evidence.profile.counters[YVEX_RUNTIME_PROFILE_H2D_BYTES],
            production.evidence.profile.counters[YVEX_RUNTIME_PROFILE_D2H_BYTES],
            production.evidence.profile.counters[YVEX_RUNTIME_PROFILE_D2D_BYTES],
            production.evidence.profile.counters[YVEX_RUNTIME_PROFILE_TARGET_FORWARDS],
            production.evidence.profile.counters[YVEX_RUNTIME_PROFILE_TARGET_ROWS],
            production.evidence.profile.counters[YVEX_RUNTIME_PROFILE_ROW_EXPERT_PAIRS],
            production.evidence.profile.counters[YVEX_RUNTIME_PROFILE_UNIQUE_EXPERTS],
            production.evidence.profile.counters[YVEX_RUNTIME_PROFILE_EXPERT_BYTES],
            production.evidence.profile.counters[YVEX_RUNTIME_PROFILE_OUTPUT_HEAD_ROWS],
            production.evidence.profile.counters[YVEX_RUNTIME_PROFILE_LOGITS_D2H_BYTES],
            production.evidence.profile.counters[
                YVEX_RUNTIME_PROFILE_REPLAYED_ACCEPTED_TARGET_ROWS],
            production.evidence.profile.phase_ns[YVEX_RUNTIME_PROFILE_ATTENTION],
            production.evidence.profile.phase_ns[YVEX_RUNTIME_PROFILE_MOE_TOTAL],
            production.evidence.profile.phase_ns[YVEX_RUNTIME_PROFILE_OUTPUT_HEAD] +
                production.evidence.profile.phase_ns[YVEX_RUNTIME_PROFILE_SAMPLING],
            production.evidence.profile.phase_ns[YVEX_RUNTIME_PROFILE_SYNCHRONIZATION_WAIT],
            production.evidence.profile.phase_ns[YVEX_RUNTIME_PROFILE_TOTAL_PREFILL],
            production.evidence.profile.phase_ns[YVEX_RUNTIME_PROFILE_FIRST_DECODE],
            production.evidence.profile.phase_ns[YVEX_RUNTIME_PROFILE_SUBSEQUENT_DECODE],
            production.evidence.profile.phase_ns[YVEX_RUNTIME_PROFILE_TOTAL_GENERATION]);
        printf("generation_worklists count=%llu pairs=%llu buckets=%llu "
               "max_bucket=%llu tc_eligible=%llu tc_executed=%llu narrow=%llu "
               "tail=%llu widths=",
               production.evidence.expert_worklists.worklist_count,
               production.evidence.expert_worklists.pair_count,
               production.evidence.expert_worklists.bucket_count,
               production.evidence.expert_worklists.maximum_bucket_population,
               production.evidence.expert_worklists.tensor_core_eligible_pairs,
               production.evidence.expert_worklists.tensor_core_executed_pairs,
               production.evidence.expert_worklists.narrow_pairs,
               production.evidence.expert_worklists.tail_rows);
        for (unsigned int index = 1u;
             index < YVEX_EXPERT_WORKLIST_HISTOGRAM_CAP; ++index)
            printf("%s%u:%llu", index == 1u ? "" : ",", index,
                   production.evidence.expert_worklists.width_histogram[index]);
        printf(" populations=");
        for (unsigned int index = 1u;
             index < YVEX_EXPERT_WORKLIST_HISTOGRAM_CAP; ++index)
            printf("%s%u:%llu", index == 1u ? "" : ",", index,
                   production.evidence.expert_worklists.population_histogram[index]);
        printf(" provenance=%llu,%llu,%llu,%llu,%llu\n",
               production.evidence.expert_worklists.provenance_counts[
                   YVEX_EXECUTION_BATCH_SINGLE_ROW],
               production.evidence.expert_worklists.provenance_counts[
                   YVEX_EXECUTION_BATCH_SPECULATIVE_VERIFICATION],
               production.evidence.expert_worklists.provenance_counts[
                   YVEX_EXECUTION_BATCH_MULTI_SESSION],
               production.evidence.expert_worklists.provenance_counts[
                   YVEX_EXECUTION_BATCH_PREFILL],
               production.evidence.expert_worklists.provenance_counts[
                   YVEX_EXECUTION_BATCH_COMPILED_COMPATIBLE]);
        printf(
            "generation_scheduler continuous_batching=%s admitted_width=%llu "
            "rendezvous=%llu multi_source_rendezvous=%llu max_rendezvous_width=%llu "
            "physical_batches=%llu multi_source_batches=%llu "
            "max_multi_source_width=%llu max_source_count=%llu "
            "multi_source_worklists=%llu\n",
            continuous.enabled ? "pass" : "not-run",
            continuous.admitted_maximum_width,
            continuous.rendezvous_steps,
            continuous.multi_source_rendezvous,
            continuous.maximum_rendezvous_width,
            continuous.physical_batches,
            continuous.multi_source_batches,
            continuous.maximum_multi_source_width,
            continuous.maximum_source_count,
            continuous.multi_source_worklists);
    }
    yvex_model_engine_close(&model);
    return rc == YVEX_OK ? 0 : 1;
}
