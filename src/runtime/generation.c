/*
 * Generation composes the admitted tokenizer, transformer, logits, sampler, and session owners
 * without taking over any of their state. A sampled token becomes visible only after that exact ID
 * is accepted as the next decode input and its state transaction commits.
 *
 * Partial progress is intentionally not rolled back across completed token transactions. Failure
 * reports the last committed model and text state so cancellation, retry, and server publication
 * all observe the same prefix.
 */
#include <yvex/internal/generation.h>
#include <yvex/internal/profile.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <yvex/internal/core.h>
#include <yvex/internal/graph_state.h>
#define GENERATION_LIFECYCLE_ACTIVE 1u
#define GENERATION_LIFECYCLE_CLOSING 2u
#define GENERATION_LIFECYCLE_CLOSED 6u
struct yvex_runtime_generation_context {
    yvex_runtime_model *model;
    yvex_runtime_execution_session *session;
    const yvex_runtime_model_view *model_view;
    const yvex_tokenizer *tokenizer;
    yvex_runtime_transformer_context *transformer;
    yvex_runtime_decode_context *decode;
    yvex_runtime_logits_context *logits;
    yvex_runtime_sampling_context *sampling;
    yvex_tokenizer_decoder *decoder;
    yvex_token_sequence *sequence;
    yvex_runtime_generation_options options;
    yvex_runtime_generation_plan_summary plan;
    unsigned int *additional_stops;
    float *hidden, *logits_row;
    unsigned long long hidden_count, logits_count, workspace_bytes;
    atomic_uint lifecycle;
    atomic_ullong admission_failures;
    pthread_mutex_t drain_mutex;
    pthread_cond_t drain_condition;
    unsigned long long execution_count, failure_count, cancellation_count;
    int drain_mutex_ready, drain_condition_ready, continuation_allowed;
};

static int generation_refuse(yvex_error *err, yvex_status status,
                             const char *reason)
{
    yvex_error_set(err, status, "runtime.generation", reason);
    return status;
}

static int generation_hash_finish(yvex_sha256 *hash,
                                  char output[YVEX_SHA256_HEX_CAP])
{
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    if (!hash || !output || !yvex_sha256_final(hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

static yvex_runtime_profile_mode generation_profile_mode(yvex_runtime_trace_policy policy)
{
    static const yvex_runtime_profile_mode modes[] = {
        YVEX_RUNTIME_PROFILE_OFF, YVEX_RUNTIME_PROFILE_SUMMARY,
        YVEX_RUNTIME_PROFILE_STAGES, YVEX_RUNTIME_PROFILE_DETAILED};
    return policy <= YVEX_RUNTIME_TRACE_FULL ? modes[policy] : YVEX_RUNTIME_PROFILE_OFF;
}
/*
 * Derive one path-free workload identity before prompt execution.
 *
 * Malformed text/messages/provider identities publish no digest. Workload identity is profiling
 * evidence, not generation-plan identity.
 */
static int generation_workload_identity(
    const yvex_runtime_generation_context *context,
    const yvex_runtime_generation_turn_request *turn,
    char output[YVEX_SHA256_HEX_CAP])
{
    const yvex_runtime_generation_request *request = turn ? turn->prompt : NULL;
    yvex_sha256 hash;
    unsigned long long index;
    if (!context || !turn || !request || !output) return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.profile.workload.v1") ||
        !yvex_sha256_update_text(&hash, context->plan.generation_plan_identity) ||
        !yvex_sha256_update_u64(&hash, request->kind) ||
        !yvex_sha256_update_u64(&hash, turn->maximum_new_tokens) ||
        !yvex_sha256_update_u64(&hash, turn->committed_prefix_token_count)) return 0;
    for (index = 0ull; index < turn->committed_prefix_token_count; ++index)
        if (!yvex_sha256_update_u64(&hash, turn->committed_prefix_token_ids[index])) return 0;
    if (request->kind == YVEX_GENERATION_INPUT_TEXT) {
        if (!request->text || request->text_bytes > SIZE_MAX ||
            !yvex_sha256_update_u64(&hash, request->text_bytes) ||
            !yvex_sha256_update(&hash, request->text, (size_t)request->text_bytes)) return 0;
    } else if (request->kind == YVEX_GENERATION_INPUT_PROVIDER) {
        if (!request->provider_request || !request->provider_request->sealed ||
            !yvex_sha256_hex_valid(request->provider_request->request_identity) ||
            !yvex_sha256_update_text(&hash, request->provider_request->request_identity)) return 0;
    } else {
        if (!request->messages || !request->message_count ||
            !yvex_sha256_update_u64(&hash, request->message_count)) return 0;
        for (index = 0ull; index < request->message_count; ++index) {
            const yvex_prompt_message *message = &request->messages[index];
            if ((!message->content && message->content_len) || message->content_len > SIZE_MAX ||
                !yvex_sha256_update_u64(&hash, message->role) ||
                !yvex_sha256_update_u64(&hash, message->content_len) ||
                !yvex_sha256_update(&hash, message->content, (size_t)message->content_len)) return 0;
        }
    }
    return generation_hash_finish(&hash, output);
}

static int generation_profile_phase(yvex_runtime_profile_record *profile,
                                    yvex_runtime_profile_phase phase,
                                    unsigned long long elapsed, yvex_error *err)
{
    return !profile || profile->mode == YVEX_RUNTIME_PROFILE_OFF || !elapsed
               ? YVEX_OK : runtime_profile_phase_add(profile, phase, elapsed, err);
}
/*
 * Project one complete Transformer transaction into generation profile evidence.
 *
 * Counter or duration overflow leaves the profile unsealed and returns typed failure.
 */
static int generation_profile_transformer(
    yvex_runtime_profile_record *profile,
    const yvex_runtime_transformer_result *value, yvex_error *err)
{
#define COUNTER(kind_, value_) do {                                                        \
        if ((value_) != 0ull && runtime_profile_counter_add(                              \
                profile, (kind_), (value_), err)                                           \
                            != YVEX_OK) return yvex_error_code(err);                        \
    } while (0)
    if (!profile || profile->mode == YVEX_RUNTIME_PROFILE_OFF || !value || !value->completed)
        return YVEX_OK;
    COUNTER(YVEX_RUNTIME_PROFILE_H2D_BYTES, value->h2d_bytes);
    COUNTER(YVEX_RUNTIME_PROFILE_D2H_BYTES, value->d2h_bytes);
    COUNTER(YVEX_RUNTIME_PROFILE_D2D_BYTES, value->d2d_bytes);
    COUNTER(YVEX_RUNTIME_PROFILE_UPLOADS, value->upload_count);
    COUNTER(YVEX_RUNTIME_PROFILE_DOWNLOADS, value->download_count);
    COUNTER(YVEX_RUNTIME_PROFILE_CACHE_HITS, value->cache_hits);
    COUNTER(YVEX_RUNTIME_PROFILE_CACHE_MISSES, value->cache_misses);
    COUNTER(YVEX_RUNTIME_PROFILE_EXPERT_SUBVIEWS, value->routed_experts * 3ull);
    COUNTER(YVEX_RUNTIME_PROFILE_KERNEL_LAUNCHES, value->kernel_launches);
    COUNTER(YVEX_RUNTIME_PROFILE_STREAM_SYNCHRONIZATIONS, value->stream_synchronizations);
    COUNTER(YVEX_RUNTIME_PROFILE_DEVICE_SYNCHRONIZATIONS, value->device_synchronizations);
#undef COUNTER
    if (generation_profile_phase(profile, YVEX_RUNTIME_PROFILE_EMBEDDING,
                                 value->embedding_ns, err) != YVEX_OK ||
        generation_profile_phase(profile, YVEX_RUNTIME_PROFILE_ATTENTION,
                                 profile->backend == YVEX_BACKEND_KIND_CUDA
                                     ? value->attention_device_ns : value->attention_ns,
                                 err) != YVEX_OK ||
        generation_profile_phase(profile, YVEX_RUNTIME_PROFILE_MOE_TOTAL,
                                 value->moe_ns, err) != YVEX_OK ||
        generation_profile_phase(profile, YVEX_RUNTIME_PROFILE_SYNCHRONIZATION_WAIT,
                                 value->synchronization_ns, err) != YVEX_OK ||
        generation_profile_phase(profile, YVEX_RUNTIME_PROFILE_FINAL_NORMALIZATION,
                                 value->final_ns, err) != YVEX_OK)
        return yvex_error_code(err);
    return YVEX_OK;
}

static int generation_profile_decode(yvex_runtime_profile_record *profile,
                                     const yvex_runtime_decode_step_result *value,
                                     yvex_error *err)
{
    yvex_runtime_transformer_result projected = {0};
    if (!value || !value->completed) return YVEX_OK;
    projected.completed = 1;
    projected.routed_experts = value->routed_experts;
    projected.h2d_bytes = value->h2d_bytes; projected.d2h_bytes = value->d2h_bytes;
    projected.d2d_bytes = value->d2d_bytes; projected.upload_count = value->upload_count;
    projected.download_count = value->download_count;
    projected.cache_hits = value->cache_hits; projected.cache_misses = value->cache_misses;
    projected.kernel_launches = value->kernel_launches;
    projected.stream_synchronizations = value->stream_synchronizations;
    projected.device_synchronizations = value->device_synchronizations;
    projected.embedding_ns = value->embedding_ns; projected.attention_ns = value->attention_ns;
    projected.attention_device_ns = value->attention_device_ns;
    projected.moe_ns = value->moe_ns; projected.final_ns = value->final_ns;
    projected.synchronization_ns = value->synchronization_ns;
    return generation_profile_transformer(profile, &projected, err);
}

static int generation_bytes_digest(const char *domain, const unsigned char *bytes,
                                   unsigned long long count,
                                   char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    if (!domain || (!bytes && count) || count > SIZE_MAX || !output) return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, domain) ||
        !yvex_sha256_update_u64(&hash, count) ||
        (count && !yvex_sha256_update(&hash, bytes, (size_t)count))) return 0;
    return generation_hash_finish(&hash, output);
}

static int generation_prefix_identity(
    const unsigned int *tokens, unsigned long long count,
    char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned long long index;
    if ((!tokens && count) || !output)
        return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.generation.prefix.v1") ||
        !yvex_sha256_update_u64(&hash, count))
        return 0;
    for (index = 0u; index < count; ++index)
        if (!yvex_sha256_update_u64(&hash, tokens[index]))
            return 0;
    return generation_hash_finish(&hash, output);
}

static int generation_state_summary(
    const yvex_runtime_execution_session *session,
    yvex_graph_attention_state_summary *summary, yvex_error *err)
{
    const yvex_runtime_session_view *view = yvex_runtime_session_view_get(session);
    if (!view || !view->attention_state_provider ||
        !view->attention_state_provider->summary ||
        view->attention_state_provider->summary(
            view->attention_state_provider->context, summary, err) != YVEX_OK)
        return generation_refuse(err, YVEX_ERR_STATE,
                                 "persistent generation state is unavailable");
    return YVEX_OK;
}

static int generation_stop_identity(
    const yvex_tokenizer_plan_summary *tokenizer,
    const unsigned int *additional, unsigned long long count,
    char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned long long index;
    if (!tokenizer || (!additional && count) || !output) return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.generation.stop.v1") ||
        !yvex_sha256_update_text(&hash, tokenizer->special_policy_identity) ||
        !yvex_sha256_update_u64(&hash, tokenizer->eos_present) ||
        !yvex_sha256_update_u64(&hash, tokenizer->eos_token_id) ||
        !yvex_sha256_update_u64(&hash, count)) return 0;
    for (index = 0ull; index < count; ++index)
        if (!yvex_sha256_update_u64(&hash, additional[index])) return 0;
    return generation_hash_finish(&hash, output);
}

static int generation_plan_identity(
    const yvex_runtime_generation_plan_summary *plan,
    char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    if (!plan || !output) return 0;
    yvex_sha256_init(&hash);
    return yvex_sha256_update_text(&hash, "yvex.runtime.generation.plan.v2") &&
           yvex_sha256_update_u64(&hash, plan->schema_version) &&
           yvex_sha256_update_u64(&hash, plan->backend) &&
           yvex_sha256_update_u64(&hash, plan->context_capacity) &&
           yvex_sha256_update_u64(&hash, plan->prefill_chunk_tokens) &&
           yvex_sha256_update_u64(&hash, plan->maximum_new_tokens) &&
           yvex_sha256_update_u64(&hash, plan->maximum_output_bytes) &&
           yvex_sha256_update_u64(&hash, plan->trace_policy) &&
           yvex_sha256_update_text(&hash, plan->runtime_model_identity) &&
           yvex_sha256_update_text(&hash, plan->runtime_binding_identity) &&
           yvex_sha256_update_text(&hash, plan->runtime_descriptor_identity) &&
           yvex_sha256_update_text(&hash, plan->tokenizer_plan_identity) &&
           yvex_sha256_update_text(&hash, plan->prompt_policy_identity) &&
           yvex_sha256_update_text(&hash, plan->transformer_plan_identity) &&
           yvex_sha256_update_text(&hash, plan->logits_plan_identity) &&
           yvex_sha256_update_text(&hash, plan->sampling_policy_identity) &&
           yvex_sha256_update_text(&hash, plan->stop_policy_identity) &&
           generation_hash_finish(&hash, output);
}

static int generation_stops_open(yvex_runtime_generation_context *context,
                                 unsigned long long vocabulary_size,
                                 yvex_error *err)
{
    unsigned long long index;
    if (!context->options.additional_stop_token_count) return YVEX_OK;
    if (!context->options.additional_stop_token_ids ||
        context->options.additional_stop_token_count > SIZE_MAX / sizeof(unsigned int))
        return generation_refuse(err, YVEX_ERR_BOUNDS,
                                 "additional stop-token extent is invalid");
    context->additional_stops = yvex_core_calloc(
        (size_t)context->options.additional_stop_token_count,
        sizeof(*context->additional_stops));
    if (!context->additional_stops)
        return generation_refuse(err, YVEX_ERR_NOMEM,
                                 "additional stop-token allocation failed");
    for (index = 0ull; index < context->options.additional_stop_token_count; ++index) {
        unsigned int token = context->options.additional_stop_token_ids[index];
        unsigned long long scan;
        if (token >= vocabulary_size)
            return generation_refuse(err, YVEX_ERR_BOUNDS,
                                     "additional stop token is outside vocabulary");
        for (scan = 0ull; scan < index; ++scan)
            if (context->additional_stops[scan] == token)
                return generation_refuse(err, YVEX_ERR_FORMAT,
                                         "additional stop tokens contain a duplicate");
        context->additional_stops[index] = token;
    }
    context->options.additional_stop_token_ids = context->additional_stops;
    return YVEX_OK;
}

static int generation_plan_build(yvex_runtime_generation_context *context,
                                 yvex_error *err)
{
    yvex_runtime_model_summary model;
    const yvex_transformer_plan_summary *transformer;
    const yvex_runtime_logits_plan_summary *logits;
    const yvex_tokenizer_plan_summary *tokenizer;
    yvex_runtime_generation_plan_summary plan;
    if (yvex_runtime_model_summary_copy(context->model, &model, err) != YVEX_OK)
        return yvex_error_code(err);
    transformer = yvex_transformer_plan_summary_get(
        yvex_runtime_transformer_context_plan(context->transformer));
    logits = yvex_runtime_logits_plan_summary_get(context->logits);
    tokenizer = yvex_tokenizer_plan_summary_get(context->tokenizer);
    if (!transformer || !logits || !tokenizer || !tokenizer->sealed ||
        !tokenizer->runtime_bound ||
        tokenizer->vocabulary_size != transformer->vocabulary_size ||
        tokenizer->vocabulary_size != logits->vocabulary_size ||
        strcmp(transformer->transformer_plan_identity,
               logits->transformer_plan_identity) != 0)
        return generation_refuse(err, YVEX_ERR_STATE,
                                 "generation lower-owner plans are incompatible");
    memset(&plan, 0, sizeof(plan));
    plan.schema_version = YVEX_RUNTIME_GENERATION_SCHEMA_V2;
    plan.backend = context->options.backend;
    plan.context_capacity = context->options.context_capacity;
    plan.prefill_chunk_tokens = context->options.prefill_chunk_tokens;
    plan.maximum_new_tokens = context->options.maximum_new_tokens;
    plan.maximum_output_bytes = context->options.maximum_output_bytes;
    plan.trace_policy = (unsigned int)context->options.trace_policy;
    yvex_runtime_identity_copy(plan.runtime_model_identity,
                               model.runtime_model_identity);
    yvex_runtime_identity_copy(plan.runtime_binding_identity,
                               context->model_view->binding->identity);
    yvex_runtime_identity_copy(plan.runtime_descriptor_identity,
                               model.runtime_descriptor_identity);
    yvex_runtime_identity_copy(plan.tokenizer_plan_identity,
                               tokenizer->tokenizer_plan_identity);
    yvex_runtime_identity_copy(plan.prompt_policy_identity,
                               tokenizer->prompt_policy_identity);
    yvex_runtime_identity_copy(plan.transformer_plan_identity,
                               transformer->transformer_plan_identity);
    yvex_runtime_identity_copy(plan.logits_plan_identity,
                               logits->output_head_plan_identity);
    yvex_runtime_identity_copy(plan.sampling_policy_identity,
                               context->options.sampling_policy.policy_identity);
    if (!generation_stop_identity(tokenizer, context->additional_stops,
                                  context->options.additional_stop_token_count,
                                  plan.stop_policy_identity) ||
        !generation_plan_identity(&plan, plan.generation_plan_identity))
        return generation_refuse(err, YVEX_ERR_STATE,
                                 "generation plan identity derivation failed");
    context->plan = plan;
    return YVEX_OK;
}

static int generation_enter(yvex_runtime_generation_context *context,
                            yvex_error *err)
{
    unsigned int expected = 0u;
    if (context && atomic_compare_exchange_strong_explicit(
                       &context->lifecycle, &expected,
                       GENERATION_LIFECYCLE_ACTIVE, memory_order_acq_rel,
                       memory_order_acquire))
        return YVEX_OK;
    if (context)
        (void)atomic_fetch_add_explicit(&context->admission_failures, 1ull,
                                        memory_order_relaxed);
    return generation_refuse(err, YVEX_ERR_STATE,
                             expected & GENERATION_LIFECYCLE_CLOSING
                                 ? "generation context is closing"
                                 : "generation context is already in use");
}
/*
 * Release exclusive admission and wake the close owner after accounting.
 *
 * Does not release context ownership.
 */
static void generation_leave(yvex_runtime_generation_context *context, int rc,
                             int executed)
{
    unsigned int observed;
    if (!context) return;
    if (executed) context->execution_count++;
    if (executed && rc != YVEX_OK) {
        context->failure_count++;
        if (rc == YVEX_ERR_CANCELLED) context->cancellation_count++;
    }
    observed = atomic_load_explicit(&context->lifecycle, memory_order_acquire);
    if ((observed & GENERATION_LIFECYCLE_CLOSING) &&
        context->drain_mutex_ready &&
        pthread_mutex_lock(&context->drain_mutex) == 0) {
        (void)atomic_fetch_and_explicit(&context->lifecycle,
                                        ~GENERATION_LIFECYCLE_ACTIVE,
                                        memory_order_release);
        if (context->drain_condition_ready)
            (void)pthread_cond_broadcast(&context->drain_condition);
        (void)pthread_mutex_unlock(&context->drain_mutex);
        return;
    }
    (void)atomic_fetch_and_explicit(&context->lifecycle,
                                    ~GENERATION_LIFECYCLE_ACTIVE,
                                    memory_order_release);
}

static int generation_cancelled(
    const yvex_runtime_generation_context *context, yvex_error *err)
{
    if (context->options.cancel_requested &&
        context->options.cancel_requested(context->options.cancel_context))
        return generation_refuse(err, YVEX_ERR_CANCELLED,
                                 "generation was cancelled");
    return YVEX_OK;
}
/*
 * Open one generation context over exactly one borrowed model/session plane.
 *
 * Sealed lower model/session and bounded immutable request options. Closes all partial local
 * ownership while leaving model/session to their caller.
 */
int yvex_runtime_generation_context_open(
    yvex_runtime_generation_context **out, yvex_runtime_model *model,
    yvex_runtime_execution_session *session,
    const yvex_runtime_generation_options *options, yvex_error *err)
{
    yvex_runtime_generation_context *context = NULL;
    yvex_runtime_transformer_options transformer_options = {0};
    yvex_runtime_decode_options decode_options = {0};
    yvex_runtime_logits_options logits_options = {0};
    yvex_runtime_sampling_options sampling_options = {0};
    yvex_tokenizer_decode_options decoder_options = {0};
    const yvex_runtime_logits_plan_summary *logits_plan;
    unsigned long long hidden_bytes, logits_bytes;
    int rc = YVEX_OK;
    if (out) *out = NULL;
    if (!out || !model || !session || !options ||
        options->schema_version != YVEX_RUNTIME_GENERATION_SCHEMA_V2 ||
        (options->backend != YVEX_BACKEND_KIND_CPU &&
         options->backend != YVEX_BACKEND_KIND_CUDA) ||
        !options->context_capacity || !options->prefill_chunk_tokens ||
        !options->maximum_new_tokens || !options->maximum_output_bytes ||
        options->trace_policy > YVEX_RUNTIME_TRACE_FULL)
        return generation_refuse(err, YVEX_ERR_INVALID_ARG,
                                 "complete bounded generation options are required");
    context = yvex_core_calloc(1u, sizeof(*context));
    if (!context)
        return generation_refuse(err, YVEX_ERR_NOMEM,
                                 "generation context allocation failed");
    context->model = model;
    context->session = session;
    context->model_view = yvex_runtime_model_view_get(model);
    context->tokenizer = context->model_view ? context->model_view->tokenizer : NULL;
    context->options = *options;
    atomic_init(&context->lifecycle, 0u);
    atomic_init(&context->admission_failures, 0ull);
    if (!context->model_view || !context->tokenizer ||
        !yvex_runtime_session_view_get(session) ||
        yvex_runtime_session_view_get(session)->model != model) {
        rc = generation_refuse(err, YVEX_ERR_STATE,
                               "generation model, session, and tokenizer are not paired");
        goto failure;
    }
    rc = generation_stops_open(context, yvex_tokenizer_vocab_size(context->tokenizer), err);
    if (rc != YVEX_OK) goto failure;
    context->options.sampling_policy = options->sampling_policy;
    rc = yvex_runtime_sampling_policy_seal(
        &context->options.sampling_policy,
        yvex_tokenizer_vocab_size(context->tokenizer), err);
    if (rc != YVEX_OK) goto failure;
    transformer_options.maximum_host_bytes = options->maximum_host_bytes;
    transformer_options.maximum_device_bytes = options->maximum_device_bytes;
    transformer_options.context_capacity = options->context_capacity;
    transformer_options.workspace_token_capacity = options->prefill_chunk_tokens;
    transformer_options.cancel_requested = options->cancel_requested;
    transformer_options.cancel_context = options->cancel_context;
    transformer_options.evidence_level = options->trace_policy == YVEX_RUNTIME_TRACE_FULL
                                             ? YVEX_ATTENTION_EVIDENCE_FULL
                                             : YVEX_ATTENTION_EVIDENCE_NONE;
    rc = yvex_runtime_transformer_context_open(
        &context->transformer, model, session, &transformer_options, err);
    if (rc != YVEX_OK) goto failure;
    logits_options.maximum_rows = 1ull;
    logits_options.maximum_host_bytes = options->maximum_host_bytes;
    logits_options.maximum_device_bytes = options->maximum_device_bytes;
    logits_options.cancel_requested = options->cancel_requested;
    logits_options.cancel_context = options->cancel_context;
    rc = yvex_runtime_logits_context_open(
        &context->logits, model, session,
        yvex_runtime_transformer_context_plan(context->transformer),
        &logits_options, err);
    if (rc != YVEX_OK) goto failure;
    rc = generation_plan_build(context, err);
    if (rc != YVEX_OK) goto failure;
    logits_plan = yvex_runtime_logits_plan_summary_get(context->logits);
    sampling_options.maximum_vocabulary_size = logits_plan->vocabulary_size;
    sampling_options.maximum_rows = 1ull;
    sampling_options.maximum_host_bytes = options->maximum_host_bytes;
    sampling_options.cancel_requested = options->cancel_requested;
    sampling_options.cancel_context = options->cancel_context;
    rc = yvex_runtime_sampling_context_open(
        &context->sampling, logits_plan, &context->options.sampling_policy,
        &sampling_options, err);
    if (rc != YVEX_OK) goto failure;
    decode_options.maximum_steps = options->maximum_new_tokens;
    decode_options.cancel_requested = options->cancel_requested;
    decode_options.cancel_context = options->cancel_context;
    rc = yvex_runtime_decode_context_open(
        &context->decode, context->transformer, session, &decode_options, err);
    if (rc != YVEX_OK) goto failure;
    decoder_options.skip_special_tokens = 1;
    decoder_options.require_complete_utf8 = 1;
    decoder_options.cancelled = options->cancel_requested;
    decoder_options.cancel_context = options->cancel_context;
    rc = yvex_tokenizer_decoder_open(&context->decoder, context->tokenizer,
                                     &decoder_options, err);
    if (rc != YVEX_OK) goto failure;
    rc = yvex_token_sequence_open(&context->sequence,
                                  options->maximum_new_tokens, err);
    if (rc != YVEX_OK) goto failure;
    context->hidden_count = logits_plan->hidden_width;
    context->logits_count = logits_plan->vocabulary_size;
    if (!yvex_core_u64_mul(context->hidden_count, sizeof(float), &hidden_bytes) ||
        !yvex_core_u64_mul(context->logits_count, sizeof(float), &logits_bytes) ||
        !yvex_core_u64_add(hidden_bytes, logits_bytes, &context->workspace_bytes) ||
        context->workspace_bytes > SIZE_MAX ||
        (options->maximum_host_bytes &&
         context->workspace_bytes > options->maximum_host_bytes)) {
        rc = generation_refuse(err, YVEX_ERR_NOMEM,
                               "generation-local workspace exceeds its budget");
        goto failure;
    }
    context->hidden = yvex_core_calloc((size_t)context->hidden_count, sizeof(float));
    context->logits_row = yvex_core_calloc((size_t)context->logits_count, sizeof(float));
    if (!context->hidden || !context->logits_row) {
        rc = generation_refuse(err, YVEX_ERR_NOMEM,
                               "generation-local workspace allocation failed");
        goto failure;
    }
    if (pthread_mutex_init(&context->drain_mutex, NULL) != 0) {
        rc = generation_refuse(err, YVEX_ERR_STATE,
                               "generation lifecycle mutex initialization failed");
        goto failure;
    }
    context->drain_mutex_ready = 1;
    if (pthread_cond_init(&context->drain_condition, NULL) != 0) {
        rc = generation_refuse(err, YVEX_ERR_STATE,
                               "generation lifecycle condition initialization failed");
        goto failure;
    }
    context->drain_condition_ready = 1;
    context->continuation_allowed = 1;
    *out = context;
    yvex_error_clear(err);
    return YVEX_OK;
failure:
    if (context) {
        yvex_error cleanup;
        yvex_error_clear(&cleanup);
        yvex_token_sequence_close(&context->sequence);
        yvex_tokenizer_decoder_close(&context->decoder);
        (void)yvex_runtime_decode_context_close(&context->decode, &cleanup);
        (void)yvex_runtime_sampling_context_close(&context->sampling, &cleanup);
        (void)yvex_runtime_logits_context_close(&context->logits, &cleanup);
        (void)yvex_runtime_transformer_context_close(&context->transformer, &cleanup);
        if (context->drain_condition_ready)
            (void)pthread_cond_destroy(&context->drain_condition);
        if (context->drain_mutex_ready)
            (void)pthread_mutex_destroy(&context->drain_mutex);
        yvex_core_free(context->logits_row);
        yvex_core_free(context->hidden);
        yvex_core_free(context->additional_stops);
        yvex_core_free(context);
    }
    return rc;
}
/* Return the immutable plan borrowed for the context lifetime. */
const yvex_runtime_generation_plan_summary *yvex_runtime_generation_plan_summary_get(
    const yvex_runtime_generation_context *context)
{
    return context ? &context->plan : NULL;
}

static int generation_token_classify(
    const yvex_runtime_generation_context *context, unsigned int token,
    yvex_tokenizer_token_classification *classification, int *additional_stop,
    yvex_error *err)
{
    unsigned long long index;
    int rc = yvex_tokenizer_token_classify(context->tokenizer, token,
                                           classification, err);
    *additional_stop = 0;
    for (index = 0ull; rc == YVEX_OK &&
                         index < context->options.additional_stop_token_count; ++index)
        if (context->additional_stops[index] == token) *additional_stop = 1;
    return rc;
}

static int generation_encode_prompt(
    const yvex_runtime_generation_context *context,
    const yvex_runtime_generation_request *request,
    yvex_rendered_prompt *rendered, yvex_tokenizer_encode_result *encoded,
    char prompt_identity[YVEX_SHA256_HEX_CAP], yvex_error *err)
{
    yvex_tokenizer_encode_options encode;
    int rc;
    memset(rendered, 0, sizeof(*rendered));
    memset(encoded, 0, sizeof(*encoded));
    if (!request || request->schema_version != YVEX_RUNTIME_GENERATION_SCHEMA_V2 ||
        request->kind > YVEX_GENERATION_INPUT_PROVIDER)
        return generation_refuse(err, YVEX_ERR_INVALID_ARG,
                                 "typed text or message input is required");
    encode = request->encode_options;
    if (!encode.maximum_tokens ||
        encode.maximum_tokens > context->options.context_capacity)
        encode.maximum_tokens = context->options.context_capacity;
    if (request->kind == YVEX_GENERATION_INPUT_TEXT) {
        if (!request->text || !request->text_bytes)
            return generation_refuse(err, YVEX_ERR_INVALID_ARG,
                                     "nonempty explicit-length prompt text is required");
        rc = yvex_tokenizer_encode(context->tokenizer, request->text,
                                   request->text_bytes, &encode, encoded, err);
        if (rc == YVEX_OK)
            yvex_runtime_identity_copy(prompt_identity,
                                       encoded->input_identity);
    } else if (request->kind == YVEX_GENERATION_INPUT_MESSAGES) {
        if (!request->messages || !request->message_count)
            return generation_refuse(err, YVEX_ERR_INVALID_ARG,
                                     "nonempty typed prompt messages are required");
        rc = yvex_tokenizer_encode_prompt(
            context->tokenizer, request->messages, request->message_count,
            &request->prompt_options, &encode, rendered, encoded, err);
        if (rc == YVEX_OK)
            yvex_runtime_identity_copy(prompt_identity,
                                       rendered->prompt_identity);
    } else {
        if (!request->provider_request)
            return generation_refuse(err, YVEX_ERR_INVALID_ARG,
                                     "sealed provider request is required");
        rc = yvex_tokenizer_encode_provider_prompt(
            context->tokenizer, request->provider_request, &encode,
            rendered, encoded, err);
        if (rc == YVEX_OK)
            yvex_runtime_identity_copy(prompt_identity,
                                       rendered->prompt_identity);
    }
    if (rc == YVEX_OK && (!encoded->completed || !encoded->tokens.len ||
                          encoded->tokens.len > context->options.context_capacity))
        rc = generation_refuse(err, YVEX_ERR_BOUNDS,
                               "encoded prompt is empty or exceeds context capacity");
    return rc;
}

static int generation_prefill(
    yvex_runtime_generation_context *context,
    const yvex_tokenizer_encode_result *encoded,
    unsigned long long reusable_prefix,
    const yvex_runtime_generation_turn_request *turn, float **final_hidden,
    unsigned long long *final_hidden_count,
    yvex_runtime_transformer_result *final_result,
    unsigned long long *completed_chunks,
    yvex_runtime_profile_record *profile, yvex_error *err)
{
    const yvex_transformer_plan_summary *plan = yvex_transformer_plan_summary_get(
        yvex_runtime_transformer_context_plan(context->transformer));
    unsigned long long offset = 0ull, suffix_count, maximum_chunk, maximum_values;
    float *buffer = NULL;
    int rc = YVEX_OK;
    *final_hidden = NULL;
    *final_hidden_count = 0ull;
    *completed_chunks = 0ull;
    memset(final_result, 0, sizeof(*final_result));
    if (reusable_prefix >= encoded->tokens.len)
        return generation_refuse(err, YVEX_ERR_STATE,
                                 "generation turn requires one exact new prompt suffix token");
    suffix_count = encoded->tokens.len - reusable_prefix;
    maximum_chunk = context->options.prefill_chunk_tokens;
    if (maximum_chunk > suffix_count) maximum_chunk = suffix_count;
    if (!plan || !yvex_core_u64_mul(maximum_chunk, plan->hidden_width,
                                    &maximum_values) ||
        maximum_values > SIZE_MAX / sizeof(float))
        return generation_refuse(err, YVEX_ERR_BOUNDS,
                                 "prompt prefill hidden extent overflowed");
    buffer = yvex_core_calloc((size_t)maximum_values, sizeof(float));
    if (!buffer)
        return generation_refuse(err, YVEX_ERR_NOMEM,
                                 "prompt prefill hidden allocation failed");
    while (offset < suffix_count && rc == YVEX_OK) {
        yvex_transformer_input_summary summary;
        yvex_transformer_input *input = NULL;
        yvex_runtime_transformer_request request = {0};
        yvex_runtime_transformer_output output = {0};
        yvex_runtime_transformer_result result;
        unsigned long long count = suffix_count - offset, values;
        if (count > maximum_chunk) count = maximum_chunk;
        memset(&summary, 0, sizeof(summary));
        summary.schema_version = YVEX_TRANSFORMER_INPUT_SCHEMA_V1;
        summary.token_start = reusable_prefix + offset;
        summary.token_count = count;
        summary.vocabulary_size = plan->vocabulary_size;
        yvex_runtime_identity_copy(summary.logical_model_identity,
                                   plan->logical_model_identity);
        yvex_runtime_identity_copy(summary.runtime_numeric_identity,
                                   plan->runtime_numeric_identity);
        yvex_runtime_identity_copy(summary.runtime_descriptor_identity,
                                   plan->runtime_descriptor_identity);
        yvex_runtime_identity_copy(summary.transformer_plan_identity,
                                   plan->transformer_plan_identity);
        rc = yvex_transformer_input_seal(&summary,
                                         encoded->tokens.ids + reusable_prefix + offset, err);
        if (rc == YVEX_OK)
            rc = yvex_transformer_input_open_memory(
                &input, &summary, encoded->tokens.ids + reusable_prefix + offset, err);
        request.chunk_tokens = count;
        request.backend = context->options.backend;
        request.phase = YVEX_TRANSFORMER_PHASE_PREFILL;
        (void)yvex_core_u64_mul(count, plan->hidden_width, &values);
        memset(buffer, 0, (size_t)values * sizeof(float));
        output.normalized_hidden = buffer;
        output.capacity = values;
        if (rc == YVEX_OK)
            rc = yvex_runtime_transformer_execute(
                context->transformer, input, &request, &output, &result, err);
        yvex_transformer_input_close(&input);
        if (rc == YVEX_OK) {
            rc = generation_profile_transformer(profile, &result, err);
        }
        if (rc == YVEX_OK) {
            *final_result = result;
            *final_hidden_count = values;
            (*completed_chunks)++;
            offset += count;
            if (turn->progress_sink)
                rc = turn->progress_sink(
                    turn->progress_context,
                    YVEX_GENERATION_PROGRESS_PREFILL_PROGRESS,
                    offset, suffix_count, err);
        }
    }
    if (rc == YVEX_OK) {
        *final_hidden = buffer;
        return YVEX_OK;
    }
    yvex_core_free(buffer);
    return rc;
}
/*
 * Derive one token-step identity from every authoritative published field.
 *
 * No pointer, padding, or native structure bytes participate.
 */
static int generation_token_identity(
    const yvex_runtime_generation_token_result *token,
    char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    if (!token || !output ||
        token->schema_version != YVEX_RUNTIME_GENERATION_SCHEMA_V2 ||
        !token->sampled || token->sampled_token_id != token->classification.token_id)
        return 0;
    yvex_sha256_init(&hash);
    return yvex_sha256_update_text(&hash, "yvex.runtime.generation.token.v2") &&
           yvex_sha256_update_u64(&hash, token->schema_version) &&
           yvex_sha256_update_u64(&hash, token->ordinal) &&
           yvex_sha256_update_u64(&hash, token->sampled_token_id) &&
           yvex_sha256_update_u64(&hash, token->decode_input_token_id) &&
           yvex_sha256_update_u64(&hash, token->sequence_state_before) &&
           yvex_sha256_update_u64(&hash, token->sequence_state_after) &&
           yvex_sha256_update_u64(&hash, token->position_before) &&
           yvex_sha256_update_u64(&hash, token->position_after) &&
           yvex_sha256_update_u64(&hash, token->persistent_generation_before) &&
           yvex_sha256_update_u64(&hash, token->persistent_generation_after) &&
           yvex_sha256_update_u64(&hash, token->text_byte_offset) &&
           yvex_sha256_update_u64(&hash, token->text_byte_count) &&
           yvex_sha256_update_u64(&hash, token->model_committed) &&
           yvex_sha256_update_u64(&hash, token->decode_submitted) &&
           yvex_sha256_update_u64(&hash, token->detokenized) &&
           yvex_sha256_update_u64(&hash, token->text_published) &&
           yvex_sha256_update_u64(&hash, token->terminal) &&
           yvex_sha256_update_u64(&hash, token->suppressed) &&
           yvex_sha256_update_u64(&hash, token->classification.special) &&
           yvex_sha256_update_u64(&hash, token->classification.eos) &&
           yvex_sha256_update_u64(&hash, token->classification.pad) &&
           yvex_sha256_update_u64(&hash, token->classification.unknown) &&
           yvex_sha256_update_u64(&hash, token->classification.stop) &&
           yvex_sha256_update_u64(&hash, token->classification.suppressed_by_default) &&
           yvex_sha256_update_text(&hash, token->source_logits_identity) &&
           yvex_sha256_update_text(&hash, token->sampling_result_identity) &&
           yvex_sha256_update_text(&hash, token->decode_execution_identity) &&
           yvex_sha256_update_text(&hash, token->persistent_state_digest) &&
           yvex_sha256_update_text(&hash, token->decoder_fragment_identity) &&
           generation_hash_finish(&hash, output);
}

static int generation_tokens_identity(
    const yvex_runtime_generation_token_result *tokens,
    unsigned long long count, char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned long long index;
    if ((!tokens && count) || !output) return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.generation.tokens.v2") ||
        !yvex_sha256_update_u64(&hash, count)) return 0;
    for (index = 0ull; index < count; ++index)
        if (!yvex_sha256_update_u64(&hash, tokens[index].sampled_token_id) ||
            !yvex_sha256_update_text(&hash, tokens[index].token_step_identity))
            return 0;
    return generation_hash_finish(&hash, output);
}
/*
 * Derive one complete-or-partial generation execution identity field by field.
 *
 * No pointer, padding, or native structure bytes participate.
 */
static int generation_execution_identity(
    const yvex_runtime_generation_result *result,
    const yvex_runtime_generation_token_result *tokens,
    char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned long long index;
    if (!result || (!tokens && result->sampled_token_count) || !output) return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.generation.execution.v2") ||
        !yvex_sha256_update_u64(&hash, result->schema_version) ||
        !yvex_sha256_update_u64(&hash, result->status) ||
        !yvex_sha256_update_u64(&hash, result->stop_reason) ||
        !yvex_sha256_update_u64(&hash, result->completed) ||
        !yvex_sha256_update_u64(&hash, result->partial) ||
        !yvex_sha256_update_u64(&hash, result->cancelled) ||
        !yvex_sha256_update_u64(&hash, result->failed) ||
        !yvex_sha256_update_u64(&hash, result->has_incomplete_token) ||
        !yvex_sha256_update_u64(&hash, result->prompt_bytes) ||
        !yvex_sha256_update_u64(&hash, result->prompt_token_count) ||
        !yvex_sha256_update_u64(&hash, result->prefill_chunk_count) ||
        !yvex_sha256_update_u64(&hash, result->requested_new_tokens) ||
        !yvex_sha256_update_u64(&hash, result->sampled_token_count) ||
        !yvex_sha256_update_u64(&hash, result->model_committed_token_count) ||
        !yvex_sha256_update_u64(&hash, result->text_published_token_count) ||
        !yvex_sha256_update_u64(&hash, result->terminal_token_count) ||
        !yvex_sha256_update_u64(&hash, result->suppressed_token_count) ||
        !yvex_sha256_update_u64(&hash, result->first_incomplete_token) ||
        !yvex_sha256_update_u64(&hash, result->logits_projection_count) ||
        !yvex_sha256_update_u64(&hash, result->sampling_draw_count) ||
        !yvex_sha256_update_u64(&hash, result->decode_step_count) ||
        !yvex_sha256_update_u64(&hash, result->initial_position) ||
        !yvex_sha256_update_u64(&hash, result->reusable_prefix_token_count) ||
        !yvex_sha256_update_u64(&hash, result->new_prefill_token_count) ||
        !yvex_sha256_update_u64(&hash, result->final_position) ||
        !yvex_sha256_update_u64(&hash, result->final_persistent_generation) ||
        !yvex_sha256_update_u64(&hash, result->generated_text_bytes) ||
        !yvex_sha256_update_text(&hash, result->prompt_identity) ||
        !yvex_sha256_update_text(&hash, result->prompt_token_identity) ||
        !yvex_sha256_update_text(&hash, result->reusable_prefix_identity) ||
        !yvex_sha256_update_text(&hash, result->initial_rng_identity) ||
        !yvex_sha256_update_text(&hash, result->final_rng_identity) ||
        !yvex_sha256_update_text(&hash, result->generated_token_identity) ||
        !yvex_sha256_update_text(&hash, result->generated_text_digest) ||
        !yvex_sha256_update_text(&hash, result->final_persistent_state_digest) ||
        !yvex_sha256_update_text(&hash, result->generation_plan_identity)) return 0;
    for (index = 0ull; index < result->sampled_token_count; ++index)
        if (!yvex_sha256_update_text(&hash, tokens[index].token_step_identity))
            return 0;
    return generation_hash_finish(&hash, output);
}

static int generation_project_sample(
    yvex_runtime_generation_context *context,
    const yvex_runtime_transformer_result *prefill,
    const float *prefill_hidden, unsigned long long prefill_hidden_count,
    const yvex_runtime_decode_step_result *decode,
    yvex_runtime_logits_row_result *logits_result,
    yvex_runtime_sampling_result *sampling_result,
    yvex_runtime_profile_record *profile, yvex_error *err)
{
    yvex_runtime_logits_source logits_source;
    yvex_runtime_sampling_source sampling_source;
    const char *stage = "normalized hidden admission";
    unsigned long long started, completed;
    int rc;
    memset(logits_result, 0, sizeof(*logits_result));
    memset(sampling_result, 0, sizeof(*sampling_result));
    if (prefill)
        rc = yvex_runtime_logits_source_from_transformer(
            context->logits, &logits_source, prefill, prefill_hidden,
            prefill_hidden_count, prefill->token_count - 1ull, err);
    else
        rc = yvex_runtime_logits_source_from_decode(
            context->logits, &logits_source, decode, context->hidden,
            context->hidden_count, err);
    if (rc == YVEX_OK) {
        stage = "complete vocabulary projection";
        started = yvex_core_monotonic_ns();
        rc = yvex_runtime_logits_project(
            context->logits, &logits_source, context->options.backend,
            context->logits_row, context->logits_count, logits_result, err);
        completed = yvex_core_monotonic_ns();
        if (rc == YVEX_OK)
            rc = generation_profile_phase(profile, YVEX_RUNTIME_PROFILE_OUTPUT_HEAD,
                                           completed - started, err);
        if (rc == YVEX_OK && profile->mode != YVEX_RUNTIME_PROFILE_OFF &&
            ((logits_result->h2d_bytes && runtime_profile_counter_add(
                  profile, YVEX_RUNTIME_PROFILE_H2D_BYTES,
                  logits_result->h2d_bytes, err) != YVEX_OK) ||
             (logits_result->d2h_bytes && runtime_profile_counter_add(
                  profile, YVEX_RUNTIME_PROFILE_D2H_BYTES,
                  logits_result->d2h_bytes, err) != YVEX_OK) ||
             (logits_result->kernel_launches && runtime_profile_counter_add(
                  profile, YVEX_RUNTIME_PROFILE_KERNEL_LAUNCHES,
                  logits_result->kernel_launches, err) != YVEX_OK)))
            rc = yvex_error_code(err);
    }
    if (rc == YVEX_OK) {
        stage = "sampling source admission";
        rc = yvex_runtime_sampling_source_from_logits(
            context->sampling, &sampling_source, context->logits_row,
            context->logits_count, logits_result, err);
    }
    if (rc == YVEX_OK) {
        stage = "token selection";
        started = yvex_core_monotonic_ns();
        rc = yvex_runtime_sampling_select(context->sampling, &sampling_source,
                                          sampling_result, err);
        completed = yvex_core_monotonic_ns();
        if (rc == YVEX_OK)
            rc = generation_profile_phase(profile, YVEX_RUNTIME_PROFILE_SAMPLING,
                                           completed - started, err);
    }
    if (rc != YVEX_OK && !yvex_error_is_set(err))
        yvex_error_set(err, (yvex_status)rc, "runtime.generation.sample", stage);
    return rc;
}

static int generation_terminal_token(
    yvex_runtime_generation_token_result *token,
    yvex_runtime_generation_result *result,
    int additional_stop, yvex_error *err)
{
    token->terminal = 1;
    token->suppressed = token->classification.suppressed_by_default;
    token->sequence_state_after = YVEX_TOKEN_APPEND_PROPOSED;
    token->position_after = token->position_before;
    token->persistent_generation_after = token->persistent_generation_before;
    result->terminal_token_count++;
    if (token->suppressed) result->suppressed_token_count++;
    result->stop_reason = token->classification.eos
                              ? YVEX_GENERATION_STOP_EOS
                              : YVEX_GENERATION_STOP_TOKENIZER_TOKEN;
    if (!token->classification.eos && !additional_stop)
        return generation_refuse(err, YVEX_ERR_STATE,
                                 "terminal generation token lacks an admitted stop fact");
    if (!generation_token_identity(token, token->token_step_identity))
        return generation_refuse(err, YVEX_ERR_STATE,
                                 "terminal token identity derivation failed");
    return YVEX_OK;
}

static int generation_commit_ordinary(
    yvex_runtime_generation_context *context,
    yvex_runtime_generation_token_result *token,
    unsigned long long sequence_ordinal, unsigned char *text,
    unsigned long long text_capacity, yvex_runtime_generation_result *result,
    yvex_runtime_decode_step_result *decode_result, yvex_error *err)
{
    yvex_tokenizer_fragment fragment;
    unsigned long long next_text, started, completed;
    int rc;
    memset(&fragment, 0, sizeof(fragment));
    rc = yvex_token_sequence_transition(
        context->sequence, sequence_ordinal, YVEX_TOKEN_APPEND_PROPOSED,
        YVEX_TOKEN_APPEND_APPENDED, err);
    token->sequence_state_after = rc == YVEX_OK
                                      ? YVEX_TOKEN_APPEND_APPENDED
                                      : YVEX_TOKEN_APPEND_PROPOSED;
    if (rc == YVEX_OK) rc = generation_cancelled(context, err);
    if (rc == YVEX_OK) {
        rc = yvex_token_sequence_transition(
            context->sequence, sequence_ordinal, YVEX_TOKEN_APPEND_APPENDED,
            YVEX_TOKEN_APPEND_SUBMITTED, err);
        if (rc == YVEX_OK) {
            token->sequence_state_after = YVEX_TOKEN_APPEND_SUBMITTED;
            token->decode_submitted = 1;
            token->decode_input_token_id = token->sampled_token_id;
        }
    }
    if (rc == YVEX_OK) {
        started = yvex_core_monotonic_ns();
        rc = yvex_runtime_decode_step(
            context->decode, token->ordinal, token->position_before,
            token->sampled_token_id, context->options.backend, context->hidden,
            context->hidden_count, decode_result, err);
        completed = yvex_core_monotonic_ns();
        if (rc == YVEX_OK)
            rc = generation_profile_phase(
                &result->profile,
                result->decode_step_count ? YVEX_RUNTIME_PROFILE_SUBSEQUENT_DECODE
                                          : YVEX_RUNTIME_PROFILE_FIRST_DECODE,
                completed - started, err);
        if (rc == YVEX_OK)
            rc = generation_profile_decode(&result->profile, decode_result, err);
    }
    if (rc == YVEX_OK) {
        rc = yvex_token_sequence_transition(
            context->sequence, sequence_ordinal, YVEX_TOKEN_APPEND_SUBMITTED,
            YVEX_TOKEN_APPEND_MODEL_COMMITTED, err);
        token->sequence_state_after = YVEX_TOKEN_APPEND_MODEL_COMMITTED;
        token->model_committed = 1;
        token->position_after = decode_result->position_after;
        token->persistent_generation_after = decode_result->generation_after;
        yvex_runtime_identity_copy(token->decode_execution_identity,
                                   decode_result->decode_step_identity);
        yvex_runtime_identity_copy(token->persistent_state_digest,
                                   decode_result->persistent_state_digest);
        result->model_committed_token_count++;
        result->decode_step_count++;
    }
    if (rc == YVEX_OK) rc = generation_cancelled(context, err);
    if (rc == YVEX_OK) {
        started = yvex_core_monotonic_ns();
        rc = yvex_tokenizer_decoder_push(context->decoder,
                                         token->sampled_token_id, &fragment, err);
        completed = yvex_core_monotonic_ns();
        if (rc == YVEX_OK)
            rc = generation_profile_phase(&result->profile,
                                           YVEX_RUNTIME_PROFILE_DETOKENIZATION,
                                           completed - started, err);
    }
    if (rc == YVEX_OK) {
        token->detokenized = 1;
        token->suppressed = fragment.suppressed;
        token->text_byte_offset = result->generated_text_bytes;
        token->text_byte_count = fragment.byte_count;
        yvex_runtime_identity_copy(token->decoder_fragment_identity,
                                   fragment.fragment_identity);
        if (!yvex_core_u64_add(result->generated_text_bytes,
                               fragment.byte_count, &next_text) ||
            next_text > text_capacity ||
            next_text > context->options.maximum_output_bytes) {
            rc = generation_refuse(err, YVEX_ERR_NOMEM,
                                   "generated text exceeds its admitted output budget");
        } else {
            rc = yvex_token_sequence_transition(
                context->sequence, sequence_ordinal,
                YVEX_TOKEN_APPEND_MODEL_COMMITTED,
                YVEX_TOKEN_APPEND_DETOKENIZED, err);
            token->sequence_state_after = YVEX_TOKEN_APPEND_DETOKENIZED;
            if (rc == YVEX_OK && fragment.byte_count)
                memcpy(text + result->generated_text_bytes, fragment.bytes,
                       (size_t)fragment.byte_count);
            if (rc == YVEX_OK)
                rc = yvex_token_sequence_transition(
                    context->sequence, sequence_ordinal,
                    YVEX_TOKEN_APPEND_DETOKENIZED,
                    YVEX_TOKEN_APPEND_TEXT_PUBLISHED, err);
            if (rc == YVEX_OK) {
                token->sequence_state_after = YVEX_TOKEN_APPEND_TEXT_PUBLISHED;
                token->text_published = 1;
                result->generated_text_bytes = next_text;
                result->text_published_token_count++;
                if (token->suppressed) result->suppressed_token_count++;
            }
        }
    }
    yvex_tokenizer_fragment_clear(&fragment);
    if (!generation_token_identity(token, token->token_step_identity) && rc == YVEX_OK)
        rc = generation_refuse(err, YVEX_ERR_STATE,
                               "ordinary token identity derivation failed");
    return rc;
}

static int generation_decoder_finish(
    yvex_runtime_generation_context *context, yvex_error *err)
{
    yvex_tokenizer_fragment fragment;
    int rc;
    memset(&fragment, 0, sizeof(fragment));
    rc = yvex_tokenizer_decoder_finish(context->decoder, &fragment, err);
    if (rc == YVEX_OK && fragment.byte_count)
        rc = generation_refuse(err, YVEX_ERR_STATE,
                               "decoder finish unexpectedly produced unowned text bytes");
    yvex_tokenizer_fragment_clear(&fragment);
    return rc;
}

static yvex_runtime_generation_stop_reason generation_failure_stop(
    int rc, int prompt_stage,
    const yvex_runtime_generation_token_result *token)
{
    if (rc == YVEX_ERR_CANCELLED) return YVEX_GENERATION_STOP_CANCELLED;
    if (prompt_stage) return YVEX_GENERATION_STOP_TOKENIZER_FAILURE;
    if (token && token->model_committed && !token->detokenized)
        return YVEX_GENERATION_STOP_TOKENIZER_FAILURE;
    if (token && token->detokenized && !token->text_published)
        return YVEX_GENERATION_STOP_OUTPUT_FAILURE;
    return YVEX_GENERATION_STOP_MODEL_FAILURE;
}
/*
 * Finalize exact aggregate progress after success, cancellation, or lower-owner failure.
 *
 * Snapshots authoritative KV/RNG state and seals field-wise identities. Identity/snapshot refusal
 * replaces success but never erases earlier lower-owner progress.
 */
static int generation_result_finish(
    yvex_runtime_generation_context *context,
    yvex_runtime_generation_token_result *tokens,
    const unsigned char *text, unsigned long long text_capacity,
    yvex_runtime_generation_result *result, int rc, yvex_error *err)
{
    yvex_graph_attention_state_summary state;
    yvex_runtime_sampling_context_summary sampling;
    char canonical[YVEX_SHA256_HEX_CAP];
    unsigned long long index;
    yvex_error secondary;
    int finish_rc;
    yvex_error_clear(&secondary);
    finish_rc = generation_state_summary(context->session, &state, &secondary);
    if (finish_rc == YVEX_OK) {
        result->final_position = state.next_position;
        result->final_persistent_generation = state.generation;
        yvex_runtime_identity_copy(result->final_persistent_state_digest,
                                   state.state_content_identity);
    } else if (rc == YVEX_OK) {
        rc = finish_rc;
        if (err) *err = secondary;
    }
    finish_rc = yvex_runtime_sampling_context_snapshot(context->sampling,
                                                       &sampling, &secondary);
    if (finish_rc == YVEX_OK)
        yvex_runtime_identity_copy(result->final_rng_identity,
                                   sampling.rng_state_identity);
    else if (rc == YVEX_OK) {
        rc = finish_rc;
        if (err) *err = secondary;
    }
    for (index = 0ull; index < result->sampled_token_count; ++index) {
        if (!tokens[index].token_step_identity[0] &&
            !generation_token_identity(&tokens[index],
                                       tokens[index].token_step_identity) &&
            rc == YVEX_OK)
            rc = generation_refuse(err, YVEX_ERR_STATE,
                                   "partial generation token identity failed");
    }
    result->has_incomplete_token =
        result->sampled_token_count >
        result->model_committed_token_count + result->terminal_token_count;
    result->first_incomplete_token = result->has_incomplete_token
                                         ? result->sampled_token_count - 1ull
                                         : result->sampled_token_count;
    if (result->profile.schema_version == YVEX_RUNTIME_PROFILE_SCHEMA_V1) {
        unsigned long long completed = yvex_core_monotonic_ns();
        finish_rc = result->profile.mode == YVEX_RUNTIME_PROFILE_OFF
                        ? YVEX_OK
                        : runtime_profile_counter_add(
                              &result->profile, YVEX_RUNTIME_PROFILE_GENERATED_TOKENS,
                              result->model_committed_token_count, &secondary);
        if (finish_rc == YVEX_OK && result->profile.mode != YVEX_RUNTIME_PROFILE_OFF &&
            completed > result->profile.started_ns)
            finish_rc = runtime_profile_phase_add(
                &result->profile, YVEX_RUNTIME_PROFILE_TOTAL_GENERATION,
                completed - result->profile.started_ns, &secondary);
        if (finish_rc == YVEX_OK)
            finish_rc = runtime_profile_finish(&result->profile, &secondary);
        if (finish_rc != YVEX_OK && rc == YVEX_OK) {
            rc = finish_rc;
            if (err) *err = secondary;
        }
    }
    result->completed = rc == YVEX_OK;
    result->cancelled = rc == YVEX_ERR_CANCELLED;
    result->failed = rc != YVEX_OK && rc != YVEX_ERR_CANCELLED;
    result->partial = rc != YVEX_OK &&
                      (result->prefill_chunk_count || result->sampled_token_count ||
                       result->model_committed_token_count ||
                       result->generated_text_bytes);
    result->status = result->completed
                         ? YVEX_GENERATION_STATUS_COMPLETE
                         : (result->cancelled
                                ? YVEX_GENERATION_STATUS_CANCELLED
                                : (result->partial ? YVEX_GENERATION_STATUS_PARTIAL
                                                   : YVEX_GENERATION_STATUS_FAILED));
    if (!generation_tokens_identity(tokens, result->sampled_token_count,
                                    result->generated_token_identity) ||
        !generation_bytes_digest("yvex.runtime.generation.text.v1", text,
                                 result->generated_text_bytes,
                                 result->generated_text_digest) ||
        !generation_execution_identity(result, tokens,
                                       result->generation_execution_identity)) {
        if (rc == YVEX_OK)
            rc = generation_refuse(err, YVEX_ERR_STATE,
                                   "generation aggregate identity failed");
    } else if (!generation_execution_identity(
                   result, tokens, canonical) ||
               strcmp(canonical, result->generation_execution_identity) != 0) {
        if (rc == YVEX_OK)
            rc = generation_refuse(err, YVEX_ERR_STATE,
                                   "generation aggregate identity is not canonical");
    }
    if (text && text_capacity > result->generated_text_bytes)
        ((unsigned char *)text)[result->generated_text_bytes] = '\0';
    return rc;
}

static int generation_turn_prepare(
    yvex_runtime_generation_context *context,
    const yvex_runtime_generation_turn_request *turn,
    yvex_tokenizer_encode_result *encoded, yvex_rendered_prompt *rendered,
    const yvex_transformer_plan_summary **transformer,
    yvex_runtime_generation_result *result, yvex_error *err)
{
    const yvex_runtime_generation_request *request = turn->prompt;
    yvex_runtime_sampling_context_summary initial_sampling;
    yvex_graph_attention_state_summary initial_state;
    unsigned long long index;
    int rc;
    if (!context->continuation_allowed)
        return generation_refuse(
            err, YVEX_ERR_STATE,
            "generation context requires reset after an incomplete turn");
    rc = generation_state_summary(context->session, &initial_state, err);
    if (rc == YVEX_OK &&
        (initial_state.next_position != turn->committed_prefix_token_count ||
         initial_state.committed_sequence_length !=
             turn->committed_prefix_token_count ||
         initial_state.transaction_active))
        rc = generation_refuse(
            err, YVEX_ERR_STATE,
            "committed token prefix does not match persistent session extent");
    if (rc == YVEX_OK) {
        result->initial_position = initial_state.next_position;
        result->reusable_prefix_token_count =
            turn->committed_prefix_token_count;
        if (!generation_prefix_identity(turn->committed_prefix_token_ids,
                                        turn->committed_prefix_token_count,
                                        result->reusable_prefix_identity))
            rc = generation_refuse(err, YVEX_ERR_STATE,
                                   "committed prefix identity derivation failed");
    }
    if (rc == YVEX_OK) rc = yvex_tokenizer_decoder_reset(context->decoder, err);
    if (rc == YVEX_OK) rc = yvex_token_sequence_reset(context->sequence, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_sampling_context_snapshot(
            context->sampling, &initial_sampling, err);
    if (rc == YVEX_OK)
        yvex_runtime_identity_copy(result->initial_rng_identity,
                                   initial_sampling.rng_state_identity);
    if (rc == YVEX_OK) rc = generation_cancelled(context, err);
    if (rc == YVEX_OK)
        rc = generation_encode_prompt(context, request, rendered, encoded,
                                      result->prompt_identity, err);
    if (rc == YVEX_OK) {
        result->prompt_bytes = request->kind == YVEX_GENERATION_INPUT_TEXT
                                   ? request->text_bytes
                                   : rendered->len;
        result->prompt_token_count = encoded->tokens.len;
        yvex_runtime_identity_copy(result->prompt_token_identity,
                                   encoded->token_ids_identity);
    }
    *transformer = yvex_transformer_plan_summary_get(
        yvex_runtime_transformer_context_plan(context->transformer));
    if (rc == YVEX_OK &&
        (!*transformer ||
         encoded->tokens.len > context->options.context_capacity ||
         encoded->tokens.len <= turn->committed_prefix_token_count ||
         turn->prompt_token_capacity < encoded->tokens.len))
        rc = generation_refuse(
            err, YVEX_ERR_BOUNDS,
            "prompt suffix or caller token capacity is incompatible");
    for (index = 0ull;
         rc == YVEX_OK && index < turn->committed_prefix_token_count; ++index)
        if (encoded->tokens.ids[index] !=
            turn->committed_prefix_token_ids[index])
            rc = generation_refuse(
                err, YVEX_ERR_STATE,
                "committed model tokens are not an exact prompt prefix");
    if (rc == YVEX_OK)
        memcpy(turn->prompt_token_ids, encoded->tokens.ids,
               (size_t)encoded->tokens.len * sizeof(*turn->prompt_token_ids));
    if (rc == YVEX_OK)
        result->new_prefill_token_count =
            encoded->tokens.len - turn->committed_prefix_token_count;
    if (rc == YVEX_OK && turn->progress_sink)
        rc = turn->progress_sink(
            turn->progress_context, YVEX_GENERATION_PROGRESS_PROMPT_ACCEPTED,
            encoded->tokens.len, turn->committed_prefix_token_count, err);
    return rc;
}

int yvex_runtime_generation_turn_execute(
    yvex_runtime_generation_context *context,
    const yvex_runtime_generation_turn_request *turn,
    yvex_runtime_generation_token_result *tokens,
    unsigned long long token_capacity, unsigned char *text,
    unsigned long long text_capacity, yvex_runtime_generation_result *result,
    yvex_error *err)
{
    const yvex_runtime_generation_request *request = turn ? turn->prompt : NULL;
    const yvex_transformer_plan_summary *transformer;
    yvex_tokenizer_encode_result encoded;
    yvex_rendered_prompt rendered;
    yvex_runtime_transformer_result prefill;
    yvex_runtime_decode_step_result last_decode;
    yvex_graph_attention_state_summary before;
    float *prefill_hidden = NULL;
    unsigned long long prefill_values = 0ull, prefill_chunks = 0ull;
    unsigned long long turn_maximum = turn ? turn->maximum_new_tokens : 0ull;
    unsigned long long started, completed;
    char workload_identity[YVEX_SHA256_HEX_CAP];
    int rc, prompt_stage = 1, use_prefill = 1;
    if (result) memset(result, 0, sizeof(*result));
    if (!context || !turn ||
        turn->schema_version != YVEX_RUNTIME_GENERATION_TURN_SCHEMA_V1 ||
        !request || !tokens || !text || !result || !turn_maximum ||
        turn_maximum > context->options.maximum_new_tokens ||
        token_capacity < turn_maximum ||
        (!turn->committed_prefix_token_ids &&
         turn->committed_prefix_token_count) ||
        (!turn->prompt_token_ids && turn->prompt_token_capacity) ||
        text_capacity < context->options.maximum_output_bytes)
        return generation_refuse(err, YVEX_ERR_INVALID_ARG,
                                 "generation outputs do not satisfy sealed capacities");
    rc = generation_enter(context, err);
    if (rc != YVEX_OK) return rc;
    memset(tokens, 0, (size_t)turn_maximum * sizeof(*tokens));
    memset(text, 0, (size_t)context->options.maximum_output_bytes);
    memset(&encoded, 0, sizeof(encoded));
    memset(&rendered, 0, sizeof(rendered));
    memset(&last_decode, 0, sizeof(last_decode));
    result->schema_version = YVEX_RUNTIME_GENERATION_SCHEMA_V2;
    result->requested_new_tokens = turn_maximum;
    yvex_runtime_identity_copy(result->generation_plan_identity,
                               context->plan.generation_plan_identity);
    rc = generation_workload_identity(context, turn, workload_identity)
             ? runtime_profile_begin(
                   &result->profile, generation_profile_mode(context->options.trace_policy),
                   YVEX_RUNTIME_PROFILE_GENERATION, context->options.backend,
                   context->model_view->binding->artifact_identity,
                   context->model_view->binding->profile_identity,
                   context->model_view->binding->identity,
                   context->plan.runtime_model_identity,
                   context->plan.generation_plan_identity, workload_identity, err)
             : generation_refuse(err, YVEX_ERR_STATE,
                                 "generation workload identity derivation failed");
    started = yvex_core_monotonic_ns();
    if (rc == YVEX_OK)
        rc = generation_turn_prepare(context, turn, &encoded, &rendered,
                                     &transformer, result, err);
    completed = yvex_core_monotonic_ns();
    if (rc == YVEX_OK)
        rc = generation_profile_phase(
            &result->profile,
            request->kind == YVEX_GENERATION_INPUT_TEXT
                ? YVEX_RUNTIME_PROFILE_TOKENIZER
                : YVEX_RUNTIME_PROFILE_PROMPT_RENDERING,
            completed - started, err);
    if (rc == YVEX_OK && result->profile.mode != YVEX_RUNTIME_PROFILE_OFF) {
        rc = runtime_profile_counter_add(
            &result->profile, YVEX_RUNTIME_PROFILE_PROMPT_TOKENS,
            result->prompt_token_count, err);
        if (rc == YVEX_OK)
            rc = runtime_profile_counter_add(
                &result->profile, YVEX_RUNTIME_PROFILE_REUSED_TOKENS,
                result->reusable_prefix_token_count, err);
        if (rc == YVEX_OK)
            rc = runtime_profile_counter_add(
                &result->profile, YVEX_RUNTIME_PROFILE_NEW_PREFILL_TOKENS,
                result->new_prefill_token_count, err);
    }
    if (rc == YVEX_OK) prompt_stage = 0;
    if (rc == YVEX_OK && turn->progress_sink)
        rc = turn->progress_sink(
            turn->progress_context, YVEX_GENERATION_PROGRESS_PREFILL_STARTED,
            result->new_prefill_token_count, context->options.prefill_chunk_tokens,
            err);
    started = yvex_core_monotonic_ns();
    if (rc == YVEX_OK)
        rc = generation_prefill(context, &encoded,
                                turn->committed_prefix_token_count, turn, &prefill_hidden,
                                &prefill_values, &prefill, &prefill_chunks,
                                &result->profile, err);
    completed = yvex_core_monotonic_ns();
    if (rc == YVEX_OK)
        rc = generation_profile_phase(&result->profile,
                                       YVEX_RUNTIME_PROFILE_TOTAL_PREFILL,
                                       completed - started, err);
    result->prefill_chunk_count = prefill_chunks;
    if (rc == YVEX_OK && turn->progress_sink)
        rc = turn->progress_sink(
            turn->progress_context, YVEX_GENERATION_PROGRESS_PREFILL_COMPLETED,
            result->new_prefill_token_count, prefill_chunks, err);
    while (rc == YVEX_OK && result->stop_reason == YVEX_GENERATION_STOP_NONE) {
        yvex_runtime_logits_row_result logits_result = {0};
        yvex_runtime_sampling_result sample = {0};
        yvex_runtime_generation_token_result *token;
        unsigned long long sequence_ordinal;
        int additional_stop = 0;
        if (result->model_committed_token_count == turn_maximum) {
            result->stop_reason = YVEX_GENERATION_STOP_MAX_NEW_TOKENS;
            break;
        }
        rc = generation_state_summary(context->session, &before, err);
        if (rc == YVEX_OK && before.next_position == context->options.context_capacity) {
            result->stop_reason = YVEX_GENERATION_STOP_CONTEXT_CAPACITY;
            break;
        }
        if (rc == YVEX_OK) rc = generation_cancelled(context, err);
        if (rc == YVEX_OK)
            rc = generation_project_sample(
                context, use_prefill ? &prefill : NULL,
                prefill_hidden, prefill_values,
                use_prefill ? NULL : &last_decode,
                &logits_result, &sample, &result->profile, err);
        if (logits_result.completed) result->logits_projection_count++;
        if (rc != YVEX_OK) break;
        token = &tokens[result->sampled_token_count];
        token->schema_version = YVEX_RUNTIME_GENERATION_SCHEMA_V2;
        token->ordinal = result->sampled_token_count;
        token->sampled = 1;
        token->sampled_token_id = sample.selected_token_id;
        token->sequence_state_before = YVEX_TOKEN_APPEND_PROPOSED;
        token->position_before = before.next_position;
        token->position_after = before.next_position;
        token->persistent_generation_before = before.generation;
        token->persistent_generation_after = before.generation;
        yvex_runtime_identity_copy(token->source_logits_identity,
                                   logits_result.logits_row_identity);
        yvex_runtime_identity_copy(token->sampling_result_identity,
                                   sample.execution_identity);
        result->sampling_draw_count += sample.rng_draw_count;
        result->sampled_token_count++;
        rc = generation_token_classify(context, token->sampled_token_id,
                                       &token->classification,
                                       &additional_stop, err);
        if (rc == YVEX_OK)
            rc = yvex_token_sequence_append(
                context->sequence, token->sampled_token_id,
                transformer->vocabulary_size, &sequence_ordinal, err);
        if (rc == YVEX_OK &&
            (token->classification.eos || token->classification.stop ||
             additional_stop))
            rc = generation_terminal_token(token, result,
                                           additional_stop, err);
        else if (rc == YVEX_OK)
            rc = generation_commit_ordinary(
                context, token, sequence_ordinal, text, text_capacity,
                result, &last_decode, err);
        if (rc == YVEX_OK && turn->fragment_sink && token->text_published) {
            started = yvex_core_monotonic_ns();
            rc = turn->fragment_sink(
                turn->fragment_context, token,
                text + token->text_byte_offset, token->text_byte_count, err);
            completed = yvex_core_monotonic_ns();
            if (rc == YVEX_OK)
                rc = generation_profile_phase(
                    &result->profile, YVEX_RUNTIME_PROFILE_PROVIDER_PUBLICATION,
                    completed - started, err);
            if (rc != YVEX_OK)
                result->stop_reason = YVEX_GENERATION_STOP_OUTPUT_FAILURE;
        }
        use_prefill = 0;
    }
    if (rc == YVEX_OK) {
        rc = generation_decoder_finish(context, err);
        if (rc != YVEX_OK)
            result->stop_reason = rc == YVEX_ERR_CANCELLED
                                      ? YVEX_GENERATION_STOP_CANCELLED
                                      : YVEX_GENERATION_STOP_TOKENIZER_FAILURE;
    }
    if (rc != YVEX_OK && result->stop_reason == YVEX_GENERATION_STOP_NONE)
        result->stop_reason = generation_failure_stop(
            rc, prompt_stage,
            result->sampled_token_count
                ? &tokens[result->sampled_token_count - 1ull]
                : NULL);
    rc = generation_result_finish(context, tokens, text, text_capacity,
                                  result, rc, err);
    yvex_core_free(prefill_hidden);
    yvex_tokenizer_encode_result_clear(&encoded);
    yvex_rendered_prompt_free(&rendered);
    context->continuation_allowed = rc == YVEX_OK;
    generation_leave(context, rc, 1);
    if (rc == YVEX_OK) yvex_error_clear(err);
    return rc;
}
/* Preserve the fresh one-shot entrypoint as a strict wrapper over reusable turn execution. */
int yvex_runtime_generation_execute(
    yvex_runtime_generation_context *context,
    const yvex_runtime_generation_request *request,
    yvex_runtime_generation_token_result *tokens,
    unsigned long long token_capacity, unsigned char *text,
    unsigned long long text_capacity, yvex_runtime_generation_result *result,
    yvex_error *err)
{
    yvex_runtime_generation_turn_request turn;
    unsigned int *prompt_tokens;
    int rc;
    if (!context || context->options.context_capacity > SIZE_MAX / sizeof(unsigned int))
        return generation_refuse(err, YVEX_ERR_INVALID_ARG,
                                 "generation context is required");
    prompt_tokens = yvex_core_calloc((size_t)context->options.context_capacity,
                                     sizeof(*prompt_tokens));
    if (!prompt_tokens)
        return generation_refuse(err, YVEX_ERR_NOMEM,
                                 "fresh generation prompt directory allocation failed");
    memset(&turn, 0, sizeof(turn));
    turn.schema_version = YVEX_RUNTIME_GENERATION_TURN_SCHEMA_V1;
    turn.prompt = request;
    turn.prompt_token_ids = prompt_tokens;
    turn.prompt_token_capacity = context->options.context_capacity;
    rc = yvex_runtime_generation_turn_execute(
        context, &turn, tokens, token_capacity, text, text_capacity, result, err);
    yvex_core_free(prompt_tokens);
    return rc;
}
/*
 * Validate every authoritative aggregate and token field against canonical identities.
 *
 * Refuses any structural, identity, ordering, or publication mutation.
 */
int yvex_runtime_generation_result_validate(
    const yvex_runtime_generation_plan_summary *plan,
    const yvex_runtime_generation_token_result *tokens,
    unsigned long long token_capacity, const unsigned char *text,
    unsigned long long text_capacity,
    const yvex_runtime_generation_result *result, yvex_error *err)
{
    char identity[YVEX_SHA256_HEX_CAP], text_digest[YVEX_SHA256_HEX_CAP];
    unsigned long long index, published_offset = 0ull, committed = 0ull;
    unsigned long long published = 0ull, terminal = 0ull, suppressed = 0ull;
    if (!plan || !result || (!tokens && result->sampled_token_count) ||
        (!text && result->generated_text_bytes) ||
        plan->schema_version != YVEX_RUNTIME_GENERATION_SCHEMA_V2 ||
        result->schema_version != YVEX_RUNTIME_GENERATION_SCHEMA_V2 ||
        result->sampled_token_count > token_capacity ||
        result->generated_text_bytes > text_capacity ||
        !result->requested_new_tokens ||
        result->requested_new_tokens > plan->maximum_new_tokens ||
        result->sampled_token_count > result->requested_new_tokens ||
        result->model_committed_token_count + result->terminal_token_count >
            result->sampled_token_count ||
        result->reusable_prefix_token_count != result->initial_position ||
        result->reusable_prefix_token_count > result->prompt_token_count ||
        result->new_prefill_token_count !=
            result->prompt_token_count - result->reusable_prefix_token_count ||
        result->profile.schema_version != YVEX_RUNTIME_PROFILE_SCHEMA_V1 ||
        runtime_profile_validate(&result->profile, NULL) != YVEX_OK ||
        !yvex_sha256_hex_valid(result->reusable_prefix_identity) ||
        strcmp(result->generation_plan_identity,
               plan->generation_plan_identity) != 0 ||
        !generation_plan_identity(plan, identity) ||
        strcmp(identity, plan->generation_plan_identity) != 0)
        return generation_refuse(err, YVEX_ERR_FORMAT,
                                 "generation result geometry or plan identity is invalid");
    for (index = 0ull; index < result->sampled_token_count; ++index) {
        const yvex_runtime_generation_token_result *token = &tokens[index];
        if (token->ordinal != index || !token->sampled ||
            token->sampled_token_id != token->classification.token_id ||
            (token->decode_submitted &&
             token->decode_input_token_id != token->sampled_token_id) ||
            (!token->decode_submitted && token->decode_input_token_id) ||
            !generation_token_identity(token, identity) ||
            strcmp(identity, token->token_step_identity) != 0)
            return generation_refuse(err, YVEX_ERR_FORMAT,
                                     "generation token identity is invalid");
        if (token->terminal) {
            if (token->decode_submitted || token->model_committed ||
                token->decode_input_token_id ||
                token->position_after != token->position_before ||
                token->persistent_generation_after !=
                    token->persistent_generation_before ||
                token->decode_execution_identity[0] ||
                token->sequence_state_after != YVEX_TOKEN_APPEND_PROPOSED)
                return generation_refuse(err, YVEX_ERR_FORMAT,
                                         "terminal token falsely claims model commit");
            terminal++;
        }
        if (token->model_committed) {
            if (token->terminal || !token->decode_submitted ||
                token->decode_input_token_id != token->sampled_token_id ||
                token->position_after != token->position_before + 1ull ||
                token->persistent_generation_after !=
                    token->persistent_generation_before + 1ull ||
                !yvex_sha256_hex_valid(token->decode_execution_identity) ||
                !yvex_sha256_hex_valid(token->persistent_state_digest))
                return generation_refuse(err, YVEX_ERR_FORMAT,
                                         "ordinary token model transition is invalid");
            committed++;
        } else if (token->position_after != token->position_before ||
                   token->persistent_generation_after !=
                       token->persistent_generation_before ||
                   token->decode_execution_identity[0] ||
                   token->sequence_state_after > YVEX_TOKEN_APPEND_SUBMITTED) {
            return generation_refuse(err, YVEX_ERR_FORMAT,
                                     "uncommitted token claims downstream progress");
        }
        if (token->text_published) {
            if (!token->model_committed || !token->detokenized ||
                token->sequence_state_after != YVEX_TOKEN_APPEND_TEXT_PUBLISHED ||
                token->text_byte_offset != published_offset ||
                !yvex_core_u64_add(published_offset, token->text_byte_count,
                                   &published_offset))
                return generation_refuse(err, YVEX_ERR_FORMAT,
                                         "generation text directory is discontinuous");
            published++;
        }
        if (token->suppressed) suppressed++;
    }
    if (committed != result->model_committed_token_count ||
        published != result->text_published_token_count ||
        terminal != result->terminal_token_count ||
        suppressed != result->suppressed_token_count ||
        published_offset != result->generated_text_bytes ||
        result->decode_step_count != committed ||
        result->logits_projection_count < result->sampled_token_count ||
        result->has_incomplete_token !=
            (result->sampled_token_count > committed + terminal) ||
        result->first_incomplete_token !=
            (result->has_incomplete_token ? result->sampled_token_count - 1ull
                                          : result->sampled_token_count))
        return generation_refuse(err, YVEX_ERR_FORMAT,
                                 "generation aggregate counters are inconsistent");
    if (result->completed &&
        (result->status != YVEX_GENERATION_STATUS_COMPLETE || result->partial ||
         result->cancelled || result->failed ||
         result->final_position != result->initial_position +
             result->new_prefill_token_count + committed))
        return generation_refuse(err, YVEX_ERR_FORMAT,
                                 "complete generation state is inconsistent");
    if (!generation_tokens_identity(tokens, result->sampled_token_count, identity) ||
        strcmp(identity, result->generated_token_identity) != 0 ||
        !generation_bytes_digest("yvex.runtime.generation.text.v1", text,
                                 result->generated_text_bytes, text_digest) ||
        strcmp(text_digest, result->generated_text_digest) != 0 ||
        !generation_execution_identity(result, tokens, identity) ||
        strcmp(identity, result->generation_execution_identity) != 0)
        return generation_refuse(err, YVEX_ERR_FORMAT,
                                 "generation aggregate evidence was mutated");
    yvex_error_clear(err);
    return YVEX_OK;
}

static int generation_context_snapshot(
    const yvex_runtime_generation_context *context,
    yvex_runtime_generation_context_summary *summary, yvex_error *err)
{
    yvex_runtime_generation_context *mutable =
        (yvex_runtime_generation_context *)context;
    yvex_runtime_sampling_context_summary sampling;
    yvex_token_sequence_summary sequence;
    unsigned int lifecycle;
    int rc;
    if (!context || !summary)
        return generation_refuse(err, YVEX_ERR_INVALID_ARG,
                                 "generation context and snapshot output are required");
    rc = generation_enter(mutable, err);
    if (rc != YVEX_OK) return rc;
    memset(summary, 0, sizeof(*summary));
    rc = yvex_runtime_sampling_context_snapshot(context->sampling, &sampling, err);
    if (rc == YVEX_OK)
        rc = yvex_token_sequence_summary_get(context->sequence, &sequence, err);
    lifecycle = atomic_load_explicit(&context->lifecycle, memory_order_acquire);
    if (rc == YVEX_OK) {
        summary->schema_version = YVEX_RUNTIME_GENERATION_SCHEMA_V2;
        summary->open = !(lifecycle & GENERATION_LIFECYCLE_CLOSING);
        summary->busy = 0;
        summary->closing = (lifecycle & GENERATION_LIFECYCLE_CLOSING) != 0u;
        summary->execution_count = context->execution_count;
        summary->failure_count = context->failure_count +
            atomic_load_explicit(&context->admission_failures, memory_order_relaxed);
        summary->cancellation_count = context->cancellation_count;
        summary->token_capacity = context->options.maximum_new_tokens;
        summary->text_capacity = context->options.maximum_output_bytes;
        summary->workspace_bytes = context->workspace_bytes + sampling.workspace_bytes;
        yvex_runtime_identity_copy(summary->generation_plan_identity,
                                   context->plan.generation_plan_identity);
        yvex_runtime_identity_copy(summary->token_sequence_identity,
                                   sequence.state_identity);
        yvex_runtime_identity_copy(summary->rng_state_identity,
                                   sampling.rng_state_identity);
    }
    generation_leave(mutable, rc, 0);
    if (rc == YVEX_OK) yvex_error_clear(err);
    return rc;
}
/*
 * Transfer unique close ownership, block new entry, drain active execution, and release
 * dependencies.
 *
 * Caller aliases are invalid after close ownership transfer.
 */
int yvex_runtime_generation_context_close(
    yvex_runtime_generation_context **context, yvex_error *err)
{
    yvex_runtime_generation_context *owner;
    unsigned int observed, desired;
    int rc;
    if (!context || !*context) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    owner = *context;
    observed = atomic_load_explicit(&owner->lifecycle, memory_order_acquire);
    while (!(observed & GENERATION_LIFECYCLE_CLOSING)) {
        desired = observed | GENERATION_LIFECYCLE_CLOSING;
        if (atomic_compare_exchange_weak_explicit(
                &owner->lifecycle, &observed, desired,
                memory_order_acq_rel, memory_order_acquire)) break;
    }
    if (owner->drain_mutex_ready) {
        if (pthread_mutex_lock(&owner->drain_mutex) != 0)
            return generation_refuse(err, YVEX_ERR_STATE,
                                     "generation close drain lock failed");
        while (atomic_load_explicit(&owner->lifecycle, memory_order_acquire) &
               GENERATION_LIFECYCLE_ACTIVE) {
            if (!owner->drain_condition_ready ||
                pthread_cond_wait(&owner->drain_condition,
                                  &owner->drain_mutex) != 0) {
                (void)pthread_mutex_unlock(&owner->drain_mutex);
                return generation_refuse(err, YVEX_ERR_STATE,
                                         "generation close drain failed");
            }
        }
        (void)pthread_mutex_unlock(&owner->drain_mutex);
    }
    rc = yvex_runtime_sampling_context_close(&owner->sampling, err);
    if (rc == YVEX_OK) rc = yvex_runtime_logits_context_close(&owner->logits, err);
    if (rc == YVEX_OK) rc = yvex_runtime_decode_context_close(&owner->decode, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_transformer_context_close(&owner->transformer, err);
    if (rc != YVEX_OK) return rc;
    yvex_tokenizer_decoder_close(&owner->decoder);
    yvex_token_sequence_close(&owner->sequence);
    if (owner->drain_condition_ready &&
        pthread_cond_destroy(&owner->drain_condition) != 0)
        return generation_refuse(err, YVEX_ERR_STATE,
                                 "generation close condition cleanup failed");
    owner->drain_condition_ready = 0;
    if (owner->drain_mutex_ready &&
        pthread_mutex_destroy(&owner->drain_mutex) != 0)
        return generation_refuse(err, YVEX_ERR_STATE,
                                 "generation close mutex cleanup failed");
    owner->drain_mutex_ready = 0;
    atomic_store_explicit(&owner->lifecycle, GENERATION_LIFECYCLE_CLOSED,
                          memory_order_release);
    yvex_core_free(owner->logits_row);
    yvex_core_free(owner->hidden);
    yvex_core_free(owner->additional_stops);
    memset(owner, 0, sizeof(*owner));
    yvex_core_free(owner);
    *context = NULL;
    yvex_error_clear(err);
    return YVEX_OK;
}

const char *yvex_runtime_generation_stop_reason_name(
    yvex_runtime_generation_stop_reason reason)
{
    static const char *const names[] = {
        "none", "eos", "tokenizer-stop-token", "max-new-tokens",
        "context-capacity", "cancelled", "model-failure",
        "tokenizer-failure", "output-failure"
    };
    return reason <= YVEX_GENERATION_STOP_OUTPUT_FAILURE
               ? names[(unsigned int)reason]
               : "unknown";
}

static int generation_context_cleanup(void **opaque, yvex_error *err)
{
    yvex_runtime_generation_context *context =
        opaque ? (yvex_runtime_generation_context *)*opaque : NULL;
    int rc = yvex_runtime_generation_context_close(&context, err);
    if (opaque) *opaque = context;
    return rc;
}

static void generation_operator_ready(yvex_generation_operator_result *result,
                                      yvex_backend_kind backend)
{
    result->generation_plan_ready = 1;
    result->generation_prompt_ready = 1;
    result->generation_prefill_ready = 1;
    result->generation_first_token_ready = result->execution.sampled_token_count > 0ull;
    result->sampled_token_feedback_ready =
        result->execution.model_committed_token_count > 0ull;
    result->generation_decode_loop_ready = 1;
    result->generation_logits_loop_ready = 1;
    result->generation_sampling_loop_ready = 1;
    result->generation_token_append_ready = 1;
    result->generation_eos_stop_ready = 1;
    result->generation_context_stop_ready = 1;
    result->generation_incremental_text_ready = 1;
    result->generation_partial_progress_ready = 1;
    result->generation_cpu_ready = backend == YVEX_BACKEND_KIND_CPU;
    result->generation_cuda_model_path_ready = backend == YVEX_BACKEND_KIND_CUDA;
    result->generation_loop_ready = 1;
    result->generation_ready = 1;
}
/*
 * Execute one operator-reachable prompt through the production generation API.
 *
 * Preserves typed partial generation and retry-safe cleanup ownership.
 */
int yvex_runtime_generation_operator_execute(
    const yvex_generation_operator_request *request,
    yvex_generation_operator_result *result,
    yvex_runtime_cleanup_lease **retained_cleanup, yvex_error *err)
{
    yvex_runtime_model_open_request model_request = {0};
    yvex_runtime_session_open_request session_request = {0};
    yvex_runtime_generation_options options = {0};
    yvex_runtime_generation_request execution_request = {0};
    yvex_runtime_model_failure failure = {0};
    yvex_runtime_cleanup_lease *cleanup = NULL;
    yvex_runtime_model *model = NULL;
    yvex_runtime_execution_session *session = NULL;
    yvex_runtime_generation_context *context = NULL;
    const yvex_runtime_model_view *view;
    unsigned long long text_allocation;
    yvex_error primary, cleanup_error, validation_error;
    int adopted = 0, rc, close_rc;
    if (result) memset(result, 0, sizeof(*result));
    if (!request || !result || !retained_cleanup || *retained_cleanup ||
        !request->target || !request->artifact_path ||
        !request->runtime_binding_path || !request->context_capacity ||
        !request->prefill_chunk_tokens || !request->maximum_new_tokens ||
        !request->maximum_output_bytes ||
        request->input_kind > YVEX_GENERATION_INPUT_MESSAGES ||
        (request->backend != YVEX_BACKEND_KIND_CPU &&
         request->backend != YVEX_BACKEND_KIND_CUDA))
        return generation_refuse(err, YVEX_ERR_INVALID_ARG,
                                 "complete generation operator arguments are required");
    yvex_core_text_copy(result->command, sizeof(result->command),
                        "execute transformer generate");
    yvex_core_text_copy(result->target, sizeof(result->target), request->target);
    yvex_core_text_copy(result->backend, sizeof(result->backend),
                        request->backend == YVEX_BACKEND_KIND_CUDA ? "cuda" : "cpu");
    yvex_core_text_copy(result->sampling_execution_kind,
                        sizeof(result->sampling_execution_kind), "common-host");
    yvex_core_text_copy(result->tokenizer_execution_kind,
                        sizeof(result->tokenizer_execution_kind), "common-host");
    model_request.artifact_path = request->artifact_path;
    model_request.runtime_binding_path = request->runtime_binding_path;
    model_request.target_id = request->target;
    model_request.maximum_host_bytes = request->maximum_host_bytes;
    session_request.backend = request->backend;
    session_request.maximum_host_bytes = request->maximum_host_bytes;
    session_request.maximum_device_bytes = request->maximum_device_bytes;
    rc = yvex_runtime_cleanup_lease_acquire(
        &cleanup, &model_request, &session_request, &model, &session,
        &failure, err);
    options.schema_version = YVEX_RUNTIME_GENERATION_SCHEMA_V2;
    options.backend = request->backend;
    options.context_capacity = request->context_capacity;
    options.prefill_chunk_tokens = request->prefill_chunk_tokens;
    options.maximum_new_tokens = request->maximum_new_tokens;
    options.maximum_output_bytes = request->maximum_output_bytes;
    options.maximum_host_bytes = request->maximum_host_bytes;
    options.maximum_device_bytes = request->maximum_device_bytes;
    options.trace_policy = YVEX_RUNTIME_TRACE_SUMMARY;
    options.sampling_policy = request->sampling_policy;
    options.cancel_requested = request->cancel_requested;
    options.cancel_context = request->cancel_context;
    if (rc == YVEX_OK)
        rc = yvex_runtime_generation_context_open(
            &context, model, session, &options, err);
    if (rc == YVEX_OK) {
        result->plan = *yvex_runtime_generation_plan_summary_get(context);
        rc = yvex_runtime_cleanup_lease_adopt(
            cleanup, context, generation_context_cleanup, err);
        adopted = rc == YVEX_OK;
    }
    if (rc == YVEX_OK &&
        (request->maximum_new_tokens > SIZE_MAX / sizeof(*result->tokens) ||
         !yvex_core_u64_add(request->maximum_output_bytes, 1ull,
                            &text_allocation) || text_allocation > SIZE_MAX))
        rc = generation_refuse(err, YVEX_ERR_BOUNDS,
                               "generation operator output extent overflowed");
    if (rc == YVEX_OK) {
        result->tokens = yvex_core_calloc(
            (size_t)request->maximum_new_tokens, sizeof(*result->tokens));
        result->text = yvex_core_calloc((size_t)text_allocation, 1u);
        if (!result->tokens || !result->text)
            rc = generation_refuse(err, YVEX_ERR_NOMEM,
                                   "generation operator output allocation failed");
    }
    execution_request.schema_version = YVEX_RUNTIME_GENERATION_SCHEMA_V2;
    execution_request.kind = request->input_kind;
    execution_request.text = request->text;
    execution_request.text_bytes = request->text_bytes;
    execution_request.messages = request->messages;
    execution_request.message_count = request->message_count;
    execution_request.prompt_options = request->prompt_options;
    execution_request.encode_options = request->encode_options;
    if (rc == YVEX_OK)
        rc = yvex_runtime_generation_execute(
            context, &execution_request, result->tokens,
            request->maximum_new_tokens, result->text, text_allocation,
            &result->execution, err);
    if (rc == YVEX_OK)
        rc = generation_context_snapshot(context, &result->context, err);
    if (result->execution.schema_version == YVEX_RUNTIME_GENERATION_SCHEMA_V2) {
        result->token_count = result->execution.sampled_token_count;
        result->text_bytes = result->execution.generated_text_bytes;
    }
    if (rc == YVEX_OK) {
        close_rc = yvex_runtime_generation_result_validate(
            &result->plan, result->tokens, request->maximum_new_tokens,
            result->text, text_allocation, &result->execution,
            &validation_error);
        if (close_rc != YVEX_OK) {
            rc = close_rc;
            if (err) *err = validation_error;
        }
    }
    view = yvex_runtime_model_view_get(model);
    if (view && view->adapter)
        yvex_core_text_copy(result->family, sizeof(result->family),
                            view->adapter->family_name);
    if (rc == YVEX_OK) generation_operator_ready(result, request->backend);
    primary = err ? *err : (yvex_error){0};
    yvex_error_clear(&cleanup_error);
    if (!adopted && context) {
        close_rc = yvex_runtime_generation_context_close(&context, &cleanup_error);
        if (rc == YVEX_OK && close_rc != YVEX_OK) {
            rc = close_rc;
            primary = cleanup_error;
        }
    }
    close_rc = yvex_runtime_cleanup_lease_close(&cleanup, &cleanup_error);
    if (close_rc != YVEX_OK) {
        rc = close_rc;
        primary = cleanup_error;
    }
    if (cleanup) *retained_cleanup = cleanup;
    if (rc == YVEX_OK) {
        result->completed = 1;
        yvex_core_text_copy(result->status, sizeof(result->status), "complete");
        yvex_error_clear(err);
    } else {
        if (err) *err = primary;
        yvex_core_text_copy(result->status, sizeof(result->status),
                            result->execution.partial ? "partial" : "refused");
        yvex_core_text_copy(result->reason, sizeof(result->reason),
                            err && yvex_error_is_set(err)
                                ? yvex_error_message(err)
                                : "generation execution refused");
    }
    return rc;
}

void yvex_runtime_generation_operator_result_release(
    yvex_generation_operator_result *result)
{
    if (!result) return;
    yvex_core_free(result->tokens);
    yvex_core_free(result->text);
    result->tokens = NULL;
    result->text = NULL;
    result->token_count = 0ull;
    result->text_bytes = 0ull;
}
