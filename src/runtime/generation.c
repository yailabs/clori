/* Generation publishes tokens only after state commit; failures preserve that committed prefix. */
#include "src/runtime/private.h"
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <yvex/internal/core.h>
#include <yvex/internal/graph_state.h>
typedef struct {
    yvex_token_sequence_transaction *sequence;
    yvex_tokenizer_decoder_transaction *decoder;
    yvex_tokenizer_fragment fragments[YVEX_SPECULATION_MAX_BLOCK + 2u];
    unsigned long long count, text_bytes;
} generation_publication;
typedef struct {
    yvex_token_sequence_transaction *sequence;
} generation_terminal_publication;
static int generation_refuse(yvex_error *err, yvex_status status, const char *reason);
static int generation_token_classify(const yvex_runtime_generation_context *context,
    unsigned int token, yvex_tokenizer_token_classification *classification,
    int *additional_stop, yvex_error *err);
static int generation_run_target_only(yvex_runtime_generation_context *context,
    const yvex_runtime_generation_turn_request *turn, const yvex_transformer_plan_summary *transformer,
    const yvex_runtime_transformer_result *prefill, const float *prefill_hidden, unsigned long long prefill_values,
    yvex_runtime_generation_token_result *tokens, unsigned char *text, unsigned long long text_capacity,
    yvex_runtime_generation_result *result, yvex_error *err);
static int generation_publication_prepare(void *opaque, yvex_error *err)
{
    generation_publication *publication = opaque;
    int rc = yvex_token_sequence_transaction_prepare(publication->sequence, err);
    if (rc == YVEX_OK)
        rc = yvex_tokenizer_decoder_transaction_prepare(publication->decoder, err);
    return rc;
}
static void generation_publication_publish(void *opaque)
{
    generation_publication *publication = opaque;
    yvex_token_sequence_transaction_publish(&publication->sequence);
    yvex_tokenizer_decoder_transaction_publish(&publication->decoder);
}
static void generation_publication_cancel(void *opaque)
{
    generation_publication *publication = opaque;
    yvex_tokenizer_decoder_transaction_abort(&publication->decoder);
    yvex_token_sequence_transaction_abort(&publication->sequence);
}
static void generation_publication_clear(generation_publication *publication)
{
    unsigned long long index;
    if (!publication) return;
    generation_publication_cancel(publication);
    for (index = 0ull; index < publication->count; ++index)
        yvex_tokenizer_fragment_clear(&publication->fragments[index]);
    memset(publication, 0, sizeof(*publication));
}
static int generation_terminal_prepare(void *opaque, yvex_error *err)
{
    generation_terminal_publication *publication = opaque;
    return yvex_token_sequence_transaction_prepare(publication->sequence, err);
}
static void generation_terminal_publish(void *opaque)
{
    generation_terminal_publication *publication = opaque;
    yvex_token_sequence_transaction_publish(&publication->sequence);
}
static void generation_terminal_cancel(void *opaque)
{
    generation_terminal_publication *publication = opaque;
    yvex_token_sequence_transaction_abort(&publication->sequence);
}
static int generation_terminal_sequence_commit(yvex_runtime_generation_context *context,
    unsigned int token_id,
    yvex_runtime_speculation_context *pending_speculation,
    unsigned long long *ordinal, yvex_error *err)
{
    generation_terminal_publication publication = {0};
    yvex_runtime_commit_participant participant = {
        .context = &publication, .prepare = generation_terminal_prepare,
        .publish = generation_terminal_publish,
        .cancel = generation_terminal_cancel};
    int rc;
    if (!pending_speculation)
        return yvex_token_sequence_append(
            context->sequence, token_id, yvex_tokenizer_vocab_size(context->tokenizer), ordinal, err);
    rc = yvex_token_sequence_transaction_begin(context->sequence, 1ull,
                                                &publication.sequence, err);
    if (rc == YVEX_OK)
        rc = yvex_token_sequence_transaction_append(
            publication.sequence, token_id, yvex_tokenizer_vocab_size(context->tokenizer),
            ordinal, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_speculation_finish_terminal(pending_speculation, &participant, err);
    if (publication.sequence) yvex_token_sequence_transaction_abort(&publication.sequence);
    return rc;
}
static int generation_publication_stage(
    yvex_runtime_generation_context *context, const unsigned int *token_ids,
    unsigned long long count, int terminal_present, unsigned int terminal_id,
    const yvex_graph_attention_state_summary *before,
    const char *anchor_source_identity, const char *anchor_sampling_identity,
    const char *tail_source_identity, const char *tail_sampling_identity,
    yvex_runtime_generation_token_result *tokens,
    const yvex_runtime_generation_result *result, unsigned long long text_capacity,
    generation_publication *publication, yvex_error *err)
{
    unsigned long long index, terminal_ordinal;
    unsigned long long next_text = result->generated_text_bytes;
    int rc;
    memset(publication, 0, sizeof(*publication));
    rc = yvex_token_sequence_transaction_begin(context->sequence,
        count + (unsigned long long)terminal_present, &publication->sequence, err);
    if (rc == YVEX_OK)
        rc = yvex_tokenizer_decoder_transaction_begin(context->decoder,
                                                       &publication->decoder, err);
    for (index = 0ull; rc == YVEX_OK && index < count; ++index) {
        yvex_runtime_generation_token_result *token =
            &tokens[result->sampled_token_count + index];
        unsigned long long ordinal;
        int additional_stop = 0;
        memset(token, 0, sizeof(*token));
        token->schema_version = YVEX_RUNTIME_GENERATION_SCHEMA_V3;
        token->ordinal = result->sampled_token_count + index;
        token->sampled = 1;
        token->sampled_token_id = token_ids[index];
        token->decode_input_token_id = token_ids[index];
        token->sequence_state_before = YVEX_TOKEN_APPEND_PROPOSED;
        token->position_before = before->next_position + index;
        token->position_after = token->position_before + 1ull;
        token->persistent_generation_before = before->generation;
        yvex_runtime_identity_copy(token->source_logits_identity,
            index ? tail_source_identity : anchor_source_identity);
        yvex_runtime_identity_copy(token->sampling_result_identity,
            index ? tail_sampling_identity : anchor_sampling_identity);
        rc = generation_token_classify(context, token_ids[index],
                                       &token->classification,
                                       &additional_stop, err);
        if (rc == YVEX_OK &&
            (token->classification.eos || token->classification.stop ||
             additional_stop))
            rc = generation_refuse(err, YVEX_ERR_STATE,
                "terminal token cannot enter a speculative commit prefix");
        if (rc == YVEX_OK)
            rc = yvex_token_sequence_transaction_append(
                publication->sequence, token_ids[index],
                yvex_tokenizer_vocab_size(context->tokenizer), &ordinal, err);
        for (yvex_token_append_state state = YVEX_TOKEN_APPEND_PROPOSED;
             rc == YVEX_OK && state < YVEX_TOKEN_APPEND_TEXT_PUBLISHED;
             state = (yvex_token_append_state)(state + 1))
            rc = yvex_token_sequence_transaction_transition(
                publication->sequence, ordinal, state,
                (yvex_token_append_state)(state + 1), err);
        if (rc == YVEX_OK)
            rc = yvex_tokenizer_decoder_transaction_push(
                publication->decoder, token_ids[index], &publication->fragments[index], err);
        if (rc == YVEX_OK) publication->count = index + 1ull;
        if (rc == YVEX_OK &&
            (!yvex_core_u64_add(next_text,
                                publication->fragments[index].byte_count,
                                &next_text) ||
             next_text > text_capacity ||
             next_text > context->options.maximum_output_bytes))
            rc = generation_refuse(err, YVEX_ERR_NOMEM,
                "speculative text exceeds its admitted output budget");
    }
    if (rc == YVEX_OK && terminal_present)
        rc = yvex_token_sequence_transaction_append(
            publication->sequence, terminal_id,
            yvex_tokenizer_vocab_size(context->tokenizer), &terminal_ordinal, err);
    if (rc == YVEX_OK) {
        publication->text_bytes = next_text - result->generated_text_bytes;
        return YVEX_OK;
    }
    generation_publication_clear(publication);
    return rc;
}
static int generation_refuse(yvex_error *err, yvex_status status, const char *reason)
{
    yvex_error_set(err, status, "runtime.generation", reason);
    return status;
}
static const unsigned long long generation_transformer_phase_facts =
    (1ull << YVEX_EXECUTION_PHASE_FACT_MOVEMENT) | (1ull << YVEX_EXECUTION_PHASE_FACT_KERNELS) |
    (1ull << YVEX_EXECUTION_PHASE_FACT_SYNCHRONIZATIONS);
static int generation_phase_time(
    yvex_runtime_generation_context *context, yvex_execution_roofline_phase phase,
    unsigned long long duration, unsigned long long work, unsigned long long committed,
    const yvex_execution_memory_facts *m, unsigned long long h2d,
    unsigned long long d2h, unsigned long long d2d,
    unsigned long long kernels, unsigned long long synchronizations,
    unsigned long long fact_mask, yvex_error *err)
{
    yvex_execution_phase_measurement delta = {
        .phase = phase, .fact_mask = fact_mask |
            YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_DURATION) |
            YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_WORK) |
            YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_COMMITTED_TOKENS),
        .active_weight_bytes = m ? m->active_weight_bytes : 0ull, .state_bytes = m ? m->state_bytes : 0ull,
        .activation_bytes = m ? m->activation_bytes : 0ull, .temporary_bytes = m ? m->temporary_bytes : 0ull,
        .h2d_bytes = h2d, .d2h_bytes = d2h, .d2d_bytes = d2d, .kernel_count = kernels,
        .synchronization_count = synchronizations, .measured_duration_ns = duration, .work_units = work,
        .committed_tokens = committed};
    if (!duration || !work) return YVEX_OK;
    return yvex_execution_phase_measurement_accumulate(
        context->phase_measurements, YVEX_EXECUTION_ROOFLINE_PHASE_COUNT,
        &context->phase_measurement_count, &delta, err);
}
static int generation_phase_physical(
    yvex_runtime_generation_context *context, yvex_execution_roofline_phase phase,
    unsigned long long duration, unsigned long long work, unsigned long long committed,
    const yvex_execution_physical_facts *physical, yvex_error *err)
{
    return generation_phase_time(
        context, phase, duration, work, committed,
        physical->memory.complete ? &physical->memory : NULL,
        physical->h2d_bytes, physical->d2h_bytes, physical->d2d_bytes,
        physical->kernel_count, physical->synchronization_count,
        generation_transformer_phase_facts |
            (physical->memory.complete ? YVEX_EXECUTION_PHASE_MEMORY_FACTS : 0ull), err);
}
static int generation_state_summary(const yvex_runtime_execution_session *session,
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
static int generation_cancelled(const yvex_runtime_generation_context *context, yvex_error *err)
{
    if (context->options.cancel_requested &&
        context->options.cancel_requested(context->options.cancel_context))
        return generation_refuse(err, YVEX_ERR_CANCELLED,
                                 "generation was cancelled");
    return YVEX_OK;
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
    if (!request || request->schema_version != YVEX_RUNTIME_GENERATION_SCHEMA_V3 ||
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
    yvex_runtime_generation_context *context, const yvex_tokenizer_encode_result *encoded,
    unsigned long long reusable_prefix, const yvex_runtime_generation_turn_request *turn,
    float **final_hidden, unsigned long long *final_hidden_count,
    yvex_runtime_transformer_result *final_result, unsigned long long *completed_chunks,
    yvex_runtime_profile_record *profile, yvex_error *err)
{
    const yvex_transformer_plan_summary *plan = yvex_transformer_plan_summary_get(
        yvex_runtime_transformer_context_plan(context->transformer));
    yvex_expert_worklist_observation worklists = {0};
    unsigned long long offset = 0ull, suffix_count, maximum_chunk, maximum_values, compiled_row_width = 0ull;
    float *buffer = NULL;
    const int device_only =
        context->options.backend == YVEX_BACKEND_KIND_CUDA &&
        context->options.mode == YVEX_GENERATION_MODE_TARGET_ONLY &&
        context->options.sampling_policy.strategy == YVEX_SAMPLING_STRATEGY_GREEDY &&
        context->options.evidence_profile == YVEX_EXECUTION_EVIDENCE_PRODUCTION;
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
    rc = yvex_runtime_model_compatible_batch_width_copy(context->model, &compiled_row_width, err);
    if (rc != YVEX_OK) return rc;
    /* Configured prefill remains bounded by the compiler-sealed routed-row envelope. */
    if (compiled_row_width > 1ull && maximum_chunk > compiled_row_width) maximum_chunk = compiled_row_width;
    if (maximum_chunk > suffix_count) maximum_chunk = suffix_count;
    if (!plan || !yvex_core_u64_mul(maximum_chunk, plan->hidden_width,
                                    &maximum_values) ||
        maximum_values > SIZE_MAX / sizeof(float))
        return generation_refuse(err, YVEX_ERR_BOUNDS,
                                 "prompt prefill hidden extent overflowed");
    if (!device_only) buffer = yvex_core_calloc((size_t)maximum_values, sizeof(float));
    if (!device_only && !buffer)
        return generation_refuse(err, YVEX_ERR_NOMEM, "prompt prefill hidden allocation failed");
    while (offset < suffix_count && rc == YVEX_OK) {
        yvex_transformer_input_summary summary;
        yvex_transformer_input *input = NULL;
        yvex_runtime_transformer_request request = {0};
        yvex_runtime_transformer_output output = {0};
        yvex_runtime_transformer_result result;
        yvex_runtime_speculation_feature_result draft_result = {0};
        unsigned long long count = suffix_count - offset, values, started, completed;
        unsigned long long synchronizations = 0ull;
        if (count > maximum_chunk) count = maximum_chunk;
        memset(&summary, 0, sizeof(summary));
        summary.schema_version = YVEX_TRANSFORMER_INPUT_SCHEMA_V1;
        summary.token_start = reusable_prefix + offset;
        summary.token_count = count;
        summary.vocabulary_size = plan->vocabulary_size;
        yvex_runtime_identity_copy(summary.logical_model_identity, plan->logical_model_identity);
        yvex_runtime_identity_copy(summary.runtime_numeric_identity, plan->runtime_numeric_identity);
        yvex_runtime_identity_copy(summary.runtime_descriptor_identity,
                                   plan->runtime_descriptor_identity);
        yvex_runtime_identity_copy(summary.transformer_plan_identity, plan->transformer_plan_identity);
        rc = yvex_transformer_input_seal(&summary,
                                         encoded->tokens.ids + reusable_prefix + offset, err);
        if (rc == YVEX_OK)
            rc = yvex_transformer_input_open_memory(
                &input, &summary, encoded->tokens.ids + reusable_prefix + offset, err);
        request.chunk_tokens = count;
        request.backend = context->options.backend;
        request.phase = YVEX_TRANSFORMER_PHASE_PREFILL;
        (void)yvex_core_u64_mul(count, plan->hidden_width, &values);
        if (buffer) memset(buffer, 0, (size_t)values * sizeof(float));
        output.normalized_hidden = buffer;
        output.capacity = buffer ? values : 0ull;
        started = yvex_core_monotonic_ns();
        if (rc == YVEX_OK && context->speculation)
            rc = yvex_runtime_speculation_prefill(
                context->speculation,
                encoded->tokens.ids + reusable_prefix + offset,
                reusable_prefix + offset, count, buffer, values,
                &result, &draft_result, err);
        else if (rc == YVEX_OK)
            rc = yvex_runtime_transformer_execute(
                context->transformer, input, &request, &output, &result, err);
        completed = yvex_core_monotonic_ns();
        yvex_transformer_input_close(&input);
        if (rc == YVEX_OK) rc = yvex_runtime_generation_profile_transformer(profile, &result, err);
        if (rc == YVEX_OK && result.expert_worklists.worklist_count)
            rc = yvex_expert_worklist_observation_add(
                &worklists, &result.expert_worklists, err);
        if (rc == YVEX_OK && !context->speculation &&
            !yvex_core_u64_add(result.stream_synchronizations, result.device_synchronizations,
                               &synchronizations))
            rc = generation_refuse(err, YVEX_ERR_BOUNDS, "prefill physical accounting overflowed");
        if (rc == YVEX_OK && context->speculation)
            rc = generation_phase_physical(
                context, YVEX_EXECUTION_ROOFLINE_PREFILL_LAYER,
                completed - started, count, 0ull, &draft_result.physical, err);
        else if (rc == YVEX_OK)
            rc = generation_phase_time(context, YVEX_EXECUTION_ROOFLINE_PREFILL_LAYER,
                completed - started, count, 0ull,
                result.memory.complete ? &result.memory : NULL,
                result.h2d_bytes, result.d2h_bytes, result.d2d_bytes,
                result.kernel_launches, synchronizations,
                generation_transformer_phase_facts |
                    (result.memory.complete ? YVEX_EXECUTION_PHASE_MEMORY_FACTS : 0ull), err);
        if (rc == YVEX_OK) {
            *final_result = result;
            final_result->expert_worklists = worklists;
            *final_hidden_count = buffer ? values : 0ull;
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
static int generation_profile_count(yvex_runtime_profile_record *profile,
    yvex_runtime_profile_counter counter, unsigned long long value, yvex_error *err)
{
    return !value || runtime_profile_counter_add(profile, counter, value, err) == YVEX_OK
               ? YVEX_OK : yvex_error_code(err);
}
static int generation_observe_worklists(
    yvex_runtime_generation_result *result,
    const yvex_expert_worklist_observation *observation, yvex_error *err)
{
    return !observation || !observation->worklist_count
               ? YVEX_OK
               : yvex_expert_worklist_observation_add(
                     &result->expert_worklists, observation, err);
}
static int generation_profile_graph_delta(yvex_runtime_profile_record *profile,
    const yvex_backend_cuda_attention_graph_summary *before,
    const yvex_backend_cuda_attention_graph_summary *after, yvex_error *err)
{
    const unsigned long long prior[] = {before->launch_count, before->capture_count,
                                        before->replay_count};
    const unsigned long long current[] = {after->launch_count, after->capture_count,
                                          after->replay_count};
    const yvex_runtime_profile_counter counters[] = {
        YVEX_RUNTIME_PROFILE_GRAPH_LAUNCHES, YVEX_RUNTIME_PROFILE_GRAPH_CAPTURES,
        YVEX_RUNTIME_PROFILE_GRAPH_REPLAYS};
    size_t index;
    for (index = 0u; index < sizeof(prior) / sizeof(prior[0]); ++index) {
        if (current[index] < prior[index])
            return generation_refuse(err, YVEX_ERR_STATE,
                "CUDA graph evidence regressed during one generation turn");
        if (generation_profile_count(
                profile, counters[index], current[index] - prior[index], err) != YVEX_OK)
            return yvex_error_code(err);
    }
    return YVEX_OK;
}
/* Derive token-step identity from published fields, never pointers, padding, or object bytes. */
static int generation_project_logits(
    yvex_runtime_generation_context *context, const yvex_runtime_transformer_result *prefill,
    const float *prefill_hidden, unsigned long long prefill_hidden_count,
    const yvex_runtime_decode_step_result *decode, yvex_runtime_logits_row_result *logits_result,
    yvex_runtime_profile_record *profile, yvex_error *err)
{
    yvex_runtime_logits_source logits_source;
    unsigned long long started, completed;
    int rc;
    memset(logits_result, 0, sizeof(*logits_result));
    if (prefill)
        rc = yvex_runtime_logits_source_from_transformer(
            context->logits, &logits_source, prefill, prefill_hidden,
            prefill_hidden_count, prefill->token_count - 1ull, err);
    else
        rc = yvex_runtime_logits_source_from_decode(
            context->logits, &logits_source, decode,
            decode->normalized_hidden_host_available ? context->hidden : NULL,
            decode->normalized_hidden_host_available ? context->hidden_count : 0ull,
            err);
    if (rc == YVEX_OK) {
        started = yvex_core_monotonic_ns();
        rc = yvex_runtime_logits_project(
            context->logits, &logits_source, context->options.backend,
            context->logits_row, context->logits_row ? context->logits_count : 0ull,
            logits_result, err);
        completed = yvex_core_monotonic_ns();
        if (rc == YVEX_OK)
            rc = yvex_runtime_generation_profile_phase(profile, YVEX_RUNTIME_PROFILE_OUTPUT_HEAD,
                                           completed - started, err);
        if (rc == YVEX_OK)
            rc = generation_phase_time(context, YVEX_EXECUTION_ROOFLINE_OUTPUT_HEAD,
                completed - started, 1ull, 0ull,
                logits_result->memory.complete ? &logits_result->memory : NULL,
                logits_result->h2d_bytes,
                logits_result->d2h_bytes, logits_result->d2d_bytes,
                logits_result->kernel_launches, logits_result->device_synchronizations,
                (logits_result->memory.complete
                     ? YVEX_EXECUTION_PHASE_MEMORY_FACTS : 0ull) |
                    YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_MOVEMENT) |
                    YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_KERNELS) |
                    YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_SYNCHRONIZATIONS), err);
        if (rc == YVEX_OK && profile->mode != YVEX_RUNTIME_PROFILE_OFF &&
            (generation_profile_count(profile, YVEX_RUNTIME_PROFILE_H2D_BYTES,
                                      logits_result->h2d_bytes, err) != YVEX_OK ||
             generation_profile_count(profile, YVEX_RUNTIME_PROFILE_D2H_BYTES,
                                      logits_result->d2h_bytes, err) != YVEX_OK ||
             generation_profile_count(profile, YVEX_RUNTIME_PROFILE_LOGITS_H2D_BYTES,
                                      logits_result->h2d_bytes, err) != YVEX_OK ||
             generation_profile_count(profile, YVEX_RUNTIME_PROFILE_LOGITS_D2H_BYTES,
                                      logits_result->d2h_bytes, err) != YVEX_OK ||
             generation_profile_count(profile, YVEX_RUNTIME_PROFILE_FULL_ARRAY_HOST_SCAN_BYTES,
                                      logits_result->full_array_host_scan_bytes, err) != YVEX_OK ||
             generation_profile_count(profile, YVEX_RUNTIME_PROFILE_KERNEL_LAUNCHES,
                                      logits_result->kernel_launches, err) != YVEX_OK))
            rc = yvex_error_code(err);
    }
    return rc;
}
static int generation_sampling_account(
    yvex_runtime_profile_record *profile, const yvex_runtime_sampling_result *sampling,
    unsigned long long elapsed, yvex_error *err)
{
    const unsigned long long d2h = sampling->d2h_bytes, scans = sampling->full_array_host_scan_bytes;
    const unsigned long long kernels = sampling->kernel_launches, streams = sampling->stream_synchronizations;
    const unsigned long long devices = sampling->device_synchronizations;
    int rc = yvex_runtime_generation_profile_phase(profile, YVEX_RUNTIME_PROFILE_SAMPLING, elapsed, err);
    if (rc == YVEX_OK && profile->mode != YVEX_RUNTIME_PROFILE_OFF &&
        (generation_profile_count(profile, YVEX_RUNTIME_PROFILE_D2H_BYTES, d2h, err) != YVEX_OK ||
         generation_profile_count(profile, YVEX_RUNTIME_PROFILE_LOGITS_D2H_BYTES, d2h, err) != YVEX_OK ||
         generation_profile_count(profile, YVEX_RUNTIME_PROFILE_FULL_ARRAY_HOST_SCAN_BYTES, scans, err) != YVEX_OK ||
         generation_profile_count(profile, YVEX_RUNTIME_PROFILE_KERNEL_LAUNCHES, kernels, err) != YVEX_OK ||
         generation_profile_count(profile, YVEX_RUNTIME_PROFILE_STREAM_SYNCHRONIZATIONS, streams, err) != YVEX_OK ||
         generation_profile_count(profile, YVEX_RUNTIME_PROFILE_DEVICE_SYNCHRONIZATIONS, devices, err) != YVEX_OK))
        rc = yvex_error_code(err);
    return rc;
}
static int generation_project_sample(
    yvex_runtime_generation_context *context,
    const yvex_runtime_transformer_result *prefill,
    const float *prefill_hidden, unsigned long long prefill_hidden_count,
    const yvex_runtime_decode_step_result *decode, yvex_runtime_logits_row_result *logits_result,
    yvex_runtime_sampling_result *sampling_result,
    yvex_runtime_profile_record *profile, yvex_error *err)
{
    yvex_runtime_sampling_source source;
    yvex_runtime_sampling_transaction *transaction = NULL;
    unsigned long long started, completed;
    int rc;
    memset(sampling_result, 0, sizeof(*sampling_result));
    rc = generation_project_logits(context, prefill, prefill_hidden, prefill_hidden_count,
                                   decode, logits_result, profile, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_sampling_source_from_logits(
            context->sampling, &source, context->logits_row,
            context->logits_row ? context->logits_count : 0ull,
            logits_result, err);
    if (rc == YVEX_OK && context->options.sampling_policy.strategy == YVEX_SAMPLING_STRATEGY_STOCHASTIC)
        rc = yvex_runtime_sampling_transaction_begin(context->sampling, &transaction, err);
    if (rc == YVEX_OK) {
        started = yvex_core_monotonic_ns();
        rc = yvex_runtime_sampling_select(context->sampling, transaction, &source, sampling_result, err);
        completed = yvex_core_monotonic_ns();
        if (rc == YVEX_OK)
            rc = generation_sampling_account(profile, sampling_result,
                                             completed - started, err);
    }
    if (transaction && rc == YVEX_OK)
        rc = yvex_runtime_sampling_transaction_prepare_commit(transaction, err);
    if (transaction && rc == YVEX_OK)
        yvex_runtime_sampling_transaction_publish_commit(&transaction);
    else if (transaction)
        (void)yvex_runtime_sampling_transaction_abort(&transaction, NULL);
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
    if (!yvex_runtime_generation_token_identity(token, token->token_step_identity))
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
    unsigned long long next_text, started, completed, synchronizations = 0ull;
    const int device_only =
        context->options.backend == YVEX_BACKEND_KIND_CUDA &&
        context->options.mode == YVEX_GENERATION_MODE_TARGET_ONLY &&
        context->options.sampling_policy.strategy == YVEX_SAMPLING_STRATEGY_GREEDY &&
        context->options.evidence_profile == YVEX_EXECUTION_EVIDENCE_PRODUCTION;
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
            token->sampled_token_id, context->options.backend,
            device_only ? NULL : context->hidden,
            device_only ? 0ull : context->hidden_count, decode_result, err);
        completed = yvex_core_monotonic_ns();
        if (rc == YVEX_OK)
            rc = yvex_runtime_generation_profile_phase(
                &result->profile,
                result->decode_step_count ? YVEX_RUNTIME_PROFILE_SUBSEQUENT_DECODE
                                          : YVEX_RUNTIME_PROFILE_FIRST_DECODE,
                completed - started, err);
        if (rc == YVEX_OK)
            rc = yvex_runtime_generation_profile_decode(&result->profile, decode_result, err);
        if (rc == YVEX_OK)
            rc = generation_observe_worklists(
                result, &decode_result->expert_worklists, err);
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
        if (!yvex_core_u64_add(decode_result->stream_synchronizations,
                               decode_result->device_synchronizations, &synchronizations))
            rc = generation_refuse(err, YVEX_ERR_BOUNDS, "decode physical accounting overflowed");
        if (rc == YVEX_OK)
            rc = generation_phase_time(context, YVEX_EXECUTION_ROOFLINE_DECODE_LAYER,
                completed - started, 1ull, 1ull,
                decode_result->memory.complete ? &decode_result->memory : NULL,
                decode_result->h2d_bytes, decode_result->d2h_bytes,
                decode_result->d2d_bytes, decode_result->kernel_launches, synchronizations,
                generation_transformer_phase_facts |
                    (decode_result->memory.complete ? YVEX_EXECUTION_PHASE_MEMORY_FACTS : 0ull), err);
    }
    if (rc == YVEX_OK) rc = generation_cancelled(context, err);
    if (rc == YVEX_OK) {
        started = yvex_core_monotonic_ns();
        rc = yvex_tokenizer_decoder_push(context->decoder,
                                         token->sampled_token_id, &fragment, err);
        completed = yvex_core_monotonic_ns();
        if (rc == YVEX_OK)
            rc = yvex_runtime_generation_profile_phase(&result->profile,
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
    if (!yvex_runtime_generation_token_identity(token, token->token_step_identity) && rc == YVEX_OK)
        rc = generation_refuse(err, YVEX_ERR_STATE,
                               "ordinary token identity derivation failed");
    return rc;
}
static int generation_speculative_terminal_find(
    const yvex_runtime_generation_context *context,
    const unsigned int *token_ids, unsigned long long count,
    int *present, unsigned long long *terminal_index, yvex_error *err)
{
    unsigned long long index;
    int rc = YVEX_OK;
    *present = 0;
    *terminal_index = count;
    for (index = 0ull; rc == YVEX_OK && index < count; ++index) {
        yvex_tokenizer_token_classification classification;
        int additional_stop = 0;
        rc = generation_token_classify(context, token_ids[index],
                                       &classification, &additional_stop, err);
        if (rc == YVEX_OK &&
            (classification.eos || classification.stop || additional_stop)) {
            *present = 1;
            *terminal_index = index;
            break;
        }
    }
    return rc;
}
static int generation_speculative_cycle_terminal(
    yvex_runtime_generation_context *context, unsigned int token_id,
    const char *source_identity, const char *sampling_identity,
    unsigned long long position, unsigned long long generation,
    yvex_runtime_generation_token_result *tokens,
    yvex_runtime_generation_result *result, yvex_error *err)
{
    yvex_runtime_generation_token_result *token =
        &tokens[result->sampled_token_count];
    int additional_stop = 0, rc;
    memset(token, 0, sizeof(*token));
    token->schema_version = YVEX_RUNTIME_GENERATION_SCHEMA_V3;
    token->ordinal = result->sampled_token_count;
    token->sampled = 1;
    token->sampled_token_id = token_id;
    token->sequence_state_before = YVEX_TOKEN_APPEND_PROPOSED;
    token->position_before = position;
    token->position_after = position;
    token->persistent_generation_before = generation;
    token->persistent_generation_after = generation;
    yvex_runtime_identity_copy(token->source_logits_identity, source_identity);
    yvex_runtime_identity_copy(token->sampling_result_identity,
                               sampling_identity);
    rc = generation_token_classify(context, token_id, &token->classification,
                                   &additional_stop, err);
    if (rc == YVEX_OK &&
        !(token->classification.eos || token->classification.stop ||
          additional_stop))
        rc = generation_refuse(err, YVEX_ERR_STATE,
                               "speculative terminal token is not terminal");
    if (rc == YVEX_OK) {
        result->sampled_token_count++;
        rc = generation_terminal_token(token, result, additional_stop, err);
    }
    return rc;
}
static int generation_speculative_publish(
    yvex_runtime_generation_context *context,
    const yvex_runtime_generation_turn_request *turn,
    const unsigned int *token_ids, unsigned long long count,
    int terminal_present, unsigned int terminal_id,
    const yvex_graph_attention_state_summary *before,
    const char *anchor_source_identity, const char *anchor_sampling_identity,
    const char *tail_source_identity, const char *tail_sampling_identity,
    yvex_runtime_generation_token_result *tokens, unsigned char *text,
    unsigned long long text_capacity, yvex_runtime_generation_result *result,
    yvex_runtime_speculation_commit_result *commit, yvex_error *err)
{
    const yvex_tokenizer_plan_summary *tokenizer = yvex_tokenizer_plan_summary_get(context->tokenizer);
    generation_publication publication;
    yvex_runtime_commit_participant participant = {
        .context = &publication, .prepare = generation_publication_prepare,
        .publish = generation_publication_publish,
        .cancel = generation_publication_cancel};
    unsigned long long index, started, completed;
    int rc = generation_publication_stage(
        context, token_ids, count, terminal_present, terminal_id, before,
        anchor_source_identity,
        anchor_sampling_identity, tail_source_identity, tail_sampling_identity,
        tokens, result, text_capacity, &publication, err);
    if (rc == YVEX_OK) {
        started = yvex_core_monotonic_ns();
        rc = yvex_runtime_speculation_commit_prefix(
            context->speculation, count, context->hidden,
            context->hidden_count, &participant, commit, err);
        completed = yvex_core_monotonic_ns();
        if (commit->completed)
            result->speculative_commit_ns += completed - started;
    }
    if (commit->completed) {
        unsigned long long offset = result->generated_text_bytes;
        for (index = 0ull; index < count; ++index) {
            yvex_runtime_generation_token_result *token =
                &tokens[result->sampled_token_count + index];
            yvex_tokenizer_fragment *fragment = &publication.fragments[index];
            token->sequence_state_after = YVEX_TOKEN_APPEND_TEXT_PUBLISHED;
            token->decode_submitted = 1;
            token->model_committed = 1;
            token->detokenized = 1;
            token->text_published = 1;
            token->suppressed = fragment->suppressed;
            token->persistent_generation_after =
                commit->target_result.generation_after;
            token->text_byte_offset = offset;
            token->text_byte_count = fragment->byte_count;
            if (tokenizer && tokenizer->explicit_reasoning_supported &&
                token->sampled_token_id == tokenizer->reasoning_end_token_id)
                result->speculation_source_boundary_token_count = token->ordinal + 1ull;
            yvex_runtime_identity_copy(token->decode_execution_identity,
                                       commit->commit_identity);
            yvex_runtime_identity_copy(token->persistent_state_digest,
                                       commit->target_state_identity);
            yvex_runtime_identity_copy(token->decoder_fragment_identity,
                                       fragment->fragment_identity);
            if (fragment->byte_count)
                memcpy(text + offset, fragment->bytes,
                       (size_t)fragment->byte_count);
            offset += fragment->byte_count;
            if (!yvex_runtime_generation_token_identity(token,
                                           token->token_step_identity) &&
                rc == YVEX_OK)
                rc = generation_refuse(
                    err, YVEX_ERR_STATE,
                    "speculative token identity derivation failed");
        }
        result->sampled_token_count += count;
        result->model_committed_token_count += count;
        result->text_published_token_count += count;
        result->decode_step_count += count;
        result->generated_text_bytes = offset;
        for (index = 0ull; index < count; ++index) {
            yvex_runtime_generation_token_result *token =
                &tokens[result->sampled_token_count - count + index];
            if (token->suppressed) result->suppressed_token_count++;
            if (rc == YVEX_OK && turn->fragment_sink) {
                started = yvex_core_monotonic_ns();
                rc = turn->fragment_sink(
                    turn->fragment_context, token,
                    text + token->text_byte_offset,
                    token->text_byte_count, err);
                completed = yvex_core_monotonic_ns();
                if (rc == YVEX_OK)
                    rc = yvex_runtime_generation_profile_phase(
                        &result->profile,
                        YVEX_RUNTIME_PROFILE_PROVIDER_PUBLICATION,
                        completed - started, err);
                if (rc != YVEX_OK)
                    result->stop_reason = YVEX_GENERATION_STOP_OUTPUT_FAILURE;
            }
        }
    }
    generation_publication_clear(&publication);
    return rc;
}
static int generation_speculative_terminal(
    yvex_runtime_generation_context *context,
    const yvex_runtime_speculation_target_step_result *step,
    const yvex_graph_attention_state_summary *before,
    yvex_runtime_generation_token_result *tokens,
    yvex_runtime_generation_result *result, yvex_error *err)
{
    yvex_runtime_generation_token_result *token =
        &tokens[result->sampled_token_count];
    unsigned long long ordinal;
    int additional_stop = 0, rc;
    memset(token, 0, sizeof(*token));
    token->schema_version = YVEX_RUNTIME_GENERATION_SCHEMA_V3;
    token->ordinal = result->sampled_token_count;
    token->sampled = 1;
    token->sampled_token_id = step->token_id;
    token->sequence_state_before = YVEX_TOKEN_APPEND_PROPOSED;
    token->position_before = before->next_position;
    token->position_after = before->next_position;
    token->persistent_generation_before = before->generation;
    token->persistent_generation_after = before->generation;
    yvex_runtime_identity_copy(token->source_logits_identity,
                               step->source_selection_identity);
    yvex_runtime_identity_copy(token->sampling_result_identity,
                               step->sampling_identity);
    rc = generation_token_classify(context, step->token_id,
                                   &token->classification,
                                   &additional_stop, err);
    if (rc == YVEX_OK &&
        !(token->classification.eos || token->classification.stop ||
          additional_stop))
        rc = generation_refuse(err, YVEX_ERR_STATE,
                               "target step is not terminal");
    if (rc == YVEX_OK)
        rc = generation_terminal_sequence_commit(
            context, step->token_id, context->speculation, &ordinal, err);
    if (rc == YVEX_OK) {
        result->sampling_draw_count += step->target_rng_draw_count;
        result->sampled_token_count++;
        rc = generation_terminal_token(token, result,
                                       additional_stop, err);
    }
    if (rc != YVEX_OK)
        (void)yvex_runtime_speculation_cycle_abort(context->speculation, NULL);
    return rc;
}
static void generation_speculative_account_confidence(
    yvex_runtime_generation_result *result,
    const yvex_runtime_speculation_cycle_result *cycle)
{
    double confidence_sum = 0.0;
    unsigned long long confidence_before = result->confidence_logit_count;
    unsigned long long index;
    for (index = 0ull; index < cycle->draft_proposed_count; ++index) {
        double value = cycle->confidence_logits[index];
        if (!result->confidence_logit_count ||
            value < result->confidence_logit_minimum)
            result->confidence_logit_minimum = value;
        if (!result->confidence_logit_count ||
            value > result->confidence_logit_maximum)
            result->confidence_logit_maximum = value;
        confidence_sum += value;
        result->confidence_logit_count++;
    }
    if (result->confidence_logit_count)
        result->confidence_logit_mean =
            (result->confidence_logit_mean * (double)confidence_before +
             confidence_sum) /
            (double)result->confidence_logit_count;
}
static void generation_speculative_account_cycle(
    yvex_runtime_generation_result *result,
    const yvex_runtime_speculation_cycle_result *cycle)
{
    result->draft_cycle_count++;
    result->draft_forward_count++;
    result->proposed_token_count += cycle->draft_proposed_count;
    result->selected_verification_token_count += cycle->candidate_count;
    result->target_verification_count += cycle->target_verification_count;
    /* The draft projects its complete source-authored block even when output or
     * context limits select a shorter prefix for target verification. The target
     * anchor was already counted by generation_speculative_current_step(). */
    result->logits_projection_count +=
        cycle->draft_proposed_count + cycle->candidate_count + 1ull;
    result->accepted_draft_token_count +=
        cycle->acceptance.accepted_draft_count;
    result->rejected_draft_token_count +=
        cycle->acceptance.rejected_draft_count;
    result->discarded_draft_token_count +=
        cycle->draft_proposed_count - cycle->acceptance.accepted_draft_count -
        cycle->acceptance.rejected_draft_count;
    result->target_correction_or_bonus_token_count +=
        cycle->acceptance.correction_present || cycle->acceptance.bonus_present;
    if (cycle->acceptance.accepted_draft_count >
        result->maximum_accepted_prefix)
        result->maximum_accepted_prefix =
            cycle->acceptance.accepted_draft_count;
    result->draft_ns += cycle->draft_ns;
    result->verification_ns += cycle->verification_ns;
    generation_speculative_account_confidence(result, cycle);
}
/* Failed cycles retain work evidence; uncommitted proposals acquire no output meaning. */
static void generation_speculative_account_incomplete(
    yvex_runtime_generation_result *result,
    const yvex_runtime_speculation_cycle_result *cycle)
{
    if (!cycle->draft_started) return;
    result->draft_cycle_count++;
    result->draft_ns += cycle->draft_ns;
    if (!cycle->draft_completed) return;
    result->draft_forward_count++;
    result->proposed_token_count += cycle->draft_proposed_count;
    result->discarded_draft_token_count += cycle->draft_proposed_count;
    result->logits_projection_count += cycle->draft_proposed_count;
    generation_speculative_account_confidence(result, cycle);
    if (!cycle->verification_started) return;
    result->selected_verification_token_count += cycle->candidate_count;
    result->verification_ns += cycle->verification_ns;
    if (!cycle->verification_completed) return;
    result->target_verification_count += cycle->target_verification_count;
    result->logits_projection_count += cycle->candidate_count + 1ull;
}
static int generation_speculation_progress(
    const yvex_runtime_generation_turn_request *turn,
    const yvex_runtime_generation_result *result,
    yvex_runtime_speculation_progress_kind kind,
    const yvex_runtime_speculation_cycle_result *cycle,
    double seconds, yvex_error *err)
{
    yvex_runtime_speculation_progress progress;
    if (!turn->speculation_progress_sink) return YVEX_OK;
    memset(&progress, 0, sizeof(progress));
    progress.schema_version = YVEX_RUNTIME_GENERATION_SCHEMA_V3;
    progress.kind = kind;
    progress.cycle = result->draft_cycle_count +
                     (kind <= YVEX_SPECULATION_PROGRESS_VERIFICATION_COMPLETED);
    if (cycle) {
        progress.proposed_tokens = cycle->draft_proposed_count;
        progress.selected_verification_tokens = cycle->candidate_count;
        progress.accepted_tokens = cycle->acceptance.accepted_draft_count;
        progress.rejected_tokens = cycle->acceptance.rejected_draft_count;
        progress.discarded_tokens =
            cycle->draft_proposed_count - cycle->acceptance.accepted_draft_count -
            cycle->acceptance.rejected_draft_count;
        progress.verification_count = cycle->target_verification_count;
        if (cycle->completed) {
            unsigned long long index;
            double total = 0.0;
            progress.confidence_logit_count = cycle->draft_proposed_count;
            for (index = 0ull; index < cycle->draft_proposed_count; ++index) {
                double value = cycle->confidence_logits[index];
                if (!index || value < progress.confidence_logit_minimum)
                    progress.confidence_logit_minimum = value;
                if (!index || value > progress.confidence_logit_maximum)
                    progress.confidence_logit_maximum = value;
                total += value;
            }
            progress.confidence_logit_mean =
                total / (double)progress.confidence_logit_count;
        }
    }
    progress.seconds = seconds;
    yvex_runtime_identity_copy(progress.policy_identity,
                               result->speculation_policy_identity);
    return turn->speculation_progress_sink(
        turn->speculation_progress_context, &progress, err);
}
typedef struct {
    const yvex_runtime_generation_turn_request *turn;
    const yvex_runtime_generation_result *result;
    unsigned long long proposed_count, candidate_count;
} generation_speculation_phase_context;
static int generation_speculation_phase(
    void *opaque, yvex_runtime_speculation_phase phase,
    unsigned long long elapsed_ns, yvex_error *err)
{
    generation_speculation_phase_context *context = opaque;
    yvex_runtime_speculation_cycle_result cycle = {0};
    yvex_runtime_speculation_progress_kind kind;
    if (!context || !context->turn || !context->result)
        return generation_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "speculation progress bridge requires an active generation turn");
    cycle.draft_proposed_count = context->proposed_count;
    cycle.candidate_count = context->candidate_count;
    switch (phase) {
    case YVEX_RUNTIME_SPECULATION_PHASE_DRAFT_STARTED:
        kind = YVEX_SPECULATION_PROGRESS_DRAFT_STARTED;
        break;
    case YVEX_RUNTIME_SPECULATION_PHASE_DRAFT_COMPLETED:
        kind = YVEX_SPECULATION_PROGRESS_DRAFT_COMPLETED;
        break;
    case YVEX_RUNTIME_SPECULATION_PHASE_VERIFICATION_STARTED:
        kind = YVEX_SPECULATION_PROGRESS_VERIFICATION_STARTED;
        break;
    case YVEX_RUNTIME_SPECULATION_PHASE_VERIFICATION_COMPLETED:
        kind = YVEX_SPECULATION_PROGRESS_VERIFICATION_COMPLETED;
        cycle.target_verification_count = 1ull;
        break;
    default:
        return generation_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "speculation progress phase is not admitted");
    }
    return generation_speculation_progress(
        context->turn, context->result, kind, &cycle,
        (double)elapsed_ns / 1000000000.0, err);
}
static int generation_speculative_current_step(
    yvex_runtime_generation_context *context,
    const yvex_runtime_transformer_result *anchor, const float *anchor_hidden,
    unsigned long long anchor_hidden_count,
    const yvex_graph_attention_state_summary *before,
    yvex_runtime_generation_token_result *tokens, yvex_runtime_generation_result *result,
    yvex_runtime_speculation_target_step_result *step,
    int *terminal, yvex_error *err)
{
    yvex_runtime_logits_row_result logits = {0};
    yvex_runtime_sampling_source source = {0};
    yvex_runtime_sampling_result selection = {0};
    unsigned long long terminal_index = 0ull, started, completed;
    int rc;
    *terminal = 0;
    memset(step, 0, sizeof(*step));
    rc = generation_cancelled(context, err);
    if (rc == YVEX_OK)
        rc = generation_project_logits(context, anchor, anchor_hidden,
            anchor_hidden_count, NULL, &logits, &result->profile, err);
    if (rc == YVEX_OK && context->device_selection &&
        (!logits.device_values_available || logits.full_array_host_scan_bytes))
        rc = generation_refuse(err, YVEX_ERR_STATE,
            "production DSpark target logits were materialized on the host");
    if (rc == YVEX_OK)
        rc = yvex_runtime_sampling_source_from_logits(
            context->sampling, &source, context->logits_row,
            context->logits_row ? context->logits_count : 0ull, &logits, err);
    if (rc == YVEX_OK) {
        started = yvex_core_monotonic_ns();
        rc = yvex_runtime_speculation_target_step_select(
            context->speculation, before->next_position, &source, step, &selection, err);
        completed = yvex_core_monotonic_ns();
        if (rc == YVEX_OK)
            rc = generation_sampling_account(&result->profile, &selection,
                                             completed - started, err);
    }
    if (rc == YVEX_OK && context->device_selection &&
        (!selection.device_selection || selection.full_array_host_scan_bytes))
        rc = generation_refuse(err, YVEX_ERR_STATE,
            "production DSpark target selection returned host-authored facts");
    if (logits.completed) result->logits_projection_count++;
    if (rc == YVEX_OK)
        rc = generation_speculative_terminal_find(
            context, &step->token_id, 1ull, terminal, &terminal_index, err);
    if (rc == YVEX_OK && *terminal)
        return generation_speculative_terminal(context, step, before, tokens, result, err);
    return rc;
}
static int generation_speculative_candidate_cycle(
    yvex_runtime_generation_context *context,
    const yvex_runtime_generation_turn_request *turn,
    const yvex_speculation_family_policy *policy,
    const yvex_runtime_speculation_target_step_result *anchor_step,
    yvex_runtime_generation_token_result *tokens, unsigned char *text,
    unsigned long long text_capacity, yvex_runtime_generation_result *result,
    yvex_runtime_speculation_commit_result *commit, int *terminal,
    yvex_error *err)
{
    yvex_runtime_speculation_cycle_result cycle = {0}, cycle_metrics = {0};
    yvex_speculation_commit_plan commit_plan = {0};
    yvex_graph_attention_state_summary before;
    unsigned long long remaining_output, remaining_context, candidate_count;
    unsigned long long terminal_index = 0ull, committed_count;
    unsigned long long commit_ns_before;
    unsigned int terminal_id = 0u;
    unsigned int committed[YVEX_SPECULATION_MAX_BLOCK + 2u] = {0};
    generation_speculation_phase_context phase_context = {
        .turn = turn, .result = result};
    int rc;
    *terminal = 0;
    memset(commit, 0, sizeof(*commit));
    rc = generation_state_summary(context->session, &before, err);
    if (rc != YVEX_OK) return rc;
    remaining_output = turn->maximum_new_tokens -
                       result->model_committed_token_count;
    remaining_context = context->options.context_capacity - before.next_position;
    committed[0] = anchor_step->token_id;
    rc = yvex_speculation_candidate_extent(
        policy->block_size, remaining_output, remaining_context,
        &candidate_count, err);
    if (rc != YVEX_OK) return rc;
    if (!candidate_count) {
        rc = generation_speculative_publish(
            context, turn, committed, 1ull, 0, 0u, &before,
            anchor_step->source_selection_identity, anchor_step->sampling_identity,
            anchor_step->source_selection_identity, anchor_step->sampling_identity,
            tokens, text, text_capacity, result, commit, err);
        if (commit->completed)
            result->sampling_draw_count += anchor_step->target_rng_draw_count;
        return rc;
    }
    rc = generation_cancelled(context, err);
    if (rc == YVEX_OK) {
        yvex_runtime_speculation_cycle_request request = {
            .position = before.next_position,
            .candidate_count = candidate_count,
            .conditioning_token_id = anchor_step->token_id,
            .phase_sink = generation_speculation_phase,
            .phase_context = &phase_context};
        phase_context.proposed_count = policy->block_size;
        phase_context.candidate_count = candidate_count;
        rc = yvex_runtime_speculation_cycle(
            context->speculation, &request, &cycle, err);
    }
    if (!cycle.completed) {
        (void)generation_observe_worklists(result, &cycle.draft_worklists, NULL);
        (void)generation_observe_worklists(result, &cycle.verification_worklists, NULL);
        generation_speculative_account_incomplete(result, &cycle);
        return rc;
    }
    rc = generation_observe_worklists(result, &cycle.draft_worklists, err);
    if (rc == YVEX_OK)
        rc = generation_observe_worklists(
            result, &cycle.verification_worklists, err);
    cycle_metrics = cycle;
    memcpy(committed + 1u, cycle.committed_token_ids,
           (size_t)cycle.committed_count * sizeof(*committed));
    if (rc == YVEX_OK)
        rc = generation_speculative_terminal_find(
            context, committed, cycle.committed_count + 1ull,
            terminal, &terminal_index, err);
    if (rc == YVEX_OK && *terminal && !terminal_index)
        rc = generation_refuse(
            err, YVEX_ERR_STATE,
            "a terminal target anchor escaped pre-verification admission");
    if (rc == YVEX_OK && *terminal) terminal_id = committed[terminal_index];
    if (rc == YVEX_OK)
        rc = yvex_speculation_commit_plan_build(
            &cycle.acceptance,
            *terminal ? terminal_index : cycle.committed_count + 1ull,
            &commit_plan, err);
    /* A terminal candidate ends the effective prefix. Later executed proposals remain measured,
     * but its suffix is stop-discarded rather than target-rejected. */
    if (*terminal && terminal_index <= cycle.acceptance.accepted_draft_count) {
        cycle_metrics.acceptance.accepted_draft_count = terminal_index;
        cycle_metrics.acceptance.rejected_draft_count = 0ull;
        cycle_metrics.acceptance.correction_present = 0;
        cycle_metrics.acceptance.bonus_present = 0;
    }
    generation_speculative_account_cycle(result, &cycle_metrics);
    if (rc == YVEX_OK &&
        commit_plan.publication_token_count !=
            commit_plan.state_prefix_count + (unsigned long long)commit_plan.terminal)
        rc = generation_refuse(
            err, YVEX_ERR_STATE,
            "speculative publication extent does not match its commit plan");
    if (rc == YVEX_OK && cycle_metrics.acceptance.rejected_draft_count)
        rc = generation_speculation_progress(
            turn, result, YVEX_SPECULATION_PROGRESS_CANDIDATE_REJECTED,
            &cycle_metrics, 0.0, err);
    committed_count = commit_plan.state_prefix_count;
    commit_ns_before = result->speculative_commit_ns;
    if (rc == YVEX_OK)
        rc = generation_speculative_publish(
            context, turn, committed, committed_count,
            commit_plan.terminal, terminal_id, &before,
            anchor_step->source_selection_identity, anchor_step->sampling_identity,
            cycle.acceptance.acceptance_identity,
            cycle.acceptance.acceptance_identity, tokens, text, text_capacity,
            result, commit, err);
    if (commit->completed) {
        if (rc == YVEX_OK)
            rc = generation_observe_worklists(
                result, &commit->extension_worklists, err);
        result->sampling_draw_count += anchor_step->target_rng_draw_count +
                                       cycle.target_rng_draw_count;
        if (rc == YVEX_OK)
            rc = generation_speculation_progress(
                turn, result, YVEX_SPECULATION_PROGRESS_PREFIX_ACCEPTED,
                &cycle_metrics, 0.0, err);
        if (rc == YVEX_OK)
            rc = generation_speculation_progress(
                turn, result, YVEX_SPECULATION_PROGRESS_CYCLE_COMMITTED,
                &cycle_metrics,
                (double)(result->speculative_commit_ns - commit_ns_before) /
                    1000000000.0,
                err);
        if (rc == YVEX_OK && *terminal)
            rc = generation_speculative_cycle_terminal(
                context, terminal_id, cycle.verification_execution_identity,
                cycle.acceptance.acceptance_identity,
                commit->target_result.position_after,
                commit->target_result.generation_after,
                tokens, result, err);
    }
    if (rc == YVEX_OK)
        rc = generation_phase_physical(
            context, YVEX_EXECUTION_ROOFLINE_DRAFT_SWEEP, cycle.draft_ns,
            cycle.draft_proposed_count, 0ull, &cycle.draft_physical, err);
    if (rc == YVEX_OK)
        rc = generation_phase_physical(
            context, YVEX_EXECUTION_ROOFLINE_VERIFY_SWEEP, cycle.verification_ns,
            cycle.candidate_count + 1ull, commit->completed ? commit->token_count : 0ull,
            &cycle.verification_physical, err);
    if (rc == YVEX_OK && commit->completed && commit->promotion_ns &&
        !commit->promotion_physical.available)
        rc = generation_refuse(err, YVEX_ERR_STATE,
                               "state promotion omitted its physical facts");
    if (rc == YVEX_OK && commit->completed && commit->promotion_ns)
        rc = generation_phase_physical(
            context, YVEX_EXECUTION_ROOFLINE_STATE_PROMOTION,
            commit->promotion_ns, commit->verified_prefix_count, commit->token_count,
            &commit->promotion_physical.physical, err);
    return rc;
}
static int generation_run_dspark(
    yvex_runtime_generation_context *context,
    const yvex_runtime_generation_turn_request *turn,
    const yvex_transformer_plan_summary *transformer,
    const yvex_runtime_transformer_result *prefill,
    const float *prefill_hidden, unsigned long long prefill_hidden_count,
    yvex_runtime_generation_token_result *tokens, unsigned char *text,
    unsigned long long text_capacity, yvex_runtime_generation_result *result,
    yvex_error *err)
{
    const yvex_speculation_family_policy *policy =
        yvex_runtime_speculation_policy_get(context->speculation);
    yvex_runtime_transformer_result anchor = *prefill;
    const float *anchor_hidden = prefill_hidden;
    unsigned long long anchor_hidden_count = prefill_hidden_count;
    int rc = policy ? YVEX_OK : generation_refuse(
        err, YVEX_ERR_STATE, "DSpark generation policy is unavailable");
    while (rc == YVEX_OK && result->stop_reason == YVEX_GENERATION_STOP_NONE) {
        yvex_runtime_speculation_commit_result commit = {0};
        yvex_runtime_speculation_target_step_result anchor_step = {0};
        yvex_graph_attention_state_summary before;
        int terminal = 0;
        if (result->model_committed_token_count == turn->maximum_new_tokens) {
            result->stop_reason = YVEX_GENERATION_STOP_MAX_NEW_TOKENS;
            break;
        }
        rc = generation_state_summary(context->session, &before, err);
        if (rc == YVEX_OK &&
            before.next_position == context->options.context_capacity) {
            result->stop_reason = YVEX_GENERATION_STOP_CONTEXT_CAPACITY;
            break;
        }
        if (rc == YVEX_OK)
            rc = generation_speculative_current_step(
                context, &anchor, anchor_hidden, anchor_hidden_count, &before,
                tokens, result, &anchor_step, &terminal, err);
        if (terminal || rc != YVEX_OK) break;
        rc = generation_speculative_candidate_cycle(
            context, turn, policy, &anchor_step, tokens, text,
            text_capacity, result, &commit, &terminal, err);
        if (commit.completed) {
            anchor = commit.target_result;
            anchor_hidden = context->hidden;
            anchor_hidden_count = context->hidden_count;
        }
        /* The channel transition preserves committed state, ledger, decoder and RNG. */
        if (rc == YVEX_OK && result->speculation_source_boundary_token_count) {
            rc = generation_run_target_only(
                context, turn, transformer, &anchor, anchor_hidden,
                anchor_hidden_count, tokens, text, text_capacity, result, err);
            break;
        }
        if (terminal) break;
    }
    /* No proposal or RNG transaction crosses a turn; abort discards only an incomplete next cycle
     * because every target correction or bonus has already joined the committed prefix. */
    {
        yvex_error primary = err ? *err : (yvex_error){0}, cleanup = {0};
        int abort_rc = yvex_runtime_speculation_cycle_abort(
            context->speculation, rc == YVEX_OK ? err : &cleanup);
        if (rc == YVEX_OK) rc = abort_rc;
        else if (err) *err = primary;
    }
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
static void generation_partial_turn_seal(
    yvex_runtime_generation_result *result, int status,
    const yvex_runtime_sampling_context_summary *sampling,
    const yvex_token_sequence_summary *sequence)
{
    yvex_runtime_partial_turn *partial = &result->partial_turn;
    memset(partial, 0, sizeof(*partial));
    if (status == YVEX_OK) return;
    partial->schema_version = YVEX_RUNTIME_PARTIAL_TURN_SCHEMA_V1;
    partial->available = 1;
    partial->committed_progress =
        result->final_position > result->initial_position ||
        result->model_committed_token_count || result->generated_text_bytes;
    partial->reset_required = 1;
    partial->failure_status = status;
    partial->stop_reason = result->stop_reason;
    partial->initial_position = result->initial_position;
    partial->final_committed_position = result->final_position;
    partial->committed_token_count = result->model_committed_token_count;
    partial->published_text_bytes = result->generated_text_bytes;
    partial->target_state_generation = result->final_persistent_generation;
    partial->rng_generation = sampling->stochastic_draws;
    partial->token_ledger_generation = sequence->generation;
    yvex_runtime_identity_copy(partial->target_state_identity,
                               result->final_persistent_state_digest);
    yvex_runtime_identity_copy(partial->rng_state_identity,
                               sampling->rng_state_identity);
    yvex_runtime_identity_copy(partial->token_ledger_identity,
                               sequence->state_identity);
    yvex_runtime_identity_copy(partial->published_text_identity,
                               result->generated_text_digest);
}
static int generation_profile_final_counters(
    yvex_runtime_generation_context *context,
    yvex_runtime_generation_result *result, yvex_error *err)
{
    yvex_runtime_profile_record *profile = &result->profile;
    yvex_execution_shape_registry_summary shapes = {0};
    unsigned long long target_forwards, target_rows, verified_rows;
    unsigned long long runtime_forwards, runtime_rows, discarded_rows;
    int rc = YVEX_OK;
#define ADD_COUNTER(kind_, value_)                                                \
    do {                                                                          \
        if (rc == YVEX_OK && (value_) != 0ull)                                    \
            rc = runtime_profile_counter_add(profile, (kind_), (value_), err);    \
    } while (0)
    if (profile->mode == YVEX_RUNTIME_PROFILE_OFF) return YVEX_OK;
    if (!yvex_core_u64_add(result->selected_verification_token_count,
                           result->target_verification_count, &verified_rows) ||
        !yvex_core_u64_add(result->rejected_draft_token_count,
                           result->discarded_draft_token_count,
                           &discarded_rows))
        return generation_refuse(err, YVEX_ERR_BOUNDS,
                                 "profile execution counters overflowed");
    runtime_forwards = result->decode_step_count;
    runtime_rows = result->decode_step_count;
    if (result->execution_mode == YVEX_GENERATION_MODE_DSPARK &&
        (!yvex_core_u64_add(result->target_verification_count,
                            result->target_correction_or_bonus_token_count,
                            &runtime_forwards) ||
         !yvex_core_u64_add(verified_rows,
                            result->target_correction_or_bonus_token_count,
                            &runtime_rows)))
        return generation_refuse(err, YVEX_ERR_BOUNDS,
                                 "profile target counters overflowed");
    if (!yvex_core_u64_add(result->prefill_chunk_count, runtime_forwards,
                           &target_forwards) ||
        !yvex_core_u64_add(result->new_prefill_token_count, runtime_rows,
                           &target_rows))
        return generation_refuse(err, YVEX_ERR_BOUNDS,
                                 "profile target totals overflowed");
    ADD_COUNTER(YVEX_RUNTIME_PROFILE_TARGET_FORWARDS, target_forwards);
    ADD_COUNTER(YVEX_RUNTIME_PROFILE_TARGET_ROWS, target_rows);
    ADD_COUNTER(YVEX_RUNTIME_PROFILE_DRAFT_FORWARDS, result->draft_forward_count);
    ADD_COUNTER(YVEX_RUNTIME_PROFILE_DRAFT_ROWS, result->proposed_token_count);
    ADD_COUNTER(YVEX_RUNTIME_PROFILE_TARGET_VERIFICATIONS,
                result->target_verification_count);
    ADD_COUNTER(YVEX_RUNTIME_PROFILE_VERIFIED_ROWS, verified_rows);
    ADD_COUNTER(YVEX_RUNTIME_PROFILE_ACCEPTED_DRAFT_TOKENS,
                result->accepted_draft_token_count);
    ADD_COUNTER(YVEX_RUNTIME_PROFILE_PROMOTED_TARGET_ROWS,
                result->accepted_draft_token_count);
    ADD_COUNTER(YVEX_RUNTIME_PROFILE_DISCARDED_CANDIDATE_ROWS,
                discarded_rows);
    ADD_COUNTER(YVEX_RUNTIME_PROFILE_TARGET_EXTENSIONS,
                result->target_correction_or_bonus_token_count);
    ADD_COUNTER(YVEX_RUNTIME_PROFILE_OUTPUT_HEAD_ROWS,
                result->logits_projection_count);
    if (rc == YVEX_OK && context->execution_shapes)
        rc = yvex_execution_shape_registry_summary_copy(
            context->execution_shapes, &shapes, err);
    if (rc == YVEX_OK && context->execution_shapes) {
        ADD_COUNTER(YVEX_RUNTIME_PROFILE_SHAPE_REGISTRY_HITS,
                    shapes.hit_count);
        ADD_COUNTER(YVEX_RUNTIME_PROFILE_SHAPE_REGISTRY_MISSES,
                    shapes.miss_count);
    }
#undef ADD_COUNTER
    return rc;
}
/* KV/RNG snapshots finalize progress without erasing evidence when snapshotting refuses. */
static int generation_result_finish(
    yvex_runtime_generation_context *context,
    yvex_runtime_generation_token_result *tokens,
    const unsigned char *text, unsigned long long text_capacity,
    yvex_runtime_generation_result *result, int rc, yvex_error *err)
{
    yvex_graph_attention_state_summary state = {0};
    yvex_runtime_sampling_context_summary sampling = {0};
    yvex_token_sequence_summary sequence = {0};
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
    if (finish_rc == YVEX_OK) {
        result->final_rng_generation = sampling.stochastic_draws;
        yvex_runtime_identity_copy(result->final_rng_identity,
                                   sampling.rng_state_identity);
    } else if (rc == YVEX_OK) {
        rc = finish_rc;
        if (err) *err = secondary;
    }
    finish_rc = yvex_token_sequence_summary_get(context->sequence, &sequence,
                                                &secondary);
    if (finish_rc == YVEX_OK) {
        result->final_token_ledger_generation = sequence.generation;
        yvex_runtime_identity_copy(result->final_token_ledger_identity,
                                   sequence.state_identity);
    }
    if (finish_rc != YVEX_OK && rc == YVEX_OK) {
        rc = finish_rc;
        if (err) *err = secondary;
    }
    for (index = 0ull; index < result->sampled_token_count; ++index) {
        if (!tokens[index].token_step_identity[0] &&
            !yvex_runtime_generation_token_identity(&tokens[index],
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
    if (result->profile.schema_version == YVEX_RUNTIME_PROFILE_SCHEMA_V3) {
        unsigned long long completed = yvex_core_monotonic_ns();
        if (result->draft_cycle_count)
            result->mean_accepted_prefix =
                (double)result->accepted_draft_token_count /
                (double)result->draft_cycle_count;
        if (completed > result->profile.started_ns)
            result->effective_committed_tokens_per_second =
                (double)result->model_committed_token_count * 1000000000.0 /
                (double)(completed - result->profile.started_ns);
        finish_rc = generation_profile_final_counters(context, result, &secondary);
        if (finish_rc == YVEX_OK)
            finish_rc = result->profile.mode == YVEX_RUNTIME_PROFILE_OFF
                            ? YVEX_OK
                            : runtime_profile_counter_add(
                                  &result->profile,
                                  YVEX_RUNTIME_PROFILE_GENERATED_TOKENS,
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
    if (context->options.backend == YVEX_BACKEND_KIND_CUDA &&
        context->phase_measurement_count) {
        yvex_execution_roofline_ledger_request request = {
            .schema_version = YVEX_EXECUTION_PHASE_ROOFLINE_SCHEMA_V1,
            .hardware = &context->hardware_profile,
            .artifact_identity = context->model_view->binding->artifact_identity,
            .execution_profile_identity = context->execution_profile.identity,
            .kernel_bundle_identity = context->plan.kernel_bundle_identity,
            .workload_profile_identity = context->workload_profile.identity,
            .measurements = context->phase_measurements,
            .measurement_count = context->phase_measurement_count};
        finish_rc = yvex_execution_roofline_ledger_build(&request, &result->roofline, &secondary);
        if (finish_rc == YVEX_OK) result->roofline_available = 1;
        else if (rc == YVEX_OK) {
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
    if (!yvex_runtime_generation_tokens_identity(
            tokens, result->sampled_token_count,
            result->generated_token_identity) ||
        !yvex_runtime_generation_bytes_digest(
            "yvex.runtime.generation.text.v1", text,
            result->generated_text_bytes, result->generated_text_digest)) {
        if (rc == YVEX_OK)
            rc = generation_refuse(err, YVEX_ERR_STATE,
                                   "generation aggregate identity failed");
    }
    generation_partial_turn_seal(result, rc, &sampling, &sequence);
    if (!yvex_runtime_generation_execution_identity(
            result, tokens, result->generation_execution_identity)) {
        if (rc == YVEX_OK)
            rc = generation_refuse(err, YVEX_ERR_STATE,
                                   "generation aggregate identity failed");
    } else if (!yvex_runtime_generation_execution_identity(
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
        if (!yvex_runtime_generation_prefix_identity(turn->committed_prefix_token_ids,
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
static int generation_run_target_only(
    yvex_runtime_generation_context *context,
    const yvex_runtime_generation_turn_request *turn,
    const yvex_transformer_plan_summary *transformer,
    const yvex_runtime_transformer_result *prefill, const float *prefill_hidden,
    unsigned long long prefill_values,
    yvex_runtime_generation_token_result *tokens, unsigned char *text,
    unsigned long long text_capacity, yvex_runtime_generation_result *result,
    yvex_error *err)
{
    yvex_runtime_decode_step_result last_decode = {0};
    yvex_graph_attention_state_summary before;
    unsigned long long started, completed;
    int rc = YVEX_OK, use_prefill = 1;
    while (rc == YVEX_OK &&
           result->stop_reason == YVEX_GENERATION_STOP_NONE) {
        yvex_runtime_logits_row_result logits_result = {0};
        yvex_runtime_sampling_result sample = {0};
        yvex_runtime_generation_token_result *token;
        unsigned long long sequence_ordinal;
        int additional_stop = 0;
        if (result->model_committed_token_count == turn->maximum_new_tokens) {
            result->stop_reason = YVEX_GENERATION_STOP_MAX_NEW_TOKENS;
            break;
        }
        rc = generation_state_summary(context->session, &before, err);
        if (rc == YVEX_OK &&
            before.next_position == context->options.context_capacity) {
            result->stop_reason = YVEX_GENERATION_STOP_CONTEXT_CAPACITY;
            break;
        }
        if (rc == YVEX_OK) rc = generation_cancelled(context, err);
        if (rc == YVEX_OK)
            rc = generation_project_sample(
                context, use_prefill ? prefill : NULL, prefill_hidden,
                prefill_values, use_prefill ? NULL : &last_decode,
                &logits_result, &sample, &result->profile, err);
        if (logits_result.completed) result->logits_projection_count++;
        if (rc != YVEX_OK) break;
        token = &tokens[result->sampled_token_count];
        token->schema_version = YVEX_RUNTIME_GENERATION_SCHEMA_V3;
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
            rc = generation_terminal_token(token, result, additional_stop, err);
        else if (rc == YVEX_OK)
            rc = generation_commit_ordinary(
                context, token, sequence_ordinal, text, text_capacity, result,
                &last_decode, err);
        if (rc == YVEX_OK && turn->fragment_sink && token->text_published) {
            started = yvex_core_monotonic_ns();
            rc = turn->fragment_sink(
                turn->fragment_context, token,
                text + token->text_byte_offset, token->text_byte_count, err);
            completed = yvex_core_monotonic_ns();
            if (rc == YVEX_OK)
                rc = yvex_runtime_generation_profile_phase(
                    &result->profile,
                    YVEX_RUNTIME_PROFILE_PROVIDER_PUBLICATION,
                    completed - started, err);
            if (rc != YVEX_OK)
                result->stop_reason = YVEX_GENERATION_STOP_OUTPUT_FAILURE;
        }
        use_prefill = 0;
    }
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
    yvex_backend_cuda_attention_graph_summary graph_before = {0}, graph_after = {0};
    float *prefill_hidden = NULL;
    unsigned long long prefill_values = 0ull, prefill_chunks = 0ull;
    unsigned long long turn_maximum = turn ? turn->maximum_new_tokens : 0ull;
    unsigned long long started, completed;
    char workload_identity[YVEX_SHA256_HEX_CAP];
    int rc, graph_profiled = 0, prompt_stage = 1;
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
    rc = yvex_runtime_private_generation_enter(context, err);
    if (rc != YVEX_OK) return rc;
    memset(context->phase_measurements, 0, sizeof(context->phase_measurements));
    context->phase_measurement_count = 0ull;
    memset(tokens, 0, (size_t)turn_maximum * sizeof(*tokens));
    memset(text, 0, (size_t)context->options.maximum_output_bytes);
    memset(&encoded, 0, sizeof(encoded));
    memset(&rendered, 0, sizeof(rendered));
    result->schema_version = YVEX_RUNTIME_GENERATION_RESULT_SCHEMA_V5;
    result->execution_mode = context->options.mode;
    result->requested_new_tokens = turn_maximum;
    yvex_runtime_identity_copy(result->generation_plan_identity,
                               context->plan.generation_plan_identity);
    yvex_runtime_identity_copy(result->speculation_policy_identity,
                               context->plan.speculation_policy_identity);
    rc = yvex_runtime_generation_workload_identity(
             context, turn, workload_identity)
             ? runtime_profile_begin(
                   &result->profile,
                   yvex_runtime_generation_profile_mode(
                       context->options.trace_policy),
                   YVEX_RUNTIME_PROFILE_GENERATION, context->options.backend,
                   context->model_view->binding->artifact_identity,
                   context->model_view->binding->profile_identity,
                   context->model_view->binding->identity,
                   context->plan.runtime_model_identity,
                   context->plan.generation_plan_identity, workload_identity, err)
             : generation_refuse(err, YVEX_ERR_STATE,
                                 "generation workload identity derivation failed");
    if (rc == YVEX_OK && result->profile.mode != YVEX_RUNTIME_PROFILE_OFF &&
        context->options.backend == YVEX_BACKEND_KIND_CUDA) {
        const yvex_runtime_session_view *view =
            yvex_runtime_session_view_get(context->session);
        rc = view && view->backend
                 ? yvex_backend_cuda_attention_graph_summary_get(
                       view->backend, &graph_before, err)
                 : generation_refuse(
                       err, YVEX_ERR_STATE,
                       "CUDA graph evidence owner is unavailable");
        graph_profiled = rc == YVEX_OK;
    }
    started = yvex_core_monotonic_ns();
    if (rc == YVEX_OK)
        rc = generation_turn_prepare(context, turn, &encoded, &rendered,
                                     &transformer, result, err);
    completed = yvex_core_monotonic_ns();
    if (rc == YVEX_OK)
        rc = yvex_runtime_generation_profile_phase(
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
    if (rc == YVEX_OK)
        rc = generation_observe_worklists(
            result, &prefill.expert_worklists, err);
    completed = yvex_core_monotonic_ns();
    if (rc == YVEX_OK)
        rc = yvex_runtime_generation_profile_phase(&result->profile,
                                       YVEX_RUNTIME_PROFILE_TOTAL_PREFILL,
                                       completed - started, err);
    result->prefill_chunk_count = prefill_chunks;
    if (rc == YVEX_OK && turn->progress_sink)
        rc = turn->progress_sink(
            turn->progress_context, YVEX_GENERATION_PROGRESS_PREFILL_COMPLETED,
            result->new_prefill_token_count, prefill_chunks, err);
    if (rc == YVEX_OK && context->options.mode == YVEX_GENERATION_MODE_DSPARK)
        rc = generation_run_dspark(
            context, turn, transformer, &prefill, prefill_hidden, prefill_values,
            tokens, text, text_capacity, result, err);
    if (rc == YVEX_OK &&
        context->options.mode == YVEX_GENERATION_MODE_TARGET_ONLY)
        rc = generation_run_target_only(
            context, turn, transformer, &prefill, prefill_hidden,
            prefill_values, tokens, text, text_capacity, result, err);
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
    if (rc != YVEX_OK && context->speculation) {
        yvex_error primary = err ? *err : (yvex_error){0};
        yvex_error cleanup;
        int cleanup_rc;
        yvex_error_clear(&cleanup);
        cleanup_rc = yvex_runtime_speculation_cycle_abort(
            context->speculation, &cleanup);
        if (cleanup_rc != YVEX_OK) {
            rc = cleanup_rc;
            if (err) *err = cleanup;
        } else if (err) *err = primary;
    }
    if (rc == YVEX_OK && graph_profiled) {
        const yvex_runtime_session_view *view =
            yvex_runtime_session_view_get(context->session);
        rc = view && view->backend &&
                     yvex_backend_cuda_attention_graph_summary_get(
                         view->backend, &graph_after, err) == YVEX_OK
                 ? generation_profile_graph_delta(
                       &result->profile, &graph_before, &graph_after, err)
                 : generation_refuse(
                       err, YVEX_ERR_STATE,
                       "CUDA graph evidence could not close the generation turn");
    }
    rc = generation_result_finish(context, tokens, text, text_capacity,
                                  result, rc, err);
    yvex_core_free(prefill_hidden);
    yvex_tokenizer_encode_result_clear(&encoded);
    yvex_rendered_prompt_free(&rendered);
    context->continuation_allowed = rc == YVEX_OK;
    yvex_runtime_private_generation_leave(context, rc, 1);
    if (rc == YVEX_OK) yvex_error_clear(err);
    return rc;
}
