/*
 * Generation evidence is serialized field by field so padding, pointers, and transient
 * proposal bytes never enter a stable identity. The same owner validates operator results
 * before they cross the CLI boundary.
 */
#include "src/runtime/private.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int generation_hash_finish(yvex_sha256 *hash,
                                  char output[YVEX_SHA256_HEX_CAP]);

static int generation_result_refuse(yvex_error *err, yvex_status status,
                                    const char *reason)
{
    yvex_error_set(err, status, "runtime.generation", reason);
    return status;
}

yvex_runtime_profile_mode yvex_runtime_generation_profile_mode(
    yvex_runtime_trace_policy policy)
{
    static const yvex_runtime_profile_mode modes[] = {
        YVEX_RUNTIME_PROFILE_OFF, YVEX_RUNTIME_PROFILE_SUMMARY,
        YVEX_RUNTIME_PROFILE_STAGES, YVEX_RUNTIME_PROFILE_DETAILED};
    return policy <= YVEX_RUNTIME_TRACE_FULL ? modes[policy]
                                             : YVEX_RUNTIME_PROFILE_OFF;
}

int yvex_runtime_generation_workload_identity(
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
        !yvex_sha256_update_u64(&hash, turn->committed_prefix_token_count))
        return 0;
    for (index = 0ull; index < turn->committed_prefix_token_count; ++index)
        if (!yvex_sha256_update_u64(
                &hash, turn->committed_prefix_token_ids[index]))
            return 0;
    if (request->kind == YVEX_GENERATION_INPUT_TEXT) {
        if (!request->text || request->text_bytes > SIZE_MAX ||
            !yvex_sha256_update_u64(&hash, request->text_bytes) ||
            !yvex_sha256_update(
                &hash, request->text, (size_t)request->text_bytes))
            return 0;
    } else if (request->kind == YVEX_GENERATION_INPUT_PROVIDER) {
        if (!request->provider_request || !request->provider_request->sealed ||
            !yvex_sha256_hex_valid(request->provider_request->request_identity) ||
            !yvex_sha256_update_text(
                &hash, request->provider_request->request_identity))
            return 0;
    } else {
        if (!request->messages || !request->message_count ||
            !yvex_sha256_update_u64(&hash, request->message_count))
            return 0;
        for (index = 0ull; index < request->message_count; ++index) {
            const yvex_prompt_message *message = &request->messages[index];
            if ((!message->content && message->content_len) ||
                message->content_len > SIZE_MAX ||
                !yvex_sha256_update_u64(&hash, message->role) ||
                !yvex_sha256_update_u64(&hash, message->content_len) ||
                !yvex_sha256_update(
                    &hash, message->content, (size_t)message->content_len))
                return 0;
        }
    }
    return generation_hash_finish(&hash, output);
}

int yvex_runtime_generation_profile_phase(
    yvex_runtime_profile_record *profile, yvex_runtime_profile_phase phase,
    unsigned long long elapsed, yvex_error *err)
{
    return !profile || profile->mode == YVEX_RUNTIME_PROFILE_OFF || !elapsed
               ? YVEX_OK
               : runtime_profile_phase_add(profile, phase, elapsed, err);
}

int yvex_runtime_generation_profile_transformer(
    yvex_runtime_profile_record *profile,
    const yvex_runtime_transformer_result *value, yvex_error *err)
{
#define COUNTER(kind_, value_)                                                   \
    do {                                                                         \
        if ((value_) != 0ull &&                                                  \
            runtime_profile_counter_add(profile, (kind_), (value_), err) !=      \
                YVEX_OK)                                                         \
            return yvex_error_code(err);                                         \
    } while (0)
    if (!profile || profile->mode == YVEX_RUNTIME_PROFILE_OFF || !value ||
        !value->completed)
        return YVEX_OK;
    COUNTER(YVEX_RUNTIME_PROFILE_H2D_BYTES, value->h2d_bytes);
    COUNTER(YVEX_RUNTIME_PROFILE_D2H_BYTES, value->d2h_bytes);
    COUNTER(YVEX_RUNTIME_PROFILE_D2D_BYTES, value->d2d_bytes);
    COUNTER(YVEX_RUNTIME_PROFILE_UPLOADS, value->upload_count);
    COUNTER(YVEX_RUNTIME_PROFILE_DOWNLOADS, value->download_count);
    COUNTER(YVEX_RUNTIME_PROFILE_CACHE_HITS, value->cache_hits);
    COUNTER(YVEX_RUNTIME_PROFILE_CACHE_MISSES, value->cache_misses);
    COUNTER(YVEX_RUNTIME_PROFILE_EXPERT_SUBVIEWS, value->expert_subviews_accessed);
    COUNTER(YVEX_RUNTIME_PROFILE_ROW_EXPERT_PAIRS, value->row_expert_pairs);
    COUNTER(YVEX_RUNTIME_PROFILE_UNIQUE_EXPERTS, value->unique_experts);
    COUNTER(YVEX_RUNTIME_PROFILE_EXPERT_BYTES, value->expert_weight_bytes);
    COUNTER(YVEX_RUNTIME_PROFILE_FULL_ARRAY_HOST_SCAN_BYTES,
            value->full_array_host_scan_bytes);
    COUNTER(YVEX_RUNTIME_PROFILE_KERNEL_LAUNCHES, value->kernel_launches);
    COUNTER(YVEX_RUNTIME_PROFILE_TENSOR_CORE_LAUNCHES,
            value->tensor_core_launches);
    COUNTER(YVEX_RUNTIME_PROFILE_GRAPH_LAUNCHES, value->graph_launches);
    COUNTER(YVEX_RUNTIME_PROFILE_GRAPH_CAPTURES, value->graph_captures);
    COUNTER(YVEX_RUNTIME_PROFILE_GRAPH_REPLAYS, value->graph_replays);
    COUNTER(YVEX_RUNTIME_PROFILE_STREAM_SYNCHRONIZATIONS,
            value->stream_synchronizations);
    COUNTER(YVEX_RUNTIME_PROFILE_DEVICE_SYNCHRONIZATIONS,
            value->device_synchronizations);
#undef COUNTER
    if (yvex_runtime_generation_profile_phase(
            profile, YVEX_RUNTIME_PROFILE_EMBEDDING, value->embedding_ns,
            err) != YVEX_OK ||
        yvex_runtime_generation_profile_phase(
            profile, YVEX_RUNTIME_PROFILE_ATTENTION,
            profile->backend == YVEX_BACKEND_KIND_CUDA
                ? value->attention_device_ns
                : value->attention_ns,
            err) != YVEX_OK ||
        yvex_runtime_generation_profile_phase(
            profile, YVEX_RUNTIME_PROFILE_MOE_TOTAL, value->moe_ns, err) !=
            YVEX_OK ||
        yvex_runtime_generation_profile_phase(
            profile, YVEX_RUNTIME_PROFILE_SYNCHRONIZATION_WAIT,
            value->synchronization_ns, err) != YVEX_OK ||
        yvex_runtime_generation_profile_phase(
            profile, YVEX_RUNTIME_PROFILE_FINAL_NORMALIZATION,
            value->final_ns, err) != YVEX_OK)
        return yvex_error_code(err);
    return YVEX_OK;
}

int yvex_runtime_generation_profile_decode(
    yvex_runtime_profile_record *profile,
    const yvex_runtime_decode_step_result *value, yvex_error *err)
{
    yvex_runtime_transformer_result projected = {0};
    if (!value || !value->completed) return YVEX_OK;
    projected.completed = 1;
    projected.routed_experts = value->routed_experts;
    projected.row_expert_pairs = value->row_expert_pairs;
    projected.unique_experts = value->unique_experts;
    projected.grouped_expert_operations = value->grouped_expert_operations;
    projected.expert_subviews_accessed = value->expert_subviews_accessed;
    projected.embedding_bytes = value->embedding_weight_bytes;
    projected.attention_weight_bytes = value->attention_weight_bytes;
    projected.expert_weight_bytes = value->expert_weight_bytes;
    projected.final_weight_bytes = value->final_weight_bytes;
    projected.h2d_bytes = value->h2d_bytes;
    projected.d2h_bytes = value->d2h_bytes;
    projected.d2d_bytes = value->d2d_bytes;
    projected.upload_count = value->upload_count;
    projected.download_count = value->download_count;
    projected.cache_hits = value->cache_hits;
    projected.cache_misses = value->cache_misses;
    projected.kernel_launches = value->kernel_launches;
    projected.tensor_core_launches = value->tensor_core_launches;
    projected.graph_launches = value->graph_launches;
    projected.graph_captures = value->graph_captures;
    projected.graph_replays = value->graph_replays;
    projected.stream_synchronizations = value->stream_synchronizations;
    projected.device_synchronizations = value->device_synchronizations;
    projected.embedding_ns = value->embedding_ns;
    projected.attention_ns = value->attention_ns;
    projected.attention_device_ns = value->attention_device_ns;
    projected.moe_ns = value->moe_ns;
    projected.final_ns = value->final_ns;
    projected.synchronization_ns = value->synchronization_ns;
    return yvex_runtime_generation_profile_transformer(
        profile, &projected, err);
}

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
        return generation_result_refuse(err, YVEX_ERR_INVALID_ARG,
                                        "generation context is required");
    prompt_tokens = yvex_core_calloc((size_t)context->options.context_capacity,
                                     sizeof(*prompt_tokens));
    if (!prompt_tokens)
        return generation_result_refuse(
            err, YVEX_ERR_NOMEM,
            "fresh generation prompt directory allocation failed");
    memset(&turn, 0, sizeof(turn));
    turn.schema_version = YVEX_RUNTIME_GENERATION_TURN_SCHEMA_V1;
    turn.prompt = request;
    turn.maximum_new_tokens = context->options.maximum_new_tokens;
    turn.prompt_token_ids = prompt_tokens;
    turn.prompt_token_capacity = context->options.context_capacity;
    rc = yvex_runtime_generation_turn_execute(
        context, &turn, tokens, token_capacity, text, text_capacity, result, err);
    yvex_core_free(prompt_tokens);
    return rc;
}

static int generation_hash_finish(yvex_sha256 *hash,
                                  char output[YVEX_SHA256_HEX_CAP])
{
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    if (!hash || !output || !yvex_sha256_final(hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

int yvex_runtime_generation_bytes_digest(const char *domain, const unsigned char *bytes,
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

int yvex_runtime_generation_prefix_identity(
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

int yvex_runtime_generation_stop_identity(
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

int yvex_runtime_generation_plan_identity(
    const yvex_runtime_generation_plan_summary *plan,
    char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    if (!plan || !output) return 0;
    yvex_sha256_init(&hash);
    return yvex_sha256_update_text(&hash, "yvex.runtime.generation.plan.v5") &&
           yvex_sha256_update_u64(&hash, plan->schema_version) &&
           yvex_sha256_update_u64(&hash, plan->backend) &&
           yvex_sha256_update_u64(&hash, plan->mode) &&
           yvex_sha256_update_u64(&hash, plan->context_capacity) &&
           yvex_sha256_update_u64(&hash, plan->prefill_chunk_tokens) &&
           yvex_sha256_update_u64(&hash, plan->maximum_new_tokens) &&
           yvex_sha256_update_u64(&hash, plan->maximum_output_bytes) &&
           yvex_sha256_update_u64(&hash, plan->trace_policy) &&
           yvex_sha256_update_u64(&hash, plan->evidence_profile) &&
           yvex_sha256_update_u64(&hash, plan->execution_class) &&
           yvex_sha256_update_text(&hash, plan->runtime_model_identity) &&
           yvex_sha256_update_text(&hash, plan->runtime_binding_identity) &&
           yvex_sha256_update_text(&hash, plan->runtime_descriptor_identity) &&
           yvex_sha256_update_text(&hash, plan->tokenizer_plan_identity) &&
           yvex_sha256_update_text(&hash, plan->prompt_policy_identity) &&
           yvex_sha256_update_text(&hash, plan->transformer_plan_identity) &&
           yvex_sha256_update_text(&hash, plan->logits_plan_identity) &&
           yvex_sha256_update_text(&hash, plan->sampling_policy_identity) &&
           yvex_sha256_update_text(&hash, plan->speculation_policy_identity) &&
           yvex_sha256_update_text(&hash, plan->stop_policy_identity) &&
           yvex_sha256_update_text(&hash, plan->kernel_bundle_identity) &&
           yvex_sha256_update_text(&hash, plan->execution_profile_identity) &&
           yvex_sha256_update_text(&hash, plan->workload_profile_identity) &&
           yvex_sha256_update_text(&hash, plan->hardware_profile) &&
           generation_hash_finish(&hash, output);
}

int yvex_runtime_generation_token_identity(
    const yvex_runtime_generation_token_result *token,
    char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    if (!token || !output ||
        token->schema_version != YVEX_RUNTIME_GENERATION_SCHEMA_V3 ||
        !token->sampled || token->sampled_token_id != token->classification.token_id)
        return 0;
    yvex_sha256_init(&hash);
    return yvex_sha256_update_text(&hash, "yvex.runtime.generation.token.v3") &&
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

int yvex_runtime_generation_tokens_identity(
    const yvex_runtime_generation_token_result *tokens,
    unsigned long long count, char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned long long index;
    if ((!tokens && count) || !output) return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.generation.tokens.v3") ||
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
int yvex_runtime_generation_execution_identity(
    const yvex_runtime_generation_result *result,
    const yvex_runtime_generation_token_result *tokens,
    char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned long long index;
    if (!result || (!tokens && result->sampled_token_count) || !output) return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.generation.execution.v5") ||
        !yvex_sha256_update_u64(&hash, result->schema_version) ||
        !yvex_sha256_update_u64(&hash, result->execution_mode) ||
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
        !yvex_sha256_update_u64(&hash, result->draft_cycle_count) ||
        !yvex_sha256_update_u64(&hash, result->draft_forward_count) ||
        !yvex_sha256_update_u64(&hash, result->proposed_token_count) ||
        !yvex_sha256_update_u64(&hash, result->selected_verification_token_count) ||
        !yvex_sha256_update_u64(&hash, result->target_verification_count) ||
        !yvex_sha256_update_u64(&hash, result->accepted_draft_token_count) ||
        !yvex_sha256_update_u64(&hash, result->rejected_draft_token_count) ||
        !yvex_sha256_update_u64(&hash, result->discarded_draft_token_count) ||
        !yvex_sha256_update_u64(&hash,
                                result->target_correction_or_bonus_token_count) ||
        !yvex_sha256_update_u64(&hash, result->speculation_source_boundary_token_count) ||
        !yvex_sha256_update_u64(&hash, result->maximum_accepted_prefix) ||
        !yvex_sha256_update_u64(&hash, result->confidence_logit_count) ||
        !yvex_sha256_update_u64(&hash, result->initial_position) ||
        !yvex_sha256_update_u64(&hash, result->reusable_prefix_token_count) ||
        !yvex_sha256_update_u64(&hash, result->new_prefill_token_count) ||
        !yvex_sha256_update_u64(&hash, result->final_position) ||
        !yvex_sha256_update_u64(&hash, result->final_persistent_generation) ||
        !yvex_sha256_update_u64(&hash, result->final_rng_generation) ||
        !yvex_sha256_update_u64(
            &hash, result->final_token_ledger_generation) ||
        !yvex_sha256_update_u64(&hash, result->generated_text_bytes) ||
        !yvex_sha256_update_text(&hash, result->prompt_identity) ||
        !yvex_sha256_update_text(&hash, result->prompt_token_identity) ||
        !yvex_sha256_update_text(&hash, result->reusable_prefix_identity) ||
        !yvex_sha256_update_text(&hash, result->initial_rng_identity) ||
        !yvex_sha256_update_text(&hash, result->final_rng_identity) ||
        !yvex_sha256_update_text(&hash,
                                 result->final_token_ledger_identity) ||
        !yvex_sha256_update_text(&hash, result->generated_token_identity) ||
        !yvex_sha256_update_text(&hash, result->generated_text_digest) ||
        !yvex_sha256_update_text(&hash, result->final_persistent_state_digest) ||
        !yvex_sha256_update_text(&hash, result->generation_plan_identity) ||
        !yvex_sha256_update_text(&hash,
                                 result->speculation_policy_identity) ||
        !yvex_sha256_update_u64(&hash, result->partial_turn.schema_version) ||
        !yvex_sha256_update_u64(&hash, result->partial_turn.available) ||
        !yvex_sha256_update_u64(&hash, result->partial_turn.committed_progress) ||
        !yvex_sha256_update_u64(&hash, result->partial_turn.reset_required) ||
        !yvex_sha256_update_u64(
            &hash, result->partial_turn.draft_state_generation_available) ||
        !yvex_sha256_update_u64(
            &hash, result->partial_turn.detokenizer_generation_available) ||
        !yvex_sha256_update_u64(&hash,
                                (uint32_t)result->partial_turn.failure_status) ||
        !yvex_sha256_update_u64(&hash, result->partial_turn.stop_reason) ||
        !yvex_sha256_update_u64(&hash, result->partial_turn.initial_position) ||
        !yvex_sha256_update_u64(
            &hash, result->partial_turn.final_committed_position) ||
        !yvex_sha256_update_u64(&hash,
                                result->partial_turn.committed_token_count) ||
        !yvex_sha256_update_u64(&hash,
                                result->partial_turn.published_text_bytes) ||
        !yvex_sha256_update_u64(&hash,
                                result->partial_turn.target_state_generation) ||
        !yvex_sha256_update_u64(&hash,
                                result->partial_turn.draft_state_generation) ||
        !yvex_sha256_update_u64(&hash, result->partial_turn.rng_generation) ||
        !yvex_sha256_update_u64(&hash,
                                result->partial_turn.token_ledger_generation) ||
        !yvex_sha256_update_u64(
            &hash, result->partial_turn.detokenizer_generation) ||
        !yvex_sha256_update_text(&hash,
                                 result->partial_turn.target_state_identity) ||
        !yvex_sha256_update_text(&hash,
                                 result->partial_turn.rng_state_identity) ||
        !yvex_sha256_update_text(&hash,
                                 result->partial_turn.token_ledger_identity) ||
        !yvex_sha256_update_text(&hash,
                                 result->partial_turn.published_text_identity)) return 0;
    for (index = 0ull; index < result->sampled_token_count; ++index)
        if (!yvex_sha256_update_text(&hash, tokens[index].token_step_identity))
            return 0;
    return generation_hash_finish(&hash, output);
}

static int generation_partial_turn_validate(
    const yvex_runtime_generation_result *result)
{
    const yvex_runtime_partial_turn *partial = &result->partial_turn;
    int progress = result->final_position > result->initial_position ||
                   result->model_committed_token_count ||
                   result->generated_text_bytes;
    if (result->completed)
        return !partial->available && !partial->schema_version &&
               !partial->committed_progress && !partial->reset_required;
    return partial->schema_version == YVEX_RUNTIME_PARTIAL_TURN_SCHEMA_V1 &&
           partial->available && partial->reset_required &&
           partial->failure_status != YVEX_OK &&
           partial->stop_reason == result->stop_reason &&
           partial->committed_progress == progress &&
           partial->initial_position == result->initial_position &&
           partial->final_committed_position == result->final_position &&
           partial->committed_token_count == result->model_committed_token_count &&
           partial->published_text_bytes == result->generated_text_bytes &&
           partial->target_state_generation == result->final_persistent_generation &&
           partial->rng_generation == result->final_rng_generation &&
           partial->token_ledger_generation ==
               result->final_token_ledger_generation &&
           !partial->draft_state_generation_available &&
           !partial->draft_state_generation &&
           !partial->detokenizer_generation_available &&
           !partial->detokenizer_generation &&
           strcmp(partial->target_state_identity,
                  result->final_persistent_state_digest) == 0 &&
           strcmp(partial->rng_state_identity, result->final_rng_identity) == 0 &&
           strcmp(partial->token_ledger_identity,
                  result->final_token_ledger_identity) == 0 &&
           strcmp(partial->published_text_identity,
                  result->generated_text_digest) == 0 &&
           yvex_sha256_hex_valid(partial->token_ledger_identity);
}

static int generation_roofline_validate(
    const yvex_runtime_generation_plan_summary *plan,
    const yvex_runtime_generation_result *result)
{
    const yvex_execution_roofline_ledger *ledger = &result->roofline;
    unsigned long long index, measured_count = 0ull;
    unsigned long long phase_mask = 0ull, rooflined_mask = 0ull;
    if (!result->roofline_available)
        return !ledger->schema_version && !ledger->identity[0];
    if (ledger->schema_version != YVEX_EXECUTION_PHASE_ROOFLINE_SCHEMA_V1 ||
        ledger->phase_count != YVEX_EXECUTION_ROOFLINE_PHASE_COUNT ||
        !ledger->measured_phase_count ||
        ledger->measured_phase_count > YVEX_EXECUTION_ROOFLINE_PHASE_COUNT ||
        strcmp(ledger->artifact_identity, result->profile.artifact_identity) != 0 ||
        strcmp(ledger->execution_profile_identity, plan->execution_profile_identity) != 0 ||
        strcmp(ledger->kernel_bundle_identity, plan->kernel_bundle_identity) != 0 ||
        strcmp(ledger->workload_profile_identity, plan->workload_profile_identity) != 0 ||
        !yvex_sha256_hex_valid(ledger->hardware_profile_identity) ||
        !yvex_sha256_hex_valid(ledger->identity)) return 0;
    for (index = 0ull; index < ledger->phase_count; ++index) {
        const yvex_execution_phase_roofline *phase = &ledger->phases[index];
        unsigned long long bit = 1ull << index;
        if (phase->measurement.phase != (yvex_execution_roofline_phase)index ||
            phase->missing_fact_mask !=
                (YVEX_EXECUTION_PHASE_FACT_ALL & ~phase->measurement.fact_mask)) return 0;
        if (!phase->available) continue;
        phase_mask |= bit;
        measured_count++;
        if (!phase->measurement.measured_duration_ns || !phase->measurement.work_units ||
            !(phase->measurement.fact_mask &
              YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_DURATION)) ||
            !(phase->measurement.fact_mask &
              YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_WORK))) return 0;
        if (phase->roofline_available) rooflined_mask |= bit;
    }
    return phase_mask == ledger->measured_phase_mask &&
           rooflined_mask == ledger->rooflined_phase_mask &&
           ledger->missing_phase_mask ==
               (((1ull << YVEX_EXECUTION_ROOFLINE_PHASE_COUNT) - 1ull) & ~phase_mask) &&
           ledger->measured_phase_count == measured_count &&
           ledger->priority_provisional == (rooflined_mask != phase_mask);
}

int yvex_runtime_generation_result_validate(
    const yvex_runtime_generation_plan_summary *plan,
    const yvex_runtime_generation_token_result *tokens,
    unsigned long long token_capacity, const unsigned char *text,
    unsigned long long text_capacity,
    const yvex_runtime_generation_result *result, yvex_error *err)
{
    char identity[YVEX_SHA256_HEX_CAP], text_digest[YVEX_SHA256_HEX_CAP];
    yvex_expert_worklist_observation worklists = {0};
    unsigned long long index, published_offset = 0ull, committed = 0ull;
    unsigned long long published = 0ull, terminal = 0ull, suppressed = 0ull;
    unsigned long long classified_selected = 0ull, classified_proposed = 0ull;
    unsigned long long completed_samples = 0ull, expected_final_position = 0ull;
    int result_extents_valid, speculation_counts_valid, worklists_valid;
    worklists_valid = !result || !result->expert_worklists.worklist_count ||
        yvex_expert_worklist_observation_add(
            &worklists, &result->expert_worklists, NULL) == YVEX_OK;
    result_extents_valid = result &&
        yvex_core_u64_add(result->model_committed_token_count,
                          result->terminal_token_count,
                          &completed_samples) &&
        yvex_core_u64_add(result->initial_position,
                          result->new_prefill_token_count,
                          &expected_final_position) &&
        yvex_core_u64_add(expected_final_position,
                          result->model_committed_token_count,
                          &expected_final_position);
    if (!plan || !result || (!tokens && result->sampled_token_count) ||
        (!text && result->generated_text_bytes) ||
        plan->schema_version != YVEX_RUNTIME_GENERATION_SCHEMA_V5 ||
        result->schema_version != YVEX_RUNTIME_GENERATION_RESULT_SCHEMA_V5 ||
        plan->evidence_profile > YVEX_EXECUTION_EVIDENCE_FORENSIC ||
        plan->execution_class > YVEX_EXECUTION_CLASS_FORENSIC_REFERENCE ||
        !yvex_sha256_hex_valid(plan->kernel_bundle_identity) ||
        !yvex_sha256_hex_valid(plan->execution_profile_identity) ||
        !yvex_sha256_hex_valid(plan->workload_profile_identity) || !plan->hardware_profile[0] ||
        result->execution_mode != plan->mode ||
        result->sampled_token_count > token_capacity ||
        result->generated_text_bytes > text_capacity ||
        !result->requested_new_tokens ||
        result->requested_new_tokens > plan->maximum_new_tokens ||
        result->sampled_token_count > result->requested_new_tokens ||
        !result_extents_valid || !worklists_valid ||
        completed_samples > result->sampled_token_count ||
        result->reusable_prefix_token_count != result->initial_position ||
        result->reusable_prefix_token_count > result->prompt_token_count ||
        result->new_prefill_token_count !=
            result->prompt_token_count - result->reusable_prefix_token_count ||
        result->profile.schema_version != YVEX_RUNTIME_PROFILE_SCHEMA_V4 ||
        runtime_profile_validate(&result->profile, NULL) != YVEX_OK ||
        !generation_roofline_validate(plan, result) ||
        !yvex_sha256_hex_valid(result->reusable_prefix_identity) ||
        !generation_partial_turn_validate(result) ||
        strcmp(result->speculation_policy_identity,
               plan->speculation_policy_identity) != 0 ||
        strcmp(result->generation_plan_identity,
               plan->generation_plan_identity) != 0 ||
        !yvex_runtime_generation_plan_identity(plan, identity) ||
        strcmp(identity, plan->generation_plan_identity) != 0)
        return generation_result_refuse(err, YVEX_ERR_FORMAT,
                                 "generation result geometry or plan identity is invalid");
    for (index = 0ull; index < result->sampled_token_count; ++index) {
        const yvex_runtime_generation_token_result *token = &tokens[index];
        if (token->ordinal != index || !token->sampled ||
            token->sampled_token_id != token->classification.token_id ||
            (token->decode_submitted &&
             token->decode_input_token_id != token->sampled_token_id) ||
            (!token->decode_submitted && token->decode_input_token_id) ||
            !yvex_runtime_generation_token_identity(token, identity) ||
            strcmp(identity, token->token_step_identity) != 0)
            return generation_result_refuse(err, YVEX_ERR_FORMAT,
                                     "generation token identity is invalid");
        if (token->terminal) {
            if (token->decode_submitted || token->model_committed ||
                token->decode_input_token_id ||
                token->position_after != token->position_before ||
                token->persistent_generation_after !=
                    token->persistent_generation_before ||
                token->decode_execution_identity[0] ||
                token->sequence_state_after != YVEX_TOKEN_APPEND_PROPOSED)
                return generation_result_refuse(err, YVEX_ERR_FORMAT,
                                         "terminal token falsely claims model commit");
            terminal++;
        }
        if (token->model_committed) {
            if (token->terminal || !token->decode_submitted ||
                token->decode_input_token_id != token->sampled_token_id ||
                token->position_after != token->position_before + 1ull ||
                token->persistent_generation_after <=
                    token->persistent_generation_before ||
                !yvex_sha256_hex_valid(token->decode_execution_identity) ||
                !yvex_sha256_hex_valid(token->persistent_state_digest))
                return generation_result_refuse(err, YVEX_ERR_FORMAT,
                                         "ordinary token model transition is invalid");
            committed++;
        } else if (token->position_after != token->position_before ||
                   token->persistent_generation_after !=
                       token->persistent_generation_before ||
                   token->decode_execution_identity[0] ||
                   token->sequence_state_after > YVEX_TOKEN_APPEND_SUBMITTED) {
            return generation_result_refuse(err, YVEX_ERR_FORMAT,
                                     "uncommitted token claims downstream progress");
        }
        if (token->text_published) {
            if (!token->model_committed || !token->detokenized ||
                token->sequence_state_after != YVEX_TOKEN_APPEND_TEXT_PUBLISHED ||
                token->text_byte_offset != published_offset ||
                !yvex_core_u64_add(published_offset, token->text_byte_count,
                                   &published_offset))
                return generation_result_refuse(err, YVEX_ERR_FORMAT,
                                         "generation text directory is discontinuous");
            published++;
        }
        if (token->suppressed) suppressed++;
    }
    speculation_counts_valid =
        yvex_core_u64_add(result->accepted_draft_token_count,
                          result->rejected_draft_token_count,
                          &classified_selected) &&
        yvex_core_u64_add(classified_selected,
                          result->discarded_draft_token_count,
                          &classified_proposed);
    if (committed != result->model_committed_token_count ||
        published != result->text_published_token_count ||
        terminal != result->terminal_token_count ||
        suppressed != result->suppressed_token_count ||
        published_offset != result->generated_text_bytes ||
        result->decode_step_count != committed ||
        result->logits_projection_count < result->sampled_token_count ||
        result->has_incomplete_token !=
            (result->sampled_token_count > completed_samples) ||
        result->first_incomplete_token !=
            (result->has_incomplete_token ? result->sampled_token_count - 1ull
                                          : result->sampled_token_count) ||
        (result->execution_mode == YVEX_GENERATION_MODE_TARGET_ONLY &&
         (result->draft_cycle_count || result->draft_forward_count || result->proposed_token_count ||
          result->selected_verification_token_count || result->target_verification_count ||
          result->accepted_draft_token_count || result->rejected_draft_token_count ||
          result->discarded_draft_token_count || result->target_correction_or_bonus_token_count ||
          result->speculation_source_boundary_token_count || result->maximum_accepted_prefix ||
          result->confidence_logit_count || result->confidence_logit_minimum ||
          result->confidence_logit_maximum || result->confidence_logit_mean || result->draft_ns ||
          result->verification_ns || result->speculative_commit_ns ||
          result->speculation_policy_identity[0])) ||
        (result->execution_mode == YVEX_GENERATION_MODE_DSPARK &&
         (!speculation_counts_valid ||
          !yvex_sha256_hex_valid(result->speculation_policy_identity) ||
          result->speculation_source_boundary_token_count > result->model_committed_token_count ||
          result->draft_forward_count > result->draft_cycle_count ||
           result->target_verification_count > result->draft_forward_count ||
           (result->completed &&
            (result->draft_cycle_count != result->draft_forward_count ||
             result->draft_forward_count != result->target_verification_count)) ||
          result->selected_verification_token_count >
              result->proposed_token_count ||
          classified_selected >
              result->selected_verification_token_count ||
          classified_proposed != result->proposed_token_count ||
          result->maximum_accepted_prefix >
              result->accepted_draft_token_count ||
          result->confidence_logit_count != result->proposed_token_count ||
          (result->confidence_logit_count &&
           (!isfinite(result->confidence_logit_minimum) ||
            !isfinite(result->confidence_logit_maximum) ||
            !isfinite(result->confidence_logit_mean) ||
            result->confidence_logit_minimum >
                result->confidence_logit_mean ||
            result->confidence_logit_mean >
                result->confidence_logit_maximum)) ||
          (!result->confidence_logit_count &&
           (result->confidence_logit_minimum ||
            result->confidence_logit_maximum ||
            result->confidence_logit_mean)) ||
          result->target_correction_or_bonus_token_count >
              result->target_verification_count)))
        return generation_result_refuse(err, YVEX_ERR_FORMAT,
                                 "generation aggregate counters are inconsistent");
    if (result->completed &&
        (result->status != YVEX_GENERATION_STATUS_COMPLETE || result->partial ||
         result->cancelled || result->failed ||
         result->final_position != expected_final_position))
        return generation_result_refuse(err, YVEX_ERR_FORMAT,
                                 "complete generation state is inconsistent");
    if (!yvex_runtime_generation_tokens_identity(tokens, result->sampled_token_count, identity) ||
        strcmp(identity, result->generated_token_identity) != 0 ||
        !yvex_runtime_generation_bytes_digest("yvex.runtime.generation.text.v1", text,
                                 result->generated_text_bytes, text_digest) ||
        strcmp(text_digest, result->generated_text_digest) != 0 ||
        !yvex_runtime_generation_execution_identity(result, tokens, identity) ||
        strcmp(identity, result->generation_execution_identity) != 0)
        return generation_result_refuse(err, YVEX_ERR_FORMAT,
                                 "generation aggregate evidence was mutated");
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
    result->dspark_ready =
        result->execution.execution_mode == YVEX_GENERATION_MODE_DSPARK;
    result->speculative_execution_ready = result->dspark_ready &&
        result->execution.draft_cycle_count > 0ull &&
        result->execution.target_verification_count > 0ull;
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
    yvex_model_engine_open_request model_request = {0};
    yvex_runtime_session_open_request session_request = {0};
    yvex_runtime_generation_options options = {0};
    yvex_runtime_generation_request execution_request = {0};
    yvex_model_engine_failure failure = {0};
    yvex_runtime_cleanup_lease *cleanup = NULL;
    yvex_model_engine *model = NULL;
    yvex_runtime_execution_session *session = NULL;
    yvex_runtime_generation_context *context = NULL;
    const yvex_model_engine_view *view;
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
        request->mode > YVEX_GENERATION_MODE_DSPARK ||
        (request->backend != YVEX_BACKEND_KIND_CPU &&
         request->backend != YVEX_BACKEND_KIND_CUDA))
        return generation_result_refuse(err, YVEX_ERR_INVALID_ARG,
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
    options.schema_version = YVEX_RUNTIME_GENERATION_SCHEMA_V5;
    options.backend = request->backend;
    options.mode = request->mode;
    options.context_capacity = request->context_capacity;
    options.prefill_chunk_tokens = request->prefill_chunk_tokens;
    options.maximum_new_tokens = request->maximum_new_tokens;
    options.maximum_output_bytes = request->maximum_output_bytes;
    options.maximum_host_bytes = request->maximum_host_bytes;
    options.maximum_device_bytes = request->maximum_device_bytes;
    options.trace_policy = YVEX_RUNTIME_TRACE_SUMMARY;
    options.evidence_profile = YVEX_EXECUTION_EVIDENCE_PRODUCTION;
    options.sampling_policy = request->sampling_policy;
    options.cancel_requested = request->cancel_requested;
    options.cancel_context = request->cancel_context;
    model_request.artifact_path = request->artifact_path;
    model_request.runtime_binding_path = request->runtime_binding_path;
    model_request.target_id = request->target;
    model_request.startup_generation = &options;
    model_request.residency_backend = request->backend;
    model_request.maximum_host_bytes = request->maximum_host_bytes;
    model_request.maximum_device_bytes = request->maximum_device_bytes;
    session_request.backend = request->backend;
    session_request.maximum_host_bytes = request->maximum_host_bytes;
    session_request.maximum_device_bytes = request->maximum_device_bytes;
    rc = yvex_runtime_cleanup_lease_acquire(
        &cleanup, &model_request, &session_request, &model, &session,
        &failure, err);
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
        rc = generation_result_refuse(err, YVEX_ERR_BOUNDS,
                               "generation operator output extent overflowed");
    if (rc == YVEX_OK) {
        result->tokens = yvex_core_calloc(
            (size_t)request->maximum_new_tokens, sizeof(*result->tokens));
        result->text = yvex_core_calloc((size_t)text_allocation, 1u);
        if (!result->tokens || !result->text)
            rc = generation_result_refuse(err, YVEX_ERR_NOMEM,
                                   "generation operator output allocation failed");
    }
    execution_request.schema_version = YVEX_RUNTIME_GENERATION_SCHEMA_V3;
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
        rc = yvex_runtime_generation_context_summary_copy(context, &result->context, err);
    if (result->execution.schema_version == YVEX_RUNTIME_GENERATION_RESULT_SCHEMA_V5) {
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
    view = yvex_model_engine_view_get(model);
    if (view && view->target_id)
        yvex_core_text_copy(result->family, sizeof(result->family),
                            view->target_id);
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
