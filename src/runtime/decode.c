/* Owner: runtime repeated decode.
 * Owns: explicit decode admission, one-token steps, repeated lifecycle, partial progress, and operator workflow.
 * Does not own: transformer math, token-file serialization, persistent-state storage, logits, or sampling.
 * Invariants: the session is authoritative for position/generation and each completed step commits exactly once.
 * Boundary: family-neutral orchestration over one borrowed transformer context and its paired session.
 * Purpose: execute teacher-forced numeric-token decode while preserving warm transformer and KV resources.
 * Inputs: admitted token inputs, exact expected positions, backend selection, and caller-owned output storage.
 * Effects: advances committed session state per successful token and publishes bounded step evidence.
 * Failure: preserves completed steps, never publishes the failing step, and reports exact partial progress. */
#include <yvex/internal/decode.h>

#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <yvex/internal/core.h>
#include <yvex/internal/runtime.h>

struct yvex_runtime_decode_context {
    yvex_runtime_transformer_context *transformer;
    yvex_runtime_execution_session *session;
    yvex_runtime_decode_options options;
    unsigned long long execution_count;
    pthread_mutex_t mutex;
    int mutex_ready, busy;
};

/* Purpose: publish one stable decode refusal.
 * Inputs: status, reason, and optional error output. Effects: replaces only the error.
 * Failure: returns the supplied status. Boundary: no runtime state mutation. */
static int decode_refuse(yvex_error *err, yvex_status status, const char *reason)
{
    yvex_error_set(err, status, "runtime.decode", reason);
    return status;
}

/* Purpose: copy the authoritative persistent-state summary from the paired session.
 * Inputs: live session and caller summary. Effects: invokes the provider summary protocol.
 * Failure: absent provider or provider failure refuses. Boundary: read-only state inspection. */
static int decode_state_summary(const yvex_runtime_execution_session *session,
                                yvex_graph_attention_state_summary *summary,
                                yvex_error *err)
{
    const yvex_runtime_session_view *view = yvex_runtime_session_view_get(session);
    if (!view || !view->attention_state_provider ||
        !view->attention_state_provider->summary ||
        view->attention_state_provider->summary(
            view->attention_state_provider->context, summary, err) != YVEX_OK)
        return decode_refuse(err, YVEX_ERR_STATE,
                             "decode persistent state is unavailable");
    return YVEX_OK;
}

/* Purpose: hash canonical F32 values without native object representation.
 * Inputs: initialized hash and finite value span. Effects: appends canonical value bits.
 * Failure: rejects absent or non-finite input. Boundary: no digest finalization. */
static int decode_hash_values(yvex_sha256 *hash, const float *values,
                              unsigned long long count)
{
    unsigned long long index;
    uint32_t bits;
    if (!hash || (!values && count)) return 0;
    for (index = 0ull; index < count; ++index) {
        if (!isfinite(values[index])) return 0;
        memcpy(&bits, &values[index], sizeof(bits));
        if (!yvex_sha256_update_u64(hash, bits)) return 0;
    }
    return 1;
}

/* Purpose: identify one complete step from ordered canonical fields.
 * Inputs: complete step evidence and caller digest. Effects: writes one SHA-256 identity.
 * Failure: inconsistent transition or identity facts refuse. Boundary: excludes native layout. */
int yvex_runtime_decode_step_identity(
    const yvex_runtime_decode_step_result *result,
    char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    if (output) output[0] = '\0';
    if (!result || !output || result->schema_version != YVEX_RUNTIME_DECODE_SCHEMA_V1 ||
        !result->completed || result->position_after != result->position_before + 1ull ||
        result->generation_after != result->generation_before + 1ull ||
        !yvex_sha256_hex_valid(result->embedding_digest) ||
        !yvex_sha256_hex_valid(result->routing_digest) ||
        !yvex_sha256_hex_valid(result->layer_digest) ||
        !yvex_sha256_hex_valid(result->normalized_hidden_digest) ||
        !yvex_sha256_hex_valid(result->persistent_state_digest) ||
        !yvex_sha256_hex_valid(result->transformer_execution_identity)) return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.decode.step.v1") ||
        !yvex_sha256_update_u64(&hash, result->schema_version) ||
        !yvex_sha256_update_u64(&hash, result->step_ordinal) ||
        !yvex_sha256_update_u64(&hash, result->token_id) ||
        !yvex_sha256_update_u64(&hash, result->position_before) ||
        !yvex_sha256_update_u64(&hash, result->position_after) ||
        !yvex_sha256_update_u64(&hash, result->generation_before) ||
        !yvex_sha256_update_u64(&hash, result->generation_after) ||
        !yvex_sha256_update_u64(&hash, result->layers_executed) ||
        !yvex_sha256_update_u64(&hash, result->swa_layers) ||
        !yvex_sha256_update_u64(&hash, result->csa_layers) ||
        !yvex_sha256_update_u64(&hash, result->hca_layers) ||
        !yvex_sha256_update_u64(&hash, result->hash_routers) ||
        !yvex_sha256_update_u64(&hash, result->learned_routers) ||
        !yvex_sha256_update_u64(&hash, result->routed_experts) ||
        !yvex_sha256_update_u64(&hash, result->shared_experts) ||
        !yvex_sha256_update_u64(&hash, result->h2d_bytes) ||
        !yvex_sha256_update_u64(&hash, result->d2h_bytes) ||
        !yvex_sha256_update_u64(&hash, result->kernel_launches) ||
        !yvex_sha256_update_text(&hash, result->embedding_digest) ||
        !yvex_sha256_update_text(&hash, result->routing_digest) ||
        !yvex_sha256_update_text(&hash, result->layer_digest) ||
        !yvex_sha256_update_text(&hash, result->normalized_hidden_digest) ||
        !yvex_sha256_update_text(&hash, result->persistent_state_digest) ||
        !yvex_sha256_update_text(&hash, result->transformer_execution_identity) ||
        !yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

/* Purpose: identify an ordered complete-or-partial repeated decode result.
 * Inputs: aggregate facts, completed step directory, and output. Effects: writes one identity.
 * Failure: inconsistent prefix or missing identities refuse. Boundary: excludes timing/pointers. */
int yvex_runtime_decode_result_identity(
    const yvex_runtime_decode_result *result,
    const yvex_runtime_decode_step_result *steps,
    char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;
    if (output) output[0] = '\0';
    if (!result || !output || (!steps && result->completed_steps) ||
        result->schema_version != YVEX_RUNTIME_DECODE_SCHEMA_V1 ||
        result->completed_steps > result->requested_steps ||
        result->final_committed_prefix !=
            result->initial_committed_prefix + result->completed_steps ||
        !yvex_sha256_hex_valid(result->input_identity) ||
        !yvex_sha256_hex_valid(result->aggregate_hidden_digest) ||
        !yvex_sha256_hex_valid(result->aggregate_state_digest)) return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.decode.execution.v1") ||
        !yvex_sha256_update_u64(&hash, result->schema_version) ||
        !yvex_sha256_update_u64(&hash, result->requested_steps) ||
        !yvex_sha256_update_u64(&hash, result->completed_steps) ||
        !yvex_sha256_update_u64(&hash, result->status) ||
        !yvex_sha256_update_u64(&hash, (unsigned int)result->completed) ||
        !yvex_sha256_update_u64(&hash, (unsigned int)result->partial) ||
        !yvex_sha256_update_u64(&hash, (unsigned int)result->has_incomplete_step) ||
        !yvex_sha256_update_u64(&hash, result->first_incomplete_step) ||
        !yvex_sha256_update_u64(&hash, result->initial_committed_prefix) ||
        !yvex_sha256_update_u64(&hash, result->final_committed_prefix) ||
        !yvex_sha256_update_u64(&hash, result->generation_before) ||
        !yvex_sha256_update_u64(&hash, result->generation_after) ||
        !yvex_sha256_update_u64(&hash, result->layers_executed) ||
        !yvex_sha256_update_u64(&hash, result->swa_layers) ||
        !yvex_sha256_update_u64(&hash, result->csa_layers) ||
        !yvex_sha256_update_u64(&hash, result->hca_layers) ||
        !yvex_sha256_update_u64(&hash, result->hash_routers) ||
        !yvex_sha256_update_u64(&hash, result->learned_routers) ||
        !yvex_sha256_update_u64(&hash, result->routed_experts) ||
        !yvex_sha256_update_u64(&hash, result->shared_experts) ||
        !yvex_sha256_update_u64(&hash, result->h2d_bytes) ||
        !yvex_sha256_update_u64(&hash, result->d2h_bytes) ||
        !yvex_sha256_update_u64(&hash, result->kernel_launches) ||
        !yvex_sha256_update_text(&hash, result->input_identity) ||
        !yvex_sha256_update_text(&hash, result->aggregate_hidden_digest) ||
        !yvex_sha256_update_text(&hash, result->aggregate_state_digest)) return 0;
    for (index = 0ull; index < result->completed_steps; ++index)
        if (!yvex_sha256_hex_valid(steps[index].decode_step_identity) ||
            !yvex_sha256_update_text(&hash, steps[index].decode_step_identity)) return 0;
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

/* Purpose: enter one decode coordinator exclusively.
 * Inputs: live context. Effects: sets the busy lifecycle flag under its mutex.
 * Failure: lock or concurrent use refuses. Boundary: does not enter transformer execution. */
static int decode_enter(yvex_runtime_decode_context *context, yvex_error *err)
{
    if (!context || pthread_mutex_lock(&context->mutex) != 0)
        return decode_refuse(err, YVEX_ERR_STATE, "decode context lock failed");
    if (context->busy) {
        (void)pthread_mutex_unlock(&context->mutex);
        return decode_refuse(err, YVEX_ERR_STATE,
                             "decode context is busy");
    }
    context->busy = 1;
    (void)pthread_mutex_unlock(&context->mutex);
    return YVEX_OK;
}

/* Purpose: leave one decode coordinator and retain its reusable lifecycle.
 * Inputs: held context and completion fact. Effects: clears busy and advances local evidence.
 * Failure: a mutex failure cannot alter model/session state. Boundary: decode-local counters only. */
static void decode_leave(yvex_runtime_decode_context *context, int completed)
{
    if (context && pthread_mutex_lock(&context->mutex) == 0) {
        context->busy = 0;
        if (completed) context->execution_count++;
        (void)pthread_mutex_unlock(&context->mutex);
    }
}

/* Purpose: allocate one decode lifecycle over an already-open paired transformer/session.
 * Inputs: borrowed exact owners and bounded step policy. Effects: owns only mutex/counters.
 * Failure: owner mismatch or allocation failure leaves no context. Boundary: no model resource opens. */
int yvex_runtime_decode_context_open(
    yvex_runtime_decode_context **out,
    yvex_runtime_transformer_context *transformer,
    yvex_runtime_execution_session *session,
    const yvex_runtime_decode_options *options, yvex_error *err)
{
    yvex_runtime_decode_context *context;
    if (out) *out = NULL;
    if (!out || !transformer || !session || !options || !options->maximum_steps ||
        yvex_runtime_transformer_context_session(transformer) != session)
        return decode_refuse(err, YVEX_ERR_INVALID_ARG,
                             "decode requires one paired transformer/session and step budget");
    context = (yvex_runtime_decode_context *)calloc(1u, sizeof(*context));
    if (!context)
        return decode_refuse(err, YVEX_ERR_NOMEM, "decode context allocation failed");
    context->transformer = transformer;
    context->session = session;
    context->options = *options;
    if (pthread_mutex_init(&context->mutex, NULL) != 0) {
        free(context);
        return decode_refuse(err, YVEX_ERR_STATE, "decode mutex initialization failed");
    }
    context->mutex_ready = 1;
    *out = context;
    yvex_error_clear(err);
    return YVEX_OK;
}

/* Purpose: seal one bounded one-token view against the existing transformer input schema.
 * Inputs: plan, position, and numeric token ID. Effects: allocates one immutable input owner.
 * Failure: vocabulary, identity, or allocation refusal propagates. Boundary: no external file. */
static int decode_token_input(yvex_transformer_input **out,
                              const yvex_transformer_plan_summary *plan,
                              unsigned long long position, unsigned int token_id,
                              yvex_error *err)
{
    yvex_transformer_input_summary summary;
    memset(&summary, 0, sizeof(summary));
    summary.schema_version = YVEX_TRANSFORMER_INPUT_SCHEMA_V1;
    summary.token_start = position;
    summary.token_count = 1ull;
    summary.vocabulary_size = plan->vocabulary_size;
    yvex_runtime_identity_copy(summary.logical_model_identity,
                               plan->logical_model_identity);
    yvex_runtime_identity_copy(summary.runtime_numeric_identity,
                               plan->runtime_numeric_identity);
    yvex_runtime_identity_copy(summary.runtime_descriptor_identity,
                               plan->runtime_descriptor_identity);
    yvex_runtime_identity_copy(summary.transformer_plan_identity,
                               plan->transformer_plan_identity);
    if (yvex_transformer_input_seal(&summary, &token_id, err) != YVEX_OK)
        return yvex_error_code(err);
    return yvex_transformer_input_open_memory(out, &summary, &token_id, err);
}

/* Purpose: execute one step while the decode coordinator is exclusively held.
 * Inputs: exact expected position/token/backend and prevalidated caller row capacity.
 * Effects: delegates one explicit decode-phase transformer transaction and publishes evidence.
 * Failure: no failing-step KV or hidden row is published. Boundary: no repeated-loop policy. */
static int decode_step_locked(
    yvex_runtime_decode_context *context, unsigned long long step_ordinal,
    unsigned long long expected_position, unsigned int token_id,
    yvex_backend_kind backend, float *normalized_hidden,
    unsigned long long normalized_hidden_capacity,
    yvex_runtime_decode_step_result *result, yvex_error *err)
{
    const yvex_transformer_plan_summary *plan = yvex_transformer_plan_summary_get(
        yvex_runtime_transformer_context_plan(context->transformer));
    yvex_graph_attention_state_summary before = {0}, after = {0};
    yvex_runtime_transformer_request request = {0};
    yvex_runtime_transformer_output output = {0};
    yvex_runtime_transformer_result transformer;
    yvex_transformer_input *input = NULL;
    int rc;
    memset(result, 0, sizeof(*result));
    if (!plan || !expected_position || !normalized_hidden ||
        normalized_hidden_capacity < plan->hidden_width)
        return decode_refuse(err, YVEX_ERR_INVALID_ARG,
                             "decode step output or nonzero position is invalid");
    rc = decode_state_summary(context->session, &before, err);
    if (rc == YVEX_OK &&
        (before.transaction_active || !before.position_consistent ||
         before.prepared_layer_count != before.layer_count ||
         before.next_position != expected_position ||
         before.committed_sequence_length != expected_position))
        rc = decode_refuse(err, YVEX_ERR_STATE,
                           "decode step does not match committed persistent state");
    if (rc == YVEX_OK)
        rc = decode_token_input(&input, plan, expected_position, token_id, err);
    request.chunk_tokens = 1ull;
    request.backend = backend;
    request.phase = YVEX_TRANSFORMER_PHASE_DECODE;
    output.normalized_hidden = normalized_hidden;
    output.capacity = normalized_hidden_capacity;
    if (rc == YVEX_OK)
        rc = yvex_runtime_transformer_execute(context->transformer, input, &request,
                                              &output, &transformer, err);
    if (rc == YVEX_OK) rc = decode_state_summary(context->session, &after, err);
    if (rc == YVEX_OK &&
        (!transformer.completed || transformer.phase != YVEX_TRANSFORMER_PHASE_DECODE ||
         transformer.token_count != 1ull || transformer.chunk_count != 1ull ||
         transformer.position_before != expected_position ||
         transformer.position_after != expected_position + 1ull ||
         after.next_position != expected_position + 1ull ||
         after.committed_sequence_length != expected_position + 1ull ||
         after.generation != before.generation + 1ull ||
         transformer.layers_executed != 43ull || transformer.swa_layers != 2ull ||
         transformer.csa_layers != 21ull || transformer.hca_layers != 20ull ||
         transformer.hash_routers != 3ull || transformer.learned_routers != 40ull ||
         transformer.routed_experts != 258ull || transformer.shared_experts != 43ull))
        rc = decode_refuse(err, YVEX_ERR_STATE,
                           "decode step structural or state invariants failed");
    if (rc == YVEX_OK) {
        result->schema_version = YVEX_RUNTIME_DECODE_SCHEMA_V1;
        result->completed = 1;
        result->step_ordinal = step_ordinal;
        result->token_id = token_id;
        result->position_before = expected_position;
        result->position_after = after.next_position;
        result->generation_before = before.generation;
        result->generation_after = after.generation;
        result->layers_executed = transformer.layers_executed;
        result->swa_layers = transformer.swa_layers;
        result->csa_layers = transformer.csa_layers;
        result->hca_layers = transformer.hca_layers;
        result->hash_routers = transformer.hash_routers;
        result->learned_routers = transformer.learned_routers;
        result->routed_experts = transformer.routed_experts;
        result->shared_experts = transformer.shared_experts;
        result->h2d_bytes = transformer.h2d_bytes;
        result->d2h_bytes = transformer.d2h_bytes;
        result->kernel_launches = transformer.kernel_launches;
        yvex_runtime_identity_copy(result->embedding_digest,
                                   transformer.embedding_digest);
        yvex_runtime_identity_copy(result->routing_digest,
                                   transformer.routing_digest);
        yvex_runtime_identity_copy(result->layer_digest, transformer.layer_digest);
        yvex_runtime_identity_copy(result->normalized_hidden_digest,
                                   transformer.normalized_hidden_digest);
        yvex_runtime_identity_copy(result->persistent_state_digest,
                                   transformer.persistent_state_digest);
        yvex_runtime_identity_copy(result->transformer_execution_identity,
                                   transformer.execution_identity);
        if (!yvex_runtime_decode_step_identity(result, result->decode_step_identity))
            rc = decode_refuse(err, YVEX_ERR_STATE,
                               "decode step identity derivation failed");
    }
    yvex_transformer_input_close(&input);
    return rc;
}

/* Purpose: execute one public decode step with exclusive context lifecycle.
 * Inputs: exact position/token/backend and caller-owned hidden row. Effects: commits one step.
 * Failure: preserves prior KV and leaves the caller row unpublished. Boundary: no repeated loop. */
int yvex_runtime_decode_step(
    yvex_runtime_decode_context *context, unsigned long long step_ordinal,
    unsigned long long expected_position, unsigned int token_id,
    yvex_backend_kind backend, float *normalized_hidden,
    unsigned long long normalized_hidden_capacity,
    yvex_runtime_decode_step_result *result, yvex_error *err)
{
    int rc;
    if (result) memset(result, 0, sizeof(*result));
    if (!context || !result)
        return decode_refuse(err, YVEX_ERR_INVALID_ARG,
                             "decode step context and result are required");
    rc = decode_enter(context, err);
    if (rc != YVEX_OK) return rc;
    rc = decode_step_locked(context, step_ordinal, expected_position, token_id,
                            backend, normalized_hidden,
                            normalized_hidden_capacity, result, err);
    decode_leave(context, rc == YVEX_OK);
    return rc;
}

/* Purpose: accumulate exact structural counters from one completed step.
 * Inputs: aggregate and completed step facts. Effects: adds bounded execution counters.
 * Failure: none for admitted structural bounds. Boundary: identities remain separately ordered. */
static void decode_accumulate(yvex_runtime_decode_result *result,
                              const yvex_runtime_decode_step_result *step)
{
    result->layers_executed += step->layers_executed;
    result->swa_layers += step->swa_layers;
    result->csa_layers += step->csa_layers;
    result->hca_layers += step->hca_layers;
    result->hash_routers += step->hash_routers;
    result->learned_routers += step->learned_routers;
    result->routed_experts += step->routed_experts;
    result->shared_experts += step->shared_experts;
    result->h2d_bytes += step->h2d_bytes;
    result->d2h_bytes += step->d2h_bytes;
    result->kernel_launches += step->kernel_launches;
}

/* Purpose: finalize ordered aggregate evidence over completed caller-owned hidden rows.
 * Inputs: aggregate, output directory, and hidden width. Effects: writes digest identities.
 * Failure: non-finite output or invalid step identity refuses. Boundary: no state rehash/read. */
static int decode_finalize(yvex_runtime_decode_result *result,
                           const yvex_runtime_decode_output *output,
                           unsigned long long hidden_width,
                           const char *initial_state_identity, yvex_error *err)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.decode.hidden.v1") ||
        !decode_hash_values(&hash, output->normalized_hidden,
                            result->completed_steps * hidden_width) ||
        !yvex_sha256_final(&hash, digest))
        return decode_refuse(err, YVEX_ERR_STATE,
                             "decode aggregate hidden digest failed");
    yvex_sha256_hex(digest, result->aggregate_hidden_digest);
    if (result->completed_steps)
        yvex_runtime_identity_copy(
            result->aggregate_state_digest,
            output->steps[result->completed_steps - 1ull].persistent_state_digest);
    if (!result->completed_steps)
        yvex_runtime_identity_copy(result->aggregate_state_digest,
                                   initial_state_identity);
    if (!yvex_runtime_decode_result_identity(
            result, output->steps, result->decode_execution_identity))
        return decode_refuse(err, YVEX_ERR_STATE,
                             "decode execution identity derivation failed");
    return YVEX_OK;
}

/* Purpose: execute ordered teacher-forced tokens as independently committed decode steps.
 * Inputs: validated existing token input, backend, and fully preallocated hidden/step outputs.
 * Effects: publishes every completed step in order and preserves typed partial progress.
 * Failure: returns the first failing status with prior rows/state intact. Boundary: no token choice. */
int yvex_runtime_decode_execute(
    yvex_runtime_decode_context *context,
    const yvex_transformer_input *input,
    const yvex_runtime_decode_request *request,
    yvex_runtime_decode_output *output,
    yvex_runtime_decode_result *result, yvex_error *err)
{
    const yvex_transformer_input_summary *summary =
        yvex_transformer_input_summary_get(input);
    const yvex_transformer_plan_summary *plan = context
        ? yvex_transformer_plan_summary_get(
              yvex_runtime_transformer_context_plan(context->transformer)) : NULL;
    const unsigned int *tokens = yvex_transformer_input_token_ids(input);
    yvex_graph_attention_state_summary state = {0};
    yvex_error primary;
    unsigned long long index, hidden_count, final_position = 0ull;
    int rc, final_rc, state_ready = 0;
    if (result) memset(result, 0, sizeof(*result));
    if (!context || !input || !summary || !plan || !tokens || !request || !output ||
        !output->normalized_hidden || !output->steps || !result || !summary->token_start ||
        !summary->token_count || summary->token_count > context->options.maximum_steps ||
        output->step_capacity < summary->token_count ||
        !yvex_core_u64_mul(summary->token_count, plan->hidden_width, &hidden_count) ||
        output->normalized_hidden_capacity < hidden_count)
        return decode_refuse(err, YVEX_ERR_INVALID_ARG,
                             "decode request or caller output capacity is invalid");
    rc = decode_enter(context, err);
    if (rc != YVEX_OK) return rc;
    rc = yvex_runtime_transformer_context_validate_input(context->transformer,
                                                         input, err);
    if (rc == YVEX_OK) {
        rc = decode_state_summary(context->session, &state, err);
        state_ready = rc == YVEX_OK;
    }
    if (rc == YVEX_OK &&
        (state.transaction_active || !state.position_consistent ||
         state.next_position != summary->token_start ||
         state.committed_sequence_length != summary->token_start))
        rc = decode_refuse(err, YVEX_ERR_STATE,
                           "decode input start does not match committed prefix");
    if (rc == YVEX_OK &&
        (!yvex_core_u64_add(summary->token_start, summary->token_count,
                            &final_position) ||
         final_position > state.capacity))
        rc = decode_refuse(err, YVEX_ERR_BOUNDS,
                           "decode sequence exceeds persistent-state capacity");
    result->schema_version = YVEX_RUNTIME_DECODE_SCHEMA_V1;
    result->requested_steps = summary->token_count;
    result->initial_committed_prefix = summary->token_start;
    result->final_committed_prefix = state.next_position;
    result->generation_before = result->generation_after = state.generation;
    yvex_runtime_identity_copy(result->input_identity, summary->input_identity);
    for (index = 0ull; rc == YVEX_OK && index < summary->token_count; ++index) {
        if (context->options.cancel_requested &&
            context->options.cancel_requested(context->options.cancel_context)) {
            rc = decode_refuse(err, YVEX_ERR_CANCELLED,
                               "repeated decode cancelled between steps");
            break;
        }
        rc = decode_step_locked(
            context, index, summary->token_start + index, tokens[index],
            request->backend,
            output->normalized_hidden + index * plan->hidden_width,
            plan->hidden_width, &output->steps[index], err);
        if (rc == YVEX_OK) {
            result->completed_steps++;
            result->final_committed_prefix = output->steps[index].position_after;
            result->generation_after = output->steps[index].generation_after;
            decode_accumulate(result, &output->steps[index]);
        }
    }
    primary = err ? *err : (yvex_error){0};
    result->has_incomplete_step = result->completed_steps < result->requested_steps;
    result->first_incomplete_step = result->has_incomplete_step
                                        ? result->completed_steps : 0ull;
    result->partial = rc != YVEX_OK && result->completed_steps > 0ull;
    result->completed = rc == YVEX_OK &&
                        result->completed_steps == result->requested_steps;
    result->status = result->completed ? YVEX_RUNTIME_DECODE_STATUS_COMPLETE
                                      : (result->partial
                                             ? YVEX_RUNTIME_DECODE_STATUS_PARTIAL
                                             : YVEX_RUNTIME_DECODE_STATUS_NONE);
    final_rc = state_ready
                   ? decode_finalize(result, output, plan->hidden_width,
                                     state.state_content_identity, err)
                   : rc;
    if (rc == YVEX_OK && final_rc != YVEX_OK) rc = final_rc;
    else if (rc != YVEX_OK && err) *err = primary;
    decode_leave(context, result->completed_steps > 0ull);
    return rc;
}

/* Purpose: release decode-local lifecycle without closing borrowed transformer/session owners.
 * Inputs: idle context handle. Effects: destroys mutex and frees only decode-local memory.
 * Failure: busy or lock failure leaves ownership retryable. Boundary: borrowed owners remain open. */
int yvex_runtime_decode_context_close(yvex_runtime_decode_context **context,
                                      yvex_error *err)
{
    if (!context || !*context) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    if (pthread_mutex_lock(&(*context)->mutex) != 0)
        return decode_refuse(err, YVEX_ERR_STATE, "decode close lock failed");
    if ((*context)->busy) {
        (void)pthread_mutex_unlock(&(*context)->mutex);
        return decode_refuse(err, YVEX_ERR_STATE, "busy decode context cannot close");
    }
    (void)pthread_mutex_unlock(&(*context)->mutex);
    if ((*context)->mutex_ready) (void)pthread_mutex_destroy(&(*context)->mutex);
    memset(*context, 0, sizeof(**context));
    free(*context);
    *context = NULL;
    yvex_error_clear(err);
    return YVEX_OK;
}

/* Purpose: adapt one transformer owner to cleanup-lease release.
 * Inputs: opaque owned transformer handle. Effects: invokes canonical transformer close.
 * Failure: propagates retryable cleanup status. Boundary: cleanup lease adapter only. */
static int decode_transformer_cleanup(void **opaque, yvex_error *err)
{
    return yvex_runtime_transformer_context_close(
        (yvex_runtime_transformer_context **)opaque, err);
}

/* Purpose: project an admitted input slice through the existing token-input schema.
 * Inputs: source identities, bounded token span, start, and count. Effects: allocates one input.
 * Failure: invalid extent, token, or allocation refuses. Boundary: never rewrites external input. */
static int decode_input_slice(yvex_transformer_input **out,
                              const yvex_transformer_input_summary *source,
                              const unsigned int *tokens,
                              unsigned long long start,
                              unsigned long long count, yvex_error *err)
{
    yvex_transformer_input_summary summary = *source;
    summary.token_start = start;
    summary.token_count = count;
    summary.payload_bytes = 0ull;
    summary.payload_digest[0] = summary.input_identity[0] = '\0';
    if (yvex_transformer_input_seal(&summary, tokens, err) != YVEX_OK)
        return yvex_error_code(err);
    return yvex_transformer_input_open_memory(out, &summary, tokens, err);
}

/* Purpose: release operator-owned step directory without changing capability facts.
 * Inputs: operator result. Effects: frees its step allocation and clears ownership fields.
 * Failure: none. Boundary: no model/session/transformer cleanup. */
void yvex_runtime_decode_operator_result_release(
    yvex_decode_operator_result *result)
{
    if (!result) return;
    free(result->steps);
    result->steps = NULL;
    result->step_count = 0ull;
}

/* Purpose: publish one operator refusal while preserving partial decode facts.
 * Inputs: mutable operator result and primary error. Effects: writes status and reason only.
 * Failure: none. Boundary: cannot promote or erase completed step evidence. */
static void decode_operator_refuse(yvex_decode_operator_result *result,
                                   const yvex_error *err)
{
    yvex_core_text_copy(result->status, sizeof(result->status),
                        result->decode.partial ? "partial" : "refused");
    yvex_core_text_copy(result->reason, sizeof(result->reason),
                        err && yvex_error_is_set(err) ? yvex_error_message(err)
                                                     : "decode execution refused");
}

/* Purpose: execute the installed prefill-to-decode workflow over one shared warm context.
 * Inputs: exact artifact/binding/token stream, split, backend, and resource budgets.
 * Effects: opens once, prefills once, decodes remaining tokens stepwise, and publishes typed evidence.
 * Failure: retains cleanup ownership when necessary and preserves typed partial progress.
 * Boundary: one installed operator workflow; it never tokenizes, selects tokens, or samples. */
int yvex_runtime_decode_operator_execute(
    const yvex_decode_operator_request *request,
    yvex_decode_operator_result *result,
    yvex_runtime_cleanup_lease **retained_cleanup, yvex_error *err)
{
    yvex_runtime_model_open_request model_request = {0};
    yvex_runtime_session_open_request session_request = {0};
    yvex_runtime_transformer_options transformer_options = {0};
    yvex_runtime_decode_options decode_options = {0};
    yvex_runtime_model_failure failure = {0};
    yvex_runtime_cleanup_lease *cleanup = NULL;
    yvex_runtime_model *model = NULL;
    yvex_runtime_execution_session *session = NULL;
    yvex_runtime_transformer_context *transformer = NULL;
    yvex_runtime_decode_context *decode = NULL;
    yvex_transformer_input *input = NULL, *prefill_input = NULL, *decode_input = NULL;
    const yvex_transformer_input_summary *input_summary;
    const yvex_transformer_plan_summary *plan;
    const yvex_runtime_model_view *model_view;
    const unsigned int *tokens;
    yvex_transformer_input_limits limits;
    yvex_runtime_transformer_request prefill_request = {0};
    yvex_runtime_transformer_output prefill_output = {0};
    yvex_runtime_decode_request decode_request = {0};
    yvex_runtime_decode_output decode_output = {0};
    yvex_error primary = {0}, cleanup_error;
    float *prefill_hidden = NULL, *decode_hidden = NULL;
    unsigned long long prefill_count, decode_count, hidden_count;
    int adopted = 0, rc, close_rc;
    if (result) memset(result, 0, sizeof(*result));
    if (!request || !result || !retained_cleanup || *retained_cleanup ||
        !request->target || !request->artifact_path || !request->runtime_binding_path ||
        !request->input_path || !request->prefill_tokens ||
        !request->prefill_chunk_tokens || !request->context_capacity)
        return decode_refuse(err, YVEX_ERR_INVALID_ARG,
                             "complete decode operator arguments are required");
    yvex_core_text_copy(result->command, sizeof(result->command),
                        "graph transformer decode");
    yvex_core_text_copy(result->target, sizeof(result->target), request->target);
    yvex_core_text_copy(result->backend, sizeof(result->backend),
                        request->backend == YVEX_BACKEND_KIND_CUDA ? "cuda" : "cpu");
    yvex_core_text_copy(result->phase, sizeof(result->phase), "decode");
    model_request.artifact_path = request->artifact_path;
    model_request.runtime_binding_path = request->runtime_binding_path;
    model_request.target_id = request->target;
    model_request.maximum_host_bytes = request->maximum_host_bytes;
    session_request.backend = request->backend;
    session_request.maximum_host_bytes = request->maximum_host_bytes;
    session_request.maximum_device_bytes = request->maximum_device_bytes;
    rc = yvex_runtime_cleanup_lease_acquire(
        &cleanup, &model_request, &session_request, &model, &session, &failure, err);
    limits.maximum_file_bytes = request->maximum_host_bytes
                                    ? request->maximum_host_bytes : 1ull << 30u;
    if (rc == YVEX_OK)
        rc = yvex_transformer_input_open_file(&input, request->input_path,
                                              &limits, err);
    transformer_options.maximum_host_bytes = request->maximum_host_bytes;
    transformer_options.maximum_device_bytes = request->maximum_device_bytes;
    transformer_options.context_capacity = request->context_capacity;
    transformer_options.cancel_requested = request->cancel_requested;
    transformer_options.cancel_context = request->cancel_context;
    if (rc == YVEX_OK)
        rc = yvex_runtime_transformer_context_open(
            &transformer, model, session, &transformer_options, err);
    if (rc == YVEX_OK) {
        rc = yvex_runtime_cleanup_lease_adopt(
            cleanup, transformer, decode_transformer_cleanup, err);
        adopted = rc == YVEX_OK;
    }
    input_summary = yvex_transformer_input_summary_get(input);
    tokens = yvex_transformer_input_token_ids(input);
    plan = yvex_transformer_plan_summary_get(
        yvex_runtime_transformer_context_plan(transformer));
    model_view = yvex_runtime_model_view_get(model);
    if (rc == YVEX_OK &&
        (!input_summary || !tokens || !plan || !model_view || input_summary->token_start ||
         request->prefill_tokens >= input_summary->token_count ||
         request->context_capacity < input_summary->token_count))
        rc = decode_refuse(err, YVEX_ERR_BOUNDS,
                           "decode operator prefill split or context capacity is invalid");
    prefill_count = request->prefill_tokens;
    decode_count = input_summary ? input_summary->token_count - prefill_count : 0ull;
    if (rc == YVEX_OK)
        rc = decode_input_slice(&prefill_input, input_summary, tokens, 0ull,
                                prefill_count, err);
    if (rc == YVEX_OK)
        rc = decode_input_slice(&decode_input, input_summary, tokens + prefill_count,
                                prefill_count, decode_count, err);
    if (rc == YVEX_OK &&
        (!yvex_core_u64_mul(prefill_count, plan->hidden_width, &hidden_count) ||
         hidden_count > SIZE_MAX / sizeof(float)))
        rc = decode_refuse(err, YVEX_ERR_BOUNDS,
                           "decode operator prefill output overflowed");
    if (rc == YVEX_OK) {
        prefill_hidden = (float *)calloc((size_t)hidden_count, sizeof(float));
        if (!prefill_hidden)
            rc = decode_refuse(err, YVEX_ERR_NOMEM,
                               "decode operator prefill output allocation failed");
        prefill_output.normalized_hidden = prefill_hidden;
        prefill_output.capacity = hidden_count;
    }
    if (rc == YVEX_OK &&
        (!yvex_core_u64_mul(decode_count, plan->hidden_width, &hidden_count) ||
         hidden_count > SIZE_MAX / sizeof(float) || decode_count > SIZE_MAX))
        rc = decode_refuse(err, YVEX_ERR_BOUNDS,
                           "decode operator output extent overflowed");
    if (rc == YVEX_OK) {
        decode_hidden = (float *)calloc((size_t)hidden_count, sizeof(float));
        result->steps = (yvex_runtime_decode_step_result *)calloc(
            (size_t)decode_count, sizeof(*result->steps));
        if (!decode_hidden || !result->steps)
            rc = decode_refuse(err, YVEX_ERR_NOMEM,
                               "decode operator output allocation failed");
        result->step_count = decode_count;
        decode_output.normalized_hidden = decode_hidden;
        decode_output.normalized_hidden_capacity = hidden_count;
        decode_output.steps = result->steps;
        decode_output.step_capacity = decode_count;
    }
    decode_options.maximum_steps = decode_count;
    decode_options.cancel_requested = request->cancel_requested;
    decode_options.cancel_context = request->cancel_context;
    if (rc == YVEX_OK)
        rc = yvex_runtime_decode_context_open(
            &decode, transformer, session, &decode_options, err);
    prefill_request.chunk_tokens = request->prefill_chunk_tokens;
    prefill_request.backend = request->backend;
    prefill_request.phase = YVEX_TRANSFORMER_PHASE_PREFILL;
    if (rc == YVEX_OK)
        rc = yvex_runtime_transformer_execute(
            transformer, prefill_input, &prefill_request, &prefill_output,
            &result->prefill, err);
    decode_request.backend = request->backend;
    if (rc == YVEX_OK)
        rc = yvex_runtime_decode_execute(decode, decode_input, &decode_request,
                                         &decode_output, &result->decode, err);
    result->step_count = result->decode.completed_steps;
    if (plan && model_view) {
        yvex_core_text_copy(result->family, sizeof(result->family),
                            model_view->adapter->family_name);
        yvex_runtime_identity_copy(result->artifact_identity,
                                   model_view->binding->artifact_identity);
        yvex_runtime_identity_copy(result->runtime_binding_identity,
                                   model_view->binding->identity);
        yvex_runtime_identity_copy(result->transformer_plan_identity,
                                   plan->transformer_plan_identity);
        result->hidden_width = plan->hidden_width;
        result->layer_count = plan->layer_count;
        result->prefill_tokens_committed = result->prefill.committed_prefix;
        result->decode_step_ready = result->decode_repeat_ready = 1;
        result->decode_hidden_state_ready = result->decode_partial_progress_ready = 1;
        result->moe_decode_composed = result->model_decode_ready = 1;
    }
    primary = err ? *err : (yvex_error){0};
    close_rc = yvex_runtime_decode_context_close(&decode, &cleanup_error);
    if (rc == YVEX_OK && close_rc != YVEX_OK) { rc = close_rc; primary = cleanup_error; }
    yvex_transformer_input_close(&decode_input);
    yvex_transformer_input_close(&prefill_input);
    yvex_transformer_input_close(&input);
    free(prefill_hidden);
    free(decode_hidden);
    if (!adopted && transformer) {
        close_rc = yvex_runtime_transformer_context_close(&transformer, &cleanup_error);
        if (rc == YVEX_OK && close_rc != YVEX_OK) { rc = close_rc; primary = cleanup_error; }
    }
    close_rc = yvex_runtime_cleanup_lease_close(&cleanup, &cleanup_error);
    if (close_rc != YVEX_OK) { rc = close_rc; primary = cleanup_error; }
    if (cleanup) *retained_cleanup = cleanup;
    if (err && rc != YVEX_OK) *err = primary;
    if (rc == YVEX_OK) {
        result->completed = 1;
        yvex_core_text_copy(result->status, sizeof(result->status), "complete");
        yvex_error_clear(err);
    } else {
        decode_operator_refuse(result, err);
    }
    return rc;
}
