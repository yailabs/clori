/* Project one completed generation profile into scoped server evidence. */
#include "src/server/private.h"

#include <string.h>
#include <yvex/internal/core.h>
#include <yvex/internal/engine_scheduler.h>

void yvex_server_decode_measurement(
    unsigned long long first_fragment_ns,
    unsigned long long committed_tokens,
    const unsigned long long *decode_commit_ns,
    unsigned int decode_commit_count,
    unsigned long long now,
    yvex_execution_measurement *measurement)
{
    unsigned long long end = decode_commit_count
                                 ? decode_commit_ns[decode_commit_count - 1u]
                                 : now;
    unsigned long long duration = end > first_fragment_ns
                                      ? end - first_fragment_ns : 0ull;
    unsigned long long units = committed_tokens > 1ull
                                   ? committed_tokens - 1ull : 0ull;
    memset(measurement, 0, sizeof(*measurement));
    measurement->schema_version = YVEX_EXECUTION_MEASUREMENT_SCHEMA_V1;
    measurement->scope = YVEX_EXECUTION_SCOPE_SUBSEQUENT_DECODE;
    measurement->clock = YVEX_EXECUTION_CLOCK_HOST_WALL;
    measurement->composition = YVEX_EXECUTION_COMPOSITION_NESTED;
    measurement->work_unit = YVEX_EXECUTION_WORK_TOKENS;
    measurement->completed_units = units;
    if (duration) {
        measurement->available |=
            YVEX_EXECUTION_MEASUREMENT_DURATION_AVAILABLE;
        measurement->duration_ns = duration;
    }
    if (units && duration) {
        measurement->available |=
            YVEX_EXECUTION_MEASUREMENT_CUMULATIVE_RATE_AVAILABLE;
        measurement->cumulative_rate =
            (double)units * 1000000000.0 / (double)duration;
    }
    if (decode_commit_count > 1u) {
        unsigned long long first = decode_commit_ns[0];
        unsigned long long rolling_duration = end > first ? end - first : 0ull;
        if (rolling_duration) {
            measurement->available |=
                YVEX_EXECUTION_MEASUREMENT_ROLLING_RATE_AVAILABLE;
            measurement->rolling_units = decode_commit_count - 1u;
            measurement->rolling_duration_ns = rolling_duration;
            measurement->rolling_window_units =
                YVEX_SERVER_DECODE_RATE_WINDOW_TOKENS;
            measurement->rolling_rate =
                (double)measurement->rolling_units * 1000000000.0 /
                (double)rolling_duration;
        }
    }
}

static yvex_execution_measurement profile_measurement(
    yvex_execution_measurement_scope scope,
    yvex_execution_measurement_clock clock,
    yvex_execution_measurement_composition composition,
    yvex_execution_work_unit unit, unsigned long long units,
    unsigned long long duration_ns)
{
    yvex_execution_measurement value = {0};
    value.schema_version = YVEX_EXECUTION_MEASUREMENT_SCHEMA_V1;
    value.scope = scope;
    value.clock = clock;
    value.composition = composition;
    value.work_unit = unit;
    value.completed_units = units;
    if (units) {
        value.available |= YVEX_EXECUTION_MEASUREMENT_DENOMINATOR_AVAILABLE;
        value.total_units = units;
    }
    if (duration_ns) {
        value.available |= YVEX_EXECUTION_MEASUREMENT_DURATION_AVAILABLE;
        value.duration_ns = duration_ns;
    }
    if (duration_ns && units && unit != YVEX_EXECUTION_WORK_NONE) {
        value.available |= YVEX_EXECUTION_MEASUREMENT_CUMULATIVE_RATE_AVAILABLE;
        value.cumulative_rate =
            (double)units * 1000000000.0 / (double)duration_ns;
    }
    return value;
}

static int profile_emit(
    server_session_registry *registry, const server_session *session,
    const yvex_client_request *request, const char *request_id,
    const char *turn_id, const char *phase, unsigned long long value_a,
    unsigned long long value_b, unsigned long long value_c,
    unsigned long long duration_ns,
    const yvex_execution_measurement *measurement, yvex_error *err)
{
    return yvex_server_telemetry_emit_provider(
        registry->telemetry, &registry->event_scope,
        YVEX_SERVER_EVENT_GENERATION_PROFILE, YVEX_SERVER_SEVERITY_DEBUG,
        session->name, request_id, turn_id, phase, value_a, value_b, value_c,
        (double)duration_ns / 1000000000.0, 0.0, NULL,
        request->provider_request, measurement, NULL, err);
}

static int profile_emit_scoped(
    server_session_registry *registry, const server_session *session,
    const yvex_client_request *request, const char *request_id,
    const char *turn_id, const char *phase, unsigned long long value_a,
    unsigned long long value_b, unsigned long long value_c,
    unsigned long long duration_ns, yvex_execution_measurement_scope scope,
    yvex_execution_measurement_clock clock,
    yvex_execution_measurement_composition composition,
    yvex_execution_work_unit unit, unsigned long long units, yvex_error *err)
{
    yvex_execution_measurement measurement = profile_measurement(
        scope, clock, composition, unit, units, duration_ns);
    return profile_emit(registry, session, request, request_id, turn_id, phase,
                        value_a, value_b, value_c, duration_ns, &measurement,
                        err);
}

static int profile_summary_publish(
    server_session_registry *registry, const server_session *session,
    const yvex_client_request *request, const char *request_id,
    const char *turn_id, const yvex_runtime_generation_result *result,
    const yvex_runtime_profile_record *profile, yvex_error *err)
{
    const yvex_execution_measurement_composition decode_composition =
        result->execution_mode == YVEX_GENERATION_MODE_SPECULATIVE
            ? YVEX_EXECUTION_COMPOSITION_ENCLOSING
            : YVEX_EXECUTION_COMPOSITION_NESTED;
    int rc = profile_emit_scoped(
        registry, session, request, request_id, turn_id, "target",
        profile->counters[YVEX_RUNTIME_PROFILE_TARGET_FORWARDS],
        profile->counters[YVEX_RUNTIME_PROFILE_TARGET_ROWS],
        profile->counters[YVEX_RUNTIME_PROFILE_REPLAYED_ACCEPTED_TARGET_ROWS],
        profile->phase_ns[YVEX_RUNTIME_PROFILE_TOTAL_GENERATION],
        YVEX_EXECUTION_SCOPE_TOTAL_OPERATION, YVEX_EXECUTION_CLOCK_HOST_WALL,
        YVEX_EXECUTION_COMPOSITION_TOP_LEVEL, YVEX_EXECUTION_WORK_TOKENS,
        result->model_committed_token_count, err);
    if (rc == YVEX_OK)
        rc = profile_emit_scoped(
            registry, session, request, request_id, turn_id, "prefill",
            profile->counters[YVEX_RUNTIME_PROFILE_PROMPT_TOKENS],
            profile->counters[YVEX_RUNTIME_PROFILE_REUSED_TOKENS],
            profile->counters[YVEX_RUNTIME_PROFILE_NEW_PREFILL_TOKENS],
            profile->phase_ns[YVEX_RUNTIME_PROFILE_TOTAL_PREFILL],
            YVEX_EXECUTION_SCOPE_PREFILL, YVEX_EXECUTION_CLOCK_HOST_WALL,
            YVEX_EXECUTION_COMPOSITION_NESTED, YVEX_EXECUTION_WORK_TOKENS,
            profile->counters[YVEX_RUNTIME_PROFILE_NEW_PREFILL_TOKENS], err);
    if (rc == YVEX_OK)
        rc = profile_emit_scoped(
            registry, session, request, request_id, turn_id, "first-decode",
            profile->phase_calls[YVEX_RUNTIME_PROFILE_FIRST_DECODE], 0ull, 0ull,
            profile->phase_ns[YVEX_RUNTIME_PROFILE_FIRST_DECODE],
            YVEX_EXECUTION_SCOPE_FIRST_DECODE, YVEX_EXECUTION_CLOCK_HOST_WALL,
            decode_composition,
            YVEX_EXECUTION_WORK_OPERATIONS,
            profile->phase_calls[YVEX_RUNTIME_PROFILE_FIRST_DECODE], err);
    if (rc == YVEX_OK)
        rc = profile_emit_scoped(
            registry, session, request, request_id, turn_id,
            "subsequent-decode",
            profile->phase_calls[YVEX_RUNTIME_PROFILE_SUBSEQUENT_DECODE], 0ull,
            0ull, profile->phase_ns[YVEX_RUNTIME_PROFILE_SUBSEQUENT_DECODE],
            YVEX_EXECUTION_SCOPE_SUBSEQUENT_DECODE,
            YVEX_EXECUTION_CLOCK_HOST_WALL,
            decode_composition,
            YVEX_EXECUTION_WORK_OPERATIONS,
            profile->phase_calls[YVEX_RUNTIME_PROFILE_SUBSEQUENT_DECODE], err);
    return rc;
}

int yvex_server_profile_reconcile(
    const yvex_runtime_profile_record *profile,
    yvex_runtime_generation_mode mode,
    unsigned long long *attributed_ns,
    unsigned long long *unattributed_ns)
{
    static const yvex_runtime_profile_phase enclosing[] = {
        YVEX_RUNTIME_PROFILE_TOKENIZER,
        YVEX_RUNTIME_PROFILE_PROMPT_RENDERING,
        YVEX_RUNTIME_PROFILE_TOTAL_PREFILL,
        YVEX_RUNTIME_PROFILE_FIRST_DECODE,
        YVEX_RUNTIME_PROFILE_SUBSEQUENT_DECODE,
    };
    static const yvex_runtime_profile_phase ordinary_children[] = {
        YVEX_RUNTIME_PROFILE_OUTPUT_HEAD,
        YVEX_RUNTIME_PROFILE_LOGITS_PUBLICATION,
        YVEX_RUNTIME_PROFILE_SAMPLING,
        YVEX_RUNTIME_PROFILE_STATE_VALIDATION,
        YVEX_RUNTIME_PROFILE_KV_COMMIT,
        YVEX_RUNTIME_PROFILE_DETOKENIZATION,
        YVEX_RUNTIME_PROFILE_PROVIDER_PUBLICATION,
    };
    unsigned long long total = 0ull;
    size_t index;
    if (attributed_ns) *attributed_ns = 0ull;
    if (unattributed_ns) *unattributed_ns = 0ull;
    if (!profile || !attributed_ns || !unattributed_ns ||
        mode > YVEX_GENERATION_MODE_SPECULATIVE)
        return 0;
    for (index = 0u; index < sizeof(enclosing) / sizeof(enclosing[0]); ++index)
        if (!yvex_core_u64_add(total, profile->phase_ns[enclosing[index]],
                               &total))
            return 0;
    /* Ordinary decode measures only the model step. Speculative decode is an
     * enclosing iteration, so its child output/state/publication phases must
     * remain visible but must not be summed into the same wall a second time. */
    if (mode == YVEX_GENERATION_MODE_TARGET_ONLY)
        for (index = 0u;
             index < sizeof(ordinary_children) / sizeof(ordinary_children[0]);
             ++index)
            if (!yvex_core_u64_add(
                    total, profile->phase_ns[ordinary_children[index]],
                    &total))
                return 0;
    if (total > profile->phase_ns[YVEX_RUNTIME_PROFILE_TOTAL_GENERATION])
        return 0;
    *attributed_ns = total;
    *unattributed_ns =
        profile->phase_ns[YVEX_RUNTIME_PROFILE_TOTAL_GENERATION] - total;
    return 1;
}

static int profile_stage_publish(
    server_session_registry *registry, const server_session *session,
    const yvex_client_request *request, const char *request_id,
    const char *turn_id, const yvex_runtime_generation_result *result,
    const yvex_runtime_profile_record *profile,
    yvex_error *err)
{
#define PHASE(name_, a_, b_, c_, phase_, scope_, unit_, units_)                 \
    do {                                                                         \
        if (rc == YVEX_OK &&                                                     \
            ((a_) || (b_) || (c_) || (units_) ||                                \
             profile->phase_ns[(phase_)]))                                      \
            rc = profile_emit_scoped(                                            \
                registry, session, request, request_id, turn_id, (name_),       \
                (a_), (b_), (c_), profile->phase_ns[(phase_)], (scope_),         \
                YVEX_EXECUTION_CLOCK_HOST_WALL,                                  \
                ((scope_) == YVEX_EXECUTION_SCOPE_FIRST_DECODE ||               \
                 (scope_) == YVEX_EXECUTION_SCOPE_SUBSEQUENT_DECODE) &&         \
                        result->execution_mode ==                                \
                            YVEX_GENERATION_MODE_SPECULATIVE                    \
                    ? YVEX_EXECUTION_COMPOSITION_ENCLOSING                     \
                    : (scope_) == YVEX_EXECUTION_SCOPE_ATTENTION ||             \
                        (scope_) == YVEX_EXECUTION_SCOPE_MODEL_COMPONENT         \
                    ? YVEX_EXECUTION_COMPOSITION_OVERLAPPING                     \
                    : YVEX_EXECUTION_COMPOSITION_NESTED,                         \
                (unit_), (units_), err);                                         \
    } while (0)
    unsigned long long attributed_ns = 0ull, remainder_ns = 0ull;
    int reconciled, rc = YVEX_OK;
    PHASE("tokenizer", profile->phase_calls[YVEX_RUNTIME_PROFILE_TOKENIZER],
          profile->counters[YVEX_RUNTIME_PROFILE_PROMPT_TOKENS], 0ull,
          YVEX_RUNTIME_PROFILE_TOKENIZER, YVEX_EXECUTION_SCOPE_TOKENIZER,
          YVEX_EXECUTION_WORK_TOKENS,
          profile->phase_calls[YVEX_RUNTIME_PROFILE_TOKENIZER]
              ? profile->counters[YVEX_RUNTIME_PROFILE_PROMPT_TOKENS] : 0ull);
    PHASE("prompt-rendering",
          profile->phase_calls[YVEX_RUNTIME_PROFILE_PROMPT_RENDERING],
          profile->counters[YVEX_RUNTIME_PROFILE_PROMPT_TOKENS], 0ull,
          YVEX_RUNTIME_PROFILE_PROMPT_RENDERING,
          YVEX_EXECUTION_SCOPE_PROMPT_RENDERING,
          YVEX_EXECUTION_WORK_TOKENS,
          profile->phase_calls[YVEX_RUNTIME_PROFILE_PROMPT_RENDERING]
              ? profile->counters[YVEX_RUNTIME_PROFILE_PROMPT_TOKENS] : 0ull);
    PHASE("attention", profile->phase_calls[YVEX_RUNTIME_PROFILE_ATTENTION],
          profile->counters[YVEX_RUNTIME_PROFILE_CACHE_HITS],
          profile->counters[YVEX_RUNTIME_PROFILE_CACHE_MISSES],
          YVEX_RUNTIME_PROFILE_ATTENTION, YVEX_EXECUTION_SCOPE_ATTENTION,
          YVEX_EXECUTION_WORK_OPERATIONS,
          profile->phase_calls[YVEX_RUNTIME_PROFILE_ATTENTION]);
    PHASE("model-component",
          profile->counters[YVEX_RUNTIME_PROFILE_ROW_EXPERT_PAIRS],
          profile->counters[YVEX_RUNTIME_PROFILE_EXPERT_SUBVIEWS],
          profile->counters[YVEX_RUNTIME_PROFILE_EXPERT_BYTES],
          YVEX_RUNTIME_PROFILE_MOE_TOTAL,
          YVEX_EXECUTION_SCOPE_MODEL_COMPONENT,
          YVEX_EXECUTION_WORK_OPERATIONS,
          profile->phase_calls[YVEX_RUNTIME_PROFILE_MOE_TOTAL]);
    PHASE("output-head",
          profile->counters[YVEX_RUNTIME_PROFILE_OUTPUT_HEAD_ROWS],
          profile->counters[YVEX_RUNTIME_PROFILE_LOGITS_D2H_BYTES], 0ull,
          YVEX_RUNTIME_PROFILE_OUTPUT_HEAD, YVEX_EXECUTION_SCOPE_OUTPUT,
          YVEX_EXECUTION_WORK_OPERATIONS,
          profile->phase_calls[YVEX_RUNTIME_PROFILE_OUTPUT_HEAD]);
    PHASE("logits-publication",
          profile->phase_calls[YVEX_RUNTIME_PROFILE_LOGITS_PUBLICATION], 0ull,
          0ull, YVEX_RUNTIME_PROFILE_LOGITS_PUBLICATION,
          YVEX_EXECUTION_SCOPE_LOGITS_PUBLICATION,
          YVEX_EXECUTION_WORK_OPERATIONS,
          profile->phase_calls[YVEX_RUNTIME_PROFILE_LOGITS_PUBLICATION]);
    PHASE("sampling",
          profile->phase_calls[YVEX_RUNTIME_PROFILE_SAMPLING], 0ull, 0ull,
          YVEX_RUNTIME_PROFILE_SAMPLING, YVEX_EXECUTION_SCOPE_SAMPLING,
          YVEX_EXECUTION_WORK_OPERATIONS,
          profile->phase_calls[YVEX_RUNTIME_PROFILE_SAMPLING]);
    PHASE("state-validation",
          profile->phase_calls[YVEX_RUNTIME_PROFILE_STATE_VALIDATION], 0ull, 0ull,
          YVEX_RUNTIME_PROFILE_STATE_VALIDATION,
          YVEX_EXECUTION_SCOPE_STATE_COMMIT, YVEX_EXECUTION_WORK_OPERATIONS,
          profile->phase_calls[YVEX_RUNTIME_PROFILE_STATE_VALIDATION]);
    PHASE("state-commit", profile->phase_calls[YVEX_RUNTIME_PROFILE_KV_COMMIT],
          0ull, 0ull, YVEX_RUNTIME_PROFILE_KV_COMMIT,
          YVEX_EXECUTION_SCOPE_STATE_COMMIT, YVEX_EXECUTION_WORK_OPERATIONS,
          profile->phase_calls[YVEX_RUNTIME_PROFILE_KV_COMMIT]);
    PHASE("synchronization",
          profile->counters[YVEX_RUNTIME_PROFILE_QUEUE_SYNCHRONIZATIONS],
          profile->counters[YVEX_RUNTIME_PROFILE_EVENT_SYNCHRONIZATIONS],
          profile->counters[YVEX_RUNTIME_PROFILE_DEVICE_SYNCHRONIZATIONS],
          YVEX_RUNTIME_PROFILE_SYNCHRONIZATION_WAIT,
          YVEX_EXECUTION_SCOPE_SYNCHRONIZATION,
          YVEX_EXECUTION_WORK_OPERATIONS,
          profile->phase_calls[YVEX_RUNTIME_PROFILE_SYNCHRONIZATION_WAIT]);
    PHASE("detokenization",
          profile->phase_calls[YVEX_RUNTIME_PROFILE_DETOKENIZATION], 0ull, 0ull,
          YVEX_RUNTIME_PROFILE_DETOKENIZATION,
          YVEX_EXECUTION_SCOPE_DETOKENIZATION,
          YVEX_EXECUTION_WORK_OPERATIONS,
          profile->phase_calls[YVEX_RUNTIME_PROFILE_DETOKENIZATION]);
    PHASE("publication",
          profile->phase_calls[YVEX_RUNTIME_PROFILE_PROVIDER_PUBLICATION], 0ull,
          0ull, YVEX_RUNTIME_PROFILE_PROVIDER_PUBLICATION,
          YVEX_EXECUTION_SCOPE_CLIENT_PUBLICATION,
          YVEX_EXECUTION_WORK_OPERATIONS,
          profile->phase_calls[YVEX_RUNTIME_PROFILE_PROVIDER_PUBLICATION]);
#undef PHASE
    reconciled = yvex_server_profile_reconcile(
        profile, result->execution_mode, &attributed_ns, &remainder_ns);
    if (rc == YVEX_OK)
        rc = profile_emit_scoped(
            registry, session, request, request_id, turn_id, "unattributed",
            attributed_ns,
            profile->phase_ns[YVEX_RUNTIME_PROFILE_TOTAL_GENERATION],
            reconciled ? remainder_ns : 0ull,
            reconciled ? remainder_ns : 0ull,
            YVEX_EXECUTION_SCOPE_UNATTRIBUTED,
            YVEX_EXECUTION_CLOCK_HOST_WALL,
            YVEX_EXECUTION_COMPOSITION_NESTED,
            YVEX_EXECUTION_WORK_OPERATIONS, 0ull, err);
    return rc;
}

static int profile_detail_publish(
    server_session_registry *registry, const server_session *session,
    const yvex_client_request *request, const char *request_id,
    const char *turn_id, const yvex_runtime_generation_result *result,
    const yvex_runtime_generation_evidence *evidence, yvex_error *err)
{
    const yvex_runtime_profile_record *profile = &evidence->profile;
    yvex_engine_scheduler_summary batches = {0};
    int rc;
#define RAW(name_, a_, b_, c_, ns_)                                             \
    profile_emit(registry, session, request, request_id, turn_id, (name_),       \
                 (a_), (b_), (c_), (ns_), NULL, err)
    rc = RAW("movement", profile->counters[YVEX_RUNTIME_PROFILE_H2D_BYTES],
             profile->counters[YVEX_RUNTIME_PROFILE_D2H_BYTES],
             profile->counters[YVEX_RUNTIME_PROFILE_D2D_BYTES], 0ull);
    if (rc == YVEX_OK)
        rc = RAW("transfers", profile->counters[YVEX_RUNTIME_PROFILE_UPLOADS],
                 profile->counters[YVEX_RUNTIME_PROFILE_DOWNLOADS],
                 profile->counters[YVEX_RUNTIME_PROFILE_EXPERT_SUBVIEWS], 0ull);
    if (rc == YVEX_OK)
        rc = RAW("launches",
                 profile->counters[YVEX_RUNTIME_PROFILE_KERNEL_LAUNCHES],
                 profile->counters[YVEX_RUNTIME_PROFILE_QUEUE_SYNCHRONIZATIONS],
                 profile->counters[YVEX_RUNTIME_PROFILE_DEVICE_SYNCHRONIZATIONS],
                 profile->phase_ns[YVEX_RUNTIME_PROFILE_SYNCHRONIZATION_WAIT]);
    if (rc == YVEX_OK)
        rc = RAW("graphs", profile->counters[YVEX_RUNTIME_PROFILE_GRAPH_LAUNCHES],
                 profile->counters[YVEX_RUNTIME_PROFILE_GRAPH_CAPTURES],
                 profile->counters[YVEX_RUNTIME_PROFILE_GRAPH_REPLAYS], 0ull);
    if (rc == YVEX_OK)
        rc = RAW("speculation",
                 profile->counters[YVEX_RUNTIME_PROFILE_DRAFT_FORWARDS],
                 profile->counters[YVEX_RUNTIME_PROFILE_VERIFIED_ROWS],
                 profile->counters[YVEX_RUNTIME_PROFILE_PROMOTED_TARGET_ROWS], 0ull);
    if (rc == YVEX_OK && result->speculation_source_boundary_token_count)
        rc = RAW("source-boundary",
                 result->speculation_source_boundary_token_count,
                 result->model_committed_token_count -
                     result->speculation_source_boundary_token_count,
                 profile->counters[
                     YVEX_RUNTIME_PROFILE_REPLAYED_ACCEPTED_TARGET_ROWS],
                 0ull);
    if (rc == YVEX_OK)
        rc = RAW("candidate",
                 profile->counters[YVEX_RUNTIME_PROFILE_ACCEPTED_DRAFT_TOKENS],
                 profile->counters[
                     YVEX_RUNTIME_PROFILE_DISCARDED_CANDIDATE_ROWS],
                 profile->counters[YVEX_RUNTIME_PROFILE_TARGET_EXTENSIONS], 0ull);
    if (rc == YVEX_OK && evidence->expert_worklists.worklist_count)
        rc = RAW("expert-worklist", evidence->expert_worklists.worklist_count,
                 evidence->expert_worklists.pair_count,
                 evidence->expert_worklists.bucket_count, 0ull);
    if (rc == YVEX_OK && evidence->expert_worklists.worklist_count)
        rc = RAW("expert-width",
                 evidence->expert_worklists.maximum_bucket_population,
                 evidence->expert_worklists.matrix_tile_eligible_pairs,
                 evidence->expert_worklists.matrix_tile_executed_pairs, 0ull);
    if (rc == YVEX_OK && yvex_model_engine_scheduler_summary_copy(
                                 registry->model, &batches, err) == YVEX_OK &&
        batches.enabled)
        rc = RAW("execution-batches", batches.physical_batches,
                 batches.multi_source_batches, batches.maximum_width, 0ull);
    if (rc == YVEX_OK && batches.enabled)
        rc = RAW("execution-scheduling", batches.progress_submissions,
                 batches.cooperative_yields, batches.cooperative_resumes,
                 batches.scheduler_wait_nanoseconds);
    if (rc == YVEX_OK && batches.enabled)
        rc = RAW("execution-quanta", batches.progress_completions,
                 batches.maximum_ready_sequence_work,
                 batches.maximum_active_sequences,
                 batches.maximum_quantum_nanoseconds);
    if (rc == YVEX_OK && batches.enabled)
        rc = RAW("execution-batch-coalescing", batches.coalescing_waits,
                 batches.coalescing_timeouts, batches.registered_producers,
                 batches.coalescing_ns);
    if (rc == YVEX_OK && batches.enabled)
        rc = RAW("execution-step-rendezvous", batches.rendezvous_submissions,
                 batches.multi_source_rendezvous,
                 batches.maximum_rendezvous_width, 0ull);
#undef RAW
    return rc;
}

int yvex_server_session_profile_publish(
    server_session_registry *registry, const server_session *session,
    const yvex_client_request *request, const char *request_id,
    const char *turn_id, const yvex_runtime_generation_result *result,
    const yvex_runtime_generation_evidence *evidence, yvex_error *err)
{
    const yvex_runtime_profile_record *profile =
        evidence ? &evidence->profile : NULL;
    int rc;
    if (!profile || profile->mode == YVEX_RUNTIME_PROFILE_OFF) return YVEX_OK;
    rc = profile_summary_publish(registry, session, request, request_id,
                                 turn_id, result, profile, err);
    if (rc == YVEX_OK && profile->mode >= YVEX_RUNTIME_PROFILE_STAGES)
        rc = profile_stage_publish(registry, session, request, request_id,
                                   turn_id, result, profile, err);
    if (rc == YVEX_OK && profile->mode == YVEX_RUNTIME_PROFILE_DETAILED)
        rc = profile_detail_publish(registry, session, request, request_id,
                                    turn_id, result, evidence, err);
    return rc;
}
