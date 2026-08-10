/*
 * Compose admitted lower owners into one bounded autoregressive lifecycle.
 *
 * Published tokens are always target-authored: ordinary generation commits one decode step at a
 * time, while speculative generation may commit a target-verified prefix atomically. This is the
 * internal runtime/operator ABI from exact text/messages to model-backed incremental text.
 */
#ifndef INCLUDE_YVEX_INTERNAL_GENERATION_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_GENERATION_H_INCLUDED
#include <limits.h>
#include <string.h>
#include <yvex/internal/core.h>
#include <yvex/internal/sampling.h>
#include <yvex/tokenizer.h>
#ifdef __cplusplus
extern "C" {
#endif
#define YVEX_RUNTIME_PROFILE_SCHEMA_V2 2u
typedef enum {
    YVEX_RUNTIME_PROFILE_OFF = 0, YVEX_RUNTIME_PROFILE_SUMMARY, YVEX_RUNTIME_PROFILE_STAGES,
    YVEX_RUNTIME_PROFILE_DETAILED
} yvex_runtime_profile_mode;
typedef enum {
    YVEX_RUNTIME_PROFILE_STARTUP = 0, YVEX_RUNTIME_PROFILE_PREFILL, YVEX_RUNTIME_PROFILE_DECODE,
    YVEX_RUNTIME_PROFILE_GENERATION
} yvex_runtime_profile_scope;
typedef enum {
    YVEX_RUNTIME_PROFILE_QUEUE = 0, YVEX_RUNTIME_PROFILE_TOKENIZER,
    YVEX_RUNTIME_PROFILE_PROMPT_RENDERING, YVEX_RUNTIME_PROFILE_EMBEDDING,
    YVEX_RUNTIME_PROFILE_ATTENTION, YVEX_RUNTIME_PROFILE_MOE_INGRESS,
    YVEX_RUNTIME_PROFILE_ROUTER_PROJECTION, YVEX_RUNTIME_PROFILE_ROUTER_TOPK,
    YVEX_RUNTIME_PROFILE_SELECTED_EXPERT_PREPARATION, YVEX_RUNTIME_PROFILE_ROUTED_EXPERTS,
    YVEX_RUNTIME_PROFILE_SHARED_EXPERTS, YVEX_RUNTIME_PROFILE_MOE_POST, YVEX_RUNTIME_PROFILE_MOE_TOTAL,
    YVEX_RUNTIME_PROFILE_FINAL_NORMALIZATION, YVEX_RUNTIME_PROFILE_OUTPUT_HEAD,
    YVEX_RUNTIME_PROFILE_LOGITS_PUBLICATION, YVEX_RUNTIME_PROFILE_SAMPLING,
    YVEX_RUNTIME_PROFILE_STATE_VALIDATION, YVEX_RUNTIME_PROFILE_SYNCHRONIZATION_WAIT,
    YVEX_RUNTIME_PROFILE_KV_COMMIT, YVEX_RUNTIME_PROFILE_DETOKENIZATION, YVEX_RUNTIME_PROFILE_PROVIDER_PUBLICATION,
    YVEX_RUNTIME_PROFILE_TOTAL_PREFILL, YVEX_RUNTIME_PROFILE_FIRST_DECODE,
    YVEX_RUNTIME_PROFILE_SUBSEQUENT_DECODE, YVEX_RUNTIME_PROFILE_TOTAL_GENERATION, YVEX_RUNTIME_PROFILE_PHASE_COUNT
} yvex_runtime_profile_phase;
typedef enum {
    YVEX_RUNTIME_PROFILE_HOST_PAYLOAD_READS = 0, YVEX_RUNTIME_PROFILE_MAPPED_BYTES_TOUCHED,
    YVEX_RUNTIME_PROFILE_H2D_BYTES, YVEX_RUNTIME_PROFILE_D2H_BYTES,
    YVEX_RUNTIME_PROFILE_D2D_BYTES, YVEX_RUNTIME_PROFILE_MANAGED_PREFETCH_BYTES,
    YVEX_RUNTIME_PROFILE_UPLOADS, YVEX_RUNTIME_PROFILE_DOWNLOADS, YVEX_RUNTIME_PROFILE_CACHE_HITS,
    YVEX_RUNTIME_PROFILE_CACHE_MISSES, YVEX_RUNTIME_PROFILE_CACHE_EVICTIONS,
    YVEX_RUNTIME_PROFILE_EXPERT_SUBVIEWS, YVEX_RUNTIME_PROFILE_KERNEL_LAUNCHES,
    YVEX_RUNTIME_PROFILE_GRAPH_LAUNCHES, YVEX_RUNTIME_PROFILE_GRAPH_CAPTURES,
    YVEX_RUNTIME_PROFILE_GRAPH_REPLAYS, YVEX_RUNTIME_PROFILE_STREAM_SYNCHRONIZATIONS,
    YVEX_RUNTIME_PROFILE_EVENT_SYNCHRONIZATIONS, YVEX_RUNTIME_PROFILE_DEVICE_SYNCHRONIZATIONS,
    YVEX_RUNTIME_PROFILE_DEVICE_ALLOCATIONS, YVEX_RUNTIME_PROFILE_HOST_ALLOCATIONS,
    YVEX_RUNTIME_PROFILE_WORKSPACE_RESETS, YVEX_RUNTIME_PROFILE_PROMPT_TOKENS,
    YVEX_RUNTIME_PROFILE_REUSED_TOKENS, YVEX_RUNTIME_PROFILE_NEW_PREFILL_TOKENS,
    YVEX_RUNTIME_PROFILE_GENERATED_TOKENS,
    YVEX_RUNTIME_PROFILE_TARGET_FORWARDS, YVEX_RUNTIME_PROFILE_TARGET_ROWS,
    YVEX_RUNTIME_PROFILE_DRAFT_FORWARDS, YVEX_RUNTIME_PROFILE_DRAFT_ROWS,
    YVEX_RUNTIME_PROFILE_TARGET_VERIFICATIONS, YVEX_RUNTIME_PROFILE_VERIFIED_ROWS,
    YVEX_RUNTIME_PROFILE_ACCEPTED_DRAFT_TOKENS, YVEX_RUNTIME_PROFILE_PROMOTED_TARGET_ROWS,
    YVEX_RUNTIME_PROFILE_DISCARDED_CANDIDATE_ROWS, YVEX_RUNTIME_PROFILE_TARGET_EXTENSIONS,
    YVEX_RUNTIME_PROFILE_REPLAYED_ACCEPTED_TARGET_ROWS, YVEX_RUNTIME_PROFILE_OUTPUT_HEAD_ROWS,
    YVEX_RUNTIME_PROFILE_LOGITS_H2D_BYTES, YVEX_RUNTIME_PROFILE_LOGITS_D2H_BYTES,
    YVEX_RUNTIME_PROFILE_LOGITS_D2D_BYTES, YVEX_RUNTIME_PROFILE_FULL_ARRAY_HOST_SCAN_BYTES,
    YVEX_RUNTIME_PROFILE_ROW_EXPERT_PAIRS, YVEX_RUNTIME_PROFILE_UNIQUE_EXPERTS,
    YVEX_RUNTIME_PROFILE_EXPERT_BYTES, YVEX_RUNTIME_PROFILE_SHAPE_REGISTRY_HITS,
    YVEX_RUNTIME_PROFILE_SHAPE_REGISTRY_MISSES, YVEX_RUNTIME_PROFILE_COUNTER_COUNT
} yvex_runtime_profile_counter;
typedef struct {
    unsigned int schema_version, backend;
    yvex_runtime_profile_mode mode;
    yvex_runtime_profile_scope scope;
    int sealed;
    unsigned long long started_ns, completed_ns, counters[YVEX_RUNTIME_PROFILE_COUNTER_COUNT];
    unsigned long long phase_ns[YVEX_RUNTIME_PROFILE_PHASE_COUNT], phase_calls[YVEX_RUNTIME_PROFILE_PHASE_COUNT];
    char artifact_identity[YVEX_SHA256_HEX_BYTES], physical_variant_identity[YVEX_SHA256_HEX_BYTES];
    char runtime_binding_identity[YVEX_SHA256_HEX_BYTES], runtime_model_identity[YVEX_SHA256_HEX_BYTES];
    char execution_plan_identity[YVEX_SHA256_HEX_BYTES], workload_identity[YVEX_SHA256_HEX_BYTES];
    char profile_identity[YVEX_SHA256_HEX_BYTES];
} yvex_runtime_profile_record;
static inline int runtime_profile_refuse(yvex_error *err, yvex_status status, const char *reason)
{
    yvex_error_set(err, status, "runtime.profile", reason);
    return status;
}
static inline int runtime_profile_identity_copy(char output[YVEX_SHA256_HEX_BYTES], const char *input)
{
    if (!output || !yvex_sha256_hex_valid(input)) return 0;
    yvex_core_text_copy(output, YVEX_SHA256_HEX_BYTES, input);
    return 1;
}
/* Derive the canonical field-wise identity of one complete profile. */
static inline int runtime_profile_identity(const yvex_runtime_profile_record *record,
                                           char output[YVEX_SHA256_HEX_BYTES])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;
    yvex_sha256_init(&hash);
    if (!record || !output ||
        !yvex_sha256_update_text(&hash, "yvex.runtime.profile.v2") ||
        !yvex_sha256_update_u64(&hash, record->schema_version) ||
        !yvex_sha256_update_u64(&hash, record->mode) ||
        !yvex_sha256_update_u64(&hash, record->scope) ||
        !yvex_sha256_update_u64(&hash, record->backend) ||
        !yvex_sha256_update_u64(&hash, record->started_ns) ||
        !yvex_sha256_update_u64(&hash, record->completed_ns) ||
        !yvex_sha256_update_text(&hash, record->artifact_identity) ||
        !yvex_sha256_update_text(&hash, record->physical_variant_identity) ||
        !yvex_sha256_update_text(&hash, record->runtime_binding_identity) ||
        !yvex_sha256_update_text(&hash, record->runtime_model_identity) ||
        !yvex_sha256_update_text(&hash, record->execution_plan_identity) ||
        !yvex_sha256_update_text(&hash, record->workload_identity)) return 0;
    for (index = 0ull; index < YVEX_RUNTIME_PROFILE_PHASE_COUNT; ++index)
        if (!yvex_sha256_update_u64(&hash, record->phase_ns[index]) ||
            !yvex_sha256_update_u64(&hash, record->phase_calls[index])) return 0;
    for (index = 0ull; index < YVEX_RUNTIME_PROFILE_COUNTER_COUNT; ++index)
        if (!yvex_sha256_update_u64(&hash, record->counters[index])) return 0;
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}
static inline int runtime_profile_begin(yvex_runtime_profile_record *record,
                                        yvex_runtime_profile_mode mode,
                                        yvex_runtime_profile_scope scope, unsigned int backend,
                                        const char *artifact_identity, const char *physical_variant_identity,
                                        const char *runtime_binding_identity, const char *runtime_model_identity,
                                        const char *execution_plan_identity, const char *workload_identity,
                                        yvex_error *err)
{
    if (!record)
        return runtime_profile_refuse(err, YVEX_ERR_INVALID_ARG,
                                      "profile record is required");
    memset(record, 0, sizeof(*record));
    if (mode > YVEX_RUNTIME_PROFILE_DETAILED || scope > YVEX_RUNTIME_PROFILE_GENERATION ||
        !runtime_profile_identity_copy(record->artifact_identity, artifact_identity) ||
        !runtime_profile_identity_copy(record->physical_variant_identity,
                                       physical_variant_identity) ||
        !runtime_profile_identity_copy(record->runtime_binding_identity,
                                       runtime_binding_identity) ||
        !runtime_profile_identity_copy(record->runtime_model_identity, runtime_model_identity) ||
        !runtime_profile_identity_copy(record->execution_plan_identity,
                                       execution_plan_identity) ||
        !runtime_profile_identity_copy(record->workload_identity, workload_identity))
        return runtime_profile_refuse(err, YVEX_ERR_INVALID_ARG,
                                      "profile mode, scope, and exact identities are required");
    record->schema_version = YVEX_RUNTIME_PROFILE_SCHEMA_V2;
    record->mode = mode;
    record->scope = scope;
    record->backend = backend;
    record->started_ns = yvex_core_monotonic_ns();
    if (!record->started_ns) {
        memset(record, 0, sizeof(*record));
        return runtime_profile_refuse(err, YVEX_ERR_STATE,
                                      "monotonic profile clock is unavailable");
    }
    yvex_error_clear(err);
    return YVEX_OK;
}
static inline int runtime_profile_counter_add(yvex_runtime_profile_record *record,
    yvex_runtime_profile_counter counter, unsigned long long value, yvex_error *err)
{
    if (!record || record->schema_version != YVEX_RUNTIME_PROFILE_SCHEMA_V2 ||
        record->sealed || counter >= YVEX_RUNTIME_PROFILE_COUNTER_COUNT)
        return runtime_profile_refuse(err, YVEX_ERR_STATE,
                                      "profile counter mutation is invalid");
    if (ULLONG_MAX - record->counters[counter] < value)
        return runtime_profile_refuse(err, YVEX_ERR_BOUNDS, "profile counter overflowed");
    record->counters[counter] += value;
    yvex_error_clear(err);
    return YVEX_OK;
}
static inline int runtime_profile_phase_add(yvex_runtime_profile_record *record,
    yvex_runtime_profile_phase phase, unsigned long long elapsed_ns, yvex_error *err)
{
    if (!record || record->schema_version != YVEX_RUNTIME_PROFILE_SCHEMA_V2 ||
        record->sealed || phase >= YVEX_RUNTIME_PROFILE_PHASE_COUNT || !elapsed_ns)
        return runtime_profile_refuse(err, YVEX_ERR_STATE,
                                      "profile phase mutation is invalid");
    if (ULLONG_MAX - record->phase_ns[phase] < elapsed_ns ||
        record->phase_calls[phase] == ULLONG_MAX)
        return runtime_profile_refuse(err, YVEX_ERR_BOUNDS,
                                      "profile phase accounting overflowed");
    record->phase_ns[phase] += elapsed_ns;
    record->phase_calls[phase]++;
    yvex_error_clear(err);
    return YVEX_OK;
}
static inline int runtime_profile_finish(yvex_runtime_profile_record *record, yvex_error *err)
{
    if (!record || record->schema_version != YVEX_RUNTIME_PROFILE_SCHEMA_V2 || record->sealed)
        return runtime_profile_refuse(err, YVEX_ERR_STATE, "profile finish is invalid");
    record->completed_ns = yvex_core_monotonic_ns();
    if (record->completed_ns <= record->started_ns ||
        !runtime_profile_identity(record, record->profile_identity)) {
        record->completed_ns = 0ull;
        return runtime_profile_refuse(err, YVEX_ERR_STATE,
                                      "profile identity could not be sealed");
    }
    record->sealed = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}
/* Validate schema, monotonic bounds, identities, and canonical profile identity. */
static inline int runtime_profile_validate(const yvex_runtime_profile_record *record,
                                           yvex_error *err)
{
    char identity[YVEX_SHA256_HEX_BYTES];
    if (!record || record->schema_version != YVEX_RUNTIME_PROFILE_SCHEMA_V2 ||
        !record->sealed || record->mode > YVEX_RUNTIME_PROFILE_DETAILED ||
        record->scope > YVEX_RUNTIME_PROFILE_GENERATION ||
        record->completed_ns <= record->started_ns ||
        !yvex_sha256_hex_valid(record->artifact_identity) ||
        !yvex_sha256_hex_valid(record->physical_variant_identity) ||
        !yvex_sha256_hex_valid(record->runtime_binding_identity) ||
        !yvex_sha256_hex_valid(record->runtime_model_identity) ||
        !yvex_sha256_hex_valid(record->execution_plan_identity) ||
        !yvex_sha256_hex_valid(record->workload_identity) ||
        !runtime_profile_identity(record, identity) ||
        strcmp(identity, record->profile_identity) != 0)
        return runtime_profile_refuse(err, YVEX_ERR_FORMAT,
                                      "profile record is stale or malformed");
    yvex_error_clear(err);
    return YVEX_OK;
}
static inline const char *runtime_profile_mode_name(yvex_runtime_profile_mode mode)
{
    static const char *const names[] = {"off", "summary", "stages", "detailed"};
    return mode <= YVEX_RUNTIME_PROFILE_DETAILED ? names[mode] : "invalid";
}
static inline const char *runtime_profile_scope_name(yvex_runtime_profile_scope scope)
{
    static const char *const names[] = {"startup", "prefill", "decode", "generation"};
    return scope <= YVEX_RUNTIME_PROFILE_GENERATION ? names[scope] : "invalid";
}
static inline const char *runtime_profile_phase_name(yvex_runtime_profile_phase phase)
{
    static const char *const names[] = {"queue", "tokenizer", "prompt_rendering", "embedding", "attention",
        "moe_ingress", "router_projection", "router_topk", "selected_expert_preparation", "routed_experts",
        "shared_experts", "moe_post", "moe_total", "final_normalization", "output_head", "logits_publication",
        "sampling", "state_validation", "synchronization_wait", "kv_commit", "detokenization",
        "provider_publication", "total_prefill", "first_decode", "subsequent_decode", "total_generation"};
    return phase < YVEX_RUNTIME_PROFILE_PHASE_COUNT ? names[phase] : "invalid";
}
static inline const char *runtime_profile_counter_name(yvex_runtime_profile_counter counter)
{
    static const char *const names[] = {"host_payload_reads", "mapped_bytes_touched",
        "h2d_bytes", "d2h_bytes", "d2d_bytes", "managed_prefetch_bytes", "uploads", "downloads",
        "cache_hits", "cache_misses", "cache_evictions", "expert_subviews", "kernel_launches",
        "graph_launches", "graph_captures", "graph_replays", "stream_synchronizations",
        "event_synchronizations", "device_synchronizations", "device_allocations", "host_allocations",
        "workspace_resets", "prompt_tokens", "reused_tokens", "new_prefill_tokens",
        "generated_tokens", "target_forwards", "target_rows", "draft_forwards",
        "draft_rows", "target_verifications", "verified_rows", "accepted_draft_tokens",
        "promoted_target_rows", "discarded_candidate_rows", "target_extensions",
        "replayed_accepted_target_rows", "output_head_rows", "logits_h2d_bytes",
        "logits_d2h_bytes", "logits_d2d_bytes", "full_array_host_scan_bytes",
        "row_expert_pairs", "unique_experts", "expert_bytes", "shape_registry_hits",
        "shape_registry_misses"};
    return counter < YVEX_RUNTIME_PROFILE_COUNTER_COUNT ? names[counter] : "invalid";
}
#define YVEX_RUNTIME_GENERATION_SCHEMA_V3 3u
#define YVEX_RUNTIME_GENERATION_SCHEMA_V5 5u
#define YVEX_RUNTIME_GENERATION_RESULT_SCHEMA_V4 4u
#define YVEX_RUNTIME_GENERATION_TURN_SCHEMA_V1 1u
#define YVEX_RUNTIME_PARTIAL_TURN_SCHEMA_V1 1u
typedef enum {
    YVEX_GENERATION_MODE_TARGET_ONLY = 0,
    YVEX_GENERATION_MODE_DSPARK
} yvex_runtime_generation_mode;
typedef enum {
    YVEX_GENERATION_INPUT_TEXT = 0,
    YVEX_GENERATION_INPUT_MESSAGES = 1,
    YVEX_GENERATION_INPUT_PROVIDER = 2
} yvex_runtime_generation_input_kind;
typedef enum {
    YVEX_GENERATION_STOP_NONE = 0,
    YVEX_GENERATION_STOP_EOS,
    YVEX_GENERATION_STOP_TOKENIZER_TOKEN,
    YVEX_GENERATION_STOP_MAX_NEW_TOKENS,
    YVEX_GENERATION_STOP_CONTEXT_CAPACITY,
    YVEX_GENERATION_STOP_CANCELLED,
    YVEX_GENERATION_STOP_MODEL_FAILURE,
    YVEX_GENERATION_STOP_TOKENIZER_FAILURE,
    YVEX_GENERATION_STOP_OUTPUT_FAILURE
} yvex_runtime_generation_stop_reason;
typedef enum {
    YVEX_GENERATION_STATUS_NONE = 0,
    YVEX_GENERATION_STATUS_COMPLETE,
    YVEX_GENERATION_STATUS_PARTIAL,
    YVEX_GENERATION_STATUS_CANCELLED,
    YVEX_GENERATION_STATUS_FAILED
} yvex_runtime_generation_status;
typedef struct {
    unsigned int schema_version;
    yvex_backend_kind backend;
    yvex_runtime_generation_mode mode;
    unsigned long long context_capacity, prefill_chunk_tokens, maximum_new_tokens;
    unsigned long long maximum_output_bytes, maximum_host_bytes, maximum_device_bytes;
    yvex_runtime_trace_policy trace_policy;
    yvex_execution_evidence_profile evidence_profile;
    yvex_runtime_sampling_policy sampling_policy;
    const unsigned int *additional_stop_token_ids;
    unsigned long long additional_stop_token_count;
    int (*cancel_requested)(void *context);
    void *cancel_context;
} yvex_runtime_generation_options;
typedef struct {
    unsigned int schema_version;
    yvex_runtime_generation_input_kind kind;
    const unsigned char *text;
    unsigned long long text_bytes;
    const yvex_prompt_message *messages;
    unsigned long long message_count;
    yvex_prompt_options prompt_options;
    yvex_tokenizer_encode_options encode_options;
    const yvex_provider_request *provider_request;
} yvex_runtime_generation_request;
typedef struct {
    unsigned int schema_version;
    yvex_backend_kind backend;
    yvex_runtime_generation_mode mode;
    unsigned long long context_capacity, prefill_chunk_tokens, maximum_new_tokens;
    unsigned long long maximum_output_bytes;
    unsigned int trace_policy;
    yvex_execution_evidence_profile evidence_profile;
    yvex_execution_class execution_class;
    char runtime_model_identity[YVEX_SHA256_HEX_CAP];
    char runtime_binding_identity[YVEX_SHA256_HEX_CAP];
    char runtime_descriptor_identity[YVEX_SHA256_HEX_CAP];
    char tokenizer_plan_identity[YVEX_SHA256_HEX_CAP];
    char prompt_policy_identity[YVEX_SHA256_HEX_CAP];
    char transformer_plan_identity[YVEX_SHA256_HEX_CAP];
    char logits_plan_identity[YVEX_SHA256_HEX_CAP];
    char sampling_policy_identity[YVEX_SHA256_HEX_CAP];
    char speculation_policy_identity[YVEX_SHA256_HEX_CAP];
    char stop_policy_identity[YVEX_SHA256_HEX_CAP];
    char kernel_bundle_identity[YVEX_SHA256_HEX_CAP];
    char execution_profile_identity[YVEX_SHA256_HEX_CAP], workload_profile_identity[YVEX_SHA256_HEX_CAP];
    char hardware_profile[YVEX_EXECUTION_TEXT_CAP];
    char generation_plan_identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_generation_plan_summary;
typedef struct {
    unsigned int schema_version;
    unsigned long long ordinal;
    unsigned int sampled_token_id;
    unsigned int decode_input_token_id;
    yvex_token_append_state sequence_state_before, sequence_state_after;
    unsigned long long position_before, position_after;
    unsigned long long persistent_generation_before, persistent_generation_after;
    unsigned long long text_byte_offset, text_byte_count;
    int sampled, decode_submitted, model_committed, detokenized, text_published;
    int terminal, suppressed;
    yvex_tokenizer_token_classification classification;
    char source_logits_identity[YVEX_SHA256_HEX_CAP];
    char sampling_result_identity[YVEX_SHA256_HEX_CAP];
    char decode_execution_identity[YVEX_SHA256_HEX_CAP];
    char persistent_state_digest[YVEX_SHA256_HEX_CAP];
    char decoder_fragment_identity[YVEX_SHA256_HEX_CAP];
    char token_step_identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_generation_token_result;

/*
 * A failed turn reports the exact committed boundary independently of its failure class. Facts
 * without a current owner remain explicitly unavailable instead of being inferred from counters.
 */
typedef struct {
    unsigned int schema_version;
    int available, committed_progress, reset_required;
    int draft_state_generation_available, detokenizer_generation_available;
    int failure_status;
    yvex_runtime_generation_stop_reason stop_reason;
    unsigned long long initial_position, final_committed_position;
    unsigned long long committed_token_count, published_text_bytes;
    unsigned long long target_state_generation, draft_state_generation;
    unsigned long long rng_generation, token_ledger_generation;
    unsigned long long detokenizer_generation;
    char target_state_identity[YVEX_SHA256_HEX_CAP];
    char rng_state_identity[YVEX_SHA256_HEX_CAP];
    char token_ledger_identity[YVEX_SHA256_HEX_CAP];
    char published_text_identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_partial_turn;

typedef struct {
    unsigned int schema_version;
    yvex_runtime_generation_mode execution_mode;
    yvex_runtime_generation_status status;
    yvex_runtime_generation_stop_reason stop_reason;
    int completed, partial, cancelled, failed, has_incomplete_token;
    unsigned long long prompt_bytes, prompt_token_count, prefill_chunk_count;
    unsigned long long requested_new_tokens, sampled_token_count;
    unsigned long long model_committed_token_count, text_published_token_count;
    unsigned long long terminal_token_count, suppressed_token_count;
    unsigned long long first_incomplete_token;
    /* A decode step is one committed sequence position. Target forward and
     * block-verification counts remain separate because one verification may
     * commit several positions. */
    unsigned long long logits_projection_count, sampling_draw_count, decode_step_count;
    unsigned long long draft_cycle_count, draft_forward_count, proposed_token_count;
    unsigned long long selected_verification_token_count, target_verification_count;
    unsigned long long accepted_draft_token_count, rejected_draft_token_count;
    unsigned long long discarded_draft_token_count;
    unsigned long long target_correction_or_bonus_token_count;
    unsigned long long maximum_accepted_prefix;
    unsigned long long confidence_logit_count;
    unsigned long long draft_ns, verification_ns, speculative_commit_ns;
    double mean_accepted_prefix, effective_committed_tokens_per_second;
    double confidence_logit_minimum, confidence_logit_maximum;
    double confidence_logit_mean;
    unsigned long long final_position, final_persistent_generation;
    unsigned long long final_rng_generation, final_token_ledger_generation;
    unsigned long long generated_text_bytes;
    unsigned long long initial_position, reusable_prefix_token_count;
    unsigned long long new_prefill_token_count;
    char prompt_identity[YVEX_SHA256_HEX_CAP];
    char prompt_token_identity[YVEX_SHA256_HEX_CAP];
    char reusable_prefix_identity[YVEX_SHA256_HEX_CAP];
    char initial_rng_identity[YVEX_SHA256_HEX_CAP];
    char final_rng_identity[YVEX_SHA256_HEX_CAP];
    char final_token_ledger_identity[YVEX_SHA256_HEX_CAP];
    char generated_token_identity[YVEX_SHA256_HEX_CAP];
    char generated_text_digest[YVEX_SHA256_HEX_CAP];
    char final_persistent_state_digest[YVEX_SHA256_HEX_CAP];
    char generation_plan_identity[YVEX_SHA256_HEX_CAP];
    char speculation_policy_identity[YVEX_SHA256_HEX_CAP];
    char generation_execution_identity[YVEX_SHA256_HEX_CAP];
    yvex_runtime_partial_turn partial_turn;
    yvex_runtime_profile_record profile;
    int roofline_available;
    yvex_execution_roofline_ledger roofline;
} yvex_runtime_generation_result;
typedef int (*yvex_runtime_generation_fragment_sink)(
    void *context, const yvex_runtime_generation_token_result *token,
    const unsigned char *bytes, unsigned long long byte_count,
    yvex_error *err);
typedef enum {
    YVEX_GENERATION_PROGRESS_PROMPT_ACCEPTED = 0,
    YVEX_GENERATION_PROGRESS_PREFILL_STARTED,
    YVEX_GENERATION_PROGRESS_PREFILL_PROGRESS,
    YVEX_GENERATION_PROGRESS_PREFILL_COMPLETED
} yvex_runtime_generation_progress_kind;
typedef int (*yvex_runtime_generation_progress_sink)(
    void *context, yvex_runtime_generation_progress_kind kind,
    unsigned long long value_a, unsigned long long value_b,
    yvex_error *err);
typedef enum {
    YVEX_SPECULATION_PROGRESS_DRAFT_STARTED = 0,
    YVEX_SPECULATION_PROGRESS_DRAFT_COMPLETED,
    YVEX_SPECULATION_PROGRESS_VERIFICATION_STARTED,
    YVEX_SPECULATION_PROGRESS_VERIFICATION_COMPLETED,
    YVEX_SPECULATION_PROGRESS_PREFIX_ACCEPTED,
    YVEX_SPECULATION_PROGRESS_CANDIDATE_REJECTED,
    YVEX_SPECULATION_PROGRESS_CYCLE_COMMITTED
} yvex_runtime_speculation_progress_kind;
typedef struct {
    unsigned int schema_version;
    yvex_runtime_speculation_progress_kind kind;
    unsigned long long cycle, proposed_tokens, selected_verification_tokens;
    unsigned long long accepted_tokens, rejected_tokens, discarded_tokens;
    unsigned long long verification_count;
    unsigned long long confidence_logit_count;
    double confidence_logit_minimum, confidence_logit_maximum;
    double confidence_logit_mean, seconds;
    char policy_identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_speculation_progress;
typedef int (*yvex_runtime_speculation_progress_sink)(
    void *context, const yvex_runtime_speculation_progress *progress,
    yvex_error *err);
typedef struct {
    unsigned int schema_version;
    const yvex_runtime_generation_request *prompt;
    const unsigned int *committed_prefix_token_ids;
    unsigned long long committed_prefix_token_count;
    unsigned long long maximum_new_tokens;
    unsigned int *prompt_token_ids;
    unsigned long long prompt_token_capacity;
    yvex_runtime_generation_fragment_sink fragment_sink;
    void *fragment_context;
    yvex_runtime_generation_progress_sink progress_sink;
    void *progress_context;
    yvex_runtime_speculation_progress_sink speculation_progress_sink;
    void *speculation_progress_context;
} yvex_runtime_generation_turn_request;
typedef struct {
    unsigned int schema_version;
    int open, busy, closing;
    unsigned long long execution_count, failure_count, cancellation_count;
    unsigned long long token_capacity, text_capacity, workspace_bytes;
    unsigned long long artifact_reopens, model_rebuilds, output_head_reuploads;
    char generation_plan_identity[YVEX_SHA256_HEX_CAP];
    char token_sequence_identity[YVEX_SHA256_HEX_CAP];
    char rng_state_identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_generation_context_summary;

typedef struct yvex_runtime_generation_context yvex_runtime_generation_context;
int yvex_runtime_generation_bytes_digest(
    const char *domain, const unsigned char *bytes, unsigned long long count,
    char output[YVEX_SHA256_HEX_CAP]);
int yvex_runtime_generation_prefix_identity(
    const unsigned int *tokens, unsigned long long count,
    char output[YVEX_SHA256_HEX_CAP]);
int yvex_runtime_generation_stop_identity(
    const yvex_tokenizer_plan_summary *tokenizer,
    const unsigned int *additional, unsigned long long count,
    char output[YVEX_SHA256_HEX_CAP]);
int yvex_runtime_generation_plan_identity(
    const yvex_runtime_generation_plan_summary *plan,
    char output[YVEX_SHA256_HEX_CAP]);
int yvex_runtime_generation_token_identity(
    const yvex_runtime_generation_token_result *token,
    char output[YVEX_SHA256_HEX_CAP]);
int yvex_runtime_generation_tokens_identity(
    const yvex_runtime_generation_token_result *tokens,
    unsigned long long count, char output[YVEX_SHA256_HEX_CAP]);
int yvex_runtime_generation_execution_identity(
    const yvex_runtime_generation_result *result,
    const yvex_runtime_generation_token_result *tokens,
    char output[YVEX_SHA256_HEX_CAP]);
int yvex_runtime_generation_context_summary_copy(
    const yvex_runtime_generation_context *context,
    yvex_runtime_generation_context_summary *summary, yvex_error *err);
int yvex_runtime_generation_context_open(
    yvex_runtime_generation_context **out, yvex_runtime_model *model,
    yvex_runtime_execution_session *session,
    const yvex_runtime_generation_options *options, yvex_error *err);
const yvex_runtime_generation_plan_summary *yvex_runtime_generation_plan_summary_get(
    const yvex_runtime_generation_context *context);
int yvex_runtime_generation_execute(
    yvex_runtime_generation_context *context,
    const yvex_runtime_generation_request *request,
    yvex_runtime_generation_token_result *tokens,
    unsigned long long token_capacity, unsigned char *text,
    unsigned long long text_capacity, yvex_runtime_generation_result *result,
    yvex_error *err);
int yvex_runtime_generation_turn_execute(
    yvex_runtime_generation_context *context,
    const yvex_runtime_generation_turn_request *turn,
    yvex_runtime_generation_token_result *tokens,
    unsigned long long token_capacity, unsigned char *text,
    unsigned long long text_capacity, yvex_runtime_generation_result *result,
    yvex_error *err);
int yvex_runtime_generation_result_validate(
    const yvex_runtime_generation_plan_summary *plan,
    const yvex_runtime_generation_token_result *tokens,
    unsigned long long token_capacity, const unsigned char *text,
    unsigned long long text_capacity,
    const yvex_runtime_generation_result *result, yvex_error *err);
int yvex_runtime_generation_context_close(
    yvex_runtime_generation_context **context, yvex_error *err);
typedef struct {
    const char *target, *artifact_path, *runtime_binding_path;
    yvex_backend_kind backend;
    yvex_runtime_generation_mode mode;
    yvex_runtime_generation_input_kind input_kind;
    const unsigned char *text;
    unsigned long long text_bytes;
    const yvex_prompt_message *messages;
    unsigned long long message_count;
    yvex_prompt_options prompt_options;
    yvex_tokenizer_encode_options encode_options;
    unsigned long long context_capacity, prefill_chunk_tokens, maximum_new_tokens;
    unsigned long long maximum_output_bytes, maximum_host_bytes, maximum_device_bytes;
    yvex_runtime_sampling_policy sampling_policy;
    int (*cancel_requested)(void *context);
    void *cancel_context;
} yvex_generation_operator_request;
typedef struct {
    int completed;
    char status[32], command[64], target[128], family[32], backend[16];
    char sampling_execution_kind[32], tokenizer_execution_kind[32], reason[256];
    yvex_runtime_generation_plan_summary plan;
    yvex_runtime_generation_result execution;
    yvex_runtime_generation_context_summary context;
    yvex_runtime_generation_token_result *tokens;
    unsigned long long token_count;
    unsigned char *text;
    unsigned long long text_bytes;
    int generation_plan_ready, generation_prompt_ready, generation_prefill_ready;
    int generation_first_token_ready, sampled_token_feedback_ready;
    int generation_decode_loop_ready, generation_logits_loop_ready;
    int generation_sampling_loop_ready, generation_token_append_ready;
    int generation_eos_stop_ready, generation_context_stop_ready;
    int generation_incremental_text_ready, generation_partial_progress_ready;
    int generation_cpu_ready, generation_cuda_model_path_ready;
    int generation_loop_ready, generation_ready;
    int cli_generate_ready, repl_ready, interactive_chat_ready, server_generation_ready;
    int model_behavior_evaluation_ready, full_model_benchmark_ready;
    int release_qualification_ready, dspark_ready, speculative_execution_ready;
} yvex_generation_operator_result;
int yvex_runtime_generation_operator_execute(
    const yvex_generation_operator_request *request,
    yvex_generation_operator_result *result,
    yvex_runtime_cleanup_lease **retained_cleanup, yvex_error *err);
void yvex_runtime_generation_operator_result_release(
    yvex_generation_operator_result *result);
const char *yvex_runtime_generation_stop_reason_name(
    yvex_runtime_generation_stop_reason reason);
#ifdef __cplusplus
}
#endif
#endif
