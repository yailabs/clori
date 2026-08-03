/*
 * Represent production-adjacent startup, prefill, decode, and generation measurements.
 *
 * Every counter is checked, every duration is monotonic, and identities are field-wise sealed.
 * Internal typed evidence shared by runtime, backend aggregation, operator, and tests.
 */
#ifndef INCLUDE_YVEX_INTERNAL_PROFILE_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_PROFILE_H_INCLUDED
#include <limits.h>
#include <string.h>
#include <yvex/core.h>
#include <yvex/internal/core.h>
#ifdef __cplusplus
extern "C" {
#endif
#define YVEX_RUNTIME_PROFILE_SCHEMA_V1 1u
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
    YVEX_RUNTIME_PROFILE_ACCEPTED_DRAFT_TOKENS,
    YVEX_RUNTIME_PROFILE_PROMOTED_TARGET_ROWS,
    YVEX_RUNTIME_PROFILE_DISCARDED_CANDIDATE_ROWS,
    YVEX_RUNTIME_PROFILE_TARGET_EXTENSIONS,
    YVEX_RUNTIME_PROFILE_REPLAYED_ACCEPTED_TARGET_ROWS,
    YVEX_RUNTIME_PROFILE_OUTPUT_HEAD_ROWS,
    YVEX_RUNTIME_PROFILE_LOGITS_H2D_BYTES,
    YVEX_RUNTIME_PROFILE_LOGITS_D2H_BYTES,
    YVEX_RUNTIME_PROFILE_LOGITS_D2D_BYTES,
    YVEX_RUNTIME_PROFILE_FULL_ARRAY_HOST_SCAN_BYTES,
    YVEX_RUNTIME_PROFILE_ROW_EXPERT_PAIRS,
    YVEX_RUNTIME_PROFILE_UNIQUE_EXPERTS,
    YVEX_RUNTIME_PROFILE_EXPERT_BYTES,
    YVEX_RUNTIME_PROFILE_SHAPE_REGISTRY_HITS,
    YVEX_RUNTIME_PROFILE_SHAPE_REGISTRY_MISSES,
    YVEX_RUNTIME_PROFILE_COUNTER_COUNT
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
/* Publish one stable profile refusal without exporting a helper symbol. */
static inline int runtime_profile_refuse(yvex_error *err, yvex_status status, const char *reason)
{
    yvex_error_set(err, status, "runtime.profile", reason);
    return status;
}
/* Copy one exact SHA-256 identity into a profile record. */
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
/* Initialize one identity-bound mutable profile record. */
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
/* Add one authoritative checked counter to a mutable profile. */
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
/* Add one device-complete or host-complete measured stage duration. */
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
/* Seal one completed record after its monotonic outer boundary. */
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
/* Return one stable profile-mode vocabulary item. */
static inline const char *runtime_profile_mode_name(yvex_runtime_profile_mode mode)
{
    static const char *const names[] = {"off", "summary", "stages", "detailed"};
    return mode <= YVEX_RUNTIME_PROFILE_DETAILED ? names[mode] : "invalid";
}
/* Return one stable profile-scope vocabulary item. */
static inline const char *runtime_profile_scope_name(yvex_runtime_profile_scope scope)
{
    static const char *const names[] = {"startup", "prefill", "decode", "generation"};
    return scope <= YVEX_RUNTIME_PROFILE_GENERATION ? names[scope] : "invalid";
}
/* Return one stable profile-phase vocabulary item. */
static inline const char *runtime_profile_phase_name(yvex_runtime_profile_phase phase)
{
    static const char *const names[] = {"queue", "tokenizer", "prompt_rendering", "embedding", "attention",
        "moe_ingress", "router_projection", "router_topk", "selected_expert_preparation", "routed_experts",
        "shared_experts", "moe_post", "moe_total", "final_normalization", "output_head", "logits_publication",
        "sampling", "state_validation", "synchronization_wait", "kv_commit", "detokenization",
        "provider_publication", "total_prefill", "first_decode", "subsequent_decode", "total_generation"};
    return phase < YVEX_RUNTIME_PROFILE_PHASE_COUNT ? names[phase] : "invalid";
}
/* Return one stable profile-counter vocabulary item. */
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
#ifdef __cplusplus
}
#endif
#endif
