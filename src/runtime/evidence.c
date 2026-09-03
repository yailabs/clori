/* Assemble operator and forensic evidence without making rich reports part of engine dispatch. */
#include <yvex/internal/evidence.h>
#include <yvex/internal/runtime_operator.h>

#include <yvex/internal/backend.h>
#include <yvex/internal/core.h>
#include <yvex/internal/graph_state.h>

#include <errno.h>
#include <limits.h>
#include <stddef.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <build_commit.h>

static int runtime_profile_refuse(
    yvex_error *err, yvex_status status, const char *reason)
{
    yvex_error_set(err, status, "runtime.profile", reason);
    return status;
}
static int runtime_profile_identity_copy(
    char output[YVEX_SHA256_HEX_BYTES], const char *input)
{
    if (!output || !yvex_sha256_hex_valid(input)) return 0;
    yvex_core_text_copy(output, YVEX_SHA256_HEX_BYTES, input);
    return 1;
}

static int runtime_profile_identity(
    const yvex_runtime_profile_record *record,
    char output[YVEX_SHA256_HEX_BYTES])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;
    yvex_sha256_init(&hash);
    if (!record || !output ||
        !yvex_sha256_update_text(&hash, "yvex.runtime.profile.v4") ||
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
        !yvex_sha256_update_text(&hash, record->workload_identity))
        return 0;
    for (index = 0ull; index < YVEX_RUNTIME_PROFILE_PHASE_COUNT; ++index)
        if (!yvex_sha256_update_u64(&hash, record->phase_ns[index]) ||
            !yvex_sha256_update_u64(&hash, record->phase_calls[index]))
            return 0;
    for (index = 0ull; index < YVEX_RUNTIME_PROFILE_COUNTER_COUNT; ++index)
        if (!yvex_sha256_update_u64(&hash, record->counters[index])) return 0;
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

int yvex_runtime_profile_begin(
    yvex_runtime_profile_record *record, yvex_runtime_profile_mode mode,
    yvex_runtime_profile_scope scope, unsigned int backend,
    const char *artifact_identity, const char *physical_variant_identity,
    const char *runtime_binding_identity, const char *runtime_model_identity,
    const char *execution_plan_identity, const char *workload_identity,
    yvex_error *err)
{
    if (!record)
        return runtime_profile_refuse(
            err, YVEX_ERR_INVALID_ARG, "profile record is required");
    memset(record, 0, sizeof(*record));
    if (mode > YVEX_RUNTIME_PROFILE_DETAILED ||
        scope > YVEX_RUNTIME_PROFILE_GENERATION ||
        !runtime_profile_identity_copy(record->artifact_identity, artifact_identity) ||
        !runtime_profile_identity_copy(
            record->physical_variant_identity, physical_variant_identity) ||
        !runtime_profile_identity_copy(
            record->runtime_binding_identity, runtime_binding_identity) ||
        !runtime_profile_identity_copy(
            record->runtime_model_identity, runtime_model_identity) ||
        !runtime_profile_identity_copy(
            record->execution_plan_identity, execution_plan_identity) ||
        !runtime_profile_identity_copy(record->workload_identity, workload_identity))
        return runtime_profile_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "profile mode, scope, and exact identities are required");
    record->schema_version = YVEX_RUNTIME_PROFILE_SCHEMA_V4;
    record->mode = mode;
    record->scope = scope;
    record->backend = backend;
    record->started_ns = yvex_core_monotonic_ns();
    if (!record->started_ns) {
        memset(record, 0, sizeof(*record));
        return runtime_profile_refuse(
            err, YVEX_ERR_STATE, "monotonic profile clock is unavailable");
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_runtime_profile_counter_add(
    yvex_runtime_profile_record *record, yvex_runtime_profile_counter counter,
    unsigned long long value, yvex_error *err)
{
    if (!record || record->schema_version != YVEX_RUNTIME_PROFILE_SCHEMA_V4 ||
        record->sealed || counter >= YVEX_RUNTIME_PROFILE_COUNTER_COUNT)
        return runtime_profile_refuse(
            err, YVEX_ERR_STATE, "profile counter mutation is invalid");
    if (ULLONG_MAX - record->counters[counter] < value)
        return runtime_profile_refuse(
            err, YVEX_ERR_BOUNDS, "profile counter overflowed");
    record->counters[counter] += value;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_runtime_profile_phase_add(
    yvex_runtime_profile_record *record, yvex_runtime_profile_phase phase,
    unsigned long long elapsed_ns, yvex_error *err)
{
    if (!record || record->schema_version != YVEX_RUNTIME_PROFILE_SCHEMA_V4 ||
        record->sealed || phase >= YVEX_RUNTIME_PROFILE_PHASE_COUNT || !elapsed_ns)
        return runtime_profile_refuse(
            err, YVEX_ERR_STATE, "profile phase mutation is invalid");
    if (ULLONG_MAX - record->phase_ns[phase] < elapsed_ns ||
        record->phase_calls[phase] == ULLONG_MAX)
        return runtime_profile_refuse(
            err, YVEX_ERR_BOUNDS, "profile phase accounting overflowed");
    record->phase_ns[phase] += elapsed_ns;
    record->phase_calls[phase]++;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_runtime_profile_finish(
    yvex_runtime_profile_record *record, yvex_error *err)
{
    if (!record || record->schema_version != YVEX_RUNTIME_PROFILE_SCHEMA_V4 ||
        record->sealed)
        return runtime_profile_refuse(
            err, YVEX_ERR_STATE, "profile finish is invalid");
    record->completed_ns = yvex_core_monotonic_ns();
    if (record->completed_ns <= record->started_ns ||
        !runtime_profile_identity(record, record->profile_identity)) {
        record->completed_ns = 0ull;
        return runtime_profile_refuse(
            err, YVEX_ERR_STATE, "profile identity could not be sealed");
    }
    record->sealed = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_runtime_profile_validate(
    const yvex_runtime_profile_record *record, yvex_error *err)
{
    char identity[YVEX_SHA256_HEX_BYTES];
    if (!record || record->schema_version != YVEX_RUNTIME_PROFILE_SCHEMA_V4 ||
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
        return runtime_profile_refuse(
            err, YVEX_ERR_FORMAT, "profile record is stale or malformed");
    yvex_error_clear(err);
    return YVEX_OK;
}

const char *yvex_runtime_profile_mode_name(yvex_runtime_profile_mode mode)
{
    static const char *const names[] = {"off", "summary", "stages", "detailed"};
    return mode <= YVEX_RUNTIME_PROFILE_DETAILED ? names[mode] : "invalid";
}

const char *yvex_runtime_profile_phase_name(yvex_runtime_profile_phase phase)
{
    static const char *const names[] = {
        "queue", "tokenizer", "prompt_rendering", "embedding", "attention",
        "moe_ingress", "router_projection", "router_topk",
        "selected_expert_preparation", "routed_experts", "shared_experts",
        "moe_post", "moe_total", "final_normalization", "output_head",
        "logits_publication", "sampling", "state_validation",
        "synchronization_wait", "kv_commit", "detokenization",
        "provider_publication", "total_prefill", "first_decode",
        "subsequent_decode", "total_generation"};
    return phase < YVEX_RUNTIME_PROFILE_PHASE_COUNT ? names[phase] : "invalid";
}

const char *yvex_runtime_profile_counter_name(
    yvex_runtime_profile_counter counter)
{
    static const char *const names[] = {
        "host_payload_reads", "mapped_bytes_touched", "h2d_bytes", "d2h_bytes",
        "d2d_bytes", "managed_prefetch_bytes", "uploads", "downloads",
        "cache_hits", "cache_misses", "cache_evictions", "expert_subviews",
        "kernel_launches", "accelerated_matrix_launches", "graph_launches",
        "graph_captures", "graph_replays", "queue_synchronizations",
        "event_synchronizations", "device_synchronizations", "device_allocations",
        "host_allocations", "workspace_resets", "prompt_tokens", "reused_tokens",
        "new_prefill_tokens", "generated_tokens", "target_forwards", "target_rows",
        "draft_forwards", "draft_rows", "target_verifications", "verified_rows",
        "accepted_draft_tokens", "promoted_target_rows",
        "discarded_candidate_rows", "target_extensions",
        "replayed_accepted_target_rows", "output_head_rows", "logits_h2d_bytes",
        "logits_d2h_bytes", "logits_d2d_bytes", "full_array_host_scan_bytes",
        "row_expert_pairs", "unique_experts", "expert_bytes"};
    return counter < YVEX_RUNTIME_PROFILE_COUNTER_COUNT ? names[counter] : "invalid";
}

typedef enum {
    EVIDENCE_FIELD_TEXT,
    EVIDENCE_FIELD_U64,
    EVIDENCE_FIELD_UINT,
    EVIDENCE_FIELD_INT
} evidence_field_kind;

typedef struct {
    evidence_field_kind kind;
    size_t offset;
} evidence_field;

typedef yvex_runtime_operator_execution_facts operator_execution_facts;

/* The operator projection uses memcpy over these scalar runs. Reject ABI drift before a report
 * can silently copy padding or truncate a field. */
#define FIELD_RUN_BYTES(type, first, last) \
    (offsetof(type, last) + sizeof(((type *)0)->last) - offsetof(type, first))
#define ASSERT_FIELD_RUN(type, first, last, count, field_type) \
    _Static_assert(FIELD_RUN_BYTES(type, first, last) == (count) * sizeof(field_type), \
                   "operator projection field run contains padding")
ASSERT_FIELD_RUN(yvex_artifact_physical_compatibility, physical_payload_compatible,
                 payload_digest_equal, 8u, int);
ASSERT_FIELD_RUN(yvex_graph_attention_operator_result, physical_payload_compatible,
                 payload_digest_equal, 8u, int);
ASSERT_FIELD_RUN(yvex_graph_attention_state_summary, layer_count,
                 prepared_layer_count, 2u, unsigned long long);
ASSERT_FIELD_RUN(yvex_graph_attention_operator_result, state_layer_count,
                 state_prepared_layer_count, 2u, unsigned long long);
ASSERT_FIELD_RUN(yvex_graph_attention_state_summary, commit_count,
                 reset_count, 4u, unsigned long long);
ASSERT_FIELD_RUN(yvex_graph_attention_operator_result, state_commit_count,
                 state_reset_count, 4u, unsigned long long);
ASSERT_FIELD_RUN(yvex_backend_cuda_attention_graph_summary, graph_count,
                 replay_count, 5u, unsigned long long);
ASSERT_FIELD_RUN(yvex_graph_attention_operator_result, cuda_graph_count,
                 cuda_graph_replay_count, 5u, unsigned long long);
ASSERT_FIELD_RUN(yvex_backend_cuda_attention_graph_summary, launch_count,
                 memset_node_count, 5u, unsigned long long);
ASSERT_FIELD_RUN(yvex_graph_attention_operator_result, cuda_graph_launch_count,
                 cuda_graph_memset_node_count, 5u, unsigned long long);
ASSERT_FIELD_RUN(yvex_backend_cuda_attention_graph_summary, update_count,
                 update_pending_count, 2u, unsigned long long);
ASSERT_FIELD_RUN(yvex_graph_attention_operator_result, cuda_graph_update_count,
                 cuda_graph_update_pending_count, 2u, unsigned long long);
ASSERT_FIELD_RUN(yvex_backend_cuda_attention_graph_summary, capture_elapsed_ns,
                 last_replay_elapsed_ns, 4u, unsigned long long);
ASSERT_FIELD_RUN(yvex_graph_attention_operator_result, cuda_graph_capture_elapsed_ns,
                 cuda_graph_last_replay_elapsed_ns, 4u, unsigned long long);
ASSERT_FIELD_RUN(yvex_runtime_session_summary, resident_binding_count,
                 device_resident_bytes, 4u, unsigned long long);
ASSERT_FIELD_RUN(yvex_graph_attention_operator_result, resident_binding_count,
                 device_resident_bytes, 4u, unsigned long long);
ASSERT_FIELD_RUN(yvex_runtime_session_summary, host_workspace_bytes,
                 host_workspace_peak_bytes, 2u, unsigned long long);
ASSERT_FIELD_RUN(yvex_graph_attention_operator_result, pinned_host_bytes,
                 pinned_host_peak_bytes, 2u, unsigned long long);
ASSERT_FIELD_RUN(yvex_runtime_session_summary, upload_bytes,
                 upload_count, 2u, unsigned long long);
ASSERT_FIELD_RUN(yvex_graph_attention_operator_result, upload_bytes,
                 upload_count, 2u, unsigned long long);
#undef ASSERT_FIELD_RUN
#undef FIELD_RUN_BYTES

static const evidence_field operator_execution_fields[] = {
    {EVIDENCE_FIELD_UINT, offsetof(operator_execution_facts, schema_version)},
    {EVIDENCE_FIELD_TEXT, offsetof(operator_execution_facts, runtime_model_identity)},
    {EVIDENCE_FIELD_TEXT, offsetof(operator_execution_facts, runtime_binding_identity)},
    {EVIDENCE_FIELD_TEXT, offsetof(operator_execution_facts, artifact_identity)},
    {EVIDENCE_FIELD_TEXT, offsetof(operator_execution_facts, runtime_numeric_identity)},
    {EVIDENCE_FIELD_TEXT, offsetof(operator_execution_facts, runtime_descriptor_identity)},
    {EVIDENCE_FIELD_TEXT, offsetof(operator_execution_facts, semantic_graph_identity)},
    {EVIDENCE_FIELD_TEXT, offsetof(operator_execution_facts, executable_graph_identity)},
    {EVIDENCE_FIELD_U64, offsetof(operator_execution_facts, family_adapter_id)},
    {EVIDENCE_FIELD_U64, offsetof(operator_execution_facts, family_adapter_version)},
    {EVIDENCE_FIELD_UINT, offsetof(operator_execution_facts, probe)},
    {EVIDENCE_FIELD_UINT, offsetof(operator_execution_facts, probe_scope)},
    {EVIDENCE_FIELD_UINT, offsetof(operator_execution_facts, operation_scope)},
    {EVIDENCE_FIELD_UINT, offsetof(operator_execution_facts, phase)},
    {EVIDENCE_FIELD_UINT, offsetof(operator_execution_facts, backend)},
    {EVIDENCE_FIELD_UINT, offsetof(operator_execution_facts, requested_mode)},
    {EVIDENCE_FIELD_TEXT, offsetof(operator_execution_facts, selected_mode)},
    {EVIDENCE_FIELD_TEXT, offsetof(operator_execution_facts, capture_bucket)},
    {EVIDENCE_FIELD_INT, offsetof(operator_execution_facts, compare_backends)},
    {EVIDENCE_FIELD_U64, offsetof(operator_execution_facts, token_count)},
    {EVIDENCE_FIELD_U64, offsetof(operator_execution_facts, request_count)},
    {EVIDENCE_FIELD_U64, offsetof(operator_execution_facts, start_position)},
    {EVIDENCE_FIELD_U64, offsetof(operator_execution_facts, layer_start)},
    {EVIDENCE_FIELD_U64, offsetof(operator_execution_facts, layer_count)},
    {EVIDENCE_FIELD_U64, offsetof(operator_execution_facts, selection_key)},
    {EVIDENCE_FIELD_U64, offsetof(operator_execution_facts, binding_count)},
    {EVIDENCE_FIELD_U64, offsetof(operator_execution_facts, maximum_compression_ratio)},
    {EVIDENCE_FIELD_U64, offsetof(operator_execution_facts, maximum_topk_capacity)},
    {EVIDENCE_FIELD_UINT, offsetof(operator_execution_facts, trace_policy)},
    {EVIDENCE_FIELD_U64, offsetof(operator_execution_facts, maximum_host_bytes)},
    {EVIDENCE_FIELD_U64, offsetof(operator_execution_facts, maximum_device_bytes)},
    {EVIDENCE_FIELD_TEXT, offsetof(operator_execution_facts, residency_identity)},
    {EVIDENCE_FIELD_U64, offsetof(operator_execution_facts, residency_generation)},
    {EVIDENCE_FIELD_U64, offsetof(operator_execution_facts, resident_binding_count)},
    {EVIDENCE_FIELD_U64, offsetof(operator_execution_facts, resident_encoded_bytes)},
    {EVIDENCE_FIELD_TEXT, offsetof(operator_execution_facts, workspace_identity)},
    {EVIDENCE_FIELD_U64, offsetof(operator_execution_facts, workspace_bytes)},
    {EVIDENCE_FIELD_U64, offsetof(operator_execution_facts, workspace_generation)},
    {EVIDENCE_FIELD_TEXT, offsetof(operator_execution_facts, capacity_plan_identity)},
    {EVIDENCE_FIELD_TEXT, offsetof(operator_execution_facts, state_layout_identity)},
    {EVIDENCE_FIELD_U64, offsetof(operator_execution_facts, prepared_state_layers)},
    {EVIDENCE_FIELD_U64, offsetof(operator_execution_facts, state_allocated_bytes)},
    {EVIDENCE_FIELD_U64, offsetof(operator_execution_facts, state_generation)},
    {EVIDENCE_FIELD_UINT, offsetof(operator_execution_facts, device_kind)},
    {EVIDENCE_FIELD_INT, offsetof(operator_execution_facts, device_index)},
    {EVIDENCE_FIELD_INT, offsetof(operator_execution_facts, compute_capability_major)},
    {EVIDENCE_FIELD_INT, offsetof(operator_execution_facts, compute_capability_minor)},
    {EVIDENCE_FIELD_U64, offsetof(operator_execution_facts, total_device_bytes)},
};

static int evidence_refuse(yvex_error *err, yvex_status status,
                           const char *where, const char *reason)
{
    yvex_error_set(err, status, where, reason);
    return status;
}

static int evidence_hash_finish(yvex_sha256 *hash,
                                char output[YVEX_SHA256_HEX_CAP])
{
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    if (!yvex_sha256_final(hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

static int evidence_field_hash(yvex_sha256 *hash,
                               const operator_execution_facts *facts,
                               const evidence_field *field)
{
    const unsigned char *address = (const unsigned char *)facts + field->offset;
    if (field->kind == EVIDENCE_FIELD_TEXT) {
        const char *value;
        memcpy(&value, address, sizeof(value));
        return yvex_sha256_update_text(hash, value);
    }
    if (field->kind == EVIDENCE_FIELD_U64) {
        unsigned long long value;
        memcpy(&value, address, sizeof(value));
        return yvex_sha256_update_u64(hash, value);
    }
    if (field->kind == EVIDENCE_FIELD_UINT) {
        unsigned int value;
        memcpy(&value, address, sizeof(value));
        return yvex_sha256_update_u64(hash, value);
    }
    if (field->kind == EVIDENCE_FIELD_INT) {
        int value;
        memcpy(&value, address, sizeof(value));
        return yvex_sha256_update_u64(hash, (unsigned long long)value);
    }
    return 0;
}

int yvex_runtime_operator_execution_identity_compute(
    const yvex_runtime_operator_execution_facts *facts,
    char output[YVEX_SHA256_HEX_CAP], yvex_error *err)
{
    const char *const identities[] = {
        facts ? facts->runtime_model_identity : NULL,
        facts ? facts->runtime_binding_identity : NULL,
        facts ? facts->artifact_identity : NULL,
        facts ? facts->runtime_numeric_identity : NULL,
        facts ? facts->runtime_descriptor_identity : NULL,
        facts ? facts->semantic_graph_identity : NULL,
        facts ? facts->executable_graph_identity : NULL,
        facts ? facts->residency_identity : NULL,
        facts ? facts->workspace_identity : NULL,
        facts ? facts->capacity_plan_identity : NULL,
        facts ? facts->state_layout_identity : NULL,
    };
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned int index;
    if (output) output[0] = '\0';
    if (!facts || !output ||
        facts->schema_version != YVEX_RUNTIME_EXECUTION_DESCRIPTOR_SCHEMA_V2 ||
        facts->probe != YVEX_ATTENTION_PROBE_CANONICAL_V2 ||
        !facts->selected_mode || !facts->selected_mode[0] ||
        !facts->capture_bucket || !facts->capture_bucket[0] ||
        !facts->token_count || !facts->request_count || !facts->layer_count ||
        !facts->residency_generation || !facts->workspace_generation ||
        !facts->state_generation)
        return evidence_refuse(err, YVEX_ERR_INVALID_ARG,
                               "runtime.attention.descriptor",
                               "complete versioned execution descriptor facts are required");
    for (index = 0u; index < sizeof(identities) / sizeof(identities[0]); ++index)
        if (!yvex_sha256_hex_valid(identities[index]))
            return evidence_refuse(err, YVEX_ERR_FORMAT,
                                   "runtime.attention.descriptor",
                                   "execution descriptor identity fact is malformed");
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash,
                                 "yvex.runtime.attention.execution-descriptor.v2"))
        return evidence_refuse(err, YVEX_ERR_STATE,
                               "runtime.attention.descriptor",
                               "execution descriptor serialization failed");
    for (index = 0u;
         index < sizeof(operator_execution_fields) / sizeof(operator_execution_fields[0]);
         ++index)
        if (!evidence_field_hash(&hash, facts, &operator_execution_fields[index]))
            return evidence_refuse(err, YVEX_ERR_STATE,
                                   "runtime.attention.descriptor",
                                   "execution descriptor serialization failed");
    for (index = 0u; index < YVEX_ATTENTION_STATE_BINDING_COUNT; ++index)
        if (facts->state_component_entries[index] >
                facts->state_component_capacities[index] ||
            !yvex_sha256_update_u64(&hash, facts->state_component_entries[index]) ||
            !yvex_sha256_update_u64(&hash, facts->state_component_capacities[index]))
            return evidence_refuse(err, YVEX_ERR_FORMAT,
                                   "runtime.attention.descriptor",
                                   "execution state component facts are malformed");
    for (index = 0u; index < YVEX_RUNTIME_DESCRIPTOR_QTYPE_CAP; ++index)
        if (!yvex_sha256_update_u64(&hash, facts->qtype_binding_counts[index]) ||
            !yvex_sha256_update_u64(&hash, facts->qtype_bytes[index]))
            return evidence_refuse(err, YVEX_ERR_STATE,
                                   "runtime.attention.descriptor",
                                   "execution qtype requirements could not be serialized");
    if (!yvex_sha256_final(&hash, digest))
        return evidence_refuse(err, YVEX_ERR_STATE,
                               "runtime.attention.descriptor",
                               "execution descriptor identity finalization failed");
    yvex_sha256_hex(digest, output);
    yvex_error_clear(err);
    return YVEX_OK;
}

#define QUALIFICATION_FILE_MAX (16u * 1024u)

typedef enum {
    QUALIFICATION_FIELD_TEXT = 0,
    QUALIFICATION_FIELD_U64,
    QUALIFICATION_FIELD_UINT,
    QUALIFICATION_FIELD_INT
} qualification_field_kind;

typedef struct {
    const char *name;
    size_t offset, capacity;
    qualification_field_kind kind;
} qualification_field;

#define QUAL_TEXT(name)                                                        \
    {#name, offsetof(yvex_execution_qualification_record, name),               \
     sizeof(((yvex_execution_qualification_record *)0)->name),                 \
     QUALIFICATION_FIELD_TEXT}
#define QUAL_U64(name)                                                         \
    {#name, offsetof(yvex_execution_qualification_record, name),               \
     sizeof(unsigned long long), QUALIFICATION_FIELD_U64}
#define QUAL_UINT(name)                                                        \
    {#name, offsetof(yvex_execution_qualification_record, name),               \
     sizeof(unsigned int), QUALIFICATION_FIELD_UINT}
#define QUAL_INT(name)                                                         \
    {#name, offsetof(yvex_execution_qualification_record, name),               \
     sizeof(int), QUALIFICATION_FIELD_INT}

static const qualification_field qualification_fields[] = {
    QUAL_UINT(schema_version), QUAL_UINT(source_relation),
    QUAL_UINT(warm_state), QUAL_UINT(contention_state), QUAL_INT(seed_present),
    QUAL_TEXT(build_commit), QUAL_TEXT(build_source_tree),
    QUAL_TEXT(build_source_state), QUAL_TEXT(build_source_delta_identity),
    QUAL_TEXT(build_identity), QUAL_TEXT(source_repository),
    QUAL_TEXT(source_revision), QUAL_TEXT(product_model),
    QUAL_TEXT(specialization), QUAL_TEXT(artifact_identity),
    QUAL_TEXT(runtime_binding_identity), QUAL_TEXT(deployment_profile),
    QUAL_TEXT(engine_kind), QUAL_TEXT(execution_strategy),
    QUAL_TEXT(runtime_model_identity), QUAL_TEXT(runtime_descriptor_identity),
    QUAL_TEXT(semantic_graph_identity), QUAL_TEXT(executable_graph_identity),
    QUAL_TEXT(backend), QUAL_TEXT(device), QUAL_TEXT(hardware_profile_identity),
    QUAL_TEXT(execution_profile_identity), QUAL_TEXT(prompt_identity),
    QUAL_TEXT(prompt_token_identity), QUAL_TEXT(generation_plan_identity),
    QUAL_TEXT(sampling_policy_identity), QUAL_TEXT(reasoning_policy),
    QUAL_TEXT(generation_execution_identity), QUAL_TEXT(generated_text_digest),
    QUAL_TEXT(measurement_identity), QUAL_TEXT(stop_reason),
    QUAL_TEXT(execution_identity), QUAL_TEXT(environment_identity),
    QUAL_TEXT(run_identity), QUAL_U64(context_capacity),
    QUAL_U64(maximum_new_tokens), QUAL_U64(maximum_output_bytes),
    QUAL_U64(seed), QUAL_U64(prompt_tokens), QUAL_U64(generated_tokens),
    QUAL_U64(final_position), QUAL_U64(time_to_first_token_ns),
    QUAL_U64(total_generation_ns),
};

#undef QUAL_TEXT
#undef QUAL_U64
#undef QUAL_UINT
#undef QUAL_INT

static int qualification_text_valid(const char *text, size_t capacity)
{
    size_t index, length;
    if (!text || !text[0]) return 0;
    length = strnlen(text, capacity);
    if (length == capacity) return 0;
    for (index = 0u; index < length; ++index)
        if ((unsigned char)text[index] < 0x20u || text[index] == 0x7f)
            return 0;
    return 1;
}

static int qualification_hex_length(const char *text, size_t length)
{
    size_t index;
    if (!text || strlen(text) != length) return 0;
    for (index = 0u; index < length; ++index)
        if (!((text[index] >= '0' && text[index] <= '9') ||
              (text[index] >= 'a' && text[index] <= 'f')))
            return 0;
    return 1;
}

static int qualification_copy(char *out, size_t capacity, const char *input)
{
    if (!out || !qualification_text_valid(input, capacity)) return 0;
    yvex_core_text_copy(out, capacity, input);
    return 1;
}

static int qualification_execution_identity(
    const yvex_execution_qualification_record *record,
    char output[YVEX_SHA256_HEX_BYTES])
{
    yvex_sha256 hash;
    yvex_sha256_init(&hash);
    return yvex_sha256_update_text(&hash, "yvex.execution.qualification.execution.v1") &&
           yvex_sha256_update_u64(&hash, record->source_relation) &&
           yvex_sha256_update_text(&hash, record->source_repository) &&
           yvex_sha256_update_text(&hash, record->source_revision) &&
           yvex_sha256_update_text(&hash, record->product_model) &&
           yvex_sha256_update_text(&hash, record->specialization) &&
           yvex_sha256_update_text(&hash, record->artifact_identity) &&
           yvex_sha256_update_text(&hash, record->runtime_binding_identity) &&
           yvex_sha256_update_text(&hash, record->engine_kind) &&
           yvex_sha256_update_text(&hash, record->execution_strategy) &&
           yvex_sha256_update_text(&hash, record->runtime_model_identity) &&
           yvex_sha256_update_text(&hash, record->runtime_descriptor_identity) &&
           yvex_sha256_update_text(&hash, record->semantic_graph_identity) &&
           yvex_sha256_update_text(&hash, record->executable_graph_identity) &&
           yvex_sha256_update_text(&hash, record->backend) &&
           yvex_sha256_update_u64(&hash, record->context_capacity) &&
           evidence_hash_finish(&hash, output);
}

static int qualification_environment_identity(
    const yvex_execution_qualification_record *record,
    char output[YVEX_SHA256_HEX_BYTES])
{
    yvex_sha256 hash;
    yvex_sha256_init(&hash);
    return yvex_sha256_update_text(&hash, "yvex.execution.qualification.environment.v1") &&
           yvex_sha256_update_text(&hash, record->device) &&
           yvex_sha256_update_text(&hash, record->hardware_profile_identity) &&
           yvex_sha256_update_u64(&hash, record->warm_state) &&
           yvex_sha256_update_u64(&hash, record->contention_state) &&
           evidence_hash_finish(&hash, output);
}

static int qualification_run_identity(
    const yvex_execution_qualification_record *record,
    char output[YVEX_SHA256_HEX_BYTES])
{
    yvex_sha256 hash;
    yvex_sha256_init(&hash);
    return yvex_sha256_update_text(&hash, "yvex.execution.qualification.run.v1") &&
           yvex_sha256_update_text(&hash, record->build_commit) &&
           yvex_sha256_update_text(&hash, record->build_source_tree) &&
           yvex_sha256_update_text(&hash, record->build_source_state) &&
           yvex_sha256_update_text(&hash, record->build_source_delta_identity) &&
           yvex_sha256_update_text(&hash, record->build_identity) &&
           yvex_sha256_update_text(&hash, record->execution_identity) &&
           yvex_sha256_update_text(&hash, record->environment_identity) &&
           yvex_sha256_update_text(&hash, record->deployment_profile) &&
           yvex_sha256_update_text(&hash, record->execution_profile_identity) &&
           yvex_sha256_update_text(&hash, record->prompt_identity) &&
           yvex_sha256_update_text(&hash, record->prompt_token_identity) &&
           yvex_sha256_update_text(&hash, record->generation_plan_identity) &&
           yvex_sha256_update_text(&hash, record->sampling_policy_identity) &&
           yvex_sha256_update_text(&hash, record->reasoning_policy) &&
           yvex_sha256_update_u64(&hash, record->maximum_new_tokens) &&
           yvex_sha256_update_u64(&hash, record->maximum_output_bytes) &&
           yvex_sha256_update_u64(&hash, (unsigned long long)record->seed_present) &&
           yvex_sha256_update_u64(&hash, record->seed) &&
           yvex_sha256_update_text(&hash, record->generation_execution_identity) &&
           yvex_sha256_update_text(&hash, record->generated_text_digest) &&
           yvex_sha256_update_text(&hash, record->measurement_identity) &&
           yvex_sha256_update_text(&hash, record->stop_reason) &&
           yvex_sha256_update_u64(&hash, record->prompt_tokens) &&
           yvex_sha256_update_u64(&hash, record->generated_tokens) &&
           yvex_sha256_update_u64(&hash, record->final_position) &&
           yvex_sha256_update_u64(&hash, record->time_to_first_token_ns) &&
           yvex_sha256_update_u64(&hash, record->total_generation_ns) &&
           evidence_hash_finish(&hash, output);
}

static int qualification_record_shape_valid(
    const yvex_execution_qualification_record *record)
{
    size_t index;
    if (!record) return 0;
    const char *const identities[] = {
        record->build_source_delta_identity, record->build_identity,
        record->artifact_identity, record->runtime_binding_identity,
        record->runtime_model_identity,
        record->runtime_descriptor_identity, record->semantic_graph_identity,
        record->executable_graph_identity, record->prompt_identity,
        record->prompt_token_identity, record->generation_plan_identity,
        record->sampling_policy_identity, record->generation_execution_identity,
        record->generated_text_digest, record->measurement_identity,
        record->hardware_profile_identity, record->execution_profile_identity,
        record->execution_identity, record->environment_identity,
        record->run_identity};
    if (record->schema_version != YVEX_EXECUTION_QUALIFICATION_SCHEMA_V1 ||
        record->source_relation > YVEX_EXECUTION_SOURCE_UNAVAILABLE ||
        record->warm_state > YVEX_EXECUTION_WARM_STATE_WARM ||
        record->contention_state > YVEX_EXECUTION_CONTENTION_UNKNOWN ||
        (record->seed_present != 0 && record->seed_present != 1) ||
        !qualification_hex_length(record->build_commit, 40u) ||
        !qualification_hex_length(record->build_source_tree, 40u) ||
        (strcmp(record->build_source_state, "clean") &&
         strcmp(record->build_source_state, "dirty")) ||
        !record->context_capacity || !record->maximum_new_tokens ||
        !record->maximum_output_bytes || !record->prompt_tokens ||
        !record->generated_tokens ||
        record->generated_tokens > record->maximum_new_tokens ||
        !record->final_position || !record->time_to_first_token_ns ||
        record->time_to_first_token_ns > record->total_generation_ns)
        return 0;
    for (index = 0u; index < sizeof(identities) / sizeof(identities[0]); ++index)
        if (!yvex_sha256_hex_valid(identities[index])) return 0;
    for (index = 0u; index < sizeof(qualification_fields) /
                                  sizeof(qualification_fields[0]); ++index)
        if (qualification_fields[index].kind == QUALIFICATION_FIELD_TEXT &&
            !qualification_text_valid(
                (const char *)record + qualification_fields[index].offset,
                qualification_fields[index].capacity))
            return 0;
    if (record->source_relation == YVEX_EXECUTION_SOURCE_AUTHENTICATED)
        return strcmp(record->source_repository, "not-applicable") != 0 &&
               (qualification_hex_length(record->source_revision, 40u) ||
                qualification_hex_length(record->source_revision, 64u));
    return !strcmp(record->source_repository, "not-applicable") &&
           !strcmp(record->source_revision, "not-applicable");
}

int yvex_execution_qualification_seal(
    const yvex_execution_qualification_request *request,
    yvex_execution_qualification_record *record, yvex_error *err)
{
    const char *repository, *revision;
    if (record) memset(record, 0, sizeof(*record));
    if (!request || !record ||
        request->schema_version != YVEX_EXECUTION_QUALIFICATION_SCHEMA_V1 ||
        request->source_relation > YVEX_EXECUTION_SOURCE_UNAVAILABLE ||
        request->warm_state > YVEX_EXECUTION_WARM_STATE_WARM ||
        request->contention_state > YVEX_EXECUTION_CONTENTION_UNKNOWN)
        return evidence_refuse(err, YVEX_ERR_INVALID_ARG,
                               "execution.qualification",
                               "current bounded qualification request is required");
    repository = request->source_relation == YVEX_EXECUTION_SOURCE_AUTHENTICATED
                     ? request->source_repository : "not-applicable";
    revision = request->source_relation == YVEX_EXECUTION_SOURCE_AUTHENTICATED
                   ? request->source_revision : "not-applicable";
    record->schema_version = request->schema_version;
    record->source_relation = request->source_relation;
    record->warm_state = request->warm_state;
    record->contention_state = request->contention_state;
    record->seed_present = request->seed_present;
    record->context_capacity = request->context_capacity;
    record->maximum_new_tokens = request->maximum_new_tokens;
    record->maximum_output_bytes = request->maximum_output_bytes;
    record->seed = request->seed;
    record->prompt_tokens = request->prompt_tokens;
    record->generated_tokens = request->generated_tokens;
    record->final_position = request->final_position;
    record->time_to_first_token_ns = request->time_to_first_token_ns;
    record->total_generation_ns = request->total_generation_ns;
#define COPY_RECORD(field, value)                                              \
    qualification_copy(record->field, sizeof(record->field), (value))
    if (!COPY_RECORD(build_commit, YVEX_BUILD_COMMIT) ||
        !COPY_RECORD(build_source_tree, YVEX_BUILD_SOURCE_TREE) ||
        !COPY_RECORD(build_source_state, YVEX_BUILD_SOURCE_STATE) ||
        !COPY_RECORD(build_source_delta_identity, YVEX_BUILD_SOURCE_DELTA_IDENTITY) ||
        !COPY_RECORD(build_identity, YVEX_BUILD_IDENTITY) ||
        !COPY_RECORD(source_repository, repository) ||
        !COPY_RECORD(source_revision, revision) ||
        !COPY_RECORD(product_model, request->product_model) ||
        !COPY_RECORD(specialization, request->specialization) ||
        !COPY_RECORD(artifact_identity, request->artifact_identity) ||
        !COPY_RECORD(runtime_binding_identity, request->runtime_binding_identity) ||
        !COPY_RECORD(deployment_profile, request->deployment_profile) ||
        !COPY_RECORD(engine_kind, request->engine_kind) ||
        !COPY_RECORD(execution_strategy, request->execution_strategy) ||
        !COPY_RECORD(runtime_model_identity, request->runtime_model_identity) ||
        !COPY_RECORD(runtime_descriptor_identity,
                     request->runtime_descriptor_identity) ||
        !COPY_RECORD(semantic_graph_identity, request->semantic_graph_identity) ||
        !COPY_RECORD(executable_graph_identity,
                     request->executable_graph_identity) ||
        !COPY_RECORD(backend, request->backend) ||
        !COPY_RECORD(device, request->device) ||
        !COPY_RECORD(hardware_profile_identity,
                     request->hardware_profile_identity) ||
        !COPY_RECORD(execution_profile_identity,
                     request->execution_profile_identity) ||
        !COPY_RECORD(prompt_identity, request->prompt_identity) ||
        !COPY_RECORD(prompt_token_identity, request->prompt_token_identity) ||
        !COPY_RECORD(generation_plan_identity,
                     request->generation_plan_identity) ||
        !COPY_RECORD(sampling_policy_identity,
                     request->sampling_policy_identity) ||
        !COPY_RECORD(reasoning_policy, request->reasoning_policy) ||
        !COPY_RECORD(generation_execution_identity,
                     request->generation_execution_identity) ||
        !COPY_RECORD(generated_text_digest, request->generated_text_digest) ||
        !COPY_RECORD(measurement_identity, request->measurement_identity) ||
        !COPY_RECORD(stop_reason, request->stop_reason) ||
        !qualification_execution_identity(record, record->execution_identity) ||
        !qualification_environment_identity(record, record->environment_identity) ||
        !qualification_run_identity(record, record->run_identity) ||
        !qualification_record_shape_valid(record)) {
        memset(record, 0, sizeof(*record));
        return evidence_refuse(err, YVEX_ERR_FORMAT,
                               "execution.qualification",
                               "qualification facts are incomplete or malformed");
    }
#undef COPY_RECORD
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_execution_qualification_validate(
    const yvex_execution_qualification_record *record, yvex_error *err)
{
    char execution[YVEX_SHA256_HEX_BYTES];
    char environment[YVEX_SHA256_HEX_BYTES];
    char run[YVEX_SHA256_HEX_BYTES];
    if (!qualification_record_shape_valid(record) ||
        !qualification_execution_identity(record, execution) ||
        strcmp(execution, record->execution_identity) ||
        !qualification_environment_identity(record, environment) ||
        strcmp(environment, record->environment_identity) ||
        !qualification_run_identity(record, run) ||
        strcmp(run, record->run_identity))
        return evidence_refuse(err, YVEX_ERR_FORMAT,
                               "execution.qualification",
                               "qualification identity chain is stale or malformed");
    yvex_error_clear(err);
    return YVEX_OK;
}

static int qualification_bytes_format(yvex_core_bytes *bytes,
                                      const char *format, ...)
{
    char buffer[512];
    va_list arguments;
    int count;
    va_start(arguments, format);
    count = vsnprintf(buffer, sizeof(buffer), format, arguments);
    va_end(arguments);
    return count >= 0 && count < (int)sizeof(buffer) &&
           yvex_core_bytes_append(bytes, buffer, (size_t)count);
}

static int qualification_serialize(
    const yvex_execution_qualification_record *record, yvex_core_bytes *bytes)
{
    const unsigned char *base = (const unsigned char *)record;
    size_t index;
    memset(bytes, 0, sizeof(*bytes));
    bytes->maximum = QUALIFICATION_FILE_MAX;
    bytes->initial_capacity = 2048u;
    if (!qualification_bytes_format(bytes, "YVEX_EXECUTION_QUALIFICATION\t1\n"))
        return 0;
    for (index = 0u; index < sizeof(qualification_fields) /
                                  sizeof(qualification_fields[0]); ++index) {
        const qualification_field *field = &qualification_fields[index];
        if (field->kind == QUALIFICATION_FIELD_TEXT) {
            if (!qualification_bytes_format(bytes, "%s\t%s\n", field->name,
                                             (const char *)base + field->offset))
                return 0;
        } else if (field->kind == QUALIFICATION_FIELD_U64) {
            unsigned long long value;
            memcpy(&value, base + field->offset, sizeof(value));
            if (!qualification_bytes_format(bytes, "%s\t%llu\n", field->name,
                                             value)) return 0;
        } else if (field->kind == QUALIFICATION_FIELD_UINT) {
            unsigned int value;
            memcpy(&value, base + field->offset, sizeof(value));
            if (!qualification_bytes_format(bytes, "%s\t%u\n", field->name,
                                             value)) return 0;
        } else {
            int value;
            memcpy(&value, base + field->offset, sizeof(value));
            if (!qualification_bytes_format(bytes, "%s\t%d\n", field->name,
                                             value)) return 0;
        }
    }
    return 1;
}

int yvex_execution_qualification_write(
    const char *path, const yvex_execution_qualification_record *record,
    yvex_execution_qualification_publication *publication, yvex_error *err)
{
    yvex_core_bytes bytes = {0};
    yvex_core_file_result result = {0};
    int rc;
    if (publication) memset(publication, 0, sizeof(*publication));
    if (!path || path[0] != '/' || !publication)
        return evidence_refuse(err, YVEX_ERR_INVALID_ARG,
                               "execution.qualification.write",
                               "absolute path and valid qualification are required");
    if (yvex_execution_qualification_validate(record, err) != YVEX_OK)
        return yvex_error_code(err);
    if (!qualification_serialize(record, &bytes)) {
        free(bytes.data);
        return evidence_refuse(err, YVEX_ERR_BOUNDS,
                               "execution.qualification.write",
                               "bounded qualification serialization failed");
    }
    rc = yvex_core_file_publish_noreplace(path, bytes.data, bytes.count,
                                          NULL, NULL, NULL, &result, err);
    if (rc == YVEX_OK) {
        publication->published = 1;
        publication->file_bytes = bytes.count;
        yvex_core_text_copy(publication->path, sizeof(publication->path), path);
        yvex_core_text_copy(publication->run_identity,
                            sizeof(publication->run_identity),
                            record->run_identity);
    }
    free(bytes.data);
    return rc;
}

static int qualification_parse_line(char **cursor, char *end,
                                    const char *name, char **value)
{
    char *line_end, *separator;
    size_t length = strlen(name);
    if (!cursor || !*cursor || *cursor >= end) return 0;
    line_end = memchr(*cursor, '\n', (size_t)(end - *cursor));
    if (!line_end) return 0;
    separator = memchr(*cursor, '\t', (size_t)(line_end - *cursor));
    if (!separator || (size_t)(separator - *cursor) != length ||
        memcmp(*cursor, name, length) || separator + 1 == line_end)
        return 0;
    *separator = '\0';
    *line_end = '\0';
    *value = separator + 1;
    *cursor = line_end + 1;
    return 1;
}

static int qualification_parse_field(
    yvex_execution_qualification_record *record,
    const qualification_field *field, const char *value)
{
    unsigned char *base = (unsigned char *)record;
    char *end = NULL;
    unsigned long long parsed;
    if (field->kind == QUALIFICATION_FIELD_TEXT)
        return qualification_copy((char *)base + field->offset,
                                  field->capacity, value);
    if (!value[0] || value[0] < '0' || value[0] > '9') return 0;
    errno = 0;
    parsed = strtoull(value, &end, 10);
    if (errno || !end || *end) return 0;
    if (field->kind == QUALIFICATION_FIELD_U64) {
        memcpy(base + field->offset, &parsed, sizeof(parsed));
        return 1;
    }
    if (field->kind == QUALIFICATION_FIELD_UINT && parsed <= UINT_MAX) {
        unsigned int narrowed = (unsigned int)parsed;
        memcpy(base + field->offset, &narrowed, sizeof(narrowed));
        return 1;
    }
    if (field->kind == QUALIFICATION_FIELD_INT && parsed <= INT_MAX) {
        int narrowed = (int)parsed;
        memcpy(base + field->offset, &narrowed, sizeof(narrowed));
        return 1;
    }
    return 0;
}

int yvex_execution_qualification_open(
    const char *path, yvex_execution_qualification_record *record,
    yvex_error *err)
{
    static const char magic[] = "YVEX_EXECUTION_QUALIFICATION\t1\n";
    yvex_core_file_result result = {0};
    unsigned char *bytes = NULL;
    size_t count = 0u, index;
    char *cursor, *end, *value;
    int rc;
    if (record) memset(record, 0, sizeof(*record));
    if (!path || path[0] != '/' || !record)
        return evidence_refuse(err, YVEX_ERR_INVALID_ARG,
                               "execution.qualification.open",
                               "absolute path and output record are required");
    rc = yvex_core_file_read_snapshot(path, QUALIFICATION_FILE_MAX,
                                      &bytes, &count, &result, err);
    if (rc != YVEX_OK) return rc;
    if (count < sizeof(magic) - 1u ||
        memcmp(bytes, magic, sizeof(magic) - 1u)) {
        free(bytes);
        return evidence_refuse(err, YVEX_ERR_FORMAT,
                               "execution.qualification.open",
                               "qualification header is malformed");
    }
    cursor = (char *)bytes + sizeof(magic) - 1u;
    end = (char *)bytes + count;
    for (index = 0u; index < sizeof(qualification_fields) /
                                  sizeof(qualification_fields[0]); ++index) {
        const qualification_field *field = &qualification_fields[index];
        if (!qualification_parse_line(&cursor, end, field->name, &value) ||
            !qualification_parse_field(record, field, value)) {
            free(bytes);
            memset(record, 0, sizeof(*record));
            return evidence_refuse(err, YVEX_ERR_FORMAT,
                                   "execution.qualification.open",
                                   "qualification field order or value is malformed");
        }
    }
    if (cursor != end ||
        yvex_execution_qualification_validate(record, err) != YVEX_OK) {
        free(bytes);
        memset(record, 0, sizeof(*record));
        return evidence_refuse(err, YVEX_ERR_FORMAT,
                               "execution.qualification.open",
                               "qualification identity chain is malformed");
    }
    free(bytes);
    yvex_error_clear(err);
    return YVEX_OK;
}

static int evidence_add(unsigned long long *total, unsigned long long value)
{
    return yvex_core_u64_add(*total, value, total);
}

/* Compute an exact scaled ratio without requiring the numerator product to fit. */
static int evidence_mul_div(unsigned long long left, unsigned long long right,
                            unsigned long long divisor, int round_up,
                            unsigned long long *result)
{
    unsigned long long quotient = 0ull, remainder = 0ull;
    unsigned long long right_quotient, right_remainder;
    unsigned int bit;
    if (!divisor || !result) return 0;
    right_quotient = right / divisor;
    right_remainder = right % divisor;
    for (bit = 64u; bit-- > 0u;) {
        if (!yvex_core_u64_mul(quotient, 2ull, &quotient)) return 0;
        if (remainder >= divisor - remainder) {
            remainder -= divisor - remainder;
            if (!evidence_add(&quotient, 1ull)) return 0;
        } else {
            remainder += remainder;
        }
        if (!(left & (1ull << bit))) continue;
        if (!evidence_add(&quotient, right_quotient)) return 0;
        if (right_remainder && remainder >= divisor - right_remainder) {
            remainder -= divisor - right_remainder;
            if (!evidence_add(&quotient, 1ull)) return 0;
        } else {
            remainder += right_remainder;
        }
    }
    if (round_up && remainder && !evidence_add(&quotient, 1ull)) return 0;
    *result = quotient;
    return 1;
}

static int roofline_rate(unsigned long long bytes, unsigned long long duration,
                         unsigned long long scale, unsigned long long *result)
{
    return evidence_mul_div(bytes, scale, duration, 0, result);
}

static int roofline_minimum_time(unsigned long long bytes,
                                 unsigned long long bytes_per_second,
                                 unsigned long long *nanoseconds)
{
    return evidence_mul_div(bytes, 1000000000ull, bytes_per_second, 1,
                            nanoseconds);
}

static int roofline_measurement_build(
    const yvex_execution_hardware_profile *hardware,
    const yvex_execution_phase_measurement *measurement,
    yvex_execution_phase_roofline *phase)
{
    const unsigned long long active_mask =
        YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_ACTIVE_WEIGHT) |
        YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_STATE) |
        YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_ACTIVATION) |
        YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_TEMPORARY);
    unsigned long long device_ns, transfer_ns, mask;
    if (!measurement || measurement->phase >= YVEX_EXECUTION_ROOFLINE_PHASE_COUNT ||
        (measurement->fact_mask & ~YVEX_EXECUTION_PHASE_FACT_ALL)) return 0;
    mask = measurement->fact_mask ? measurement->fact_mask : YVEX_EXECUTION_PHASE_FACT_ALL;
    if (!(mask & YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_DURATION)) ||
        !(mask & YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_WORK)) ||
        !measurement->measured_duration_ns || !measurement->work_units ||
        ((mask & YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_OCCUPANCY)) &&
         measurement->occupancy_parts_per_million > 1000000ull) ||
        (!(mask & YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_ACTIVE_WEIGHT)) &&
         measurement->active_weight_bytes) ||
        (!(mask & YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_STATE)) &&
         measurement->state_bytes) ||
        (!(mask & YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_ACTIVATION)) &&
         measurement->activation_bytes) ||
        (!(mask & YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_TEMPORARY)) &&
         measurement->temporary_bytes) ||
        (!(mask & YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_MOVEMENT)) &&
         (measurement->h2d_bytes || measurement->d2h_bytes || measurement->d2d_bytes)) ||
        (!(mask & YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_KERNELS)) &&
         measurement->kernel_count) ||
        (!(mask & YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_SYNCHRONIZATIONS)) &&
         measurement->synchronization_count) ||
        (!(mask & YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_OCCUPANCY)) &&
         measurement->occupancy_parts_per_million) ||
        (!(mask & YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_COMMITTED_TOKENS)) &&
         measurement->committed_tokens)) return 0;
    memset(phase, 0, sizeof(*phase));
    phase->measurement = *measurement;
    phase->measurement.fact_mask = mask;
    phase->available = 1;
    phase->missing_fact_mask = YVEX_EXECUTION_PHASE_FACT_ALL & ~mask;
    if ((mask & active_mask) != active_mask ||
        !(mask & YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_MOVEMENT))) return 1;
    if (!evidence_add(&phase->active_device_bytes, measurement->active_weight_bytes) ||
        !evidence_add(&phase->active_device_bytes, measurement->state_bytes) ||
        !evidence_add(&phase->active_device_bytes, measurement->activation_bytes) ||
        !evidence_add(&phase->active_device_bytes, measurement->temporary_bytes) ||
        !evidence_add(&phase->transfer_bytes, measurement->h2d_bytes) ||
        !evidence_add(&phase->transfer_bytes, measurement->d2h_bytes) ||
        !evidence_add(&phase->transfer_bytes, measurement->d2d_bytes) ||
        !roofline_minimum_time(phase->active_device_bytes,
                               hardware->sustainable_read_bytes_per_second,
                               &device_ns) ||
        !roofline_minimum_time(phase->transfer_bytes,
                               hardware->sustainable_copy_bytes_per_second,
                               &transfer_ns)) return 0;
    phase->minimum_memory_time_ns = device_ns > transfer_ns ? device_ns : transfer_ns;
    if (!roofline_rate(phase->active_device_bytes, measurement->measured_duration_ns,
                       1000000000ull, &phase->measured_bytes_per_second) ||
        !roofline_rate(phase->minimum_memory_time_ns,
                       measurement->measured_duration_ns, 1000000ull,
                       &phase->roofline_utilization_parts_per_million)) return 0;
    if (phase->roofline_utilization_parts_per_million > 1000000ull)
        phase->roofline_utilization_parts_per_million = 1000000ull;
    phase->roofline_available = 1;
    if (measurement->measured_duration_ns > phase->minimum_memory_time_ns)
        phase->optimization_headroom_ns =
            measurement->measured_duration_ns - phase->minimum_memory_time_ns;
    return 1;
}

int yvex_execution_phase_measurement_accumulate(
    yvex_execution_phase_measurement *measurements,
    unsigned long long measurement_capacity,
    unsigned long long *measurement_count,
    const yvex_execution_phase_measurement *delta, yvex_error *err)
{
    const unsigned long long required =
        YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_DURATION) |
        YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_WORK) |
        YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_COMMITTED_TOKENS);
    const unsigned long long occupancy =
        YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_OCCUPANCY);
    yvex_execution_phase_measurement *measurement = NULL;
    yvex_execution_phase_measurement candidate;
    unsigned long long index;
    if (!measurements || !measurement_count || !delta || !measurement_capacity ||
        *measurement_count > measurement_capacity ||
        delta->phase >= YVEX_EXECUTION_ROOFLINE_PHASE_COUNT ||
        (delta->fact_mask & required) != required ||
        (delta->fact_mask & ~YVEX_EXECUTION_PHASE_FACT_ALL) ||
        ((delta->fact_mask & occupancy) &&
         delta->occupancy_parts_per_million > 1000000ull) ||
        (!(delta->fact_mask & occupancy) && delta->occupancy_parts_per_million) ||
        !delta->measured_duration_ns || !delta->work_units)
        return evidence_refuse(err, YVEX_ERR_INVALID_ARG,
                               "runtime.execution.roofline",
                               "phase measurement delta is incomplete");
    for (index = 0ull; index < *measurement_count; ++index)
        if (measurements[index].phase == delta->phase) measurement = &measurements[index];
    if (!measurement) {
        if (*measurement_count == measurement_capacity)
            return evidence_refuse(err, YVEX_ERR_BOUNDS,
                                   "runtime.execution.roofline",
                                   "phase measurement capacity is exhausted");
        measurements[*measurement_count] = *delta;
        (*measurement_count)++;
        yvex_error_clear(err);
        return YVEX_OK;
    }
    if (measurement->fact_mask != delta->fact_mask)
        return evidence_refuse(err, YVEX_ERR_STATE,
                               "runtime.execution.roofline",
                               "phase fact availability changed");
    candidate = *measurement;
    if (delta->fact_mask & occupancy) {
        unsigned long long previous, incoming, total, work;
        if (!yvex_core_u64_mul(measurement->occupancy_parts_per_million,
                               measurement->work_units, &previous) ||
            !yvex_core_u64_mul(delta->occupancy_parts_per_million,
                               delta->work_units, &incoming) ||
            !yvex_core_u64_add(previous, incoming, &total) ||
            !yvex_core_u64_add(measurement->work_units, delta->work_units, &work))
            return evidence_refuse(err, YVEX_ERR_BOUNDS,
                                   "runtime.execution.roofline",
                                   "phase occupancy accumulation overflowed");
        candidate.occupancy_parts_per_million = total / work;
    }
#define ACCUMULATE(field_) evidence_add(&candidate.field_, delta->field_)
    if (!ACCUMULATE(active_weight_bytes) || !ACCUMULATE(state_bytes) ||
        !ACCUMULATE(activation_bytes) || !ACCUMULATE(temporary_bytes) ||
        !ACCUMULATE(h2d_bytes) || !ACCUMULATE(d2h_bytes) ||
        !ACCUMULATE(d2d_bytes) || !ACCUMULATE(kernel_count) ||
        !ACCUMULATE(synchronization_count) || !ACCUMULATE(measured_duration_ns) ||
        !ACCUMULATE(work_units) || !ACCUMULATE(committed_tokens))
        return evidence_refuse(err, YVEX_ERR_BOUNDS,
                               "runtime.execution.roofline",
                               "phase measurement counters overflowed");
#undef ACCUMULATE
    *measurement = candidate;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int roofline_ledger_identity(yvex_execution_roofline_ledger *ledger)
{
    yvex_sha256 hash;
    unsigned long long index;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.execution-phase-roofline.v1") ||
        !yvex_sha256_update_text(&hash, ledger->hardware_profile_identity) ||
        !yvex_sha256_update_text(&hash, ledger->artifact_identity) ||
        !yvex_sha256_update_text(&hash, ledger->execution_profile_identity) ||
        !yvex_sha256_update_text(&hash, ledger->kernel_bundle_identity) ||
        !yvex_sha256_update_text(&hash, ledger->workload_profile_identity) ||
        !yvex_sha256_update_u64(&hash, ledger->measured_phase_mask) ||
        !yvex_sha256_update_u64(&hash, ledger->rooflined_phase_mask) ||
        !yvex_sha256_update_u64(&hash, (unsigned long long)ledger->priority_provisional))
        return 0;
    for (index = 0ull; index < ledger->phase_count; ++index) {
        const yvex_execution_phase_roofline *phase = &ledger->phases[index];
        const yvex_execution_phase_measurement *value = &phase->measurement;
        if (!yvex_sha256_update_u64(&hash, (unsigned long long)phase->available) ||
            !yvex_sha256_update_u64(&hash, value->phase) ||
            !yvex_sha256_update_u64(&hash, value->fact_mask) ||
            !yvex_sha256_update_u64(&hash, value->active_weight_bytes) ||
            !yvex_sha256_update_u64(&hash, value->state_bytes) ||
            !yvex_sha256_update_u64(&hash, value->activation_bytes) ||
            !yvex_sha256_update_u64(&hash, value->temporary_bytes) ||
            !yvex_sha256_update_u64(&hash, value->h2d_bytes) ||
            !yvex_sha256_update_u64(&hash, value->d2h_bytes) ||
            !yvex_sha256_update_u64(&hash, value->d2d_bytes) ||
            !yvex_sha256_update_u64(&hash, value->kernel_count) ||
            !yvex_sha256_update_u64(&hash, value->synchronization_count) ||
            !yvex_sha256_update_u64(&hash, value->occupancy_parts_per_million) ||
            !yvex_sha256_update_u64(&hash, value->measured_duration_ns) ||
            !yvex_sha256_update_u64(&hash, value->work_units) ||
            !yvex_sha256_update_u64(&hash, value->committed_tokens) ||
            !yvex_sha256_update_u64(&hash, phase->minimum_memory_time_ns) ||
            !yvex_sha256_update_u64(&hash, phase->optimization_headroom_ns) ||
            !yvex_sha256_update_u64(&hash, phase->optimization_priority)) return 0;
    }
    return evidence_hash_finish(&hash, ledger->identity);
}

int yvex_execution_roofline_ledger_build(
    const yvex_execution_roofline_ledger_request *request,
    yvex_execution_roofline_ledger *ledger, yvex_error *err)
{
    unsigned long long index, other, seen = 0ull;
    const char *identities[4];
    if (ledger) memset(ledger, 0, sizeof(*ledger));
    if (!request || !ledger ||
        request->schema_version != YVEX_EXECUTION_PHASE_ROOFLINE_SCHEMA_V1 ||
        !request->hardware ||
        request->hardware->schema_version != YVEX_EXECUTION_HARDWARE_PROFILE_SCHEMA_V1 ||
        !yvex_sha256_hex_valid(request->hardware->identity) || !request->measurements ||
        !(request->hardware->admitted_fact_mask &
          YVEX_EXECUTION_HARDWARE_FACT_BIT(YVEX_EXECUTION_HARDWARE_FACT_BANDWIDTH)) ||
        !request->measurement_count ||
        request->measurement_count > YVEX_EXECUTION_ROOFLINE_PHASE_COUNT)
        return evidence_refuse(err, YVEX_ERR_INVALID_ARG,
                               "runtime.execution.roofline",
                               "one or more unique causal phase measurements are required");
    identities[0] = request->artifact_identity;
    identities[1] = request->execution_profile_identity;
    identities[2] = request->kernel_bundle_identity;
    identities[3] = request->workload_profile_identity;
    for (index = 0ull; index < 4ull; ++index)
        if (!yvex_sha256_hex_valid(identities[index]))
            return evidence_refuse(err, YVEX_ERR_FORMAT,
                                   "runtime.execution.roofline",
                                   "roofline evidence identity is invalid");
    for (index = 0ull; index < request->measurement_count; ++index) {
        unsigned long long phase = request->measurements[index].phase;
        if (phase >= YVEX_EXECUTION_ROOFLINE_PHASE_COUNT || (seen & (1ull << phase)) ||
            !roofline_measurement_build(request->hardware,
                                        &request->measurements[index],
                                        &ledger->phases[phase]))
            return evidence_refuse(err, YVEX_ERR_INVALID_ARG,
                                   "runtime.execution.roofline",
                                   "causal phase measurement is incomplete or duplicated");
        seen |= 1ull << phase;
        if (!evidence_add(&ledger->measured_duration_ns,
                          request->measurements[index].measured_duration_ns) ||
            ((ledger->phases[phase].measurement.fact_mask &
              YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_COMMITTED_TOKENS)) &&
             !evidence_add(&ledger->committed_tokens,
                           request->measurements[index].committed_tokens)))
            return evidence_refuse(err, YVEX_ERR_BOUNDS,
                                   "runtime.execution.roofline",
                                   "roofline ledger totals overflowed");
    }
    for (index = 0ull; index < YVEX_EXECUTION_ROOFLINE_PHASE_COUNT; ++index) {
        unsigned long long priority = 1ull;
        unsigned long long score;
        if (!ledger->phases[index].available) continue;
        score = ledger->phases[index].roofline_available
                    ? ledger->phases[index].optimization_headroom_ns
                    : ledger->phases[index].measurement.measured_duration_ns;
        for (other = 0ull; other < YVEX_EXECUTION_ROOFLINE_PHASE_COUNT; ++other) {
            unsigned long long other_score;
            if (!ledger->phases[other].available) continue;
            other_score = ledger->phases[other].roofline_available
                              ? ledger->phases[other].optimization_headroom_ns
                              : ledger->phases[other].measurement.measured_duration_ns;
            if (other_score > score || (other_score == score && other < index)) ++priority;
        }
        ledger->phases[index].optimization_priority = priority;
    }
    ledger->schema_version = YVEX_EXECUTION_PHASE_ROOFLINE_SCHEMA_V1;
    ledger->phase_count = YVEX_EXECUTION_ROOFLINE_PHASE_COUNT;
    ledger->measured_phase_count = request->measurement_count;
    ledger->measured_phase_mask = seen;
    ledger->missing_phase_mask =
        ((1ull << YVEX_EXECUTION_ROOFLINE_PHASE_COUNT) - 1ull) & ~seen;
    for (index = 0ull; index < YVEX_EXECUTION_ROOFLINE_PHASE_COUNT; ++index) {
        if (!ledger->phases[index].available) {
            ledger->phases[index].measurement.phase = (yvex_execution_roofline_phase)index;
            ledger->phases[index].missing_fact_mask = YVEX_EXECUTION_PHASE_FACT_ALL;
        } else if (ledger->phases[index].roofline_available) {
            ledger->rooflined_phase_mask |= 1ull << index;
        }
    }
    ledger->priority_provisional =
        ledger->rooflined_phase_mask != ledger->measured_phase_mask;
    yvex_core_text_copy(ledger->hardware_profile_identity,
                        sizeof(ledger->hardware_profile_identity), request->hardware->identity);
    yvex_core_text_copy(ledger->artifact_identity,
                        sizeof(ledger->artifact_identity), request->artifact_identity);
    yvex_core_text_copy(ledger->execution_profile_identity,
                        sizeof(ledger->execution_profile_identity),
                        request->execution_profile_identity);
    yvex_core_text_copy(ledger->kernel_bundle_identity,
                        sizeof(ledger->kernel_bundle_identity), request->kernel_bundle_identity);
    yvex_core_text_copy(ledger->workload_profile_identity,
                        sizeof(ledger->workload_profile_identity),
                        request->workload_profile_identity);
    if (!roofline_ledger_identity(ledger))
        return evidence_refuse(err, YVEX_ERR_STATE,
                               "runtime.execution.roofline",
                               "phase roofline ledger identity derivation failed");
    yvex_error_clear(err);
    return YVEX_OK;
}
