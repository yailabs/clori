/*
 * Construct immutable runtime descriptors from admitted materialization.
 *
 * Descriptor identity binds artifact identity, materialization plan identity, tensor names, roles,
 * coordinates, qtypes, placement, and byte counts, but never stores pointer addresses as identity
 * material. Descriptor construction makes the next graph milestone possible; it does not execute
 * the graph.
 */
#include <yvex/internal/runtime.h>

#include <yvex/internal/graph_state.h>

#include <yvex/internal/backend.h>
#include <yvex/internal/core.h>

#include <limits.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    unsigned long long hash;
    unsigned long long index_plus_one;
} runtime_name_slot;

struct yvex_runtime_descriptor {
    yvex_runtime_tensor_binding *bindings;
    runtime_name_slot *name_index;
    unsigned long long name_index_capacity;
    unsigned long long count;
    yvex_runtime_descriptor_summary summary;
};

typedef enum {
    RUNTIME_EXECUTION_FIELD_TEXT,
    RUNTIME_EXECUTION_FIELD_U64,
    RUNTIME_EXECUTION_FIELD_UINT,
    RUNTIME_EXECUTION_FIELD_INT
} runtime_execution_field_kind;

typedef struct {
    runtime_execution_field_kind kind;
    size_t offset;
} runtime_execution_field;

typedef yvex_runtime_execution_descriptor_facts runtime_execution_facts;

/* Compile-time contract for the scalar runs projected with memcpy by the
 * operator owner. Every run is padding-free in both source and destination;
 * changing either ABI fails the build before a projection can corrupt facts. */
#define FIELD_RUN_BYTES(type, first, last) \
    (offsetof(type, last) + sizeof(((type *)0)->last) - offsetof(type, first))
#define ASSERT_FIELD_RUN(type, first, last, count, field_type) \
    _Static_assert(FIELD_RUN_BYTES(type, first, last) == (count) * sizeof(field_type), \
                   "runtime projection field run contains padding")
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

static const runtime_execution_field runtime_execution_fields[] = {
    {RUNTIME_EXECUTION_FIELD_UINT, offsetof(runtime_execution_facts, schema_version)},
    {RUNTIME_EXECUTION_FIELD_TEXT, offsetof(runtime_execution_facts, runtime_model_identity)},
    {RUNTIME_EXECUTION_FIELD_TEXT, offsetof(runtime_execution_facts, runtime_binding_identity)},
    {RUNTIME_EXECUTION_FIELD_TEXT, offsetof(runtime_execution_facts, artifact_identity)},
    {RUNTIME_EXECUTION_FIELD_TEXT, offsetof(runtime_execution_facts, runtime_numeric_identity)},
    {RUNTIME_EXECUTION_FIELD_TEXT, offsetof(runtime_execution_facts, runtime_descriptor_identity)},
    {RUNTIME_EXECUTION_FIELD_TEXT, offsetof(runtime_execution_facts, semantic_graph_identity)},
    {RUNTIME_EXECUTION_FIELD_TEXT, offsetof(runtime_execution_facts, executable_graph_identity)},
    {RUNTIME_EXECUTION_FIELD_U64, offsetof(runtime_execution_facts, family_adapter_id)},
    {RUNTIME_EXECUTION_FIELD_U64, offsetof(runtime_execution_facts, family_adapter_version)},
    {RUNTIME_EXECUTION_FIELD_UINT, offsetof(runtime_execution_facts, probe)},
    {RUNTIME_EXECUTION_FIELD_UINT, offsetof(runtime_execution_facts, probe_scope)},
    {RUNTIME_EXECUTION_FIELD_UINT, offsetof(runtime_execution_facts, operation_scope)},
    {RUNTIME_EXECUTION_FIELD_UINT, offsetof(runtime_execution_facts, phase)},
    {RUNTIME_EXECUTION_FIELD_UINT, offsetof(runtime_execution_facts, backend)},
    {RUNTIME_EXECUTION_FIELD_UINT, offsetof(runtime_execution_facts, requested_mode)},
    {RUNTIME_EXECUTION_FIELD_TEXT, offsetof(runtime_execution_facts, selected_mode)},
    {RUNTIME_EXECUTION_FIELD_TEXT, offsetof(runtime_execution_facts, capture_bucket)},
    {RUNTIME_EXECUTION_FIELD_INT, offsetof(runtime_execution_facts, compare_backends)},
    {RUNTIME_EXECUTION_FIELD_U64, offsetof(runtime_execution_facts, token_count)},
    {RUNTIME_EXECUTION_FIELD_U64, offsetof(runtime_execution_facts, request_count)},
    {RUNTIME_EXECUTION_FIELD_U64, offsetof(runtime_execution_facts, start_position)},
    {RUNTIME_EXECUTION_FIELD_U64, offsetof(runtime_execution_facts, layer_start)},
    {RUNTIME_EXECUTION_FIELD_U64, offsetof(runtime_execution_facts, layer_count)},
    {RUNTIME_EXECUTION_FIELD_U64, offsetof(runtime_execution_facts, selection_key)},
    {RUNTIME_EXECUTION_FIELD_U64, offsetof(runtime_execution_facts, binding_count)},
    {RUNTIME_EXECUTION_FIELD_U64,
     offsetof(runtime_execution_facts, maximum_compression_ratio)},
    {RUNTIME_EXECUTION_FIELD_U64,
     offsetof(runtime_execution_facts, maximum_topk_capacity)},
    {RUNTIME_EXECUTION_FIELD_UINT, offsetof(runtime_execution_facts, trace_policy)},
    {RUNTIME_EXECUTION_FIELD_U64, offsetof(runtime_execution_facts, maximum_host_bytes)},
    {RUNTIME_EXECUTION_FIELD_U64, offsetof(runtime_execution_facts, maximum_device_bytes)},
    {RUNTIME_EXECUTION_FIELD_TEXT, offsetof(runtime_execution_facts, residency_identity)},
    {RUNTIME_EXECUTION_FIELD_U64, offsetof(runtime_execution_facts, residency_generation)},
    {RUNTIME_EXECUTION_FIELD_U64, offsetof(runtime_execution_facts, resident_binding_count)},
    {RUNTIME_EXECUTION_FIELD_U64, offsetof(runtime_execution_facts, resident_encoded_bytes)},
    {RUNTIME_EXECUTION_FIELD_TEXT, offsetof(runtime_execution_facts, workspace_identity)},
    {RUNTIME_EXECUTION_FIELD_U64, offsetof(runtime_execution_facts, workspace_bytes)},
    {RUNTIME_EXECUTION_FIELD_U64, offsetof(runtime_execution_facts, workspace_generation)},
    {RUNTIME_EXECUTION_FIELD_TEXT, offsetof(runtime_execution_facts, capacity_plan_identity)},
    {RUNTIME_EXECUTION_FIELD_TEXT, offsetof(runtime_execution_facts, state_layout_identity)},
    {RUNTIME_EXECUTION_FIELD_U64, offsetof(runtime_execution_facts, prepared_state_layers)},
    {RUNTIME_EXECUTION_FIELD_U64, offsetof(runtime_execution_facts, state_allocated_bytes)},
    {RUNTIME_EXECUTION_FIELD_U64, offsetof(runtime_execution_facts, state_generation)},
    {RUNTIME_EXECUTION_FIELD_UINT, offsetof(runtime_execution_facts, device_kind)},
    {RUNTIME_EXECUTION_FIELD_INT, offsetof(runtime_execution_facts, device_index)},
    {RUNTIME_EXECUTION_FIELD_INT, offsetof(runtime_execution_facts, compute_capability_major)},
    {RUNTIME_EXECUTION_FIELD_INT, offsetof(runtime_execution_facts, compute_capability_minor)},
    {RUNTIME_EXECUTION_FIELD_U64, offsetof(runtime_execution_facts, total_device_bytes)},
};

static int runtime_execution_field_hash(
    yvex_sha256 *hash, const yvex_runtime_execution_descriptor_facts *facts,
    const runtime_execution_field *field) {
    const unsigned char *address = (const unsigned char *)facts + field->offset;

    if (field->kind == RUNTIME_EXECUTION_FIELD_TEXT) {
        const char *value;
        memcpy(&value, address, sizeof(value));
        return yvex_sha256_update_text(hash, value);
    }
    if (field->kind == RUNTIME_EXECUTION_FIELD_U64) {
        unsigned long long value;
        memcpy(&value, address, sizeof(value));
        return yvex_sha256_update_u64(hash, value);
    }
    if (field->kind == RUNTIME_EXECUTION_FIELD_UINT) {
        unsigned int value;
        memcpy(&value, address, sizeof(value));
        return yvex_sha256_update_u64(hash, value);
    }
    if (field->kind == RUNTIME_EXECUTION_FIELD_INT) {
        int value;
        memcpy(&value, address, sizeof(value));
        return yvex_sha256_update_u64(hash, (unsigned long long)value);
    }
    return 0;
}

/*
 * Compute one versioned pre-dispatch descriptor from explicit canonical facts.
 *
 * Writes one deterministic identity without mutating runtime state. Malformed identities, schema,
 * or hash failure publishes no usable identity.
 */
int yvex_runtime_execution_descriptor_identity_compute(
    const yvex_runtime_execution_descriptor_facts *facts,
    char output[YVEX_SHA256_HEX_CAP], yvex_error *err) {
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
        !facts->state_generation) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.attention.descriptor",
                       "complete versioned execution descriptor facts are required");
        return YVEX_ERR_INVALID_ARG;
    }
    for (index = 0u; index < sizeof(identities) / sizeof(identities[0]); ++index) {
        if (!yvex_sha256_hex_valid(identities[index])) {
            yvex_error_set(err, YVEX_ERR_FORMAT, "runtime.attention.descriptor",
                           "execution descriptor identity fact is malformed");
            return YVEX_ERR_FORMAT;
        }
    }
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(
            &hash, "yvex.runtime.attention.execution-descriptor.v2")) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.attention.descriptor",
                       "execution descriptor serialization failed");
        return YVEX_ERR_STATE;
    }
    for (index = 0u; index < sizeof(runtime_execution_fields) /
                                      sizeof(runtime_execution_fields[0]); ++index) {
        if (!runtime_execution_field_hash(
                &hash, facts, &runtime_execution_fields[index])) {
            yvex_error_set(err, YVEX_ERR_STATE, "runtime.attention.descriptor",
                           "execution descriptor serialization failed");
            return YVEX_ERR_STATE;
        }
    }
    for (index = 0u; index < YVEX_ATTENTION_STATE_BINDING_COUNT; ++index) {
        if (facts->state_component_entries[index] >
                facts->state_component_capacities[index] ||
            !yvex_sha256_update_u64(
                &hash, facts->state_component_entries[index]) ||
            !yvex_sha256_update_u64(
                &hash, facts->state_component_capacities[index])) {
            yvex_error_set(err, YVEX_ERR_FORMAT,
                           "runtime.attention.descriptor",
                           "execution state component facts are malformed");
            return YVEX_ERR_FORMAT;
        }
    }
    for (index = 0u; index < YVEX_RUNTIME_DESCRIPTOR_QTYPE_CAP; ++index) {
        if (!yvex_sha256_update_u64(&hash, facts->qtype_binding_counts[index]) ||
            !yvex_sha256_update_u64(&hash, facts->qtype_bytes[index])) {
            yvex_error_set(err, YVEX_ERR_STATE, "runtime.attention.descriptor",
                           "execution qtype requirements could not be serialized");
            return YVEX_ERR_STATE;
        }
    }
    if (!yvex_sha256_final(&hash, digest)) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.attention.descriptor",
                       "execution descriptor identity finalization failed");
        return YVEX_ERR_STATE;
    }
    yvex_sha256_hex(digest, output);
    yvex_error_clear(err);
    return YVEX_OK;
}

/*
 * Derive a stable workspace identity from explicit execution capacities.
 *
 * Model/backend identity, exact budgets, arena sizes, and optional capacity-plan identity. Writes
 * one canonical SHA-256 identity without allocating workspace storage. Malformed identity input or
 * hash failure leaves no usable output.
 */
int yvex_runtime_workspace_identity_compute(
    const char *runtime_model_identity, yvex_backend_kind backend,
    unsigned long long maximum_host_bytes, unsigned long long maximum_device_bytes,
    unsigned long long workspace_bytes, unsigned long long host_workspace_bytes,
    const char *capacity_identity, char output[YVEX_SHA256_HEX_CAP], yvex_error *err) {
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];

    if (output) output[0] = '\0';
    if (!output || !yvex_sha256_hex_valid(runtime_model_identity) ||
        (capacity_identity && capacity_identity[0] &&
         !yvex_sha256_hex_valid(capacity_identity))) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "runtime.workspace.identity",
                       "workspace identity facts are malformed");
        return YVEX_ERR_FORMAT;
    }
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.workspace.v2") ||
        !yvex_sha256_update_text(&hash, runtime_model_identity) ||
        !yvex_sha256_update_u64(&hash, backend) ||
        !yvex_sha256_update_u64(&hash, maximum_host_bytes) ||
        !yvex_sha256_update_u64(&hash, maximum_device_bytes) ||
        !yvex_sha256_update_u64(&hash, workspace_bytes) ||
        !yvex_sha256_update_u64(&hash, host_workspace_bytes) ||
        !yvex_sha256_update_text(&hash, capacity_identity ? capacity_identity : "") ||
        !yvex_sha256_final(&hash, digest)) {
        yvex_error_set(err, YVEX_ERR_STATE, "runtime.workspace.identity",
                       "workspace identity serialization failed");
        return YVEX_ERR_STATE;
    }
    yvex_sha256_hex(digest, output);
    yvex_error_clear(err);
    return YVEX_OK;
}

static void runtime_failure_set(yvex_runtime_descriptor_failure *failure,
                                yvex_runtime_descriptor_failure_code code,
                                const char *name,
                                unsigned long long tensor_index,
                                unsigned long long expected,
                                unsigned long long actual,
                                const char *reason) {
    if (!failure) return;
    memset(failure, 0, sizeof(*failure));
    failure->code = code;
    failure->tensor_index = tensor_index;
    failure->expected = expected;
    failure->actual = actual;
    failure->reason = reason;
    if (name)
        yvex_core_text_copy(failure->tensor_name, sizeof(failure->tensor_name), name);
}

static int runtime_reject(yvex_runtime_descriptor_failure *failure,
                          yvex_runtime_descriptor_failure_code code,
                          const char *name,
                          unsigned long long tensor_index,
                          unsigned long long expected,
                          unsigned long long actual,
                          yvex_error *err,
                          yvex_status status,
                          const char *message) {
    runtime_failure_set(failure, code, name, tensor_index, expected, actual,
                        message);
    yvex_error_set(err, status, "runtime.descriptor", message);
    return status;
}

static int runtime_index_insert(yvex_runtime_descriptor *descriptor,
                                const char *name,
                                unsigned long long index) {
    unsigned long long hash;
    unsigned long long slot;
    unsigned long long step = 0ull;

    if (!descriptor || !descriptor->name_index ||
        !descriptor->name_index_capacity || !name)
        return 0;
    hash = yvex_core_index_hash(name);
    slot = hash & (descriptor->name_index_capacity - 1ull);
    while (step < descriptor->name_index_capacity) {
        runtime_name_slot *candidate = &descriptor->name_index[slot];
        if (!candidate->index_plus_one) {
            candidate->hash = hash;
            candidate->index_plus_one = index + 1ull;
            return 1;
        }
        if (candidate->hash == hash &&
            strcmp(descriptor->bindings[candidate->index_plus_one - 1ull]
                       .binding->name, name) == 0)
            return 0;
        slot = (slot + 1ull) & (descriptor->name_index_capacity - 1ull);
        step++;
    }
    return 0;
}

/* Encode runtime compute identity fields in canonical identity order. */
static void runtime_compute_identity(yvex_runtime_descriptor *descriptor) {
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long i;

    yvex_sha256_init(&hash);
    yvex_sha256_update_text(&hash, descriptor->summary.artifact_identity);
    yvex_sha256_update_text(&hash,
                      descriptor->summary.materialization_plan_identity);
    yvex_sha256_update_text(&hash, descriptor->summary.logical_model_identity);
    yvex_sha256_update_text(&hash, descriptor->summary.runtime_numeric_identity);
    yvex_sha256_update_u64(&hash,
                     descriptor->summary.runtime_numeric_schema_version);
    yvex_sha256_update_u64(&hash, descriptor->summary.runtime_compute_policy_count);
    yvex_sha256_update_u64(&hash,
                     descriptor->summary.runtime_activation_policy_count);
    yvex_sha256_update_u64(&hash,
                     descriptor->summary.runtime_sparse_topk_policy_count);
    yvex_sha256_update_u64(&hash, descriptor->count);
    yvex_sha256_update_u64(&hash, descriptor->summary.payload_bytes);
    yvex_sha256_update_u64(&hash, descriptor->summary.layer_count);
    yvex_sha256_update_u64(&hash, descriptor->summary.draft_layer_count);
    if (yvex_sha256_hex_valid(descriptor->summary.model_execution.identity))
        yvex_sha256_update_text(&hash, descriptor->summary.model_execution.identity);
    for (i = 0ull; i < descriptor->count; ++i) {
        const yvex_runtime_tensor_binding *binding = &descriptor->bindings[i];
        yvex_sha256_update_text(&hash, binding->binding ? binding->binding->name : "");
        yvex_sha256_update_u64(&hash, binding->tensor_id);
        yvex_sha256_update_u64(&hash, binding->descriptor_index);
        yvex_sha256_update_u64(&hash, (unsigned long long)binding->role);
        yvex_sha256_update_u64(&hash, (unsigned long long)binding->scope);
        yvex_sha256_update_u64(&hash, binding->layer_index);
        yvex_sha256_update_u64(&hash, binding->predictor_index);
        yvex_sha256_update_u64(&hash, binding->qtype);
        yvex_sha256_update_u64(&hash, (unsigned long long)binding->placement);
        yvex_sha256_update_u64(&hash,
                         binding->binding ? binding->binding->encoded_bytes : 0ull);
    }
    (void)yvex_sha256_final(&hash, digest);
    yvex_sha256_hex(digest,
                    descriptor->summary.runtime_descriptor_identity);
}

static yvex_runtime_descriptor *runtime_descriptor_alloc(
    unsigned long long count,
    yvex_runtime_descriptor_failure *failure,
    yvex_error *err) {
    yvex_runtime_descriptor *descriptor;
    unsigned long long capacity;

    if (!yvex_core_power_of_two_capacity(count, 16ull, 2ull, 3ull, &capacity)) {
        runtime_reject(failure, YVEX_RUNTIME_DESCRIPTOR_FAILURE_ALLOCATION,
                       NULL, YVEX_MATERIALIZATION_NO_INDEX, count, 0ull, err,
                       YVEX_ERR_NOMEM,
                       "runtime descriptor index capacity overflow");
        return NULL;
    }
    descriptor = (yvex_runtime_descriptor *)calloc(1u, sizeof(*descriptor));
    if (!descriptor) {
        runtime_reject(failure, YVEX_RUNTIME_DESCRIPTOR_FAILURE_ALLOCATION,
                       NULL, YVEX_MATERIALIZATION_NO_INDEX, 1ull, 0ull, err,
                       YVEX_ERR_NOMEM,
                       "runtime descriptor allocation failed");
        return NULL;
    }
    descriptor->bindings = (yvex_runtime_tensor_binding *)calloc(
        (size_t)(count ? count : 1ull), sizeof(*descriptor->bindings));
    descriptor->name_index = (runtime_name_slot *)calloc(
        (size_t)capacity, sizeof(*descriptor->name_index));
    if (!descriptor->bindings || !descriptor->name_index) {
        yvex_runtime_descriptor_close(descriptor);
        runtime_reject(failure, YVEX_RUNTIME_DESCRIPTOR_FAILURE_ALLOCATION,
                       NULL, YVEX_MATERIALIZATION_NO_INDEX, count, 0ull, err,
                       YVEX_ERR_NOMEM,
                       "runtime descriptor binding allocation failed");
        return NULL;
    }
    descriptor->name_index_capacity = capacity;
    descriptor->count = count;
    return descriptor;
}

static void runtime_fill_common_summary(
    yvex_runtime_descriptor *descriptor,
    const yvex_complete_artifact_admission *admission,
    const yvex_materialization_summary *materialization) {
    descriptor->summary.status = YVEX_RUNTIME_DESCRIPTOR_STATUS_READY;
    yvex_runtime_identity_copy(descriptor->summary.artifact_identity,
                               admission->artifact_identity);
    yvex_runtime_identity_copy(descriptor->summary.materialization_plan_identity,
                               materialization->plan_identity);
    descriptor->summary.tensor_count = materialization->tensor_count;
    descriptor->summary.payload_bytes = materialization->payload_bytes;
    memcpy(descriptor->summary.qtype_tensor_counts,
           materialization->qtype_tensor_counts,
           sizeof(descriptor->summary.qtype_tensor_counts));
    memcpy(descriptor->summary.qtype_bytes,
           materialization->qtype_bytes,
           sizeof(descriptor->summary.qtype_bytes));
    descriptor->summary.tokenizer_metadata_available =
        admission->tokenizer_complete;
    descriptor->summary.graph_execution_ready = 0;
    descriptor->summary.generation_ready = 0;
}

const char *yvex_runtime_descriptor_failure_name(
    yvex_runtime_descriptor_failure_code code) {
    switch (code) {
    case YVEX_RUNTIME_DESCRIPTOR_FAILURE_NONE: return "none";
    case YVEX_RUNTIME_DESCRIPTOR_FAILURE_INVALID_ARGUMENT: return "invalid-argument";
    case YVEX_RUNTIME_DESCRIPTOR_FAILURE_ADMISSION: return "admission";
    case YVEX_RUNTIME_DESCRIPTOR_FAILURE_MATERIALIZATION: return "materialization";
    case YVEX_RUNTIME_DESCRIPTOR_FAILURE_DUPLICATE_BINDING: return "duplicate-binding";
    case YVEX_RUNTIME_DESCRIPTOR_FAILURE_MISSING_BINDING: return "missing-binding";
    case YVEX_RUNTIME_DESCRIPTOR_FAILURE_ARCHITECTURE: return "architecture";
    case YVEX_RUNTIME_DESCRIPTOR_FAILURE_QTYPE: return "qtype";
    case YVEX_RUNTIME_DESCRIPTOR_FAILURE_ALLOCATION: return "allocation";
    }
    return "unknown";
}

int yvex_runtime_descriptor_build(
    yvex_runtime_descriptor **out,
    const yvex_complete_artifact_admission *admission,
    const yvex_materialization_session *session,
    const yvex_runtime_descriptor_family_facts *family,
    yvex_runtime_descriptor_failure *failure,
    yvex_error *err) {
    const yvex_materialization_summary *materialization;
    yvex_runtime_descriptor *descriptor;
    unsigned long long count;
    unsigned long long i;

    if (out) *out = NULL;
    if (!out || !admission || !session)
        return runtime_reject(
            failure, YVEX_RUNTIME_DESCRIPTOR_FAILURE_INVALID_ARGUMENT,
            NULL, YVEX_MATERIALIZATION_NO_INDEX, 1ull, 0ull, err,
            YVEX_ERR_INVALID_ARG,
            "runtime descriptor requires admission and materialization session");
    if (!admission->complete || !admission->materialization_input_ready ||
        admission->runtime_supported)
        return runtime_reject(
            failure, YVEX_RUNTIME_DESCRIPTOR_FAILURE_ADMISSION,
            NULL, YVEX_MATERIALIZATION_NO_INDEX, 1ull, 0ull, err,
            YVEX_ERR_STATE,
            "runtime descriptor requires complete non-runtime artifact admission");
    materialization = yvex_materialization_session_summary(session);
    if (!materialization || !materialization->committed ||
        materialization->status != YVEX_MATERIALIZATION_STATUS_COMMITTED)
        return runtime_reject(
            failure, YVEX_RUNTIME_DESCRIPTOR_FAILURE_MATERIALIZATION,
            NULL, YVEX_MATERIALIZATION_NO_INDEX, 1ull, 0ull, err,
            YVEX_ERR_STATE,
            "runtime descriptor requires committed materialization");
    count = materialization->tensor_count;
    if (count != admission->tensor_count)
        return runtime_reject(
            failure, YVEX_RUNTIME_DESCRIPTOR_FAILURE_MATERIALIZATION,
            NULL, YVEX_MATERIALIZATION_NO_INDEX, admission->tensor_count,
            count, err, YVEX_ERR_FORMAT,
            "materialization tensor count differs from admission");
    descriptor = runtime_descriptor_alloc(count, failure, err);
    if (!descriptor) return yvex_error_code(err);
    runtime_fill_common_summary(descriptor, admission, materialization);
    if (family) {
        if (family->model_execution &&
            (family->model_execution->schema_version !=
                 YVEX_MODEL_EXECUTION_DESCRIPTOR_SCHEMA_V1 ||
             !yvex_sha256_hex_valid(family->model_execution->identity) ||
             strcmp(family->logical_model_identity,
                    family->model_execution->logical_model_identity) != 0)) {
            yvex_runtime_descriptor_close(descriptor);
            return runtime_reject(
                failure, YVEX_RUNTIME_DESCRIPTOR_FAILURE_ARCHITECTURE, NULL,
                YVEX_MATERIALIZATION_NO_INDEX, 1ull, 0ull, err, YVEX_ERR_FORMAT,
                "model execution descriptor disagrees with family facts");
        }
        yvex_runtime_identity_copy(descriptor->summary.logical_model_identity,
                                   family->logical_model_identity);
        yvex_runtime_identity_copy(descriptor->summary.runtime_numeric_identity,
                                   family->runtime_numeric_identity);
        yvex_core_text_copy(descriptor->summary.runtime_hadamard_revision,
                            sizeof(descriptor->summary.runtime_hadamard_revision),
                            family->runtime_hadamard_revision);
        descriptor->summary.runtime_numeric_schema_version = family->runtime_numeric_schema_version;
        descriptor->summary.runtime_compute_policy_count = family->runtime_compute_policy_count;
        descriptor->summary.runtime_activation_policy_count = family->runtime_activation_policy_count;
        descriptor->summary.runtime_sparse_topk_policy_count = family->runtime_sparse_topk_policy_count;
        descriptor->summary.layer_count = family->layer_count;
        descriptor->summary.draft_layer_count = family->draft_layer_count;
        descriptor->summary.routed_experts = family->routed_experts;
        descriptor->summary.experts_per_token = family->experts_per_token;
        descriptor->summary.vocabulary_size = family->vocabulary_size;
        if (family->model_execution)
            descriptor->summary.model_execution = *family->model_execution;
    }
    for (i = 0ull; i < count; ++i) {
        const yvex_materialized_tensor_binding *source =
            yvex_materialization_session_tensor_at(session, i);
        yvex_runtime_tensor_binding *row = &descriptor->bindings[i];
        if (!source) {
            yvex_runtime_descriptor_close(descriptor);
            return runtime_reject(
                failure, YVEX_RUNTIME_DESCRIPTOR_FAILURE_MISSING_BINDING,
                NULL, i, 1ull, 0ull, err, YVEX_ERR_FORMAT,
                "materialization binding missing");
        }
        row->tensor_id = source->tensor_id;
        row->descriptor_index = source->descriptor_index;
        row->binding = source;
        row->role = source->role;
        row->collection = source->collection;
        row->scope = source->scope;
        row->layer_index = source->layer_index;
        row->predictor_index = source->predictor_index;
        row->qtype = source->qtype;
        row->placement = source->placement;
        row->access_mode = source->access_mode;
        if (!runtime_index_insert(descriptor, source->name, i)) {
            yvex_runtime_descriptor_close(descriptor);
            return runtime_reject(
                failure, YVEX_RUNTIME_DESCRIPTOR_FAILURE_DUPLICATE_BINDING,
                source->name, i, 1ull, 2ull, err, YVEX_ERR_FORMAT,
                "duplicate runtime descriptor binding name");
        }
        if (source->role < YVEX_TENSOR_ROLE_COUNT)
            descriptor->summary.role_counts[source->role]++;
        if (row->scope == YVEX_TENSOR_SCOPE_GLOBAL)
            descriptor->summary.global_bindings++;
        else if (row->scope == YVEX_TENSOR_SCOPE_MAIN_LAYER)
            descriptor->summary.main_layer_bindings++;
        else if (row->scope == YVEX_TENSOR_SCOPE_DRAFT)
            descriptor->summary.draft_bindings++;
        if (row->collection == YVEX_TENSOR_COLLECTION_ROUTED_EXPERT)
            descriptor->summary.routed_expert_bindings++;
        if (row->binding && row->binding->expert_count > 1ull)
            descriptor->summary.expert_subview_count += row->binding->expert_count;
    }
    runtime_compute_identity(descriptor);
    *out = descriptor;
    yvex_error_clear(err);
    return YVEX_OK;
}

/*
 * Validate the descriptor against the same sealed terminal projection used by materialization.
 * This keeps family tensor naming outside runtime while making coordinate agreement generic.
 */
int yvex_runtime_descriptor_build_projected(
    yvex_runtime_descriptor **out,
    const yvex_complete_artifact_admission *admission,
    const yvex_materialization_session *session,
    const yvex_runtime_descriptor_family_facts *family,
    const yvex_materialization_projection *projection,
    yvex_runtime_descriptor_failure *failure, yvex_error *err)
{
    yvex_runtime_descriptor *descriptor = NULL;
    unsigned long long index;
    int rc;

    if (out) *out = NULL;
    if (!out || !projection ||
        projection->schema_version != YVEX_MATERIALIZATION_PROJECTION_SCHEMA_VERSION ||
        !projection->complete || !projection->mapping_identity || !projection->find) {
        return runtime_reject(
            failure, YVEX_RUNTIME_DESCRIPTOR_FAILURE_INVALID_ARGUMENT, NULL,
            YVEX_MATERIALIZATION_NO_INDEX, 1ull, 0ull, err, YVEX_ERR_INVALID_ARG,
            "runtime descriptor requires a complete terminal projection");
    }
    rc = yvex_runtime_descriptor_build(
        &descriptor, admission, session, family, failure, err);
    if (rc != YVEX_OK) return rc;
    if (projection->descriptor_count != descriptor->count) {
        rc = runtime_reject(
            failure, YVEX_RUNTIME_DESCRIPTOR_FAILURE_MATERIALIZATION, NULL,
            YVEX_MATERIALIZATION_NO_INDEX, projection->descriptor_count,
            descriptor->count, err, YVEX_ERR_FORMAT,
            "runtime descriptor count differs from terminal projection");
        goto fail;
    }
    for (index = 0ull; index < descriptor->count; ++index) {
        const yvex_runtime_tensor_binding *row = &descriptor->bindings[index];
        yvex_materialization_terminal terminal;

        if (!row->binding || !projection->find(
                projection->context, row->binding->name, &terminal) ||
            terminal.descriptor_index != row->descriptor_index ||
            terminal.role != row->role || terminal.collection != row->collection ||
            terminal.scope != row->scope || terminal.layer_index != row->layer_index ||
            terminal.predictor_index != row->predictor_index ||
            terminal.expert_count != row->binding->expert_count) {
            rc = runtime_reject(
                failure, YVEX_RUNTIME_DESCRIPTOR_FAILURE_ARCHITECTURE,
                row->binding ? row->binding->name : NULL, index, 1ull, 0ull,
                err, YVEX_ERR_FORMAT,
                "runtime binding does not match the terminal projection");
            goto fail;
        }
    }
    *out = descriptor;
    return YVEX_OK;

fail:
    yvex_runtime_descriptor_close(descriptor);
    return rc;
}

/*
 * Restore an immutable runtime descriptor from authenticated runtime-binding facts.
 *
 * Coordinate or identity disagreement releases all candidate state.
 */
int yvex_runtime_descriptor_import(
    yvex_runtime_descriptor **out, const yvex_runtime_descriptor_summary *summary,
    const yvex_runtime_tensor_binding *bindings, unsigned long long binding_count,
    const yvex_materialization_session *session, yvex_runtime_descriptor_failure *failure,
    yvex_error *err) {
    const yvex_materialization_summary *materialization;
    yvex_runtime_descriptor *descriptor;
    char expected_identity[YVEX_RUNTIME_DESCRIPTOR_IDENTITY_CAP];
    unsigned long long i;

    if (out) *out = NULL;
    materialization = yvex_materialization_session_summary(session);
    if (!out || !summary || !bindings || !session || !materialization ||
        !materialization->committed || summary->status != YVEX_RUNTIME_DESCRIPTOR_STATUS_READY ||
        binding_count != summary->tensor_count || binding_count != materialization->tensor_count ||
        strcmp(summary->artifact_identity, materialization->artifact_identity) != 0 ||
        strcmp(summary->materialization_plan_identity, materialization->plan_identity) != 0 ||
        (summary->model_execution.schema_version &&
         (summary->model_execution.schema_version !=
              YVEX_MODEL_EXECUTION_DESCRIPTOR_SCHEMA_V1 ||
          !yvex_sha256_hex_valid(summary->model_execution.identity) ||
          strcmp(summary->logical_model_identity,
                 summary->model_execution.logical_model_identity) != 0)))
        return runtime_reject(
            failure, YVEX_RUNTIME_DESCRIPTOR_FAILURE_INVALID_ARGUMENT, NULL,
            YVEX_MATERIALIZATION_NO_INDEX, summary ? summary->tensor_count : 0ull,
            binding_count, err, YVEX_ERR_INVALID_ARG,
            "runtime binding descriptor records are incomplete or stale");
    descriptor = runtime_descriptor_alloc(binding_count, failure, err);
    if (!descriptor) return yvex_error_code(err);
    descriptor->summary = *summary;
    yvex_core_text_copy(expected_identity, sizeof(expected_identity), summary->runtime_descriptor_identity);
    for (i = 0ull; i < binding_count; ++i) {
        const yvex_runtime_tensor_binding *record = &bindings[i];
        const yvex_materialized_tensor_binding *source =
            yvex_materialization_session_tensor_at(session, record->tensor_id);
        yvex_runtime_tensor_binding *row = &descriptor->bindings[i];
        if (!source || record->tensor_id != source->tensor_id ||
            record->descriptor_index != source->descriptor_index ||
            record->role != source->role || record->collection != source->collection ||
            record->scope != source->scope || record->layer_index != source->layer_index ||
            record->predictor_index != source->predictor_index ||
            record->qtype != source->qtype || record->placement != source->placement ||
            record->access_mode != source->access_mode) {
            yvex_runtime_descriptor_close(descriptor);
            return runtime_reject(
                failure, YVEX_RUNTIME_DESCRIPTOR_FAILURE_MATERIALIZATION,
                source ? source->name : NULL, i, 1ull, 0ull, err, YVEX_ERR_FORMAT,
                "runtime binding descriptor row disagrees with materialization");
        }
        *row = *record;
        row->binding = source;
        if (!runtime_index_insert(descriptor, source->name, i)) {
            yvex_runtime_descriptor_close(descriptor);
            return runtime_reject(
                failure, YVEX_RUNTIME_DESCRIPTOR_FAILURE_DUPLICATE_BINDING,
                source->name, i, 1ull, 2ull, err, YVEX_ERR_FORMAT,
                "runtime binding descriptor name is duplicated");
        }
    }
    runtime_compute_identity(descriptor);
    if (strcmp(descriptor->summary.runtime_descriptor_identity, expected_identity) != 0) {
        yvex_runtime_descriptor_close(descriptor);
        return runtime_reject(
            failure, YVEX_RUNTIME_DESCRIPTOR_FAILURE_ARCHITECTURE, NULL,
            YVEX_MATERIALIZATION_NO_INDEX, 1ull, 0ull, err, YVEX_ERR_FORMAT,
            "runtime binding descriptor identity disagrees with its records");
    }
    *out = descriptor;
    yvex_error_clear(err);
    return YVEX_OK;
}

void yvex_runtime_descriptor_close(yvex_runtime_descriptor *descriptor) {
    if (!descriptor) return;
    free(descriptor->bindings);
    free(descriptor->name_index);
    free(descriptor);
}

const yvex_runtime_descriptor_summary *yvex_runtime_descriptor_summary_get(
    const yvex_runtime_descriptor *descriptor) {
    return descriptor ? &descriptor->summary : NULL;
}

/*
 * Return one immutable runtime descriptor row at a checked canonical ordinal.
 *
 * The row remains valid only for the descriptor lifetime.
 */
const yvex_runtime_tensor_binding *yvex_runtime_descriptor_tensor_at(
    const yvex_runtime_descriptor *descriptor, unsigned long long index) {
    return descriptor && index < descriptor->count ? &descriptor->bindings[index] : NULL;
}

const yvex_runtime_tensor_binding *yvex_runtime_descriptor_find_role(
    const yvex_runtime_descriptor *descriptor,
    yvex_tensor_role role,
    yvex_tensor_scope scope,
    unsigned long long layer_index,
    unsigned long long predictor_index) {
    unsigned long long i;

    if (!descriptor) return NULL;
    for (i = 0ull; i < descriptor->count; ++i) {
        const yvex_runtime_tensor_binding *row = &descriptor->bindings[i];
        if (row->role == role && row->scope == scope &&
            row->layer_index == layer_index &&
            row->predictor_index == predictor_index)
            return row;
    }
    return NULL;
}
