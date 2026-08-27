/* Compile identity-bound physical facts consumed without backend topology reconstruction. */
#include <yvex/internal/execution.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <yvex/internal/backend.h>
#include <yvex/internal/core.h>
struct yvex_physical_execution_ir {
    yvex_physical_execution_decision *decisions;
    yvex_physical_execution_summary summary;
};
static int physical_ir_identity(yvex_physical_execution_ir *ir, yvex_error *err);
static int execution_refuse(yvex_error *err, yvex_status status,
                            const char *where, const char *reason)
{
    yvex_error_set(err, status, where, reason);
    return status;
}
static int execution_hash_finish(yvex_sha256 *hash,
                                 char output[YVEX_SHA256_HEX_CAP])
{
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    if (!yvex_sha256_final(hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}
static yvex_execution_consumer_class execution_consumer(yvex_tensor_role role)
{
    switch (role) {
    case YVEX_TENSOR_ROLE_TOKEN_EMBEDDING:
        return YVEX_EXECUTION_CONSUMER_EMBEDDING;
    case YVEX_TENSOR_ROLE_OUTPUT_NORM:
    case YVEX_TENSOR_ROLE_DRAFT_OUTPUT_NORM:
    case YVEX_TENSOR_ROLE_HC_HEAD_FUNCTION:
    case YVEX_TENSOR_ROLE_HC_HEAD_BASE:
    case YVEX_TENSOR_ROLE_HC_HEAD_SCALE:
        return YVEX_EXECUTION_CONSUMER_FINAL_NORMALIZATION;
    case YVEX_TENSOR_ROLE_OUTPUT_HEAD:
        return YVEX_EXECUTION_CONSUMER_OUTPUT_HEAD;
    case YVEX_TENSOR_ROLE_MOE_ROUTER:
    case YVEX_TENSOR_ROLE_MOE_ROUTER_BIAS:
    case YVEX_TENSOR_ROLE_MOE_ROUTER_TABLE:
        return YVEX_EXECUTION_CONSUMER_MOE_ROUTER;
    case YVEX_TENSOR_ROLE_MOE_EXPERT_GATE:
    case YVEX_TENSOR_ROLE_MOE_EXPERT_UP:
        return YVEX_EXECUTION_CONSUMER_ROUTED_GATE_UP;
    case YVEX_TENSOR_ROLE_MOE_EXPERT_DOWN:
        return YVEX_EXECUTION_CONSUMER_ROUTED_DOWN;
    case YVEX_TENSOR_ROLE_MOE_SHARED_EXPERT_GATE:
    case YVEX_TENSOR_ROLE_MOE_SHARED_EXPERT_UP:
    case YVEX_TENSOR_ROLE_MOE_SHARED_EXPERT_DOWN:
        return YVEX_EXECUTION_CONSUMER_SHARED_EXPERT;
    case YVEX_TENSOR_ROLE_DRAFT_FEATURE_PROJECTION:
    case YVEX_TENSOR_ROLE_DRAFT_FEATURE_NORM:
        return YVEX_EXECUTION_CONSUMER_DRAFT_FEATURE_PROJECTION;
    case YVEX_TENSOR_ROLE_DRAFT_MARKOV_EMBEDDING:
    case YVEX_TENSOR_ROLE_DRAFT_MARKOV_OUTPUT:
        return YVEX_EXECUTION_CONSUMER_MARKOV;
    case YVEX_TENSOR_ROLE_DRAFT_CONFIDENCE:
        return YVEX_EXECUTION_CONSUMER_CONFIDENCE;
    case YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_KV:
    case YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_GATE:
    case YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_APE:
    case YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_NORM:
    case YVEX_TENSOR_ROLE_INDEXER_PROJECTION:
    case YVEX_TENSOR_ROLE_INDEXER_ATTENTION_Q_B:
    case YVEX_TENSOR_ROLE_INDEXER_COMPRESSOR_KV:
    case YVEX_TENSOR_ROLE_INDEXER_COMPRESSOR_GATE:
    case YVEX_TENSOR_ROLE_INDEXER_COMPRESSOR_APE:
    case YVEX_TENSOR_ROLE_INDEXER_COMPRESSOR_NORM:
        return YVEX_EXECUTION_CONSUMER_ATTENTION_STATE;
    default:
        return role >= YVEX_TENSOR_ROLE_ATTENTION_NORM &&
                       role <= YVEX_TENSOR_ROLE_HC_FFN_SCALE
                   ? YVEX_EXECUTION_CONSUMER_ATTENTION_PROJECTION
                   : YVEX_EXECUTION_CONSUMER_DRAFT_BACKBONE;
    }
}
static int physical_execution_decision_seal(
    yvex_physical_execution_decision *decision, yvex_error *err)
{
    yvex_sha256 hash;
    if (!decision ||
        decision->schema_version != YVEX_PHYSICAL_EXECUTION_SCHEMA_V5 ||
        decision->role <= YVEX_TENSOR_ROLE_UNKNOWN ||
        decision->role >= YVEX_TENSOR_ROLE_COUNT ||
        decision->scope > YVEX_TENSOR_SCOPE_DRAFT ||
        decision->consumer >= YVEX_EXECUTION_CONSUMER_COUNT ||
        decision->layout > YVEX_EXECUTION_LAYOUT_EXPERT_MAJOR ||
        decision->sharing > YVEX_EXECUTION_SHARING_ALIAS ||
        !decision->canonical_row_width || !decision->canonical_row_count ||
        !decision->encoded_bytes || !decision->alignment ||
        !yvex_sha256_hex_valid(decision->terminal_identity))
        return execution_refuse(
            err, YVEX_ERR_INVALID_ARG, "runtime.execution.physical",
            "complete physical execution decision facts are required");
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.physical-execution.decision.v5") ||
        !yvex_sha256_update_u64(&hash, decision->schema_version) ||
        !yvex_sha256_update_u64(&hash, decision->terminal_tensor_id) ||
        !yvex_sha256_update_u64(&hash, decision->role) ||
        !yvex_sha256_update_u64(&hash, decision->scope) ||
        !yvex_sha256_update_u64(&hash, decision->layer_index) ||
        !yvex_sha256_update_u64(&hash, decision->predictor_index) ||
        !yvex_sha256_update_u64(&hash, decision->expert_count) ||
        !yvex_sha256_update_u64(&hash, decision->canonical_qtype) ||
        !yvex_sha256_update_u64(&hash, decision->canonical_row_width) ||
        !yvex_sha256_update_u64(&hash, decision->canonical_row_count) ||
        !yvex_sha256_update_u64(&hash, decision->encoded_offset) ||
        !yvex_sha256_update_u64(&hash, decision->encoded_bytes) ||
        !yvex_sha256_update_u64(&hash, decision->alignment) ||
        !yvex_sha256_update_u64(&hash, decision->consumer) ||
        !yvex_sha256_update_u64(&hash, decision->layout) ||
        !yvex_sha256_update_u64(&hash, decision->sharing) ||
        !yvex_sha256_update_text(&hash, decision->terminal_identity) ||
        !execution_hash_finish(&hash, decision->decision_identity))
        return execution_refuse(
            err, YVEX_ERR_STATE, "runtime.execution.physical",
            "physical execution decision identity derivation failed");
    yvex_error_clear(err);
    return YVEX_OK;
}
static void execution_decision_from_binding(
    yvex_physical_execution_decision *decision,
    const yvex_runtime_tensor_binding *binding,
    const yvex_materialized_tensor_binding *physical)
{
    memset(decision, 0, sizeof(*decision));
    decision->schema_version = YVEX_PHYSICAL_EXECUTION_SCHEMA_V5;
    decision->terminal_tensor_id = binding->tensor_id;
    decision->role = binding->role;
    decision->scope = binding->scope;
    decision->layer_index = binding->layer_index;
    decision->predictor_index = binding->predictor_index;
    decision->expert_count = physical->expert_count;
    decision->canonical_qtype = binding->qtype;
    decision->canonical_row_width = physical->row_width;
    decision->canonical_row_count = physical->row_count;
    decision->encoded_offset = physical->absolute_offset;
    decision->encoded_bytes = physical->encoded_bytes;
    decision->alignment = physical->alignment;
    decision->consumer = execution_consumer(binding->role);
    decision->layout = physical->expert_count > 1ull
                           ? YVEX_EXECUTION_LAYOUT_EXPERT_MAJOR
                           : YVEX_EXECUTION_LAYOUT_CANONICAL_ROW;
    decision->sharing = binding->scope == YVEX_TENSOR_SCOPE_GLOBAL
                            ? YVEX_EXECUTION_SHARING_MODEL_READ_ONLY
                            : YVEX_EXECUTION_SHARING_EXCLUSIVE;
    {
        yvex_sha256 hash;
        yvex_sha256_init(&hash);
        if (yvex_sha256_update_text(&hash, "yvex.terminal-tensor.v1") &&
            yvex_sha256_update_text(&hash, physical->name) &&
            yvex_sha256_update_u64(&hash, binding->tensor_id) &&
            yvex_sha256_update_u64(&hash, binding->role) &&
            yvex_sha256_update_u64(&hash, binding->scope))
            (void)execution_hash_finish(&hash, decision->terminal_identity);
    }
}
typedef int (*physical_binding_at_fn)(
    const void *, unsigned long long, const yvex_runtime_tensor_binding **,
    const yvex_materialized_tensor_binding **);
static int physical_ir_compile(
    yvex_physical_execution_ir **out, unsigned long long count,
    const char *physical_variant_identity, physical_binding_at_fn binding_at,
    const void *context, yvex_error *err)
{
    yvex_physical_execution_ir *ir;
    unsigned long long index;
    int rc = YVEX_OK;
    if (out) *out = NULL;
    if (!out || !count || !yvex_sha256_hex_valid(physical_variant_identity) ||
        count > SIZE_MAX / sizeof(*ir->decisions) || !binding_at)
        return execution_refuse(
            err, YVEX_ERR_INVALID_ARG, "runtime.execution.physical",
            "physical execution compiler inputs are incomplete");
    ir = calloc(1u, sizeof(*ir));
    if (ir) ir->decisions = calloc((size_t)count, sizeof(*ir->decisions));
    if (!ir || !ir->decisions) {
        yvex_physical_execution_ir_close(&ir);
        return execution_refuse(err, YVEX_ERR_NOMEM,
                                "runtime.execution.physical",
                                "physical execution IR allocation failed");
    }
    ir->summary.schema_version = YVEX_PHYSICAL_EXECUTION_SCHEMA_V5;
    ir->summary.decision_count = count;
    yvex_core_text_copy(ir->summary.physical_variant_identity,
                        sizeof(ir->summary.physical_variant_identity),
                        physical_variant_identity);
    for (index = 0ull; index < count; ++index) {
        const yvex_runtime_tensor_binding *binding = NULL;
        const yvex_materialized_tensor_binding *physical = NULL;
        yvex_physical_execution_decision *decision = &ir->decisions[index];
        if (!binding_at(context, index, &binding, &physical) || !binding || !physical) {
            rc = execution_refuse(err, YVEX_ERR_FORMAT,
                                  "runtime.execution.physical",
                                  "runtime descriptor contains an orphan tensor");
            break;
        }
        execution_decision_from_binding(decision, binding, physical);
        rc = physical_execution_decision_seal(decision, err);
        if (rc != YVEX_OK) break;
        ir->summary.consumer_counts[decision->consumer]++;
        ir->summary.layout_counts[decision->layout]++;
        if (!yvex_core_u64_add(ir->summary.encoded_bytes,
                               decision->encoded_bytes,
                               &ir->summary.encoded_bytes)) {
            rc = execution_refuse(err, YVEX_ERR_BOUNDS,
                                  "runtime.execution.physical",
                                  "physical execution byte accounting overflowed");
            break;
        }
    }
    if (rc == YVEX_OK) rc = physical_ir_identity(ir, err);
    if (rc != YVEX_OK) {
        yvex_physical_execution_ir_close(&ir);
        return rc;
    }
    *out = ir;
    yvex_error_clear(err);
    return YVEX_OK;
}
typedef struct {
    const yvex_runtime_descriptor *descriptor;
} physical_descriptor_source;
static int physical_descriptor_binding_at(
    const void *opaque, unsigned long long index,
    const yvex_runtime_tensor_binding **binding,
    const yvex_materialized_tensor_binding **physical)
{
    const physical_descriptor_source *source = opaque;
    *binding = source ? yvex_runtime_descriptor_tensor_at(source->descriptor, index) : NULL;
    *physical = *binding ? (*binding)->binding : NULL;
    return *binding && *physical;
}
static int physical_ir_identity(yvex_physical_execution_ir *ir,
                                yvex_error *err)
{
    yvex_sha256 hash;
    unsigned long long index;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.physical-execution.ir.v5") ||
        !yvex_sha256_update_u64(&hash, ir->summary.schema_version) ||
        !yvex_sha256_update_text(&hash,
                                 ir->summary.physical_variant_identity) ||
        !yvex_sha256_update_u64(&hash, ir->summary.decision_count) ||
        !yvex_sha256_update_u64(&hash, ir->summary.encoded_bytes))
        goto failed;
    for (index = 0ull; index < ir->summary.decision_count; ++index)
        if (!yvex_sha256_update_text(
                &hash, ir->decisions[index].decision_identity))
            goto failed;
    if (!execution_hash_finish(&hash, ir->summary.identity)) goto failed;
    return YVEX_OK;
failed:
    return execution_refuse(err, YVEX_ERR_STATE,
                            "runtime.execution.physical",
                            "physical execution IR identity derivation failed");
}
int yvex_physical_execution_ir_build(
    yvex_physical_execution_ir **out,
    const yvex_materialization_session *materialization,
    const yvex_runtime_descriptor *descriptor,
    const char *physical_variant_identity, yvex_error *err)
{
    const yvex_runtime_descriptor_summary *descriptor_summary;
    const yvex_materialization_summary *materialization_summary;
    physical_descriptor_source source;
    if (out) *out = NULL;
    descriptor_summary = yvex_runtime_descriptor_summary_get(descriptor);
    materialization_summary =
        yvex_materialization_session_summary(materialization);
    if (!out || !descriptor_summary || !materialization_summary ||
        descriptor_summary->status != YVEX_RUNTIME_DESCRIPTOR_STATUS_READY ||
        materialization_summary->status != YVEX_MATERIALIZATION_STATUS_COMMITTED ||
        descriptor_summary->tensor_count != materialization_summary->tensor_count ||
        !descriptor_summary->tensor_count ||
        descriptor_summary->model_execution.schema_version !=
            YVEX_MODEL_EXECUTION_DESCRIPTOR_SCHEMA_V1 ||
        !yvex_sha256_hex_valid(physical_variant_identity))
        return execution_refuse(
            err, YVEX_ERR_INVALID_ARG, "runtime.execution.physical",
            "admitted descriptor, materialization, and physical variant are required");
    source.descriptor = descriptor;
    return physical_ir_compile(
        out, descriptor_summary->tensor_count, physical_variant_identity,
        physical_descriptor_binding_at, &source, err);
}

int yvex_physical_execution_ir_import(
    yvex_physical_execution_ir **out, const yvex_physical_execution_summary *summary,
    const yvex_physical_execution_decision *decisions, unsigned long long count,
    yvex_error *err)
{
    yvex_physical_execution_ir *ir;
    unsigned long long index;
    char expected[YVEX_SHA256_HEX_CAP];
    int authenticated;
    if (out) *out = NULL;
    if (!out || !summary || !decisions || !count || summary->decision_count != count ||
        summary->schema_version != YVEX_PHYSICAL_EXECUTION_SCHEMA_V5 ||
        !yvex_sha256_hex_valid(summary->physical_variant_identity) ||
        count > SIZE_MAX / sizeof(*ir->decisions))
        return execution_refuse(err, YVEX_ERR_INVALID_ARG,
                                "runtime.execution.physical",
                                "sealed physical execution records are required");
    authenticated = summary->identity[0] != '\0';
    ir = calloc(1u, sizeof(*ir));
    if (ir) ir->decisions = calloc((size_t)count, sizeof(*ir->decisions));
    if (!ir || !ir->decisions) {
        yvex_physical_execution_ir_close(&ir);
        return execution_refuse(err, YVEX_ERR_NOMEM,
                                "runtime.execution.physical",
                                "physical execution import allocation failed");
    }
    ir->summary.schema_version = summary->schema_version;
    ir->summary.decision_count = count;
    yvex_core_text_copy(ir->summary.physical_variant_identity,
                        sizeof(ir->summary.physical_variant_identity),
                        summary->physical_variant_identity);
    for (index = 0ull; index < count; ++index) {
        ir->decisions[index] = decisions[index];
        yvex_core_text_copy(expected, sizeof(expected), decisions[index].decision_identity);
        if (physical_execution_decision_seal(&ir->decisions[index], err) != YVEX_OK ||
            (authenticated && strcmp(expected, ir->decisions[index].decision_identity) != 0))
            goto refused;
        ir->summary.consumer_counts[ir->decisions[index].consumer]++;
        ir->summary.layout_counts[ir->decisions[index].layout]++;
        if (!yvex_core_u64_add(ir->summary.encoded_bytes,
                               ir->decisions[index].encoded_bytes,
                               &ir->summary.encoded_bytes))
            goto refused;
    }
    if (authenticated &&
        (ir->summary.encoded_bytes != summary->encoded_bytes ||
         memcmp(ir->summary.consumer_counts, summary->consumer_counts,
                sizeof(ir->summary.consumer_counts)) != 0 ||
         memcmp(ir->summary.layout_counts, summary->layout_counts,
                sizeof(ir->summary.layout_counts)) != 0))
        goto refused;
    yvex_core_text_copy(expected, sizeof(expected), summary->identity);
    if (physical_ir_identity(ir, err) != YVEX_OK ||
        (authenticated && strcmp(expected, ir->summary.identity) != 0))
        goto refused;
    *out = ir;
    return YVEX_OK;
refused:
    yvex_physical_execution_ir_close(&ir);
    return execution_refuse(err, YVEX_ERR_FORMAT,
                            "runtime.execution.physical",
                            "physical execution records failed identity validation");
}
const yvex_physical_execution_summary *yvex_physical_execution_ir_summary(
    const yvex_physical_execution_ir *ir)
{
    return ir ? &ir->summary : NULL;
}
const yvex_physical_execution_decision *yvex_physical_execution_ir_decision_at(
    const yvex_physical_execution_ir *ir, unsigned long long index)
{
    return ir && index < ir->summary.decision_count ? &ir->decisions[index] : NULL;
}
void yvex_physical_execution_ir_close(yvex_physical_execution_ir **ir)
{
    if (!ir || !*ir) return;
    free((*ir)->decisions);
    memset(*ir, 0, sizeof(**ir));
    free(*ir);
    *ir = NULL;
}
int yvex_execution_hardware_profile_seal(
    yvex_execution_hardware_profile *profile, yvex_error *err)
{
    yvex_sha256 hash;
    unsigned long long allowed =
        (1ull << YVEX_EXECUTION_HARDWARE_FACT_COUNT) - 1ull;
    int has_memory, has_bandwidth, has_topology, has_paging, has_native;
    if (profile) {
        has_memory = (profile->admitted_fact_mask &
                      YVEX_EXECUTION_HARDWARE_FACT_BIT(
                          YVEX_EXECUTION_HARDWARE_FACT_MEMORY)) != 0ull;
        has_bandwidth = (profile->admitted_fact_mask &
                         YVEX_EXECUTION_HARDWARE_FACT_BIT(
                             YVEX_EXECUTION_HARDWARE_FACT_BANDWIDTH)) != 0ull;
        has_topology = (profile->admitted_fact_mask &
                        YVEX_EXECUTION_HARDWARE_FACT_BIT(
                            YVEX_EXECUTION_HARDWARE_FACT_TOPOLOGY)) != 0ull;
        has_paging = (profile->admitted_fact_mask &
                      YVEX_EXECUTION_HARDWARE_FACT_BIT(
                          YVEX_EXECUTION_HARDWARE_FACT_PAGING)) != 0ull;
        has_native = (profile->admitted_fact_mask &
                      YVEX_EXECUTION_HARDWARE_FACT_BIT(
                          YVEX_EXECUTION_HARDWARE_FACT_NATIVE_CODE)) != 0ull;
    } else {
        has_memory = has_bandwidth = has_topology = has_paging = has_native = 0;
    }
    if (!profile ||
        profile->schema_version != YVEX_EXECUTION_HARDWARE_PROFILE_SCHEMA_V1 ||
        profile->backend > YVEX_BACKEND_KIND_CUDA || !profile->name[0] ||
        strnlen(profile->name, sizeof(profile->name)) >= sizeof(profile->name) ||
        !profile->admitted_fact_mask || profile->admitted_fact_mask & ~allowed ||
        (has_memory && (!profile->total_memory_bytes ||
                        !profile->usable_memory_bytes ||
                        profile->usable_memory_bytes > profile->total_memory_bytes)) ||
        (has_bandwidth && (!profile->sustainable_read_bytes_per_second ||
                           !profile->sustainable_copy_bytes_per_second)) ||
        (has_paging && (!profile->host_page_bytes ||
                        !profile->device_page_bytes)) ||
        (profile->unified_addressing != 0 && profile->unified_addressing != 1) ||
        (profile->coherent_host_memory != 0 && profile->coherent_host_memory != 1) ||
        (profile->virtual_memory != 0 && profile->virtual_memory != 1) ||
        (profile->graph_capture != 0 && profile->graph_capture != 1) ||
        (profile->native_architecture_code != 0 &&
         profile->native_architecture_code != 1) ||
        (profile->backend == YVEX_BACKEND_KIND_CUDA &&
         (!profile->compute_major || (has_topology && !profile->sm_count))) ||
        (has_native && (!profile->native_architecture_code ||
                        profile->backend != YVEX_BACKEND_KIND_CUDA)) ||
        (!has_native && profile->native_architecture_code))
        return execution_refuse(err, YVEX_ERR_INVALID_ARG,
                                "runtime.execution.hardware",
                                "complete measured hardware profile facts are required");
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.execution-hardware-profile.v1") ||
        !yvex_sha256_update_u64(&hash, profile->schema_version) ||
        !yvex_sha256_update_text(&hash, profile->name) ||
        !yvex_sha256_update_u64(&hash, profile->backend) ||
        !yvex_sha256_update_u64(&hash, profile->admitted_fact_mask) ||
        !yvex_sha256_update_u64(&hash, (unsigned long long)profile->device_index) ||
        !yvex_sha256_update_u64(&hash, (unsigned long long)profile->compute_major) ||
        !yvex_sha256_update_u64(&hash, (unsigned long long)profile->compute_minor) ||
        !yvex_sha256_update_u64(&hash, profile->sm_count) ||
        !yvex_sha256_update_u64(&hash, profile->copy_engine_count) ||
        !yvex_sha256_update_u64(&hash, profile->l2_bytes) ||
        !yvex_sha256_update_u64(&hash, profile->total_memory_bytes) ||
        !yvex_sha256_update_u64(&hash, profile->usable_memory_bytes) ||
        !yvex_sha256_update_u64(&hash, profile->sustainable_read_bytes_per_second) ||
        !yvex_sha256_update_u64(&hash, profile->sustainable_copy_bytes_per_second) ||
        !yvex_sha256_update_u64(&hash, profile->host_page_bytes) ||
        !yvex_sha256_update_u64(&hash, profile->device_page_bytes) ||
        !yvex_sha256_update_u64(&hash, (unsigned long long)profile->unified_addressing) ||
        !yvex_sha256_update_u64(&hash, (unsigned long long)profile->coherent_host_memory) ||
        !yvex_sha256_update_u64(&hash, (unsigned long long)profile->virtual_memory) ||
        !yvex_sha256_update_u64(&hash, (unsigned long long)profile->graph_capture) ||
        !yvex_sha256_update_u64(&hash,
                                (unsigned long long)profile->native_architecture_code) ||
        !execution_hash_finish(&hash, profile->identity))
        return execution_refuse(err, YVEX_ERR_STATE,
                                "runtime.execution.hardware",
                                "hardware profile identity derivation failed");
    yvex_error_clear(err);
    return YVEX_OK;
}
int yvex_execution_workload_profile_seal(
    yvex_execution_workload_profile *profile, yvex_error *err)
{
    yvex_sha256 hash;
    if (!profile ||
        profile->schema_version != YVEX_EXECUTION_WORKLOAD_PROFILE_SCHEMA_V1 ||
        profile->kind > YVEX_EXECUTION_WORKLOAD_FULL_MODEL_RESEARCH ||
        !profile->name[0] ||
        strnlen(profile->name, sizeof(profile->name)) >= sizeof(profile->name) ||
        !profile->minimum_session_context || !profile->requested_session_context ||
        profile->minimum_session_context > profile->requested_session_context ||
        !profile->concurrent_sequences || !profile->logical_batch_tokens ||
        !profile->prefill_chunk_tokens || !profile->attention_microbatch_rows ||
        !profile->moe_row_tile || !profile->output_head_rows ||
        (profile->latency_priority != 0 && profile->latency_priority != 1) ||
        (profile->continuous_batching != 0 && profile->continuous_batching != 1) ||
        (profile->prefix_sharing != 0 && profile->prefix_sharing != 1) ||
        (profile->durable_state != 0 && profile->durable_state != 1) ||
        profile->system_reserve_bytes < YVEX_EXECUTION_MINIMUM_SYSTEM_RESERVE)
        return execution_refuse(err, YVEX_ERR_INVALID_ARG,
                                "runtime.execution.workload",
                                "complete bounded workload profile facts are required");
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.execution-workload-profile.v1") ||
        !yvex_sha256_update_u64(&hash, profile->schema_version) ||
        !yvex_sha256_update_text(&hash, profile->name) ||
        !yvex_sha256_update_u64(&hash, profile->kind) ||
        !yvex_sha256_update_u64(&hash, profile->minimum_session_context) ||
        !yvex_sha256_update_u64(&hash, profile->requested_session_context) ||
        !yvex_sha256_update_u64(&hash, profile->concurrent_sequences) ||
        !yvex_sha256_update_u64(&hash, profile->logical_batch_tokens) ||
        !yvex_sha256_update_u64(&hash, profile->prefill_chunk_tokens) ||
        !yvex_sha256_update_u64(&hash, profile->attention_microbatch_rows) ||
        !yvex_sha256_update_u64(&hash, profile->moe_row_tile) ||
        !yvex_sha256_update_u64(&hash, profile->output_head_rows) ||
        !yvex_sha256_update_u64(&hash, profile->prefix_cache_bytes) ||
        !yvex_sha256_update_u64(&hash, profile->persistent_state_bytes) ||
        !yvex_sha256_update_u64(&hash, profile->system_reserve_bytes) ||
        !yvex_sha256_update_u64(&hash, (unsigned long long)profile->latency_priority) ||
        !yvex_sha256_update_u64(&hash, (unsigned long long)profile->continuous_batching) ||
        !yvex_sha256_update_u64(&hash, (unsigned long long)profile->prefix_sharing) ||
        !yvex_sha256_update_u64(&hash, (unsigned long long)profile->durable_state) ||
        !execution_hash_finish(&hash, profile->identity))
        return execution_refuse(err, YVEX_ERR_STATE,
                                "runtime.execution.workload",
                                "workload profile identity derivation failed");
    yvex_error_clear(err);
    return YVEX_OK;
}
static int capacity_add(unsigned long long *total, unsigned long long value)
{
    return yvex_core_u64_add(*total, value, total);
}
static unsigned long long capacity_gcd(unsigned long long left,
                                       unsigned long long right)
{
    while (right) {
        unsigned long long remainder = left % right;
        left = right;
        right = remainder;
    }
    return left;
}
static int capacity_lcm(unsigned long long left, unsigned long long right,
                        unsigned long long *result)
{
    unsigned long long divisor;
    if (!left || !right || !result) return 0;
    divisor = capacity_gcd(left, right);
    return yvex_core_u64_mul(left / divisor, right, result);
}
static int capacity_ceil_div(unsigned long long value, unsigned long long divisor,
                             unsigned long long *result)
{
    unsigned long long adjusted;
    if (!divisor || !result ||
        !yvex_core_u64_add(value, divisor - 1ull, &adjusted)) return 0;
    *result = adjusted / divisor;
    return 1;
}
/* Compute one exact scaled ratio without requiring the numerator product to fit. */
static int capacity_mul_div(unsigned long long left, unsigned long long right,
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
            if (!capacity_add(&quotient, 1ull)) return 0;
        } else {
            remainder += remainder;
        }
        if (!(left & (1ull << bit))) continue;
        if (!capacity_add(&quotient, right_quotient)) return 0;
        if (right_remainder && remainder >= divisor - right_remainder) {
            remainder -= divisor - right_remainder;
            if (!capacity_add(&quotient, 1ull)) return 0;
        } else {
            remainder += right_remainder;
        }
    }
    if (round_up && remainder && !capacity_add(&quotient, 1ull)) return 0;
    *result = quotient;
    return 1;
}
static int capacity_state_request_valid(
    const yvex_execution_state_class_request *request)
{
    if (!request || request->state_class >= YVEX_MODEL_STATE_CLASS_COUNT ||
        request->extent > YVEX_EXECUTION_STATE_EXTENT_PREFIX_BUDGET ||
        !request->logical_block_tokens || !request->bytes_per_block ||
        !request->alignment_bytes ||
        !request->kernel_tile_tokens || !request->promotion_granularity_tokens ||
        !request->page_table_entry_bytes ||
        (request->shared != 0 && request->shared != 1) ||
        (request->copy_on_write != 0 && request->copy_on_write != 1) ||
        (request->extent == YVEX_EXECUTION_STATE_EXTENT_FIXED &&
         !request->fixed_tokens_per_sequence) ||
        (request->extent != YVEX_EXECUTION_STATE_EXTENT_FIXED &&
         request->fixed_tokens_per_sequence)) return 0;
    if ((request->state_class == YVEX_MODEL_STATE_CANDIDATE_DELTA) !=
        (request->extent == YVEX_EXECUTION_STATE_EXTENT_CANDIDATE)) return 0;
    if ((request->state_class == YVEX_MODEL_STATE_PREFIX_CHECKPOINT) !=
        (request->extent == YVEX_EXECUTION_STATE_EXTENT_PREFIX_BUDGET)) return 0;
    return 1;
}
static int capacity_state_geometry(
    const yvex_execution_state_class_request *request,
    unsigned long long candidate_width,
    const yvex_execution_hardware_profile *hardware,
    const yvex_execution_workload_profile *workload,
    yvex_execution_state_class_plan *plan)
{
    unsigned long long alignment_blocks, alignment_tokens, base_tokens, candidate_tokens;
    unsigned long long maximum_tokens, logical_tokens, best_score = ULLONG_MAX;
    unsigned long long best_tokens = 0ull;
    if (!capacity_state_request_valid(request)) return 0;
    if (!capacity_lcm(request->logical_block_tokens,
                      request->kernel_tile_tokens,
                      &base_tokens) ||
        !capacity_lcm(base_tokens, request->promotion_granularity_tokens,
                      &base_tokens)) return 0;
    alignment_blocks =
        request->alignment_bytes /
        capacity_gcd(request->bytes_per_block, request->alignment_bytes);
    if (!yvex_core_u64_mul(request->logical_block_tokens, alignment_blocks,
                           &alignment_tokens) ||
        !capacity_lcm(base_tokens, alignment_tokens, &base_tokens)) return 0;
    maximum_tokens = hardware->device_page_bytes / request->bytes_per_block;
    if (!yvex_core_u64_mul(maximum_tokens, request->logical_block_tokens,
                           &maximum_tokens)) return 0;
    maximum_tokens -= maximum_tokens % base_tokens;
    if (maximum_tokens < base_tokens) maximum_tokens = base_tokens;
    logical_tokens = workload->requested_session_context;
    if (request->extent == YVEX_EXECUTION_STATE_EXTENT_FIXED)
        logical_tokens = request->fixed_tokens_per_sequence;
    else if (request->extent == YVEX_EXECUTION_STATE_EXTENT_CANDIDATE)
        logical_tokens = candidate_width;
    for (candidate_tokens = base_tokens; candidate_tokens <= maximum_tokens;) {
        unsigned long long pages, rounded_tokens, blocks, page_bytes, table_bytes;
        unsigned long long fragmentation, cow_bytes, promotion_bytes, score;
        if (!capacity_ceil_div(logical_tokens, candidate_tokens, &pages) ||
            !yvex_core_u64_mul(pages, candidate_tokens, &rounded_tokens) ||
            candidate_tokens % request->logical_block_tokens ||
            !(blocks = candidate_tokens / request->logical_block_tokens) ||
            !yvex_core_u64_mul(blocks, request->bytes_per_block, &page_bytes)) return 0;
        if (page_bytes % request->alignment_bytes) goto next_candidate;
        if (!yvex_core_u64_mul(pages, request->page_table_entry_bytes,
                               &table_bytes) ||
            !capacity_ceil_div(rounded_tokens - logical_tokens,
                               request->logical_block_tokens, &blocks) ||
            !yvex_core_u64_mul(blocks, request->bytes_per_block,
                               &fragmentation)) return 0;
        cow_bytes = request->copy_on_write ? page_bytes : 0ull;
        promotion_bytes = request->extent == YVEX_EXECUTION_STATE_EXTENT_CANDIDATE
                              ? fragmentation : 0ull;
        score = 0ull;
        if (!capacity_add(&score, table_bytes) ||
            !capacity_add(&score, fragmentation) ||
            !capacity_add(&score, cow_bytes) ||
            !capacity_add(&score, promotion_bytes)) return 0;
        if (score < best_score || (score == best_score && candidate_tokens < best_tokens)) {
            best_score = score;
            best_tokens = candidate_tokens;
        }
next_candidate:
        if (maximum_tokens - candidate_tokens < base_tokens) break;
        candidate_tokens += base_tokens;
    }
    if (!best_tokens || best_tokens % request->logical_block_tokens ||
        !yvex_core_u64_mul(best_tokens / request->logical_block_tokens,
                           request->bytes_per_block, &plan->page_bytes)) return 0;
    plan->state_class = request->state_class;
    plan->extent = request->extent;
    plan->logical_block_tokens = request->logical_block_tokens;
    plan->bytes_per_block = request->bytes_per_block;
    plan->page_tokens = best_tokens;
    plan->shared = request->shared;
    plan->copy_on_write = request->copy_on_write;
    return 1;
}
static int capacity_state_usage(
    const yvex_execution_capacity_plan_request *request,
    const yvex_execution_state_class_plan *geometry,
    unsigned long long admitted_context,
    yvex_execution_state_class_plan *output,
    unsigned long long *state_bytes, unsigned long long *candidate_bytes)
{
    unsigned long long index;
    *state_bytes = 0ull;
    *candidate_bytes = 0ull;
    for (index = 0ull; index < request->state_class_count; ++index) {
        const yvex_execution_state_class_request *source = &request->state_classes[index];
        yvex_execution_state_class_plan record = geometry[index];
        unsigned long long per_sequence_tokens = admitted_context;
        unsigned long long pages_per_sequence, pages, rounded_tokens, data_bytes;
        if (source->extent == YVEX_EXECUTION_STATE_EXTENT_FIXED)
            per_sequence_tokens = source->fixed_tokens_per_sequence;
        else if (source->extent == YVEX_EXECUTION_STATE_EXTENT_CANDIDATE)
            per_sequence_tokens = request->candidate_width;
        if (source->extent == YVEX_EXECUTION_STATE_EXTENT_PREFIX_BUDGET) {
            unsigned long long bytes_per_page;
            if (!yvex_core_u64_add(record.page_bytes, source->page_table_entry_bytes,
                                   &bytes_per_page)) return 0;
            record.page_count = request->workload->prefix_cache_bytes / bytes_per_page;
            if (!yvex_core_u64_mul(record.page_count, record.page_tokens,
                                   &record.pool_tokens) ||
                !yvex_core_u64_mul(record.page_count, source->page_table_entry_bytes,
                                   &record.page_table_bytes)) return 0;
            record.tokens_per_sequence = 0ull;
            record.pool_bytes = request->workload->prefix_cache_bytes;
            record.copy_on_write_tail_bytes = record.copy_on_write ? record.page_bytes : 0ull;
            if (output) output[index] = record;
            continue;
        }
        if (!capacity_ceil_div(per_sequence_tokens, record.page_tokens,
                               &pages_per_sequence) ||
            !yvex_core_u64_mul(pages_per_sequence,
                               request->workload->concurrent_sequences, &pages) ||
            !yvex_core_u64_mul(pages, record.page_tokens, &rounded_tokens) ||
            rounded_tokens % record.logical_block_tokens ||
            !yvex_core_u64_mul(rounded_tokens / record.logical_block_tokens,
                               record.bytes_per_block, &data_bytes) ||
            !yvex_core_u64_mul(pages, source->page_table_entry_bytes,
                               &record.page_table_bytes) ||
            !yvex_core_u64_add(data_bytes, record.page_table_bytes,
                               &record.pool_bytes) ||
            !yvex_core_u64_mul(per_sequence_tokens,
                               request->workload->concurrent_sequences,
                               &record.pool_tokens)) return 0;
        record.tokens_per_sequence = per_sequence_tokens;
        record.page_count = pages;
        if (!capacity_ceil_div(rounded_tokens - record.pool_tokens,
                               record.logical_block_tokens,
                               &data_bytes) ||
            !yvex_core_u64_mul(data_bytes, record.bytes_per_block,
                               &record.tail_fragmentation_bytes)) return 0;
        if (record.copy_on_write &&
            !yvex_core_u64_mul(record.page_bytes,
                               request->workload->concurrent_sequences,
                               &record.copy_on_write_tail_bytes)) return 0;
        if (source->extent == YVEX_EXECUTION_STATE_EXTENT_CANDIDATE) {
            record.promotion_fragmentation_bytes = record.tail_fragmentation_bytes;
            if (!capacity_add(candidate_bytes, record.pool_bytes)) return 0;
        } else if (!capacity_add(state_bytes, record.pool_bytes)) return 0;
        if (output) output[index] = record;
    }
    return 1;
}
static int capacity_request_validate(
    const yvex_execution_capacity_plan_request *request,
    const yvex_execution_hardware_profile **hardware,
    const yvex_execution_workload_profile **workload)
{
    unsigned long long index, seen = 0ull;
    if (!request || !(*hardware = request->hardware) ||
        !(*workload = request->workload) ||
        request->schema_version != YVEX_EXECUTION_CAPACITY_PLAN_SCHEMA_V1 ||
        !yvex_sha256_hex_valid(request->model_execution_identity) ||
        !request->semantic_maximum_context || !request->candidate_width ||
        !request->semantic_state_class_mask ||
        (request->semantic_state_class_mask &
         ~((1ull << YVEX_MODEL_STATE_CLASS_COUNT) - 1ull)) ||
        (*hardware)->schema_version != YVEX_EXECUTION_HARDWARE_PROFILE_SCHEMA_V1 ||
        (*workload)->schema_version != YVEX_EXECUTION_WORKLOAD_PROFILE_SCHEMA_V1 ||
        !yvex_sha256_hex_valid((*hardware)->identity) ||
        !yvex_sha256_hex_valid((*workload)->identity) || !request->model_bytes ||
        !((*hardware)->admitted_fact_mask &
          YVEX_EXECUTION_HARDWARE_FACT_BIT(YVEX_EXECUTION_HARDWARE_FACT_MEMORY)) ||
        !((*hardware)->admitted_fact_mask &
          YVEX_EXECUTION_HARDWARE_FACT_BIT(YVEX_EXECUTION_HARDWARE_FACT_PAGING)) ||
        !request->workspace_bytes || !request->state_classes ||
        !request->state_class_count ||
        request->state_class_count > YVEX_MODEL_STATE_CLASS_COUNT) return 0;
    for (index = 0ull; index < request->state_class_count; ++index) {
        const yvex_execution_state_class_request *state = &request->state_classes[index];
        unsigned long long bit;
        if (!capacity_state_request_valid(state) ||
            (index && state->state_class <= request->state_classes[index - 1ull].state_class))
            return 0;
        bit = YVEX_MODEL_STATE_CLASS_BIT(state->state_class);
        if ((seen & bit) || !(request->semantic_state_class_mask & bit)) return 0;
        seen |= bit;
    }
    if (seen != request->semantic_state_class_mask) return 0;
    if ((*workload)->prefix_sharing &&
        !(seen & YVEX_MODEL_STATE_CLASS_BIT(YVEX_MODEL_STATE_PREFIX_CHECKPOINT))) return 0;
    return 1;
}
static int capacity_plan_identity(yvex_execution_capacity_plan *plan)
{
    yvex_sha256 hash;
    unsigned long long index;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.execution-capacity-plan.v1") ||
        !yvex_sha256_update_text(&hash, plan->model_execution_identity) ||
        !yvex_sha256_update_text(&hash, plan->hardware_profile_identity) ||
        !yvex_sha256_update_text(&hash, plan->workload_profile_identity) ||
        !yvex_sha256_update_u64(&hash, plan->model_maximum_context) ||
        !yvex_sha256_update_u64(&hash, plan->admitted_execution_maximum) ||
        !yvex_sha256_update_u64(&hash, plan->total_logical_context_tokens) ||
        !yvex_sha256_update_u64(&hash, plan->candidate_reserve_tokens) ||
        !yvex_sha256_update_u64(&hash, plan->model_bytes) ||
        !yvex_sha256_update_u64(&hash, plan->derived_layout_bytes) ||
        !yvex_sha256_update_u64(&hash, plan->state_pool_bytes) ||
        !yvex_sha256_update_u64(&hash, plan->candidate_reserve_bytes) ||
        !yvex_sha256_update_u64(&hash, plan->workspace_bytes) ||
        !yvex_sha256_update_u64(&hash, plan->scheduler_bytes) ||
        !yvex_sha256_update_u64(&hash, plan->graph_bytes) ||
        !yvex_sha256_update_u64(&hash, plan->prefix_cache_bytes) ||
        !yvex_sha256_update_u64(&hash, plan->persistent_state_bytes) ||
        !yvex_sha256_update_u64(&hash, plan->system_reserve_bytes) ||
        !yvex_sha256_update_u64(&hash, plan->required_bytes) ||
        !yvex_sha256_update_u64(&hash, plan->state_class_count)) return 0;
    for (index = 0ull; index < plan->state_class_count; ++index) {
        const yvex_execution_state_class_plan *state = &plan->state_classes[index];
        if (!yvex_sha256_update_u64(&hash, state->state_class) ||
            !yvex_sha256_update_u64(&hash, state->extent) ||
            !yvex_sha256_update_u64(&hash, state->logical_block_tokens) ||
            !yvex_sha256_update_u64(&hash, state->bytes_per_block) ||
            !yvex_sha256_update_u64(&hash, state->page_tokens) ||
            !yvex_sha256_update_u64(&hash, state->page_bytes) ||
            !yvex_sha256_update_u64(&hash, state->pool_tokens) ||
            !yvex_sha256_update_u64(&hash, state->pool_bytes) ||
            !yvex_sha256_update_u64(&hash, state->page_table_bytes) ||
            !yvex_sha256_update_u64(&hash, state->shared) ||
            !yvex_sha256_update_u64(&hash, state->copy_on_write)) return 0;
    }
    return execution_hash_finish(&hash, plan->identity);
}
int yvex_execution_capacity_plan_build(
    const yvex_execution_capacity_plan_request *request,
    yvex_execution_capacity_plan *plan, yvex_error *err)
{
    const yvex_execution_hardware_profile *hardware;
    const yvex_execution_workload_profile *workload;
    yvex_execution_state_class_plan geometry[YVEX_MODEL_STATE_CLASS_COUNT] = {{0}};
    unsigned long long fixed_bytes = 0ull, low, high, admitted = 0ull, index;
    unsigned long long state_bytes, candidate_bytes, required_bytes;
    if (plan) memset(plan, 0, sizeof(*plan));
    if (!plan || !capacity_request_validate(request, &hardware, &workload))
        return execution_refuse(err, YVEX_ERR_INVALID_ARG,
                                "runtime.execution.capacity",
                                "complete per-state-class planning facts are required");
    for (index = 0ull; index < request->state_class_count; ++index)
        if (!capacity_state_geometry(&request->state_classes[index],
                                     request->candidate_width, hardware,
                                     workload, &geometry[index]))
            return execution_refuse(err, YVEX_ERR_BOUNDS,
                                    "runtime.execution.capacity",
                                    "state-class page geometry cannot be represented");
    if (!capacity_add(&fixed_bytes, request->model_bytes) ||
        !capacity_add(&fixed_bytes, request->derived_layout_bytes) ||
        !capacity_add(&fixed_bytes, request->workspace_bytes) ||
        !capacity_add(&fixed_bytes, request->scheduler_bytes) ||
        !capacity_add(&fixed_bytes, request->graph_bytes) ||
        !capacity_add(&fixed_bytes, workload->prefix_cache_bytes) ||
        !capacity_add(&fixed_bytes, workload->persistent_state_bytes) ||
        !capacity_add(&fixed_bytes, workload->system_reserve_bytes))
        return execution_refuse(err, YVEX_ERR_BOUNDS,
                                "runtime.execution.capacity",
                                "fixed capacity byte accounting overflowed");
    if (fixed_bytes >= hardware->usable_memory_bytes)
        return execution_refuse(err, YVEX_ERR_BOUNDS,
                                "runtime.execution.capacity",
                                "fixed runtime resources exceed usable memory");
    low = workload->minimum_session_context;
    high = workload->requested_session_context < request->semantic_maximum_context
               ? workload->requested_session_context
               : request->semantic_maximum_context;
    while (low <= high) {
        unsigned long long middle = low + (high - low) / 2ull;
        if (!capacity_state_usage(request, geometry, middle, NULL,
                                  &state_bytes, &candidate_bytes) ||
            !yvex_core_u64_add(fixed_bytes, state_bytes, &required_bytes) ||
            !capacity_add(&required_bytes, candidate_bytes))
            return execution_refuse(err, YVEX_ERR_BOUNDS,
                                    "runtime.execution.capacity",
                                    "state-class capacity accounting overflowed");
        if (required_bytes <= hardware->usable_memory_bytes) {
            admitted = middle;
            low = middle + 1ull;
        } else {
            if (!middle) break;
            high = middle - 1ull;
        }
    }
    if (admitted < workload->minimum_session_context)
        return execution_refuse(err, YVEX_ERR_BOUNDS,
                                "runtime.execution.capacity",
                                "memory cannot admit the workload minimum context");
    if (!capacity_state_usage(request, geometry, admitted, plan->state_classes,
                              &plan->state_pool_bytes,
                              &plan->candidate_reserve_bytes) ||
        !yvex_core_u64_add(fixed_bytes, plan->state_pool_bytes,
                           &plan->required_bytes) ||
        !capacity_add(&plan->required_bytes, plan->candidate_reserve_bytes) ||
        !yvex_core_u64_mul(admitted, workload->concurrent_sequences,
                           &plan->total_logical_context_tokens) ||
        !yvex_core_u64_mul(request->candidate_width,
                           workload->concurrent_sequences,
                           &plan->candidate_reserve_tokens))
        return execution_refuse(err, YVEX_ERR_BOUNDS,
                                "runtime.execution.capacity",
                                "state-pool capacity accounting overflowed");
    plan->schema_version = YVEX_EXECUTION_CAPACITY_PLAN_SCHEMA_V1;
    plan->model_maximum_context = request->semantic_maximum_context;
    plan->admitted_execution_maximum = admitted;
    plan->per_session_maximum = admitted;
    plan->per_request_maximum = admitted;
    plan->physical_state_pool_tokens = plan->total_logical_context_tokens;
    plan->concurrent_sequences = workload->concurrent_sequences;
    plan->logical_batch_tokens = workload->logical_batch_tokens;
    plan->attention_microbatch_rows = workload->attention_microbatch_rows;
    plan->moe_row_tile = workload->moe_row_tile;
    plan->output_head_rows = workload->output_head_rows;
    plan->model_bytes = request->model_bytes;
    plan->derived_layout_bytes = request->derived_layout_bytes;
    plan->workspace_bytes = request->workspace_bytes;
    plan->scheduler_bytes = request->scheduler_bytes;
    plan->graph_bytes = request->graph_bytes;
    plan->prefix_cache_bytes = workload->prefix_cache_bytes;
    plan->persistent_state_bytes = workload->persistent_state_bytes;
    plan->system_reserve_bytes = workload->system_reserve_bytes;
    plan->usable_memory_bytes = hardware->usable_memory_bytes;
    plan->unreserved_bytes = hardware->usable_memory_bytes - plan->required_bytes;
    plan->state_class_count = request->state_class_count;
    yvex_core_text_copy(plan->model_execution_identity,
                        sizeof(plan->model_execution_identity),
                        request->model_execution_identity);
    yvex_core_text_copy(plan->hardware_profile_identity,
                        sizeof(plan->hardware_profile_identity), hardware->identity);
    yvex_core_text_copy(plan->workload_profile_identity,
                        sizeof(plan->workload_profile_identity), workload->identity);
    if (!capacity_plan_identity(plan))
        return execution_refuse(err, YVEX_ERR_STATE,
                                "runtime.execution.capacity",
                                "capacity plan identity derivation failed");
    yvex_error_clear(err);
    return YVEX_OK;
}
static int capacity_plan_structure_valid(
    const yvex_execution_capacity_plan *plan)
{
    unsigned long long required = 0ull, state_bytes = 0ull;
    unsigned long long candidate_bytes = 0ull, logical_tokens;
    unsigned long long index;
    if (!plan->model_maximum_context || !plan->admitted_execution_maximum ||
        plan->admitted_execution_maximum > plan->model_maximum_context ||
        plan->per_session_maximum != plan->admitted_execution_maximum ||
        plan->per_request_maximum != plan->admitted_execution_maximum ||
        !plan->concurrent_sequences || !plan->logical_batch_tokens ||
        !plan->attention_microbatch_rows || !plan->moe_row_tile ||
        !plan->output_head_rows || !plan->model_bytes || !plan->workspace_bytes ||
        !yvex_core_u64_mul(plan->admitted_execution_maximum,
                           plan->concurrent_sequences, &logical_tokens) ||
        logical_tokens != plan->total_logical_context_tokens ||
        plan->physical_state_pool_tokens != plan->total_logical_context_tokens)
        return 0;
    for (index = 0ull; index < plan->state_class_count; ++index) {
        const yvex_execution_state_class_plan *state =
            &plan->state_classes[index];
        unsigned long long page_bytes, rounded_tokens, data_bytes;
        unsigned long long pool_tokens, fragmentation, cow_bytes;
        if (state->state_class >= YVEX_MODEL_STATE_CLASS_COUNT ||
            state->extent > YVEX_EXECUTION_STATE_EXTENT_PREFIX_BUDGET ||
            (index && state->state_class <=
                          plan->state_classes[index - 1ull].state_class) ||
            !state->logical_block_tokens || !state->bytes_per_block ||
            !state->page_tokens ||
            state->page_tokens % state->logical_block_tokens ||
            !yvex_core_u64_mul(
                state->page_tokens / state->logical_block_tokens,
                state->bytes_per_block, &page_bytes) ||
            page_bytes != state->page_bytes ||
            (state->shared != 0 && state->shared != 1) ||
            (state->copy_on_write != 0 && state->copy_on_write != 1) ||
            !yvex_core_u64_mul(state->page_count, state->page_tokens,
                               &rounded_tokens))
            return 0;
        if (state->extent == YVEX_EXECUTION_STATE_EXTENT_PREFIX_BUDGET) {
            if (state->tokens_per_sequence ||
                rounded_tokens != state->pool_tokens ||
                state->pool_bytes != plan->prefix_cache_bytes ||
                state->promotion_fragmentation_bytes ||
                state->copy_on_write_tail_bytes !=
                    (state->copy_on_write ? state->page_bytes : 0ull))
                return 0;
            continue;
        }
        if (!state->tokens_per_sequence ||
            !yvex_core_u64_mul(state->tokens_per_sequence,
                               plan->concurrent_sequences, &pool_tokens) ||
            pool_tokens != state->pool_tokens || rounded_tokens < pool_tokens ||
            rounded_tokens % state->logical_block_tokens ||
            !yvex_core_u64_mul(
                rounded_tokens / state->logical_block_tokens,
                state->bytes_per_block, &data_bytes) ||
            !yvex_core_u64_mul(
                (rounded_tokens - pool_tokens) / state->logical_block_tokens,
                state->bytes_per_block, &fragmentation) ||
            state->pool_bytes < data_bytes ||
            state->pool_bytes - data_bytes != state->page_table_bytes ||
            fragmentation != state->tail_fragmentation_bytes ||
            !yvex_core_u64_mul(state->page_bytes, plan->concurrent_sequences,
                               &cow_bytes) ||
            state->copy_on_write_tail_bytes !=
                (state->copy_on_write ? cow_bytes : 0ull) ||
            state->promotion_fragmentation_bytes !=
                (state->extent == YVEX_EXECUTION_STATE_EXTENT_CANDIDATE
                     ? fragmentation : 0ull) ||
            !capacity_add(
                state->extent == YVEX_EXECUTION_STATE_EXTENT_CANDIDATE
                    ? &candidate_bytes : &state_bytes,
                state->pool_bytes))
            return 0;
    }
    if (state_bytes != plan->state_pool_bytes ||
        candidate_bytes != plan->candidate_reserve_bytes ||
        !capacity_add(&required, plan->model_bytes) ||
        !capacity_add(&required, plan->derived_layout_bytes) ||
        !capacity_add(&required, plan->state_pool_bytes) ||
        !capacity_add(&required, plan->candidate_reserve_bytes) ||
        !capacity_add(&required, plan->workspace_bytes) ||
        !capacity_add(&required, plan->scheduler_bytes) ||
        !capacity_add(&required, plan->graph_bytes) ||
        !capacity_add(&required, plan->prefix_cache_bytes) ||
        !capacity_add(&required, plan->persistent_state_bytes) ||
        !capacity_add(&required, plan->system_reserve_bytes) ||
        required != plan->required_bytes || required > plan->usable_memory_bytes ||
        plan->unreserved_bytes != plan->usable_memory_bytes - required)
        return 0;
    return 1;
}
int yvex_execution_capacity_plan_validate(
    const yvex_execution_capacity_plan *plan, yvex_error *err)
{
    yvex_execution_capacity_plan candidate;
    char expected[YVEX_SHA256_HEX_CAP];
    if (!plan ||
        plan->schema_version != YVEX_EXECUTION_CAPACITY_PLAN_SCHEMA_V1 ||
        !plan->state_class_count ||
        plan->state_class_count > YVEX_MODEL_STATE_CLASS_COUNT ||
        !yvex_sha256_hex_valid(plan->model_execution_identity) ||
        !yvex_sha256_hex_valid(plan->hardware_profile_identity) ||
        !yvex_sha256_hex_valid(plan->workload_profile_identity) ||
        !yvex_sha256_hex_valid(plan->identity) ||
        !capacity_plan_structure_valid(plan))
        return execution_refuse(err, YVEX_ERR_FORMAT,
                                "runtime.execution.capacity",
                                "persisted capacity plan is incomplete");
    candidate = *plan;
    candidate.identity[0] = '\0';
    if (!capacity_plan_identity(&candidate))
        return execution_refuse(err, YVEX_ERR_STATE,
                                "runtime.execution.capacity",
                                "capacity plan identity derivation failed");
    yvex_core_text_copy(expected, sizeof(expected), candidate.identity);
    if (strcmp(expected, plan->identity) != 0)
        return execution_refuse(err, YVEX_ERR_FORMAT,
                                "runtime.execution.capacity",
                                "persisted capacity plan identity mismatched");
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_execution_memory_facts_add(
    yvex_execution_memory_facts *facts, unsigned long long active_weight_bytes,
    unsigned long long state_bytes, unsigned long long activation_bytes,
    unsigned long long temporary_bytes, unsigned long long measured_operations,
    unsigned long long missing_operations, yvex_error *err)
{
    yvex_execution_memory_facts candidate;
    if (!facts || (!measured_operations && !missing_operations))
        return execution_refuse(err, YVEX_ERR_INVALID_ARG,
                                "runtime.execution.memory-facts",
                                "one measured or missing operation count is required");
    candidate = *facts;
    if (!yvex_core_u64_add(candidate.active_weight_bytes, active_weight_bytes,
                           &candidate.active_weight_bytes) ||
        !yvex_core_u64_add(candidate.state_bytes, state_bytes, &candidate.state_bytes) ||
        !yvex_core_u64_add(candidate.activation_bytes, activation_bytes,
                           &candidate.activation_bytes) ||
        !yvex_core_u64_add(candidate.temporary_bytes, temporary_bytes,
                           &candidate.temporary_bytes) ||
        !yvex_core_u64_add(candidate.measured_operations, measured_operations,
                           &candidate.measured_operations) ||
        !yvex_core_u64_add(candidate.missing_operations, missing_operations,
                           &candidate.missing_operations))
        return execution_refuse(err, YVEX_ERR_BOUNDS,
                                "runtime.execution.memory-facts",
                                "compulsory memory counters overflowed");
    candidate.complete = candidate.measured_operations != 0ull &&
                         candidate.missing_operations == 0ull;
    *facts = candidate;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_execution_memory_facts_merge(
    yvex_execution_memory_facts *facts,
    const yvex_execution_memory_facts *delta, yvex_error *err)
{
    if (!delta)
        return execution_refuse(err, YVEX_ERR_INVALID_ARG,
                                "runtime.execution.memory-facts",
                                "memory fact delta is required");
    if (!delta->measured_operations && !delta->missing_operations) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    return yvex_execution_memory_facts_add(
        facts, delta->active_weight_bytes, delta->state_bytes,
        delta->activation_bytes, delta->temporary_bytes,
        delta->measured_operations, delta->missing_operations, err);
}

int yvex_execution_physical_facts_add(
    yvex_execution_physical_facts *facts,
    const yvex_execution_memory_facts *memory, unsigned long long h2d_bytes,
    unsigned long long d2h_bytes, unsigned long long d2d_bytes,
    unsigned long long kernel_count, unsigned long long stream_synchronization_count,
    unsigned long long device_synchronization_count, yvex_error *err)
{
    yvex_execution_physical_facts candidate;
    unsigned long long synchronization_count;
    int rc;
    if (!facts || !memory)
        return execution_refuse(err, YVEX_ERR_INVALID_ARG,
                                "runtime.execution.physical-facts",
                                "physical fact owners are required");
    candidate = *facts;
    rc = yvex_execution_memory_facts_merge(&candidate.memory, memory, err);
    if (rc != YVEX_OK) return rc;
    if (!yvex_core_u64_add(stream_synchronization_count, device_synchronization_count,
                           &synchronization_count) ||
        !yvex_core_u64_add(candidate.h2d_bytes, h2d_bytes, &candidate.h2d_bytes) ||
        !yvex_core_u64_add(candidate.d2h_bytes, d2h_bytes, &candidate.d2h_bytes) ||
        !yvex_core_u64_add(candidate.d2d_bytes, d2d_bytes, &candidate.d2d_bytes) ||
        !yvex_core_u64_add(candidate.kernel_count, kernel_count, &candidate.kernel_count) ||
        !yvex_core_u64_add(candidate.synchronization_count, synchronization_count,
                           &candidate.synchronization_count))
        return execution_refuse(err, YVEX_ERR_BOUNDS,
                                "runtime.execution.physical-facts",
                                "physical fact accounting overflowed");
    *facts = candidate;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_execution_f32_hash_update(
    yvex_sha256 *hash, const float *values, unsigned long long count)
{
    unsigned long long index;
    if (!hash || !values || !count) return 0;
    for (index = 0ull; index < count; ++index) {
        uint32_t bits;
        if (!isfinite(values[index])) return 0;
        memcpy(&bits, &values[index], sizeof(bits));
        if (!yvex_sha256_update_u64(hash, bits)) return 0;
    }
    return 1;
}

int yvex_execution_f32_digest(
    const char *domain, const float *values, unsigned long long count,
    char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    if (!domain || !output) return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, domain) ||
        !yvex_execution_f32_hash_update(&hash, values, count) ||
        !yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

int yvex_execution_device_view_validate(
    const yvex_execution_device_view *view, yvex_error *err)
{
    unsigned long long elements, bytes, offset_bytes, end;
    if (!view || view->schema_version != YVEX_EXECUTION_DEVICE_VIEW_SCHEMA_V1 ||
        view->kind > YVEX_EXECUTION_DEVICE_WORKSPACE || !view->backend || !view->tensor ||
        !yvex_backend_tensor_owned_by(view->backend, view->tensor) ||
        !view->resource_generation || !view->session_generation ||
        !view->state_generation || !view->rows || !view->columns ||
        !view->element_bytes ||
        view->materialization > YVEX_EXECUTION_MATERIALIZE_FORENSIC_FULL ||
        !yvex_sha256_hex_valid(view->runtime_model_identity) ||
        !yvex_sha256_hex_valid(view->execution_profile_identity) ||
        !yvex_core_u64_mul(view->rows, view->columns, &elements) ||
        !yvex_core_u64_mul(elements, view->element_bytes, &bytes) ||
        !yvex_core_u64_mul(view->element_offset, view->element_bytes,
                           &offset_bytes) ||
        !yvex_core_u64_add(offset_bytes, bytes, &end) ||
        end > view->tensor->bytes || view->dtype != view->tensor->dtype)
        return execution_refuse(
            err, YVEX_ERR_FORMAT, "runtime.execution.device-view",
            "device value view is incomplete or has incompatible extent");
    yvex_error_clear(err);
    return YVEX_OK;
}
