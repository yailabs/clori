/*
 * Execution identity is compiled from admitted semantic and physical facts. This owner keeps
 * portable reference mechanisms explicit while providing shape and device-value contracts that
 * optimized backends can consume without changing model semantics.
 */
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

struct yvex_execution_shape_registry {
    yvex_execution_shape *shapes;
    unsigned long long count, capacity, hit_count, miss_count;
};

static int execution_shape_equal(const yvex_execution_shape *left,
                                 const yvex_execution_shape *right);
static int physical_ir_identity(yvex_physical_execution_ir *ir,
                                yvex_error *err);

static int physical_policy_valid(const yvex_physical_execution_policy *policy)
{
    return policy &&
           policy->schema_version == YVEX_PHYSICAL_EXECUTION_POLICY_SCHEMA_V1 &&
           policy->activation <= YVEX_EXECUTION_ACTIVATION_DEVICE_ENCODED &&
           policy->required_backend <= YVEX_EXECUTION_BACKEND_CUDA &&
           policy->evidence <= YVEX_EXECUTION_EVIDENCE_FORENSIC &&
           policy->fallback <= YVEX_EXECUTION_CLASS_FORENSIC_REFERENCE &&
           policy->dense_kernel_family && policy->dense_kernel_family[0] &&
           strlen(policy->dense_kernel_family) < YVEX_EXECUTION_TEXT_CAP &&
           policy->expert_kernel_family && policy->expert_kernel_family[0] &&
           strlen(policy->expert_kernel_family) < YVEX_EXECUTION_TEXT_CAP;
}

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

static yvex_execution_placement_class execution_placement(
    yvex_materialization_placement placement)
{
    switch (placement) {
    case YVEX_MATERIALIZATION_PLACEMENT_STAGED_CACHE:
        return YVEX_EXECUTION_PLACEMENT_STAGED;
    case YVEX_MATERIALIZATION_PLACEMENT_BACKEND_RESIDENT_CANDIDATE:
        return YVEX_EXECUTION_PLACEMENT_CUDA_ADDRESSABLE_HOST;
    case YVEX_MATERIALIZATION_PLACEMENT_FILE_BACKED:
    default:
        return YVEX_EXECUTION_PLACEMENT_FILE_BACKED;
    }
}

static int physical_execution_decision_seal(
    yvex_physical_execution_decision *decision, yvex_error *err)
{
    yvex_sha256 hash;
    if (!decision ||
        decision->schema_version != YVEX_PHYSICAL_EXECUTION_SCHEMA_V2 ||
        decision->role <= YVEX_TENSOR_ROLE_UNKNOWN ||
        decision->role >= YVEX_TENSOR_ROLE_COUNT ||
        decision->scope > YVEX_TENSOR_SCOPE_DRAFT ||
        decision->consumer >= YVEX_EXECUTION_CONSUMER_COUNT ||
        decision->layout > YVEX_EXECUTION_LAYOUT_DERIVED_BACKEND ||
        decision->placement > YVEX_EXECUTION_PLACEMENT_STAGED ||
        decision->sharing > YVEX_EXECUTION_SHARING_ALIAS ||
        decision->activation > YVEX_EXECUTION_ACTIVATION_DEVICE_ENCODED ||
        decision->required_backend > YVEX_EXECUTION_BACKEND_CUDA ||
        !decision->canonical_row_width || !decision->canonical_row_count ||
        !decision->encoded_bytes || !decision->alignment ||
        !decision->supported_width_mask ||
        decision->evidence > YVEX_EXECUTION_EVIDENCE_FORENSIC ||
        decision->fallback > YVEX_EXECUTION_CLASS_FORENSIC_REFERENCE ||
        !decision->kernel_family[0] ||
        !yvex_sha256_hex_valid(decision->terminal_identity))
        return execution_refuse(
            err, YVEX_ERR_INVALID_ARG, "runtime.execution.physical",
            "complete physical execution decision facts are required");
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.physical-execution.decision.v2") ||
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
        !yvex_sha256_update_u64(&hash, decision->placement) ||
        !yvex_sha256_update_u64(&hash, decision->sharing) ||
        !yvex_sha256_update_u64(&hash, decision->activation) ||
        !yvex_sha256_update_u64(&hash, decision->supported_width_mask) ||
        !yvex_sha256_update_u64(&hash, decision->maximum_context) ||
        !yvex_sha256_update_u64(&hash, decision->required_backend) ||
        !yvex_sha256_update_u64(&hash, decision->required_compute_major) ||
        !yvex_sha256_update_u64(&hash, decision->required_compute_minor) ||
        !yvex_sha256_update_u64(&hash, decision->evidence) ||
        !yvex_sha256_update_u64(&hash, decision->fallback) ||
        !yvex_sha256_update_u64(&hash,
                                (unsigned long long)decision->derived_asset_required) ||
        !yvex_sha256_update_text(&hash, decision->terminal_identity) ||
        !yvex_sha256_update_text(&hash, decision->kernel_family) ||
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
    const yvex_materialized_tensor_binding *physical,
    const yvex_model_execution_descriptor *model,
    const yvex_physical_execution_policy *policy)
{
    unsigned long long maximum_width = model->verification_width_maximum
                                           ? model->verification_width_maximum
                                           : 1ull;
    memset(decision, 0, sizeof(*decision));
    decision->schema_version = YVEX_PHYSICAL_EXECUTION_SCHEMA_V2;
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
    decision->placement = execution_placement(binding->placement);
    decision->sharing = binding->scope == YVEX_TENSOR_SCOPE_GLOBAL
                            ? YVEX_EXECUTION_SHARING_MODEL_READ_ONLY
                            : YVEX_EXECUTION_SHARING_EXCLUSIVE;
    decision->activation = policy->activation;
    decision->supported_width_mask = (1ull << (maximum_width + 1ull)) - 2ull;
    decision->maximum_context = model->maximum_context;
    decision->required_backend = policy->required_backend;
    decision->required_compute_major = policy->required_compute_major;
    decision->required_compute_minor = policy->required_compute_minor;
    decision->evidence = policy->evidence;
    decision->fallback = policy->fallback;
    decision->derived_asset_required = policy->derived_asset_required;
    yvex_core_text_copy(decision->kernel_family,
                        sizeof(decision->kernel_family),
                        physical->expert_count > 1ull
                            ? policy->expert_kernel_family
                            : policy->dense_kernel_family);
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
    const void *context, const yvex_model_execution_descriptor *model,
    const yvex_physical_execution_policy *policy, yvex_error *err)
{
    yvex_physical_execution_ir *ir;
    unsigned long long index;
    int rc = YVEX_OK;

    if (out) *out = NULL;
    if (!out || !count || !yvex_sha256_hex_valid(physical_variant_identity) ||
        count > SIZE_MAX / sizeof(*ir->decisions) || !binding_at || !model ||
        model->schema_version != YVEX_MODEL_EXECUTION_DESCRIPTOR_SCHEMA_V1 ||
        !model->maximum_context || !physical_policy_valid(policy) ||
        (!model->verification_width_maximum ? 1ull
                                            : model->verification_width_maximum) >= 63ull)
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
    ir->summary.schema_version = YVEX_PHYSICAL_EXECUTION_SCHEMA_V2;
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
        execution_decision_from_binding(decision, binding, physical, model, policy);
        rc = physical_execution_decision_seal(decision, err);
        if (rc != YVEX_OK) break;
        ir->summary.consumer_counts[decision->consumer]++;
        ir->summary.layout_counts[decision->layout]++;
        ir->summary.placement_counts[decision->placement]++;
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
    if (!yvex_sha256_update_text(&hash, "yvex.physical-execution.ir.v1") ||
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
    const char *physical_variant_identity,
    const yvex_physical_execution_policy *policy, yvex_error *err)
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
        physical_descriptor_binding_at, &source,
        &descriptor_summary->model_execution, policy, err);
}

int yvex_physical_execution_ir_import(
    yvex_physical_execution_ir **out, const yvex_physical_execution_summary *summary,
    const yvex_physical_execution_decision *decisions, unsigned long long count,
    yvex_error *err)
{
    yvex_physical_execution_ir *ir;
    unsigned long long index;
    char expected[YVEX_SHA256_HEX_CAP];
    if (out) *out = NULL;
    if (!out || !summary || !decisions || !count || summary->decision_count != count ||
        summary->schema_version != YVEX_PHYSICAL_EXECUTION_SCHEMA_V2 ||
        count > SIZE_MAX / sizeof(*ir->decisions))
        return execution_refuse(err, YVEX_ERR_INVALID_ARG,
                                "runtime.execution.physical",
                                "sealed physical execution records are required");
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
            strcmp(expected, ir->decisions[index].decision_identity) != 0)
            goto refused;
        ir->summary.consumer_counts[ir->decisions[index].consumer]++;
        ir->summary.layout_counts[ir->decisions[index].layout]++;
        ir->summary.placement_counts[ir->decisions[index].placement]++;
        if (!yvex_core_u64_add(ir->summary.encoded_bytes,
                               ir->decisions[index].encoded_bytes,
                               &ir->summary.encoded_bytes))
            goto refused;
    }
    if (ir->summary.encoded_bytes != summary->encoded_bytes ||
        memcmp(ir->summary.consumer_counts, summary->consumer_counts,
               sizeof(ir->summary.consumer_counts)) != 0 ||
        memcmp(ir->summary.layout_counts, summary->layout_counts,
               sizeof(ir->summary.layout_counts)) != 0 ||
        memcmp(ir->summary.placement_counts, summary->placement_counts,
               sizeof(ir->summary.placement_counts)) != 0)
        goto refused;
    yvex_core_text_copy(expected, sizeof(expected), summary->identity);
    if (physical_ir_identity(ir, err) != YVEX_OK ||
        strcmp(expected, ir->summary.identity) != 0)
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
        profile->system_reserve_bytes < YVEX_EXECUTION_MINIMUM_SYSTEM_RESERVE ||
        (!profile->continuous_batching && profile->concurrent_sequences != 1ull))
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

static int roofline_rate(unsigned long long bytes, unsigned long long duration,
                         unsigned long long scale, unsigned long long *result)
{
    return capacity_mul_div(bytes, scale, duration, 0, result);
}

static int roofline_minimum_time(unsigned long long bytes,
                                 unsigned long long bytes_per_second,
                                 unsigned long long *nanoseconds)
{
    return capacity_mul_div(bytes, 1000000000ull, bytes_per_second, 1,
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
    if (!capacity_add(&phase->active_device_bytes, measurement->active_weight_bytes) ||
        !capacity_add(&phase->active_device_bytes, measurement->state_bytes) ||
        !capacity_add(&phase->active_device_bytes, measurement->activation_bytes) ||
        !capacity_add(&phase->active_device_bytes, measurement->temporary_bytes) ||
        !capacity_add(&phase->transfer_bytes, measurement->h2d_bytes) ||
        !capacity_add(&phase->transfer_bytes, measurement->d2h_bytes) ||
        !capacity_add(&phase->transfer_bytes, measurement->d2d_bytes) ||
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
    if (!measurements || !measurement_count || !delta ||
        !measurement_capacity || *measurement_count > measurement_capacity ||
        delta->phase >= YVEX_EXECUTION_ROOFLINE_PHASE_COUNT ||
        (delta->fact_mask & required) != required ||
        (delta->fact_mask & ~YVEX_EXECUTION_PHASE_FACT_ALL) ||
        ((delta->fact_mask & occupancy) &&
         delta->occupancy_parts_per_million > 1000000ull) ||
        (!(delta->fact_mask & occupancy) && delta->occupancy_parts_per_million) ||
        !delta->measured_duration_ns || !delta->work_units)
        return execution_refuse(err, YVEX_ERR_INVALID_ARG,
                                "runtime.execution.roofline",
                                "phase measurement delta is incomplete");
    for (index = 0ull; index < *measurement_count; ++index)
        if (measurements[index].phase == delta->phase)
            measurement = &measurements[index];
    if (!measurement) {
        if (*measurement_count == measurement_capacity)
            return execution_refuse(err, YVEX_ERR_BOUNDS,
                                    "runtime.execution.roofline",
                                    "phase measurement capacity is exhausted");
        measurements[*measurement_count] = *delta;
        (*measurement_count)++;
        yvex_error_clear(err);
        return YVEX_OK;
    }
    if (measurement->fact_mask != delta->fact_mask)
        return execution_refuse(err, YVEX_ERR_STATE,
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
            !yvex_core_u64_add(measurement->work_units, delta->work_units,
                               &work))
            return execution_refuse(err, YVEX_ERR_BOUNDS,
                                    "runtime.execution.roofline",
                                    "phase occupancy accumulation overflowed");
        candidate.occupancy_parts_per_million = total / work;
    }
#define ACCUMULATE(field_) \
    capacity_add(&candidate.field_, delta->field_)
    if (!ACCUMULATE(active_weight_bytes) || !ACCUMULATE(state_bytes) ||
        !ACCUMULATE(activation_bytes) || !ACCUMULATE(temporary_bytes) ||
        !ACCUMULATE(h2d_bytes) || !ACCUMULATE(d2h_bytes) ||
        !ACCUMULATE(d2d_bytes) || !ACCUMULATE(kernel_count) ||
        !ACCUMULATE(synchronization_count) ||
        !ACCUMULATE(measured_duration_ns) || !ACCUMULATE(work_units) ||
        !ACCUMULATE(committed_tokens))
        return execution_refuse(err, YVEX_ERR_BOUNDS,
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
        !yvex_sha256_update_u64(&hash, (unsigned long long)ledger->priority_provisional)) return 0;
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
    return execution_hash_finish(&hash, ledger->identity);
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
        return execution_refuse(err, YVEX_ERR_INVALID_ARG,
                                "runtime.execution.roofline",
                                "one or more unique causal phase measurements are required");
    identities[0] = request->artifact_identity;
    identities[1] = request->execution_profile_identity;
    identities[2] = request->kernel_bundle_identity;
    identities[3] = request->workload_profile_identity;
    for (index = 0ull; index < 4ull; ++index)
        if (!yvex_sha256_hex_valid(identities[index]))
            return execution_refuse(err, YVEX_ERR_FORMAT,
                                    "runtime.execution.roofline",
                                    "roofline evidence identity is invalid");
    for (index = 0ull; index < request->measurement_count; ++index) {
        unsigned long long phase = request->measurements[index].phase;
        if (phase >= YVEX_EXECUTION_ROOFLINE_PHASE_COUNT ||
            (seen & (1ull << phase)) ||
            !roofline_measurement_build(request->hardware,
                                        &request->measurements[index],
                                        &ledger->phases[phase]))
            return execution_refuse(err, YVEX_ERR_INVALID_ARG,
                                    "runtime.execution.roofline",
                                    "causal phase measurement is incomplete or duplicated");
        seen |= 1ull << phase;
        if (!capacity_add(&ledger->measured_duration_ns,
                          request->measurements[index].measured_duration_ns) ||
            ((ledger->phases[phase].measurement.fact_mask &
              YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_COMMITTED_TOKENS)) &&
             !capacity_add(&ledger->committed_tokens,
                           request->measurements[index].committed_tokens)))
            return execution_refuse(err, YVEX_ERR_BOUNDS,
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
        return execution_refuse(err, YVEX_ERR_STATE,
                                "runtime.execution.roofline",
                                "phase roofline ledger identity derivation failed");
    yvex_error_clear(err);
    return YVEX_OK;
}

static int execution_resolution_executable(yvex_execution_resolution resolution)
{
    return resolution == YVEX_EXECUTION_RESOLUTION_EXACT ||
           resolution == YVEX_EXECUTION_RESOLUTION_COMPATIBLE_DEGRADED;
}

int yvex_compiled_execution_profile_seal(
    const yvex_compiled_execution_profile_request *request,
    yvex_compiled_execution_profile *profile, yvex_error *err)
{
    const char *identities[7];
    yvex_sha256 hash;
    size_t index;
    if (profile) memset(profile, 0, sizeof(*profile));
    if (!request || !profile ||
        request->schema_version != YVEX_COMPILED_EXECUTION_PROFILE_SCHEMA_V2 ||
        !request->hardware_profile || !request->hardware_profile[0] ||
        strlen(request->hardware_profile) >= YVEX_EXECUTION_TEXT_CAP ||
        request->backend > YVEX_BACKEND_KIND_CUDA ||
        request->generation_mode > YVEX_EXECUTION_GENERATION_SPECULATIVE ||
        request->workload > YVEX_EXECUTION_WORKLOAD_QUALIFICATION ||
        request->evidence > YVEX_EXECUTION_EVIDENCE_FORENSIC ||
        request->execution_class > YVEX_EXECUTION_CLASS_FORENSIC_REFERENCE ||
        !execution_resolution_executable(request->attention_resolution) ||
        !execution_resolution_executable(request->moe_resolution) ||
        !execution_resolution_executable(request->sampling_resolution) ||
        !request->context_capacity)
        return execution_refuse(
            err, YVEX_ERR_INVALID_ARG, "runtime.execution.profile",
            "complete compiled execution profile facts are required");
    identities[0] = request->logical_model_identity;
    identities[1] = request->physical_variant_identity;
    identities[2] = request->physical_execution_identity;
    identities[3] = request->artifact_identity;
    identities[4] = request->materialization_identity;
    identities[5] = request->runtime_binding_identity;
    identities[6] = request->kernel_bundle_identity;
    for (index = 0u; index < sizeof(identities) / sizeof(identities[0]); ++index)
        if (!yvex_sha256_hex_valid(identities[index]))
            return execution_refuse(
                err, YVEX_ERR_FORMAT, "runtime.execution.profile",
                "compiled execution profile identity input is invalid");
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.compiled-execution-profile.v2"))
        goto identity_failed;
    for (index = 0u; index < sizeof(identities) / sizeof(identities[0]); ++index)
        if (!yvex_sha256_update_text(&hash, identities[index]))
            goto identity_failed;
    if (!yvex_sha256_update_text(&hash, request->hardware_profile) ||
        !yvex_sha256_update_u64(&hash, request->backend) ||
        !yvex_sha256_update_u64(&hash, (unsigned long long)request->device_index) ||
        !yvex_sha256_update_u64(&hash, (unsigned long long)request->compute_major) ||
        !yvex_sha256_update_u64(&hash, (unsigned long long)request->compute_minor) ||
        !yvex_sha256_update_u64(&hash, request->context_capacity) ||
        !yvex_sha256_update_u64(&hash, request->generation_mode) ||
        !yvex_sha256_update_u64(&hash, request->workload) ||
        !yvex_sha256_update_u64(&hash, request->evidence) ||
        !yvex_sha256_update_u64(&hash, request->execution_class) ||
        !yvex_sha256_update_u64(&hash, request->attention_resolution) ||
        !yvex_sha256_update_u64(&hash, request->moe_resolution) ||
        !yvex_sha256_update_u64(&hash, request->sampling_resolution) ||
        !execution_hash_finish(&hash, profile->identity))
        goto identity_failed;
    profile->schema_version = YVEX_COMPILED_EXECUTION_PROFILE_SCHEMA_V2;
    profile->backend = request->backend;
    profile->device_index = request->device_index;
    profile->compute_major = request->compute_major;
    profile->compute_minor = request->compute_minor;
    profile->context_capacity = request->context_capacity;
    profile->generation_mode = request->generation_mode;
    profile->workload = request->workload;
    profile->evidence = request->evidence;
    profile->execution_class = request->execution_class;
    profile->attention_resolution = request->attention_resolution;
    profile->moe_resolution = request->moe_resolution;
    profile->sampling_resolution = request->sampling_resolution;
    profile->resolution =
        request->attention_resolution == YVEX_EXECUTION_RESOLUTION_EXACT &&
                request->moe_resolution == YVEX_EXECUTION_RESOLUTION_EXACT &&
                request->sampling_resolution == YVEX_EXECUTION_RESOLUTION_EXACT
            ? YVEX_EXECUTION_RESOLUTION_EXACT
            : YVEX_EXECUTION_RESOLUTION_COMPATIBLE_DEGRADED;
    yvex_core_text_copy(profile->hardware_profile,
                        sizeof(profile->hardware_profile),
                        request->hardware_profile);
    yvex_core_text_copy(profile->kernel_bundle_identity,
                        sizeof(profile->kernel_bundle_identity),
                        request->kernel_bundle_identity);
    yvex_error_clear(err);
    return YVEX_OK;
identity_failed:
    memset(profile, 0, sizeof(*profile));
    return execution_refuse(err, YVEX_ERR_STATE,
                            "runtime.execution.profile",
                            "compiled execution profile identity derivation failed");
}

int yvex_execution_device_view_validate(
    const yvex_execution_device_view *view, yvex_error *err)
{
    unsigned long long elements, bytes, offset_bytes, end;
    if (!view || view->schema_version != YVEX_EXECUTION_DEVICE_VIEW_SCHEMA_V1 ||
        view->kind > YVEX_EXECUTION_DEVICE_WORKSPACE || !view->backend || !view->tensor ||
        !yvex_backend_tensor_owned_by(view->backend, view->tensor) ||
        !view->model_generation || !view->session_generation ||
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

int yvex_execution_shape_seal(yvex_execution_shape *shape, yvex_error *err)
{
    yvex_sha256 hash;
    if (!shape || shape->schema_version != YVEX_EXECUTION_SHAPE_SCHEMA_V1 ||
        shape->target_scope > YVEX_EXECUTION_SCOPE_DRAFT ||
        shape->phase > YVEX_EXECUTION_PHASE_RESET ||
        shape->operation_scope > YVEX_EXECUTION_OPERATION_RELEASE_SET ||
        !shape->token_width || shape->token_width > 64ull ||
        shape->context_band > YVEX_EXECUTION_CONTEXT_NEAR_CAPACITY ||
        shape->evidence > YVEX_EXECUTION_EVIDENCE_FORENSIC ||
        !shape->candidate_capacity || !shape->context_capacity ||
        shape->position > shape->context_capacity ||
        shape->token_width > shape->context_capacity - shape->position ||
        !yvex_sha256_hex_valid(shape->execution_profile_identity) ||
        !yvex_sha256_hex_valid(shape->attention_plan_identity) ||
        !yvex_sha256_hex_valid(shape->state_layout_identity) ||
        !yvex_sha256_hex_valid(shape->kernel_bundle_identity) ||
        !yvex_sha256_hex_valid(shape->workspace_identity))
        return execution_refuse(err, YVEX_ERR_INVALID_ARG,
                                "runtime.execution.shape",
                                "complete execution shape facts are required");
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.execution-shape.v1") ||
        !yvex_sha256_update_text(&hash, shape->execution_profile_identity) ||
        !yvex_sha256_update_text(&hash, shape->attention_plan_identity) ||
        !yvex_sha256_update_text(&hash, shape->state_layout_identity) ||
        !yvex_sha256_update_text(&hash, shape->kernel_bundle_identity) ||
        !yvex_sha256_update_u64(&hash, shape->target_scope) ||
        !yvex_sha256_update_u64(&hash, shape->phase) ||
        !yvex_sha256_update_u64(&hash, shape->operation_scope) ||
        !yvex_sha256_update_u64(&hash, shape->token_width) ||
        !yvex_sha256_update_u64(&hash,
                                (unsigned long long)shape->candidate_visible) ||
        !yvex_sha256_update_u64(&hash, shape->context_band) ||
        !yvex_sha256_update_u64(&hash, shape->position) ||
        !yvex_sha256_update_u64(&hash, shape->context_capacity) ||
        !yvex_sha256_update_u64(&hash, shape->local_capacity) ||
        !yvex_sha256_update_u64(&hash, shape->compressed_capacity) ||
        !yvex_sha256_update_u64(&hash, shape->indexer_capacity) ||
        !yvex_sha256_update_u64(&hash, shape->rolling_capacity) ||
        !yvex_sha256_update_u64(&hash, shape->candidate_capacity) ||
        !yvex_sha256_update_u64(&hash, shape->workspace_generation) ||
        !yvex_sha256_update_u64(&hash, shape->evidence) ||
        !yvex_sha256_update_text(&hash, shape->workspace_identity) ||
        !execution_hash_finish(&hash, shape->identity))
        return execution_refuse(err, YVEX_ERR_STATE,
                                "runtime.execution.shape",
                                "execution shape identity derivation failed");
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_execution_shape_registry_open(
    yvex_execution_shape_registry **out, unsigned long long capacity,
    yvex_error *err)
{
    yvex_execution_shape_registry *registry;
    if (out) *out = NULL;
    if (!out || !capacity || capacity > SIZE_MAX / sizeof(*registry->shapes))
        return execution_refuse(err, YVEX_ERR_INVALID_ARG,
                                "runtime.execution.shape-registry",
                                "bounded shape registry capacity is required");
    registry = calloc(1u, sizeof(*registry));
    if (registry)
        registry->shapes = calloc((size_t)capacity,
                                  sizeof(*registry->shapes));
    if (!registry || !registry->shapes) {
        yvex_execution_shape_registry_close(&registry);
        return execution_refuse(err, YVEX_ERR_NOMEM,
                                "runtime.execution.shape-registry",
                                "shape registry allocation failed");
    }
    registry->capacity = capacity;
    *out = registry;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_execution_shape_registry_register(
    yvex_execution_shape_registry *registry,
    const yvex_execution_shape *shape, yvex_error *err)
{
    unsigned long long index;
    if (!registry || !shape || !yvex_sha256_hex_valid(shape->identity))
        return execution_refuse(err, YVEX_ERR_INVALID_ARG,
                                "runtime.execution.shape-registry",
                                "sealed shape and registry are required");
    for (index = 0ull; index < registry->count; ++index) {
        if (strcmp(registry->shapes[index].identity, shape->identity) == 0) {
            if (execution_shape_equal(&registry->shapes[index], shape)) {
                yvex_error_clear(err);
                return YVEX_OK;
            }
            return execution_refuse(err, YVEX_ERR_STATE,
                                    "runtime.execution.shape-registry",
                                    "shape identity collision was refused");
        }
    }
    if (registry->count == registry->capacity)
        return execution_refuse(err, YVEX_ERR_BOUNDS,
                                "runtime.execution.shape-registry",
                                "shape registry capacity is exhausted");
    registry->shapes[registry->count++] = *shape;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int execution_shape_equal(const yvex_execution_shape *left,
                                 const yvex_execution_shape *right)
{
    return left->schema_version == right->schema_version &&
           left->target_scope == right->target_scope &&
           left->phase == right->phase &&
           left->operation_scope == right->operation_scope &&
           left->token_width == right->token_width &&
           left->candidate_visible == right->candidate_visible &&
           left->context_band == right->context_band &&
           left->position == right->position &&
           left->context_capacity == right->context_capacity &&
           left->local_capacity == right->local_capacity &&
           left->compressed_capacity == right->compressed_capacity &&
           left->indexer_capacity == right->indexer_capacity &&
           left->rolling_capacity == right->rolling_capacity &&
           left->candidate_capacity == right->candidate_capacity &&
           left->workspace_generation == right->workspace_generation &&
           left->evidence == right->evidence &&
           strcmp(left->execution_profile_identity,
                  right->execution_profile_identity) == 0 &&
           strcmp(left->attention_plan_identity,
                  right->attention_plan_identity) == 0 &&
           strcmp(left->state_layout_identity,
                  right->state_layout_identity) == 0 &&
           strcmp(left->kernel_bundle_identity,
                  right->kernel_bundle_identity) == 0 &&
           strcmp(left->workspace_identity, right->workspace_identity) == 0 &&
           strcmp(left->identity, right->identity) == 0;
}

static int execution_shape_key_equal(const yvex_execution_shape *left,
                                     const yvex_execution_shape *right)
{
    return left->target_scope == right->target_scope &&
           left->phase == right->phase &&
           left->operation_scope == right->operation_scope &&
           left->token_width == right->token_width &&
           left->candidate_visible == right->candidate_visible &&
           left->context_band == right->context_band &&
           left->context_capacity == right->context_capacity &&
           left->workspace_generation == right->workspace_generation &&
           left->evidence == right->evidence &&
           strcmp(left->execution_profile_identity,
                  right->execution_profile_identity) == 0 &&
           strcmp(left->attention_plan_identity,
                  right->attention_plan_identity) == 0 &&
           strcmp(left->state_layout_identity,
                  right->state_layout_identity) == 0 &&
           strcmp(left->kernel_bundle_identity,
                  right->kernel_bundle_identity) == 0 &&
           strcmp(left->workspace_identity, right->workspace_identity) == 0;
}

static int execution_shape_capacity(
    const yvex_execution_shape *configured,
    const yvex_execution_shape *required,
    yvex_execution_shape_failure *failure)
{
    const struct {
        yvex_execution_capacity_component component;
        unsigned long long configured, required;
    } capacities[] = {
        {YVEX_EXECUTION_CAPACITY_LOCAL, configured->local_capacity,
         required->local_capacity},
        {YVEX_EXECUTION_CAPACITY_COMPRESSED, configured->compressed_capacity,
         required->compressed_capacity},
        {YVEX_EXECUTION_CAPACITY_INDEXER, configured->indexer_capacity,
         required->indexer_capacity},
        {YVEX_EXECUTION_CAPACITY_ROLLING, configured->rolling_capacity,
         required->rolling_capacity},
        {YVEX_EXECUTION_CAPACITY_CANDIDATE, configured->candidate_capacity,
         required->candidate_capacity},
        {YVEX_EXECUTION_CAPACITY_CONTEXT,
         configured->context_capacity - required->position,
         required->token_width}};
    size_t index;
    for (index = 0u; index < sizeof(capacities) / sizeof(capacities[0]); ++index) {
        if (capacities[index].required <= capacities[index].configured) continue;
        if (failure) {
            failure->component = capacities[index].component;
            failure->configured = capacities[index].configured;
            failure->required = capacities[index].required;
        }
        return 0;
    }
    return 1;
}

int yvex_execution_shape_registry_select(
    yvex_execution_shape_registry *registry,
    const yvex_execution_shape *request,
    const yvex_execution_shape **selected,
    yvex_execution_shape_failure *failure, yvex_error *err)
{
    unsigned long long index;
    if (selected) *selected = NULL;
    if (failure) memset(failure, 0, sizeof(*failure));
    if (!registry || !request || !selected ||
        !yvex_sha256_hex_valid(request->identity))
        return execution_refuse(err, YVEX_ERR_INVALID_ARG,
                                "runtime.execution.shape-registry",
                                "sealed shape request and output are required");
    for (index = 0ull; index < registry->count; ++index) {
        const yvex_execution_shape *candidate = &registry->shapes[index];
        if (!execution_shape_key_equal(candidate, request)) continue;
        if (!execution_shape_capacity(candidate, request, failure)) {
            if (failure) {
                failure->position = request->position;
                failure->width = request->token_width;
                failure->target_scope = request->target_scope;
                failure->phase = request->phase;
                failure->context_band = request->context_band;
                yvex_core_text_copy(failure->shape_identity,
                                    sizeof(failure->shape_identity),
                                    candidate->identity);
                yvex_core_text_copy(failure->state_layout_identity,
                                    sizeof(failure->state_layout_identity),
                                    request->state_layout_identity);
                yvex_core_text_copy(failure->workspace_identity,
                                    sizeof(failure->workspace_identity),
                                    request->workspace_identity);
            }
            registry->miss_count++;
            return execution_refuse(err, YVEX_ERR_BOUNDS,
                                    "runtime.execution.shape-capacity",
                                    "execution request exceeds an admitted shape capacity");
        }
        registry->hit_count++;
        *selected = candidate;
        yvex_error_clear(err);
        return YVEX_OK;
    }
    registry->miss_count++;
    if (failure) {
        failure->component = YVEX_EXECUTION_CAPACITY_NONE;
        failure->required = request->token_width;
        failure->position = request->position;
        failure->width = request->token_width;
        failure->target_scope = request->target_scope;
        failure->phase = request->phase;
        failure->context_band = request->context_band;
        yvex_core_text_copy(failure->state_layout_identity,
                            sizeof(failure->state_layout_identity),
                            request->state_layout_identity);
        yvex_core_text_copy(failure->workspace_identity,
                            sizeof(failure->workspace_identity),
                            request->workspace_identity);
    }
    return execution_refuse(err, YVEX_ERR_UNSUPPORTED,
                            "runtime.execution.shape-registry",
                            "no compatible execution shape is admitted");
}

int yvex_execution_shape_registry_summary_copy(
    const yvex_execution_shape_registry *registry,
    yvex_execution_shape_registry_summary *summary, yvex_error *err)
{
    if (!registry || !summary)
        return execution_refuse(err, YVEX_ERR_INVALID_ARG,
                                "runtime.execution.shape-registry",
                                "shape registry and summary output are required");
    summary->count = registry->count;
    summary->capacity = registry->capacity;
    summary->hit_count = registry->hit_count;
    summary->miss_count = registry->miss_count;
    yvex_error_clear(err);
    return YVEX_OK;
}

void yvex_execution_shape_registry_close(
    yvex_execution_shape_registry **registry)
{
    if (!registry || !*registry) return;
    free((*registry)->shapes);
    memset(*registry, 0, sizeof(**registry));
    free(*registry);
    *registry = NULL;
}
