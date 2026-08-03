/*
 * Execution identity is compiled from admitted semantic and physical facts. This owner keeps
 * portable reference mechanisms explicit while providing shape and device-value contracts that
 * optimized backends can consume without changing model semantics.
 */
#include <yvex/internal/execution.h>

#include <limits.h>
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
        decision->schema_version != YVEX_PHYSICAL_EXECUTION_SCHEMA_V1 ||
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
    if (!yvex_sha256_update_text(&hash, "yvex.physical-execution.decision.v1") ||
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
    const yvex_runtime_tensor_binding *binding)
{
    const yvex_materialized_tensor_binding *physical = binding->binding;
    memset(decision, 0, sizeof(*decision));
    decision->schema_version = YVEX_PHYSICAL_EXECUTION_SCHEMA_V1;
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
    decision->activation = YVEX_EXECUTION_ACTIVATION_DEVICE_F32;
    decision->supported_width_mask = 0x7eull;
    decision->maximum_context = ULLONG_MAX;
    decision->required_backend = YVEX_EXECUTION_BACKEND_ANY;
    decision->evidence = YVEX_EXECUTION_EVIDENCE_PRODUCTION;
    decision->fallback = YVEX_EXECUTION_CLASS_PORTABLE_REFERENCE;
    yvex_core_text_copy(decision->kernel_family,
                        sizeof(decision->kernel_family),
                        physical->expert_count > 1ull ? "portable-expert-row"
                                                     : "portable-encoded-row");
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
    const char *physical_variant_identity, yvex_error *err)
{
    const yvex_runtime_descriptor_summary *descriptor_summary;
    const yvex_materialization_summary *materialization_summary;
    yvex_physical_execution_ir *ir;
    unsigned long long index;
    int rc = YVEX_OK;
    if (out) *out = NULL;
    descriptor_summary = yvex_runtime_descriptor_summary_get(descriptor);
    materialization_summary =
        yvex_materialization_session_summary(materialization);
    if (!out || !descriptor_summary || !materialization_summary ||
        descriptor_summary->status != YVEX_RUNTIME_DESCRIPTOR_STATUS_READY ||
        materialization_summary->status != YVEX_MATERIALIZATION_STATUS_COMMITTED ||
        descriptor_summary->tensor_count != materialization_summary->tensor_count ||
        !descriptor_summary->tensor_count ||
        !yvex_sha256_hex_valid(physical_variant_identity) ||
        descriptor_summary->tensor_count > SIZE_MAX / sizeof(*ir->decisions))
        return execution_refuse(
            err, YVEX_ERR_INVALID_ARG, "runtime.execution.physical",
            "admitted descriptor, materialization, and physical variant are required");
    ir = calloc(1u, sizeof(*ir));
    if (ir)
        ir->decisions = calloc((size_t)descriptor_summary->tensor_count,
                               sizeof(*ir->decisions));
    if (!ir || !ir->decisions) {
        yvex_physical_execution_ir_close(&ir);
        return execution_refuse(err, YVEX_ERR_NOMEM,
                                "runtime.execution.physical",
                                "physical execution IR allocation failed");
    }
    ir->summary.schema_version = YVEX_PHYSICAL_EXECUTION_SCHEMA_V1;
    ir->summary.decision_count = descriptor_summary->tensor_count;
    yvex_core_text_copy(ir->summary.physical_variant_identity,
                        sizeof(ir->summary.physical_variant_identity),
                        physical_variant_identity);
    for (index = 0ull; index < descriptor_summary->tensor_count; ++index) {
        const yvex_runtime_tensor_binding *binding =
            yvex_runtime_descriptor_tensor_at(descriptor, index);
        yvex_physical_execution_decision *decision = &ir->decisions[index];
        if (!binding || !binding->binding) {
            rc = execution_refuse(err, YVEX_ERR_FORMAT,
                                  "runtime.execution.physical",
                                  "runtime descriptor contains an orphan tensor");
            break;
        }
        execution_decision_from_binding(decision, binding);
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

const yvex_physical_execution_summary *yvex_physical_execution_ir_summary(
    const yvex_physical_execution_ir *ir)
{
    return ir ? &ir->summary : NULL;
}

void yvex_physical_execution_ir_close(yvex_physical_execution_ir **ir)
{
    if (!ir || !*ir) return;
    free((*ir)->decisions);
    memset(*ir, 0, sizeof(**ir));
    free(*ir);
    *ir = NULL;
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
        request->schema_version != YVEX_COMPILED_EXECUTION_PROFILE_SCHEMA_V1 ||
        !request->hardware_profile || !request->hardware_profile[0] ||
        strlen(request->hardware_profile) >= YVEX_EXECUTION_TEXT_CAP ||
        request->backend > YVEX_BACKEND_KIND_CUDA ||
        request->generation_mode > YVEX_EXECUTION_GENERATION_SPECULATIVE ||
        request->workload > YVEX_EXECUTION_WORKLOAD_QUALIFICATION ||
        request->evidence > YVEX_EXECUTION_EVIDENCE_FORENSIC ||
        request->execution_class > YVEX_EXECUTION_CLASS_FORENSIC_REFERENCE ||
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
    if (!yvex_sha256_update_text(&hash, "yvex.compiled-execution-profile.v1"))
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
        !yvex_sha256_update_u64(
            &hash, (unsigned long long)request->host_stochastic_reference) ||
        !yvex_sha256_update_u64(
            &hash, (unsigned long long)request->token_local_moe_reference) ||
        !yvex_sha256_update_u64(
            &hash, (unsigned long long)request->eager_attention_reference) ||
        !execution_hash_finish(&hash, profile->identity))
        goto identity_failed;
    profile->schema_version = YVEX_COMPILED_EXECUTION_PROFILE_SCHEMA_V1;
    profile->backend = request->backend;
    profile->device_index = request->device_index;
    profile->compute_major = request->compute_major;
    profile->compute_minor = request->compute_minor;
    profile->context_capacity = request->context_capacity;
    profile->generation_mode = request->generation_mode;
    profile->workload = request->workload;
    profile->evidence = request->evidence;
    profile->execution_class = request->execution_class;
    profile->host_stochastic_reference = request->host_stochastic_reference;
    profile->token_local_moe_reference = request->token_local_moe_reference;
    profile->eager_attention_reference = request->eager_attention_reference;
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
        !backend_tensor_owner_is(view->backend, view->tensor) ||
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
