/*
 * Lower admitted family policy into a transformer plan and execute bounded composition numerics.
 *
 * Component identities and typed global bindings determine one ordered immutable plan. Reusable
 * family-neutral composition from embedding values to normalized hidden values.
 */
#include <yvex/internal/transformer.h>

#include "src/graph/private.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <yvex/internal/core.h>

struct yvex_transformer_plan {
    yvex_transformer_plan_summary summary;
    yvex_transformer_layer_plan *layers;
};

static const yvex_tensor_role transformer_weight_roles[YVEX_TRANSFORMER_WEIGHT_COUNT] = {
    YVEX_TENSOR_ROLE_TOKEN_EMBEDDING, YVEX_TENSOR_ROLE_HC_HEAD_FUNCTION,
    YVEX_TENSOR_ROLE_HC_HEAD_BASE, YVEX_TENSOR_ROLE_HC_HEAD_SCALE,
    YVEX_TENSOR_ROLE_OUTPUT_NORM};

typedef struct {
    yvex_transformer_family_policy policy;
    yvex_tensor_scope tensor_scope;
    unsigned long long family_adapter_id, family_adapter_version;
    unsigned long long layer_count, vocabulary_size;
    const char *artifact_identity, *materialization_identity, *logical_model_identity;
    const char *runtime_numeric_identity, *runtime_descriptor_identity;
    yvex_transformer_weight_binding weights[YVEX_TRANSFORMER_WEIGHT_COUNT];
} transformer_plan_facts;

static int transformer_refuse(yvex_error *err, yvex_status status, const char *reason)
{
    yvex_error_set(err, status, "graph.transformer", reason);
    return status;
}

static void transformer_identity_copy(char destination[YVEX_SHA256_HEX_CAP],
                                      const char *source)
{
    size_t length = source ? strnlen(source, YVEX_SHA256_HEX_CAP - 1u) : 0u;
    memset(destination, 0, YVEX_SHA256_HEX_CAP);
    if (length) memcpy(destination, source, length);
}

static int transformer_hash_f64(yvex_sha256 *hash, double value)
{
    uint64_t bits;
    if (!isfinite(value)) return 0;
    memcpy(&bits, &value, sizeof(bits));
    return yvex_sha256_update_u64(hash, bits);
}

static int linear_physical_facts_valid(const yvex_transformer_linear_physical_plan *plan)
{
    size_t domain_length;
    if (!plan) return 0;
    domain_length = strnlen(plan->semantic_domain, sizeof(plan->semantic_domain));
    return plan->schema_version == YVEX_TRANSFORMER_LINEAR_PHYSICAL_SCHEMA_V1 &&
           domain_length > 0u && domain_length < sizeof(plan->semantic_domain) &&
           plan->operation >= YVEX_TRANSFORMER_LINEAR_OPERATION_JOINT_VIDEO_OUTPUT &&
           plan->operation <= YVEX_TRANSFORMER_LINEAR_OPERATION_JOINT_AUDIO_OUTPUT &&
           plan->implementation ==
               YVEX_TRANSFORMER_LINEAR_IMPLEMENTATION_CUBLAS_LT_F32_BIAS &&
           plan->reduction >= YVEX_TRANSFORMER_LINEAR_REDUCTION_INPLACE &&
           plan->reduction <= YVEX_TRANSFORMER_LINEAR_REDUCTION_COMPUTE_TYPE &&
           plan->stages <= YVEX_TRANSFORMER_LINEAR_STAGES_8X5 &&
           plan->backend == YVEX_BACKEND_KIND_CUDA && plan->algorithm_id &&
           plan->tile_rows && plan->tile_columns && plan->split_k > 1u &&
           plan->compute_capability_major && plan->input_width && plan->output_width &&
           plan->workspace_bytes && plan->deterministic == 1 && plan->exact == 1;
}

static int linear_operation_identity(yvex_transformer_linear_physical_plan *plan)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.transformer.linear.operation.v1") ||
        !yvex_sha256_update_text(&hash, plan->semantic_domain) ||
        !yvex_sha256_update_u64(&hash, plan->operation) ||
        !yvex_sha256_update_u64(&hash, plan->input_width) ||
        !yvex_sha256_update_u64(&hash, plan->output_width) ||
        !yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, plan->operation_identity);
    return 1;
}

static int linear_physical_identity(yvex_transformer_linear_physical_plan *plan)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.transformer.linear.physical.v1") ||
        !yvex_sha256_update_text(&hash, plan->operation_identity) ||
        !yvex_sha256_update_u64(&hash, plan->implementation) ||
        !yvex_sha256_update_u64(&hash, plan->reduction) ||
        !yvex_sha256_update_u64(&hash, plan->stages) ||
        !yvex_sha256_update_u64(&hash, plan->backend) ||
        !yvex_sha256_update_u64(&hash, plan->algorithm_id) ||
        !yvex_sha256_update_u64(&hash, plan->tile_rows) ||
        !yvex_sha256_update_u64(&hash, plan->tile_columns) ||
        !yvex_sha256_update_u64(&hash, plan->split_k) ||
        !yvex_sha256_update_u64(&hash, plan->compute_capability_major) ||
        !yvex_sha256_update_u64(&hash, plan->compute_capability_minor) ||
        !yvex_sha256_update_u64(&hash, plan->input_width) ||
        !yvex_sha256_update_u64(&hash, plan->output_width) ||
        !yvex_sha256_update_u64(&hash, plan->workspace_bytes) ||
        !yvex_sha256_update_u64(&hash, (unsigned long long)plan->deterministic) ||
        !yvex_sha256_update_u64(&hash, (unsigned long long)plan->exact) ||
        !yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, plan->physical_identity);
    return 1;
}

int yvex_transformer_linear_physical_seal(
    yvex_transformer_linear_physical_plan *plan, yvex_error *err)
{
    if (!plan)
        return transformer_refuse(err, YVEX_ERR_INVALID_ARG,
                                  "linear physical plan is required");
    plan->operation_identity[0] = '\0';
    plan->physical_identity[0] = '\0';
    if (!linear_physical_facts_valid(plan))
        return transformer_refuse(err, YVEX_ERR_FORMAT,
                                  "linear physical plan facts are malformed");
    if (!linear_operation_identity(plan) || !linear_physical_identity(plan))
        return transformer_refuse(err, YVEX_ERR_STATE,
                                  "linear physical plan identity derivation failed");
    yvex_error_clear(err);
    return YVEX_OK;
}

static int linear_physical_compile(
    const yvex_transformer_linear_physical_request *request,
    yvex_transformer_linear_physical_plan *plan, yvex_error *err)
{
    size_t domain_length;
    if (!request || !plan || !request->semantic_domain)
        return transformer_refuse(err, YVEX_ERR_INVALID_ARG,
                                  "linear physical compiler facts are required");
    domain_length = strnlen(request->semantic_domain, YVEX_TRANSFORMER_LINEAR_DOMAIN_CAP);
    if (!domain_length || domain_length >= YVEX_TRANSFORMER_LINEAR_DOMAIN_CAP)
        return transformer_refuse(err, YVEX_ERR_BOUNDS,
                                  "linear physical semantic domain exceeds its bound");
    memset(plan, 0, sizeof(*plan));
    plan->schema_version = request->schema_version;
    memcpy(plan->semantic_domain, request->semantic_domain, domain_length);
    plan->operation = request->operation;
    plan->implementation = request->implementation;
    plan->reduction = request->reduction;
    plan->stages = request->stages;
    plan->backend = request->backend;
    plan->algorithm_id = request->algorithm_id;
    plan->tile_rows = request->tile_rows;
    plan->tile_columns = request->tile_columns;
    plan->split_k = request->split_k;
    plan->compute_capability_major = request->compute_capability_major;
    plan->compute_capability_minor = request->compute_capability_minor;
    plan->input_width = request->input_width;
    plan->output_width = request->output_width;
    plan->workspace_bytes = request->workspace_bytes;
    plan->deterministic = request->deterministic;
    plan->exact = request->exact;
    return yvex_transformer_linear_physical_seal(plan, err);
}

int yvex_transformer_linear_physical_profile_compile(
    const char *semantic_domain, yvex_transformer_linear_operation operation,
    unsigned long long input_width, unsigned long long output_width,
    yvex_transformer_linear_profile profile, yvex_transformer_linear_physical_plan *plan,
    yvex_error *err)
{
    yvex_transformer_linear_physical_request request = {
        .schema_version = YVEX_TRANSFORMER_LINEAR_PHYSICAL_SCHEMA_V1,
        .semantic_domain = semantic_domain,
        .operation = operation,
        .implementation = YVEX_TRANSFORMER_LINEAR_IMPLEMENTATION_CUBLAS_LT_F32_BIAS,
        .backend = YVEX_BACKEND_KIND_CUDA,
        .compute_capability_major = 12u,
        .compute_capability_minor = 1u,
        .input_width = input_width,
        .output_width = output_width,
        .workspace_bytes = 1024ull * 1024ull,
        .deterministic = 1,
        .exact = 1,
    };
    if (profile == YVEX_TRANSFORMER_LINEAR_PROFILE_CUBLAS_LT_SM121_ALGORITHM_10) {
        request.algorithm_id = 10u;
        request.tile_rows = request.tile_columns = 32u;
        request.split_k = 10u;
        request.reduction = YVEX_TRANSFORMER_LINEAR_REDUCTION_INPLACE;
        request.stages = YVEX_TRANSFORMER_LINEAR_STAGES_DEFAULT;
    } else if (profile == YVEX_TRANSFORMER_LINEAR_PROFILE_CUBLAS_LT_SM121_ALGORITHM_20) {
        request.algorithm_id = 20u;
        request.tile_rows = 128u;
        request.tile_columns = 32u;
        request.split_k = 3u;
        request.reduction = YVEX_TRANSFORMER_LINEAR_REDUCTION_COMPUTE_TYPE;
        request.stages = YVEX_TRANSFORMER_LINEAR_STAGES_8X5;
    } else {
        return transformer_refuse(err, YVEX_ERR_UNSUPPORTED,
                                  "linear physical profile is not admitted");
    }
    return linear_physical_compile(&request, plan, err);
}

int yvex_transformer_linear_physical_validate(
    const yvex_transformer_linear_physical_plan *plan, yvex_error *err)
{
    yvex_transformer_linear_physical_plan copy;
    char expected_operation[YVEX_SHA256_HEX_CAP];
    char expected_physical[YVEX_SHA256_HEX_CAP];
    if (!plan || !yvex_sha256_hex_valid(plan->operation_identity) ||
        !yvex_sha256_hex_valid(plan->physical_identity))
        return transformer_refuse(err, YVEX_ERR_FORMAT,
                                  "sealed linear physical identities are required");
    copy = *plan;
    transformer_identity_copy(expected_operation, plan->operation_identity);
    transformer_identity_copy(expected_physical, plan->physical_identity);
    if (yvex_transformer_linear_physical_seal(&copy, err) != YVEX_OK)
        return yvex_error_code(err);
    if (strcmp(copy.operation_identity, expected_operation) != 0 ||
        strcmp(copy.physical_identity, expected_physical) != 0)
        return transformer_refuse(err, YVEX_ERR_STATE,
                                  "linear physical plan identity is stale");
    yvex_error_clear(err);
    return YVEX_OK;
}

static int transformer_layer_identity(yvex_transformer_layer_plan *out,
                                      const yvex_attention_layer_plan *attention,
                                      const yvex_moe_layer_plan *moe)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    if (!out || !attention || !moe || attention->layer_index != moe->layer_index ||
        attention->predictor_index != moe->predictor_index ||
        attention->tensor_scope != moe->tensor_scope ||
        !yvex_sha256_hex_valid(moe->layer_identity)) return 0;
    out->ordinal = moe->ordinal;
    out->layer_index = attention->layer_index;
    out->predictor_index = attention->predictor_index;
    out->tensor_scope = attention->tensor_scope;
    transformer_identity_copy(out->moe_layer_identity, moe->layer_identity);
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.transformer.layer.v2") ||
        !yvex_sha256_update_u64(&hash, out->ordinal) ||
        !yvex_sha256_update_u64(&hash, out->layer_index) ||
        !yvex_sha256_update_u64(&hash, out->predictor_index) ||
        !yvex_sha256_update_u64(&hash, out->tensor_scope) ||
        !yvex_sha256_update_u64(&hash, attention->attention_class) ||
        !yvex_sha256_update_u64(&hash, attention->hidden_dimension) ||
        !yvex_sha256_update_u64(&hash, attention->residual_stream_count) ||
        !yvex_sha256_update_u64(&hash, attention->residual_expanded_width) ||
        !yvex_sha256_update_text(&hash, moe->layer_identity) ||
        !yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, out->layer_identity);
    return 1;
}

static int transformer_binding_project(
    const yvex_materialization_session *materialization,
    const yvex_runtime_descriptor *descriptor, yvex_tensor_role role,
    yvex_tensor_scope scope, unsigned long long layer_index,
    unsigned long long predictor_index, yvex_transformer_weight_binding *out,
    yvex_error *err)
{
    const yvex_runtime_tensor_binding *runtime = yvex_runtime_descriptor_find_role(
        descriptor, role, scope, layer_index, predictor_index);
    const yvex_materialized_tensor_binding *binding = runtime
        ? yvex_materialization_session_tensor_at(materialization, runtime->tensor_id)
        : NULL;
    const yvex_quant_numeric_capability *capability = binding
        ? yvex_quant_numeric_capability_at(binding->qtype) : NULL;
    if (!binding || binding->role != role || !binding->encoded_bytes ||
        binding->expert_count > 1ull || !binding->backend_compatible)
        return transformer_refuse(err, YVEX_ERR_FORMAT,
                                  "transformer global binding is unavailable");
    if (!capability || !capability->reference_decoder_available ||
        !capability->dedicated_cpu_compute_available ||
        !capability->dedicated_cuda_compute_available)
        return transformer_refuse(
            err, YVEX_ERR_UNSUPPORTED,
            "transformer global binding qtype lacks CPU/CUDA execution");
    *out = (yvex_transformer_weight_binding){
        .tensor_id = binding->tensor_id,
        .row_width = binding->row_width,
        .row_count = binding->row_count,
        .encoded_bytes = binding->encoded_bytes,
        .role = binding->role,
        .tensor_scope = scope,
        .layer_index = layer_index,
        .predictor_index = predictor_index,
        .qtype = binding->qtype};
    return YVEX_OK;
}

/*
 * Identify the plan field-by-field.
 *
 * Writes identity.
 */
static int transformer_plan_identity(yvex_transformer_plan *plan)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;
    yvex_transformer_plan_summary *s = plan ? &plan->summary : NULL;
    if (!s) return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.transformer.plan.v2") ||
        !yvex_sha256_update_u64(&hash, s->schema_version) ||
        !yvex_sha256_update_u64(&hash, s->tensor_scope) ||
        !yvex_sha256_update_u64(&hash, s->family_adapter_id) ||
        !yvex_sha256_update_u64(&hash, s->family_adapter_version) ||
        !yvex_sha256_update_text(&hash, s->artifact_identity) ||
        !yvex_sha256_update_text(&hash, s->materialization_identity) ||
        !yvex_sha256_update_text(&hash, s->logical_model_identity) ||
        !yvex_sha256_update_text(&hash, s->runtime_numeric_identity) ||
        !yvex_sha256_update_text(&hash, s->runtime_descriptor_identity) ||
        !yvex_sha256_update_text(&hash, s->attention_plan_identity) ||
        !yvex_sha256_update_text(&hash, s->moe_plan_identity) ||
        !yvex_sha256_update_u64(&hash, s->layer_count) ||
        !yvex_sha256_update_u64(&hash, s->hidden_width) ||
        !yvex_sha256_update_u64(&hash, s->residual_streams) ||
        !yvex_sha256_update_u64(&hash, s->expanded_width) ||
        !yvex_sha256_update_u64(&hash, s->maximum_context) ||
        !yvex_sha256_update_u64(&hash, s->vocabulary_size) ||
        !yvex_sha256_update_u64(&hash, s->initial_policy) ||
        !yvex_sha256_update_u64(&hash, s->final_policy) ||
        !yvex_sha256_update_u64(&hash, s->sinkhorn_iterations) ||
        !transformer_hash_f64(&hash, s->mhc_epsilon) ||
        !transformer_hash_f64(&hash, s->output_norm_epsilon)) return 0;
    for (index = 0ull; index < YVEX_TRANSFORMER_WEIGHT_COUNT; ++index) {
        const yvex_transformer_weight_binding *weight = &s->weights[index];
        if (!yvex_sha256_update_u64(&hash, weight->tensor_id) ||
            !yvex_sha256_update_u64(&hash, weight->role) ||
            !yvex_sha256_update_u64(&hash, weight->tensor_scope) ||
            !yvex_sha256_update_u64(&hash, weight->layer_index) ||
            !yvex_sha256_update_u64(&hash, weight->predictor_index) ||
            !yvex_sha256_update_u64(&hash, weight->qtype) ||
            !yvex_sha256_update_u64(&hash, weight->row_width) ||
            !yvex_sha256_update_u64(&hash, weight->row_count) ||
            !yvex_sha256_update_u64(&hash, weight->encoded_bytes)) return 0;
    }
    for (index = 0ull; index < s->layer_count; ++index)
        if (!yvex_sha256_update_text(&hash, plan->layers[index].layer_identity)) return 0;
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, s->transformer_plan_identity);
    return 1;
}

static int transformer_plan_build(yvex_transformer_plan **out,
                                  const transformer_plan_facts *facts,
                                  const yvex_attention_plan *attention,
                                  const yvex_moe_plan *moe, yvex_error *err)
{
    yvex_transformer_plan *plan = NULL;
    const yvex_transformer_family_policy *policy = facts ? &facts->policy : NULL;
    const yvex_attention_summary *attention_summary = yvex_attention_plan_summary(attention);
    const yvex_moe_plan_summary *moe_summary = yvex_moe_plan_summary_get(moe);
    unsigned long long index;
    if (out) *out = NULL;
    if (!out || !facts || !policy || !attention_summary || !moe_summary)
        return transformer_refuse(err, YVEX_ERR_INVALID_ARG,
                                  "transformer plan requires complete family, attention, and MoE facts");
    if (policy->schema_version != YVEX_TRANSFORMER_PLAN_SCHEMA_V2)
        return transformer_refuse(err, YVEX_ERR_FORMAT,
                                  "transformer family policy schema is stale");
    if (facts->tensor_scope > YVEX_TENSOR_SCOPE_DRAFT ||
        attention_summary->tensor_scope != facts->tensor_scope ||
        moe_summary->tensor_scope != facts->tensor_scope)
        return transformer_refuse(err, YVEX_ERR_FORMAT,
                                  "transformer component tensor scopes disagree");
    if (!policy->attention_then_moe || !policy->deferred_ffn_post ||
        !policy->final_norm_after_head)
        return transformer_refuse(err, YVEX_ERR_FORMAT,
                                  "transformer family composition policy is incomplete");
    if (!policy->hidden_width || !policy->residual_streams ||
        !policy->expanded_width ||
        policy->expanded_width != policy->hidden_width * policy->residual_streams)
        return transformer_refuse(err, YVEX_ERR_FORMAT,
                                  "transformer family residual geometry is incompatible");
    if (facts->layer_count != attention_summary->layer_count ||
        facts->layer_count != moe_summary->layer_count)
        return transformer_refuse(err, YVEX_ERR_FORMAT,
                                  "transformer attention and MoE layer counts disagree");
    plan = (yvex_transformer_plan *)calloc(1u, sizeof(*plan));
    if (!plan) return transformer_refuse(err, YVEX_ERR_NOMEM, "transformer plan allocation failed");
    plan->layers = (yvex_transformer_layer_plan *)calloc(
        (size_t)facts->layer_count, sizeof(*plan->layers));
    if (!plan->layers) goto allocation;
    plan->summary.schema_version = YVEX_TRANSFORMER_PLAN_SCHEMA_V2;
    plan->summary.tensor_scope = facts->tensor_scope;
    plan->summary.family_adapter_id = facts->family_adapter_id;
    plan->summary.family_adapter_version = facts->family_adapter_version;
    plan->summary.layer_count = facts->layer_count;
    plan->summary.hidden_width = policy->hidden_width;
    plan->summary.residual_streams = policy->residual_streams;
    plan->summary.expanded_width = policy->expanded_width;
    plan->summary.maximum_context = policy->maximum_context;
    plan->summary.vocabulary_size = facts->vocabulary_size;
    plan->summary.initial_policy = policy->initial_policy;
    plan->summary.final_policy = policy->final_policy;
    plan->summary.sinkhorn_iterations = policy->sinkhorn_iterations;
    plan->summary.mhc_epsilon = policy->mhc_epsilon;
    plan->summary.output_norm_epsilon = policy->output_norm_epsilon;
#define COPY_ID(member_, value_) transformer_identity_copy(plan->summary.member_, (value_))
    COPY_ID(artifact_identity, facts->artifact_identity);
    COPY_ID(materialization_identity, facts->materialization_identity);
    COPY_ID(logical_model_identity, facts->logical_model_identity);
    COPY_ID(runtime_numeric_identity, facts->runtime_numeric_identity);
    COPY_ID(runtime_descriptor_identity, facts->runtime_descriptor_identity);
    COPY_ID(attention_plan_identity, attention_summary->attention_plan_identity);
    COPY_ID(moe_plan_identity, moe_summary->moe_plan_identity);
#undef COPY_ID
    for (index = 0ull; index < YVEX_TRANSFORMER_WEIGHT_COUNT; ++index) {
        unsigned long long width = index == YVEX_TRANSFORMER_WEIGHT_EMBEDDING ||
                                           index == YVEX_TRANSFORMER_WEIGHT_OUTPUT_NORM
                                       ? policy->hidden_width
                                       : index == YVEX_TRANSFORMER_WEIGHT_FINAL_FUNCTION
                                             ? policy->expanded_width
                                             : index == YVEX_TRANSFORMER_WEIGHT_FINAL_BASE
                                                   ? policy->residual_streams : 1ull;
        unsigned long long row_count = index == YVEX_TRANSFORMER_WEIGHT_EMBEDDING
                                           ? facts->vocabulary_size
                                           : index == YVEX_TRANSFORMER_WEIGHT_FINAL_FUNCTION
                                                 ? policy->residual_streams : 1ull;
        if (!facts->weights[index].encoded_bytes ||
            facts->weights[index].row_width != width ||
            facts->weights[index].row_count != row_count)
            goto geometry;
        plan->summary.weights[index] = facts->weights[index];
    }
    for (index = 0ull; index < facts->layer_count; ++index)
        if (!transformer_layer_identity(&plan->layers[index],
                                        yvex_attention_plan_layer_at(attention, index),
                                        yvex_moe_plan_layer_at(moe, index)))
            goto identity;
    if (!transformer_plan_identity(plan)) goto identity;
    *out = plan;
    yvex_error_clear(err);
    return YVEX_OK;
allocation:
    transformer_refuse(err, YVEX_ERR_NOMEM, "transformer layer directory allocation failed");
    goto failure;
identity:
    transformer_refuse(err, YVEX_ERR_STATE, "transformer plan identity derivation failed");
    goto failure;
geometry:
    transformer_refuse(err, YVEX_ERR_FORMAT,
                       "transformer global binding geometry is incompatible");
failure:
    yvex_transformer_plan_close(&plan);
    return yvex_error_code(err);
}

/* Compile global bindings and family policy before runtime model publication. */
int yvex_transformer_plan_compile(
    yvex_transformer_plan **out, const yvex_transformer_family_policy *policy,
    unsigned long long family_adapter_id,
    unsigned long long family_adapter_version,
    const yvex_materialization_session *materialization,
    const yvex_runtime_descriptor *descriptor,
    const yvex_attention_plan *attention, const yvex_moe_plan *moe,
    yvex_tensor_scope execution_scope, yvex_error *err)
{
    const yvex_runtime_descriptor_summary *runtime =
        yvex_runtime_descriptor_summary_get(descriptor);
    const yvex_materialization_summary *material =
        yvex_materialization_session_summary(materialization);
    const yvex_attention_summary *attention_summary =
        yvex_attention_plan_summary(attention);
    const yvex_attention_layer_plan *last;
    transformer_plan_facts facts = {0};
    unsigned long long slot;
    if (out) *out = NULL;
    if (!out || !policy || !family_adapter_id || !family_adapter_version ||
        !runtime || !material || !attention_summary ||
        !attention_summary->layer_count || !moe ||
        (execution_scope != YVEX_TENSOR_SCOPE_GLOBAL &&
         execution_scope != YVEX_TENSOR_SCOPE_DRAFT))
        return transformer_refuse(err, YVEX_ERR_INVALID_ARG,
                                  "transformer compiler facts are incomplete");
    facts.policy = *policy;
    facts.family_adapter_id = family_adapter_id;
    facts.family_adapter_version = family_adapter_version;
    facts.tensor_scope = execution_scope == YVEX_TENSOR_SCOPE_GLOBAL
                             ? YVEX_TENSOR_SCOPE_MAIN_LAYER
                             : YVEX_TENSOR_SCOPE_DRAFT;
    facts.layer_count = attention_summary->layer_count;
    facts.vocabulary_size = runtime->vocabulary_size;
    facts.artifact_identity = material->artifact_identity;
    facts.materialization_identity = material->plan_identity;
    facts.logical_model_identity = runtime->logical_model_identity;
    facts.runtime_numeric_identity = runtime->runtime_numeric_identity;
    facts.runtime_descriptor_identity = runtime->runtime_descriptor_identity;
    last = yvex_attention_plan_layer_at(attention,
                                        attention_summary->layer_count - 1ull);
    if (!last)
        return transformer_refuse(err, YVEX_ERR_STATE,
                                  "transformer final layer is unavailable");
    for (slot = 0ull; slot < YVEX_TRANSFORMER_WEIGHT_COUNT; ++slot) {
        yvex_tensor_role role = transformer_weight_roles[slot];
        yvex_tensor_scope scope = YVEX_TENSOR_SCOPE_GLOBAL;
        unsigned long long layer = YVEX_MATERIALIZATION_NO_INDEX;
        unsigned long long predictor = YVEX_MATERIALIZATION_NO_INDEX;
        if (execution_scope == YVEX_TENSOR_SCOPE_DRAFT &&
            slot != YVEX_TRANSFORMER_WEIGHT_EMBEDDING) {
            scope = YVEX_TENSOR_SCOPE_DRAFT;
            layer = last->layer_index;
            predictor = last->predictor_index;
            if (slot == YVEX_TRANSFORMER_WEIGHT_OUTPUT_NORM)
                role = YVEX_TENSOR_ROLE_DRAFT_OUTPUT_NORM;
        }
        if (transformer_binding_project(
                materialization, descriptor, role, scope, layer, predictor,
                &facts.weights[slot], err) != YVEX_OK)
            return yvex_error_code(err);
    }
    return transformer_plan_build(out, &facts, attention, moe, err);
}

/*
 * Independently reopen one pointer-free transformer plan representation.
 *
 * Identity/geometry mismatch publishes no plan.
 */
int yvex_transformer_plan_import(yvex_transformer_plan **out,
                                 const yvex_transformer_plan_summary *summary,
                                 const yvex_transformer_layer_plan *layers,
                                 yvex_error *err)
{
    yvex_transformer_plan *plan;
    char expected[YVEX_SHA256_HEX_CAP];
    unsigned long long index;
    if (out) *out = NULL;
    if (!out || !summary || !layers ||
        summary->schema_version != YVEX_TRANSFORMER_PLAN_SCHEMA_V2 ||
        summary->tensor_scope > YVEX_TENSOR_SCOPE_DRAFT ||
        !summary->layer_count || !summary->hidden_width || !summary->residual_streams ||
        summary->expanded_width != summary->hidden_width * summary->residual_streams ||
        !yvex_sha256_hex_valid(summary->transformer_plan_identity))
        return transformer_refuse(err, YVEX_ERR_INVALID_ARG,
                                  "transformer plan import facts are invalid");
    plan = (yvex_transformer_plan *)calloc(1u, sizeof(*plan));
    if (!plan) return transformer_refuse(err, YVEX_ERR_NOMEM,
                                        "transformer imported plan allocation failed");
    plan->layers = (yvex_transformer_layer_plan *)malloc(
        (size_t)summary->layer_count * sizeof(*plan->layers));
    if (!plan->layers) {
        free(plan);
        return transformer_refuse(err, YVEX_ERR_NOMEM,
                                  "transformer imported layers allocation failed");
    }
    plan->summary = *summary;
    memcpy(plan->layers, layers, (size_t)summary->layer_count * sizeof(*layers));
    transformer_identity_copy(expected, summary->transformer_plan_identity);
    plan->summary.transformer_plan_identity[0] = '\0';
    for (index = 0ull; index < summary->layer_count; ++index)
        if (plan->layers[index].ordinal != index ||
            plan->layers[index].tensor_scope != summary->tensor_scope ||
            !yvex_sha256_hex_valid(plan->layers[index].moe_layer_identity) ||
            !yvex_sha256_hex_valid(plan->layers[index].layer_identity))
            goto invalid;
    if (!transformer_plan_identity(plan) ||
        strcmp(plan->summary.transformer_plan_identity, expected) != 0)
        goto invalid;
    *out = plan;
    yvex_error_clear(err);
    return YVEX_OK;
invalid:
    yvex_transformer_plan_close(&plan);
    return transformer_refuse(err, YVEX_ERR_STATE,
                              "transformer imported plan identity is stale");
}

/*
 * Seal a pointer-free transformer plan representation for transactional publication.
 *
 * Writes only its canonical identity. Malformed facts leave identity empty.
 */
int yvex_transformer_plan_seal(yvex_transformer_plan_summary *summary,
                               const yvex_transformer_layer_plan *layers,
                               yvex_error *err)
{
    yvex_transformer_plan plan;
    unsigned long long index;
    if (!summary || !layers || summary->schema_version != YVEX_TRANSFORMER_PLAN_SCHEMA_V2 ||
        summary->tensor_scope > YVEX_TENSOR_SCOPE_DRAFT ||
        !summary->layer_count || !summary->hidden_width || !summary->residual_streams ||
        summary->expanded_width != summary->hidden_width * summary->residual_streams)
        return transformer_refuse(err, YVEX_ERR_INVALID_ARG,
                                  "transformer plan seal facts are invalid");
    for (index = 0ull; index < summary->layer_count; ++index)
        if (layers[index].ordinal != index ||
            layers[index].tensor_scope != summary->tensor_scope ||
            !yvex_sha256_hex_valid(layers[index].moe_layer_identity) ||
            !yvex_sha256_hex_valid(layers[index].layer_identity))
            return transformer_refuse(err, YVEX_ERR_FORMAT,
                                      "transformer plan layer directory is malformed");
    memset(&plan, 0, sizeof(plan));
    plan.summary = *summary;
    plan.summary.transformer_plan_identity[0] = '\0';
    plan.layers = (yvex_transformer_layer_plan *)layers;
    if (!transformer_plan_identity(&plan))
        return transformer_refuse(err, YVEX_ERR_STATE,
                                  "transformer plan seal identity failed");
    transformer_identity_copy(summary->transformer_plan_identity,
                              plan.summary.transformer_plan_identity);
    yvex_error_clear(err);
    return YVEX_OK;
}

/*
 * Borrow one immutable plan summary.
 *
 * Borrowed lifetime.
 */
const yvex_transformer_plan_summary *yvex_transformer_plan_summary_get(
    const yvex_transformer_plan *plan)
{
    return plan ? &plan->summary : NULL;
}

const yvex_transformer_layer_plan *yvex_transformer_plan_layer_at(
    const yvex_transformer_plan *plan, unsigned long long ordinal)
{
    return plan && ordinal < plan->summary.layer_count
               ? &plan->layers[ordinal] : NULL;
}

void yvex_transformer_plan_close(yvex_transformer_plan **plan)
{
    if (!plan || !*plan) return;
    free((*plan)->layers);
    memset(*plan, 0, sizeof(**plan));
    free(*plan);
    *plan = NULL;
}

int yvex_transformer_initial_residual(const yvex_transformer_plan *plan,
                                      const float *embedding, unsigned long long token_count,
                                      float *expanded, yvex_error *err)
{
    unsigned long long token, stream, lane;
    const yvex_transformer_plan_summary *s = yvex_transformer_plan_summary_get(plan);
    if (!s || !embedding || !expanded || !token_count ||
        s->initial_policy != YVEX_TRANSFORMER_INITIAL_REPEAT_STREAMS)
        return transformer_refuse(err, YVEX_ERR_INVALID_ARG,
                                  "transformer initial residual arguments are invalid");
    for (token = 0ull; token < token_count; ++token)
        for (stream = 0ull; stream < s->residual_streams; ++stream)
            for (lane = 0ull; lane < s->hidden_width; ++lane) {
                float value = embedding[token * s->hidden_width + lane];
                if (!isfinite(value))
                    return transformer_refuse(err, YVEX_ERR_FORMAT,
                                              "embedding contains a non-finite value");
                expanded[token * s->expanded_width + stream * s->hidden_width + lane] =
                    yvex_quant_bf16_decode(yvex_quant_bf16_encode(value));
            }
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_transformer_deferred_post(const yvex_transformer_plan *plan,
                                   const float *residual, const float *combined,
                                   const float *post, const float *combination,
                                   unsigned long long token_count, float *expanded,
                                   yvex_error *err)
{
    const yvex_transformer_plan_summary *s = yvex_transformer_plan_summary_get(plan);
    unsigned long long token, target, source, lane;
    if (!s || !residual || !combined || !post || !combination || !expanded || !token_count)
        return transformer_refuse(err, YVEX_ERR_INVALID_ARG,
                                  "deferred FFN post arguments are invalid");
    for (token = 0ull; token < token_count; ++token)
        for (target = 0ull; target < s->residual_streams; ++target)
            for (lane = 0ull; lane < s->hidden_width; ++lane) {
                double value = (double)post[token * s->residual_streams + target] *
                               (double)combined[token * s->hidden_width + lane];
                for (source = 0ull; source < s->residual_streams; ++source)
                    value += (double)combination[(token * s->residual_streams + source) *
                                                     s->residual_streams + target] *
                             (double)residual[token * s->expanded_width +
                                              source * s->hidden_width + lane];
                if (!isfinite(value))
                    return transformer_refuse(err, YVEX_ERR_FORMAT,
                                              "deferred FFN post produced a non-finite value");
                expanded[token * s->expanded_width + target * s->hidden_width + lane] =
                    yvex_quant_bf16_decode(yvex_quant_bf16_encode((float)value));
            }
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_transformer_final_stage_capture(
    const yvex_transformer_plan *plan, const float *expanded,
    unsigned long long token_count, const float *function, const float *base,
    const float *scale, const float *norm, float *pre_normalized,
    float *normalized, yvex_error *err)
{
    const yvex_transformer_plan_summary *s = yvex_transformer_plan_summary_get(plan);
    unsigned long long token, stream, lane, index;
    if (!s || !expanded || !function || !base || !scale || !norm || !normalized ||
        pre_normalized == normalized ||
        !token_count || s->final_policy != YVEX_TRANSFORMER_FINAL_SIGMOID_MHC_RMS)
        return transformer_refuse(err, YVEX_ERR_INVALID_ARG,
                                  "transformer final-stage arguments are invalid");
    for (token = 0ull; token < token_count; ++token) {
        const float *input = expanded + token * s->expanded_width;
        float *output = normalized + token * s->hidden_width;
        double squares = 0.0, inverse;
        for (index = 0ull; index < s->expanded_width; ++index)
            squares += (double)input[index] * (double)input[index];
        inverse = 1.0 / sqrt(squares / (double)s->expanded_width + s->output_norm_epsilon);
        if (!isfinite(inverse)) goto numeric;
        memset(output, 0, (size_t)s->hidden_width * sizeof(float));
        for (stream = 0ull; stream < s->residual_streams; ++stream) {
            double mix = 0.0, coefficient;
            for (index = 0ull; index < s->expanded_width; ++index)
                mix += (double)function[stream * s->expanded_width + index] *
                       (double)input[index];
            coefficient = 1.0 / (1.0 + exp(-(mix * inverse * (double)scale[0] +
                                             (double)base[stream])));
            coefficient += s->mhc_epsilon;
            for (lane = 0ull; lane < s->hidden_width; ++lane)
                output[lane] += (float)(coefficient *
                    (double)input[stream * s->hidden_width + lane]);
        }
        if (!yvex_attention_compute_round(YVEX_ATTENTION_COMPUTE_BF16_F32_RNE_V1,
                                          output, s->hidden_width))
            goto numeric;
        if (pre_normalized)
            memcpy(pre_normalized + token * s->hidden_width, output,
                   (size_t)s->hidden_width * sizeof(float));
        if (!yvex_attention_rms_norm(output, s->hidden_width, norm,
                                     s->output_norm_epsilon) ||
            !yvex_attention_compute_round(YVEX_ATTENTION_COMPUTE_BF16_F32_RNE_V1,
                                          output, s->hidden_width))
            goto numeric;
    }
    yvex_error_clear(err);
    return YVEX_OK;
numeric:
    return transformer_refuse(err, YVEX_ERR_FORMAT,
                              "transformer final head or RMSNorm produced non-finite values");
}

int yvex_transformer_final_stage(const yvex_transformer_plan *plan,
                                 const float *expanded, unsigned long long token_count,
                                 const float *function, const float *base, const float *scale,
                                 const float *norm, float *normalized, yvex_error *err)
{
    return yvex_transformer_final_stage_capture(
        plan, expanded, token_count, function, base, scale, norm, NULL,
        normalized, err);
}

int yvex_transformer_feature_normalize(float *values,
                                       unsigned long long value_count,
                                       const float *weights, double epsilon,
                                       yvex_error *err)
{
    if (!values || !value_count || !weights || !isfinite(epsilon) ||
        epsilon <= 0.0)
        return transformer_refuse(err, YVEX_ERR_INVALID_ARG,
                                  "draft feature normalization facts are invalid");
    if (!yvex_attention_rms_norm(values, value_count, weights, epsilon) ||
        !yvex_attention_compute_round(YVEX_ATTENTION_COMPUTE_BF16_F32_RNE_V1,
                                      values, value_count))
        return transformer_refuse(err, YVEX_ERR_FORMAT,
                                  "draft feature normalization produced non-finite values");
    yvex_error_clear(err);
    return YVEX_OK;
}
