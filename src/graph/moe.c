/*
 * Lower typed family MoE facts into an executable plan and canonical CPU operations.
 *
 * Plans bind exact descriptor roles and expert execution touches only selected subviews. Reusable
 * token-local MoE math and plan admission.
 */
#include <yvex/internal/moe.h>
#include "src/graph/private.h"
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <yvex/internal/core.h>
#include <yvex/internal/quant_numeric.h>
struct yvex_moe_plan {
    yvex_moe_plan_summary summary;
    yvex_moe_layer_plan *layers;
};
static const yvex_tensor_role moe_slot_roles[YVEX_MOE_WEIGHT_COUNT] = {
    YVEX_TENSOR_ROLE_FFN_NORM, YVEX_TENSOR_ROLE_HC_FFN_FUNCTION,
    YVEX_TENSOR_ROLE_HC_FFN_BASE, YVEX_TENSOR_ROLE_HC_FFN_SCALE,
    YVEX_TENSOR_ROLE_MOE_ROUTER, YVEX_TENSOR_ROLE_MOE_ROUTER_TABLE,
    YVEX_TENSOR_ROLE_MOE_ROUTER_BIAS, YVEX_TENSOR_ROLE_MOE_EXPERT_GATE,
    YVEX_TENSOR_ROLE_MOE_EXPERT_UP, YVEX_TENSOR_ROLE_MOE_EXPERT_DOWN,
    YVEX_TENSOR_ROLE_MOE_SHARED_EXPERT_GATE, YVEX_TENSOR_ROLE_MOE_SHARED_EXPERT_UP,
    YVEX_TENSOR_ROLE_MOE_SHARED_EXPERT_DOWN};

static int moe_refuse(yvex_error *err, yvex_status status, const char *reason)
{
    yvex_error_set(err, status, "graph.moe", reason);
    return status;
}
/*
 * Identify routing evidence field-by-field without native structure layout.
 *
 * Excludes padding and unused capacity.
 */
int yvex_moe_router_result_identity(const yvex_moe_router_result *router,
                                    unsigned long long routed_experts,
                                    char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;
    uint32_t bits;
    if (!router || !output || !router->selected_count ||
        router->selected_count > YVEX_MOE_MAX_SELECTED || !routed_experts ||
        routed_experts > 256ull) return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.moe.router-result.v1") ||
        !yvex_sha256_update_u64(&hash, router->selected_count)) return 0;
    for (index = 0ull; index < router->selected_count; ++index)
        if (router->selected_experts[index] >= routed_experts ||
            !yvex_sha256_update_u64(&hash, router->selected_experts[index])) return 0;
    for (index = 0ull; index < routed_experts; ++index) {
        if (!isfinite(router->router_logits[index]) ||
            !isfinite(router->router_scores[index])) return 0;
        memcpy(&bits, &router->router_logits[index], sizeof(bits));
        if (!yvex_sha256_update_u64(&hash, bits)) return 0;
        memcpy(&bits, &router->router_scores[index], sizeof(bits));
        if (!yvex_sha256_update_u64(&hash, bits)) return 0;
    }
    for (index = 0ull; index < router->selected_count; ++index) {
        if (!isfinite(router->selected_weights[index])) return 0;
        memcpy(&bits, &router->selected_weights[index], sizeof(bits));
        if (!yvex_sha256_update_u64(&hash, bits)) return 0;
    }
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

static const yvex_moe_family_api *moe_family_find(unsigned long long adapter_id,
                                                   unsigned long long adapter_version)
{
    unsigned long long index;
    for (index = 0ull;; ++index) {
        const yvex_moe_family_api *family = yvex_graph_moe_family_at(index);
        if (!family) return NULL;
        if (family->adapter_id == adapter_id && family->adapter_version == adapter_version)
            return family;
    }
}

static const yvex_materialized_tensor_binding *moe_binding_find(
    const yvex_materialization_session *materialization,
    const yvex_runtime_descriptor *descriptor, yvex_tensor_role role,
    unsigned long long layer_index)
{
    const yvex_runtime_tensor_binding *runtime = yvex_runtime_descriptor_find_role(
        descriptor, role, YVEX_TENSOR_SCOPE_MAIN_LAYER, layer_index,
        YVEX_MATERIALIZATION_NO_INDEX);
    return runtime ? yvex_materialization_session_tensor_at(materialization,
                                                             runtime->tensor_id) : NULL;
}

static int moe_binding_geometry(const yvex_materialized_tensor_binding *binding,
                                unsigned long long width, unsigned long long rows,
                                unsigned long long experts, int numerical)
{
    const yvex_quant_numeric_capability *capability;
    if (!binding || binding->row_width != width || binding->row_count != rows ||
        (experts > 1ull ? binding->expert_count != experts : binding->expert_count > 1ull) ||
        !binding->encoded_bytes)
        return 0;
    if (!numerical) return 1;
    if (!binding->backend_compatible) return 0;
    capability = yvex_quant_numeric_capability_at(binding->qtype);
    return capability && capability->dedicated_cpu_compute_available &&
           capability->dedicated_cuda_compute_available;
}

static int moe_layer_bind(yvex_moe_layer_plan *layer,
                          const yvex_materialization_session *materialization,
                          const yvex_runtime_descriptor *descriptor, yvex_error *err)
{
    const yvex_materialized_tensor_binding *bindings[YVEX_MOE_WEIGHT_COUNT] = {0};
    unsigned long long routed_rows, routed_down_rows, slot;
    if (!layer || !yvex_core_u64_mul(layer->expert_intermediate_width,
                                     layer->routed_experts, &routed_rows) ||
        !yvex_core_u64_mul(layer->hidden_width, layer->routed_experts,
                          &routed_down_rows))
        return moe_refuse(err, YVEX_ERR_BOUNDS, "MoE routed expert geometry overflowed");
    for (slot = 0ull; slot < YVEX_MOE_WEIGHT_COUNT; ++slot) {
        int absent = (slot == YVEX_MOE_WEIGHT_ROUTER_TABLE &&
                      layer->router_class != YVEX_MOE_ROUTER_HASH_TOKEN_ID) ||
                     (slot == YVEX_MOE_WEIGHT_ROUTER_BIAS &&
                      layer->router_class != YVEX_MOE_ROUTER_LEARNED_HIDDEN_STATE);
        layer->tensor_ids[slot] = YVEX_MOE_NO_TENSOR;
        if (absent) continue;
        bindings[slot] = moe_binding_find(materialization, descriptor,
                                          moe_slot_roles[slot], layer->layer_index);
        if (!bindings[slot]) {
            yvex_error_setf(err, YVEX_ERR_FORMAT, "graph.moe",
                            "MoE layer %llu is missing required role %s",
                            layer->layer_index, yvex_tensor_role_name(moe_slot_roles[slot]));
            return YVEX_ERR_FORMAT;
        }
        layer->tensor_ids[slot] = bindings[slot]->tensor_id;
        layer->qtypes[slot] = bindings[slot]->qtype;
    }
#define GEOM(slot_, width_, rows_, experts_) \
    moe_binding_geometry(bindings[(slot_)], (width_), (rows_), (experts_), 1)
    if (!GEOM(YVEX_MOE_WEIGHT_FFN_NORM, layer->hidden_width, 1ull, 1ull) ||
        !GEOM(YVEX_MOE_WEIGHT_MHC_FUNCTION, layer->expanded_width,
              layer->mhc_mixing_rows, 1ull) ||
        !GEOM(YVEX_MOE_WEIGHT_MHC_BASE, layer->mhc_mixing_rows, 1ull, 1ull) ||
        !GEOM(YVEX_MOE_WEIGHT_MHC_SCALE, 3ull, 1ull, 1ull) ||
        !GEOM(YVEX_MOE_WEIGHT_ROUTER, layer->hidden_width,
              layer->routed_experts, 1ull) ||
        !GEOM(YVEX_MOE_WEIGHT_ROUTED_GATE, layer->hidden_width, routed_rows,
              layer->routed_experts) ||
        !GEOM(YVEX_MOE_WEIGHT_ROUTED_UP, layer->hidden_width, routed_rows,
              layer->routed_experts) ||
        !GEOM(YVEX_MOE_WEIGHT_ROUTED_DOWN, layer->expert_intermediate_width,
              routed_down_rows, layer->routed_experts) ||
        !GEOM(YVEX_MOE_WEIGHT_SHARED_GATE, layer->hidden_width,
              layer->shared_intermediate_width, 1ull) ||
        !GEOM(YVEX_MOE_WEIGHT_SHARED_UP, layer->hidden_width,
              layer->shared_intermediate_width, 1ull) ||
        !GEOM(YVEX_MOE_WEIGHT_SHARED_DOWN, layer->shared_intermediate_width,
              layer->hidden_width, 1ull))
        return moe_refuse(err, YVEX_ERR_FORMAT,
                          "MoE encoded expert or envelope geometry is incompatible");
    if (layer->router_class == YVEX_MOE_ROUTER_HASH_TOKEN_ID &&
        !moe_binding_geometry(bindings[YVEX_MOE_WEIGHT_ROUTER_TABLE],
                              layer->hash_table_columns, layer->hash_table_rows,
                              1ull, 0)) {
        const yvex_materialized_tensor_binding *table =
            bindings[YVEX_MOE_WEIGHT_ROUTER_TABLE];
        yvex_error_setf(err, YVEX_ERR_FORMAT, "graph.moe",
                        "MoE hash router expected %llux%llu but artifact has %llux%llu experts=%llu bytes=%llu",
                        layer->hash_table_columns, layer->hash_table_rows,
                        table ? table->row_width : 0ull, table ? table->row_count : 0ull,
                        table ? table->expert_count : 0ull,
                        table ? table->encoded_bytes : 0ull);
        return YVEX_ERR_FORMAT;
    }
    if (layer->router_class == YVEX_MOE_ROUTER_LEARNED_HIDDEN_STATE &&
        !GEOM(YVEX_MOE_WEIGHT_ROUTER_BIAS, layer->correction_bias_width, 1ull, 1ull))
        return moe_refuse(err, YVEX_ERR_FORMAT, "MoE correction bias geometry is invalid");
#undef GEOM
    return YVEX_OK;
}

static int moe_hash_f64(yvex_sha256 *hash, double value)
{
    unsigned long long bits = 0ull;
    memcpy(&bits, &value, sizeof(bits));
    return yvex_sha256_update_u64(hash, bits);
}
/*
 * Hash one layer from canonical scalar fields and bound tensor identities.
 *
 * Seals identity.
 */
static int moe_layer_identity(yvex_moe_layer_plan *layer,
                              const yvex_moe_plan_summary *summary)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long slot;
    const unsigned long long fields[] = {
        layer->schema_version, layer->ordinal, layer->layer_index, layer->router_class,
        layer->scoring, layer->topk_policy, layer->activation, layer->hidden_width,
        layer->residual_streams, layer->expanded_width, layer->mhc_mixing_rows,
        layer->mhc_sinkhorn_iterations, layer->routed_experts, layer->shared_experts,
        layer->experts_per_token, layer->expert_intermediate_width,
        layer->shared_intermediate_width, layer->hash_table_rows,
        layer->hash_table_columns, layer->correction_bias_width,
        (unsigned long long)layer->requires_token_ids,
        (unsigned long long)layer->requires_correction_bias,
        (unsigned long long)layer->normalize_topk_probabilities};
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.moe.layer.v1") ||
        !yvex_sha256_update_text(&hash, summary->runtime_descriptor_identity)) return 0;
    for (slot = 0ull; slot < sizeof(fields) / sizeof(fields[0]); ++slot)
        if (!yvex_sha256_update_u64(&hash, fields[slot])) return 0;
    if (!moe_hash_f64(&hash, layer->rms_epsilon) ||
        !moe_hash_f64(&hash, layer->mhc_epsilon) ||
        !moe_hash_f64(&hash, layer->mhc_post_multiplier) ||
        !moe_hash_f64(&hash, layer->routed_scaling_factor) ||
        !moe_hash_f64(&hash, layer->activation_limit)) return 0;
    for (slot = 0ull; slot < YVEX_MOE_WEIGHT_COUNT; ++slot)
        if (!yvex_sha256_update_u64(&hash, layer->tensor_ids[slot]) ||
            !yvex_sha256_update_u64(&hash, layer->qtypes[slot])) return 0;
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, layer->layer_identity);
    return 1;
}

static int moe_plan_identity(yvex_moe_plan *plan)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.moe.plan.v1") ||
        !yvex_sha256_update_u64(&hash, plan->summary.schema_version) ||
        !yvex_sha256_update_u64(&hash, plan->summary.family_adapter_id) ||
        !yvex_sha256_update_u64(&hash, plan->summary.family_adapter_version) ||
        !yvex_sha256_update_text(&hash, plan->summary.artifact_identity) ||
        !yvex_sha256_update_text(&hash, plan->summary.materialization_identity) ||
        !yvex_sha256_update_text(&hash, plan->summary.logical_model_identity) ||
        !yvex_sha256_update_text(&hash, plan->summary.runtime_numeric_identity) ||
        !yvex_sha256_update_text(&hash, plan->summary.runtime_descriptor_identity) ||
        !yvex_sha256_update_text(&hash, plan->summary.attention_plan_identity)) return 0;
    for (index = 0ull; index < plan->summary.layer_count; ++index)
        if (!yvex_sha256_update_text(&hash, plan->layers[index].layer_identity)) return 0;
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, plan->summary.moe_plan_identity);
    return 1;
}

int yvex_moe_plan_build(yvex_moe_plan **out, unsigned long long adapter_id,
                        unsigned long long adapter_version,
                        const yvex_materialization_session *materialization,
                        const yvex_runtime_descriptor *descriptor,
                        const yvex_attention_plan *attention, yvex_error *err)
{
    const yvex_moe_family_api *family = moe_family_find(adapter_id, adapter_version);
    const yvex_runtime_descriptor_summary *descriptor_summary =
        yvex_runtime_descriptor_summary_get(descriptor);
    const yvex_attention_summary *attention_summary = yvex_attention_plan_summary(attention);
    const yvex_materialization_summary *material_summary =
        yvex_materialization_session_summary(materialization);
    yvex_moe_plan *plan = NULL;
    unsigned long long index;
    if (out) *out = NULL;
    if (!out || !family || !family->project_layer || !descriptor_summary ||
        !attention_summary || !material_summary || !attention_summary->layer_count)
        return moe_refuse(err, YVEX_ERR_INVALID_ARG,
                          "complete runtime facts and a typed MoE family adapter are required");
    plan = (yvex_moe_plan *)calloc(1u, sizeof(*plan));
    if (!plan || attention_summary->layer_count > SIZE_MAX / sizeof(*plan->layers) ||
        !(plan->layers = (yvex_moe_layer_plan *)calloc(
              (size_t)attention_summary->layer_count, sizeof(*plan->layers)))) {
        yvex_moe_plan_close(&plan);
        return moe_refuse(err, YVEX_ERR_NOMEM, "MoE plan allocation failed");
    }
    plan->summary.schema_version = YVEX_MOE_PLAN_SCHEMA_V1;
    plan->summary.family_adapter_id = adapter_id;
    plan->summary.family_adapter_version = adapter_version;
    plan->summary.layer_count = attention_summary->layer_count;
    plan->summary.routed_experts = descriptor_summary->routed_experts;
    plan->summary.experts_per_token = descriptor_summary->experts_per_token;
#define COPY_ID(member_, source_) \
    yvex_core_text_copy(plan->summary.member_, sizeof(plan->summary.member_), (source_))
    COPY_ID(artifact_identity, descriptor_summary->artifact_identity);
    COPY_ID(materialization_identity, material_summary->plan_identity);
    COPY_ID(logical_model_identity, descriptor_summary->logical_model_identity);
    COPY_ID(runtime_numeric_identity, descriptor_summary->runtime_numeric_identity);
    COPY_ID(runtime_descriptor_identity, descriptor_summary->runtime_descriptor_identity);
    COPY_ID(attention_plan_identity, attention_summary->attention_plan_identity);
#undef COPY_ID
    for (index = 0ull; index < plan->summary.layer_count; ++index) {
        yvex_moe_layer_plan *layer = &plan->layers[index];
        const yvex_attention_layer_plan *attention_layer =
            yvex_attention_plan_layer_at(attention, index);
        if (!attention_layer ||
            family->project_layer(index, descriptor_summary, attention_layer, layer, err) != YVEX_OK ||
            layer->ordinal != index || layer->layer_index != index ||
            moe_layer_bind(layer, materialization, descriptor, err) != YVEX_OK ||
            !moe_layer_identity(layer, &plan->summary)) {
            yvex_moe_plan_close(&plan);
            if (!yvex_error_is_set(err))
                moe_refuse(err, YVEX_ERR_FORMAT, "MoE family layer projection is inconsistent");
            return yvex_error_code(err);
        }
        plan->summary.hash_router_layer_count +=
            layer->router_class == YVEX_MOE_ROUTER_HASH_TOKEN_ID;
        plan->summary.learned_router_layer_count +=
            layer->router_class == YVEX_MOE_ROUTER_LEARNED_HIDDEN_STATE;
        plan->summary.shared_experts = layer->shared_experts;
        plan->summary.required_binding_count += YVEX_MOE_WEIGHT_COUNT - 1u;
        plan->summary.expert_subview_count += 3ull * layer->routed_experts;
    }
    if (!moe_plan_identity(plan)) {
        yvex_moe_plan_close(&plan);
        return moe_refuse(err, YVEX_ERR_STATE, "MoE plan identity could not be sealed");
    }
    *out = plan;
    yvex_error_clear(err);
    return YVEX_OK;
}
/*
 * Borrow the immutable MoE plan summary.
 *
 * Borrowed lifetime.
 */
const yvex_moe_plan_summary *yvex_moe_plan_summary_get(const yvex_moe_plan *plan)
{
    return plan ? &plan->summary : NULL;
}
/*
 * Borrow one immutable MoE layer by canonical ordinal.
 *
 * Borrowed lifetime.
 */
const yvex_moe_layer_plan *yvex_moe_plan_layer_at(const yvex_moe_plan *plan,
                                                   unsigned long long ordinal)
{
    return plan && ordinal < plan->summary.layer_count ? &plan->layers[ordinal] : NULL;
}

void yvex_moe_plan_close(yvex_moe_plan **plan)
{
    if (!plan || !*plan) return;
    free((*plan)->layers);
    free(*plan);
    *plan = NULL;
}

static int moe_decode_flat(const yvex_moe_weight_view *weight, float *out,
                           unsigned long long count)
{
    unsigned long long index;
    if (!weight || !weight->encoded || weight->row_count != 1ull ||
        weight->row_width != count || !out) return 0;
    for (index = 0ull; index < count; ++index) {
        if (weight->qtype == YVEX_GGUF_QTYPE_F32) {
            uint32_t bits;
            memcpy(&bits, weight->encoded + index * 4ull, sizeof(bits));
            memcpy(&out[index], &bits, sizeof(bits));
        } else if (weight->qtype == YVEX_GGUF_QTYPE_BF16) {
            unsigned short bits;
            memcpy(&bits, weight->encoded + index * 2ull, sizeof(bits));
            out[index] = yvex_quant_bf16_decode(bits);
        } else if (weight->qtype == YVEX_GGUF_QTYPE_I32) {
            int32_t value;
            memcpy(&value, weight->encoded + index * 4ull, sizeof(value));
            out[index] = (float)value;
        } else return 0;
        if (!isfinite(out[index])) return 0;
    }
    return 1;
}

static int moe_matvec(const yvex_moe_weight_view *weight, const float *input,
                      float *output, yvex_error *err)
{
    yvex_quant_failure failure;
    unsigned long long row;
    if (!weight || !input || !output || !weight->row_bytes ||
        weight->row_bytes > SIZE_MAX) return moe_refuse(
            err, YVEX_ERR_INVALID_ARG, "MoE encoded matvec arguments are invalid");
    for (row = 0ull; row < weight->row_count; ++row)
        if (yvex_quant_cpu_dot(weight->qtype, weight->encoded + row * weight->row_bytes,
                               (size_t)weight->row_bytes, input, weight->row_width,
                               &output[row], &failure, err) != YVEX_OK)
            return yvex_error_code(err);
    return YVEX_OK;
}

int yvex_moe_ffn_prepare_cpu(const yvex_moe_layer_job *job, float *normalized,
                             float *post, float *combination, yvex_error *err)
{
    const yvex_moe_layer_plan *layer = job ? job->layer : NULL;
    yvex_attention_layer_plan geometry = {0};
    yvex_attention_mhc_pre_args pre;
    yvex_attention_failure failure;
    float mix[64], scale[3], base[64], norm[16384];
    if (!job || !layer || !job->expanded_input || !normalized || !post || !combination ||
        layer->mhc_mixing_rows > 64ull || layer->hidden_width > 16384ull)
        return moe_refuse(err, YVEX_ERR_INVALID_ARG, "MoE FFN ingress arguments are invalid");
    if (moe_matvec(&job->weights[YVEX_MOE_WEIGHT_MHC_FUNCTION], job->expanded_input,
                   mix, err) != YVEX_OK ||
        !moe_decode_flat(&job->weights[YVEX_MOE_WEIGHT_MHC_SCALE], scale, 3ull) ||
        !moe_decode_flat(&job->weights[YVEX_MOE_WEIGHT_MHC_BASE], base,
                         layer->mhc_mixing_rows) ||
        !moe_decode_flat(&job->weights[YVEX_MOE_WEIGHT_FFN_NORM], norm,
                         layer->hidden_width))
        return yvex_error_is_set(err) ? yvex_error_code(err)
                                      : moe_refuse(err, YVEX_ERR_FORMAT,
                                                   "MoE FFN coefficients cannot be decoded");
    geometry.layer_index = layer->layer_index;
    geometry.compute_contract = YVEX_ATTENTION_COMPUTE_BF16_F32_RNE_V1;
    geometry.residual_stream_count = layer->residual_streams;
    geometry.residual_stream_width = layer->hidden_width;
    geometry.residual_expanded_width = layer->expanded_width;
    geometry.mhc_mixing_rows = layer->mhc_mixing_rows;
    geometry.mhc_mixing_columns = layer->expanded_width;
    geometry.mhc_base_width = layer->mhc_mixing_rows;
    geometry.mhc_scale_width = 3ull;
    geometry.mhc_sinkhorn_iterations = layer->mhc_sinkhorn_iterations;
    geometry.rms_norm_epsilon = layer->rms_epsilon;
    geometry.mhc_epsilon = layer->mhc_epsilon;
    geometry.mhc_residual_post_multiplier = layer->mhc_post_multiplier;
    geometry.mhc_attention_pre_and_post = 1;
    pre = (yvex_attention_mhc_pre_args){
        &geometry, job->expanded_input, mix, scale, base, 1ull, layer->expanded_width,
        layer->mhc_mixing_rows, normalized, post, combination, layer->hidden_width,
        layer->residual_streams, layer->residual_streams * layer->residual_streams};
    memset(&failure, 0, sizeof(failure));
    if (yvex_attention_mhc_pre(&pre, &failure, err) != YVEX_OK ||
        !yvex_attention_rms_norm(normalized, layer->hidden_width, norm, layer->rms_epsilon) ||
        !yvex_attention_compute_round(YVEX_ATTENTION_COMPUTE_BF16_F32_RNE_V1,
                                      normalized, layer->hidden_width))
        return yvex_error_is_set(err) ? yvex_error_code(err)
                                      : moe_refuse(err, YVEX_ERR_FORMAT,
                                                   "MoE FFN ingress produced invalid numerics");
    yvex_error_clear(err);
    return YVEX_OK;
}

static double moe_score(double value)
{
    double softplus = value > 0.0 ? value + log1p(exp(-value)) : log1p(exp(value));
    return sqrt(softplus);
}

static void moe_topk(const float *scores, unsigned long long count, unsigned long long topk,
                     unsigned long long *selected)
{
    unsigned long long rank, candidate;
    for (rank = 0ull; rank < topk; ++rank) {
        unsigned long long best = ULLONG_MAX;
        for (candidate = 0ull; candidate < count; ++candidate) {
            unsigned long long prior;
            int used = 0;
            for (prior = 0ull; prior < rank; ++prior) used |= selected[prior] == candidate;
            if (!used && (best == ULLONG_MAX || scores[candidate] > scores[best] ||
                          (scores[candidate] == scores[best] && candidate < best))) best = candidate;
        }
        selected[rank] = best;
    }
}

int yvex_moe_route_cpu(const yvex_moe_layer_job *job, const float *normalized,
                       yvex_moe_router_result *result, yvex_error *err)
{
    const yvex_moe_layer_plan *layer = job ? job->layer : NULL;
    float bias[256];
    unsigned long long expert, rank;
    double total = 0.0;
    if (result) memset(result, 0, sizeof(*result));
    if (!job || !layer || !normalized || !result || layer->routed_experts > 256ull ||
        layer->experts_per_token > YVEX_MOE_MAX_SELECTED)
        return moe_refuse(err, YVEX_ERR_INVALID_ARG, "MoE router arguments are invalid");
    if (moe_matvec(&job->weights[YVEX_MOE_WEIGHT_ROUTER], normalized,
                   result->router_logits, err) != YVEX_OK) return yvex_error_code(err);
    for (expert = 0ull; expert < layer->routed_experts; ++expert) {
        double score = moe_score(result->router_logits[expert]);
        if (!isfinite(score)) return moe_refuse(err, YVEX_ERR_FORMAT,
                                                "MoE router score is non-finite");
        result->router_scores[expert] = (float)score;
    }
    result->selected_count = layer->experts_per_token;
    if (layer->router_class == YVEX_MOE_ROUTER_HASH_TOKEN_ID) {
        const yvex_moe_weight_view *table = &job->weights[YVEX_MOE_WEIGHT_ROUTER_TABLE];
        const unsigned char *row;
        if (!job->token_id_present || job->token_id >= layer->hash_table_rows ||
            table->qtype != YVEX_GGUF_QTYPE_I32 || table->row_bytes < layer->experts_per_token * 4ull)
            return moe_refuse(err, YVEX_ERR_BOUNDS,
                              "MoE hash router requires one in-vocabulary token ID");
        row = table->encoded + (table->row_count == 1ull
                                    ? 0ull
                                    : (unsigned long long)job->token_id * table->row_bytes);
        for (rank = 0ull; rank < layer->experts_per_token; ++rank) {
            int32_t ordinal;
            memcpy(&ordinal, row + rank * 4ull, sizeof(ordinal));
            if (ordinal < 0 || (unsigned long long)ordinal >= layer->routed_experts)
                return moe_refuse(err, YVEX_ERR_FORMAT,
                                  "MoE hash router selected an invalid expert ordinal");
            result->selected_experts[rank] = (unsigned long long)ordinal;
        }
    } else {
        if (!moe_decode_flat(&job->weights[YVEX_MOE_WEIGHT_ROUTER_BIAS], bias,
                             layer->routed_experts))
            return moe_refuse(err, YVEX_ERR_FORMAT, "MoE correction bias is malformed");
        for (expert = 0ull; expert < layer->routed_experts; ++expert) bias[expert] += result->router_scores[expert];
        moe_topk(bias, layer->routed_experts, layer->experts_per_token,
                 result->selected_experts);
    }
    for (rank = 0ull; rank < result->selected_count; ++rank) {
        unsigned long long selected = result->selected_experts[rank], prior;
        for (prior = 0ull; prior < rank; ++prior)
            if (result->selected_experts[prior] == selected)
                return moe_refuse(err, YVEX_ERR_FORMAT,
                                  "MoE router selected a duplicate expert ordinal");
        result->selected_weights[rank] = result->router_scores[selected];
        total += result->selected_weights[rank];
    }
    if (layer->normalize_topk_probabilities && (!isfinite(total) || total <= 0.0))
        return moe_refuse(err, YVEX_ERR_FORMAT, "MoE routing weights cannot be normalized");
    for (rank = 0ull; rank < result->selected_count; ++rank) {
        double value = result->selected_weights[rank];
        if (layer->normalize_topk_probabilities) value /= total;
        value *= layer->routed_scaling_factor;
        result->selected_weights[rank] = (float)value;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_moe_expert_cpu(const yvex_moe_layer_plan *layer,
                        const yvex_moe_weight_view *gate,
                        const yvex_moe_weight_view *up,
                        const yvex_moe_weight_view *down, const float *input,
                        float *output, yvex_error *err)
{
    float gate_values[4096], up_values[4096], intermediate[4096];
    unsigned long long index, width;
    int rc;
    if (!layer || !gate || !up || !down || !input || !output ||
        gate->row_count != up->row_count || down->row_width != gate->row_count ||
        down->row_count != layer->hidden_width)
        return moe_refuse(err, YVEX_ERR_INVALID_ARG, "MoE expert geometry is invalid");
    width = gate->row_count;
    if (width > 4096ull)
        return moe_refuse(err, YVEX_ERR_BOUNDS, "MoE expert width exceeds bounded scratch");
    if ((rc = moe_matvec(gate, input, gate_values, err)) != YVEX_OK ||
        (rc = moe_matvec(up, input, up_values, err)) != YVEX_OK) return rc;
    for (index = 0ull; index < width; ++index) {
        double g = fmin(gate_values[index], layer->activation_limit);
        double u = fmax(-layer->activation_limit,
                        fmin(up_values[index], layer->activation_limit));
        double silu = g >= 0.0 ? g / (1.0 + exp(-g)) : g * exp(g) / (1.0 + exp(g));
        intermediate[index] = (float)(silu * u);
    }
    if (!yvex_attention_compute_round(YVEX_ATTENTION_COMPUTE_BF16_F32_RNE_V1,
                                      intermediate, width)) {
        return moe_refuse(err, YVEX_ERR_FORMAT, "MoE SwiGLU intermediate is non-finite");
    }
    rc = moe_matvec(down, intermediate, output, err);
    if (rc == YVEX_OK &&
        !yvex_attention_compute_round(YVEX_ATTENTION_COMPUTE_BF16_F32_RNE_V1,
                                      output, layer->hidden_width))
        return moe_refuse(err, YVEX_ERR_FORMAT, "MoE expert output is non-finite");
    return rc;
}
