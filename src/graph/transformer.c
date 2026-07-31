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

static int transformer_layer_identity(yvex_transformer_layer_plan *out,
                                      const yvex_attention_layer_plan *attention,
                                      const yvex_moe_layer_plan *moe)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    if (!out || !attention || !moe || attention->layer_index != moe->layer_index ||
        !yvex_sha256_hex_valid(moe->layer_identity)) return 0;
    out->ordinal = moe->ordinal;
    out->layer_index = attention->layer_index;
    transformer_identity_copy(out->moe_layer_identity, moe->layer_identity);
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.transformer.layer.v1") ||
        !yvex_sha256_update_u64(&hash, out->ordinal) ||
        !yvex_sha256_update_u64(&hash, out->layer_index) ||
        !yvex_sha256_update_u64(&hash, attention->attention_class) ||
        !yvex_sha256_update_u64(&hash, attention->hidden_dimension) ||
        !yvex_sha256_update_u64(&hash, attention->residual_stream_count) ||
        !yvex_sha256_update_u64(&hash, attention->residual_expanded_width) ||
        !yvex_sha256_update_text(&hash, moe->layer_identity) ||
        !yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, out->layer_identity);
    return 1;
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
    if (!yvex_sha256_update_text(&hash, "yvex.transformer.plan.v1") ||
        !yvex_sha256_update_u64(&hash, s->schema_version) ||
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

int yvex_transformer_plan_build(yvex_transformer_plan **out,
                                const yvex_transformer_plan_facts *facts,
                                const yvex_attention_plan *attention,
                                const yvex_moe_plan *moe, yvex_error *err)
{
    yvex_transformer_plan *plan = NULL;
    const yvex_transformer_family_policy *policy = facts ? &facts->policy : NULL;
    const yvex_attention_summary *attention_summary = yvex_attention_plan_summary(attention);
    const yvex_moe_plan_summary *moe_summary = yvex_moe_plan_summary_get(moe);
    unsigned long long index;
    if (out) *out = NULL;
    if (!out || !facts || !policy || !attention_summary || !moe_summary ||
        policy->schema_version != YVEX_TRANSFORMER_PLAN_SCHEMA_V1 ||
        !policy->attention_then_moe || !policy->deferred_ffn_post ||
        !policy->final_norm_after_head || !policy->hidden_width || !policy->residual_streams ||
        !policy->expanded_width ||
        policy->expanded_width != policy->hidden_width * policy->residual_streams ||
        facts->layer_count != attention_summary->layer_count ||
        facts->layer_count != moe_summary->layer_count)
        return transformer_refuse(err, YVEX_ERR_FORMAT,
                                  "transformer family/component plan facts are incompatible");
    plan = (yvex_transformer_plan *)calloc(1u, sizeof(*plan));
    if (!plan) return transformer_refuse(err, YVEX_ERR_NOMEM, "transformer plan allocation failed");
    plan->layers = (yvex_transformer_layer_plan *)calloc(
        (size_t)facts->layer_count, sizeof(*plan->layers));
    if (!plan->layers) goto allocation;
    plan->summary.schema_version = YVEX_TRANSFORMER_PLAN_SCHEMA_V1;
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
        summary->schema_version != YVEX_TRANSFORMER_PLAN_SCHEMA_V1 ||
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
            plan->layers[index].layer_index != index ||
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
    if (!summary || !layers || summary->schema_version != YVEX_TRANSFORMER_PLAN_SCHEMA_V1 ||
        !summary->layer_count || !summary->hidden_width || !summary->residual_streams ||
        summary->expanded_width != summary->hidden_width * summary->residual_streams)
        return transformer_refuse(err, YVEX_ERR_INVALID_ARG,
                                  "transformer plan seal facts are invalid");
    for (index = 0ull; index < summary->layer_count; ++index)
        if (layers[index].ordinal != index || layers[index].layer_index != index ||
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

int yvex_transformer_final_stage(const yvex_transformer_plan *plan,
                                 const float *expanded, unsigned long long token_count,
                                 const float *function, const float *base, const float *scale,
                                 const float *norm, float *normalized, yvex_error *err)
{
    const yvex_transformer_plan_summary *s = yvex_transformer_plan_summary_get(plan);
    unsigned long long token, stream, lane, index;
    if (!s || !expanded || !function || !base || !scale || !norm || !normalized ||
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
                                          output, s->hidden_width) ||
            !yvex_attention_rms_norm(output, s->hidden_width, norm,
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
