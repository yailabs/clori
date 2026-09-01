/*
 * Own semantic model descriptors, tensor tables, and admitted read-only model contexts.
 *
 * Model facts are built from parsed GGUF and tensor metadata; tensor payload bytes are read only
 * by their separate resource owner; domain code returns facts and errors, not operator prose.
 * Model metadata is not runtime support, generation support, evaluation evidence, benchmark
 * evidence, or release readiness.
 */

#include <yvex/model.h>
#include <yvex/artifact.h>
#include <yvex/core.h>
#include <yvex/gguf.h>
#include <yvex/qtype.h>
#include <yvex/registry.h>
#include <yvex/source.h>
#include <yvex/tokenizer.h>
#include <yvex/internal/core.h>
#include <yvex/internal/model.h>

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define MODEL_IDENTITY_FIELD(object, field) \
    yvex_sha256_update_u64_be(hash, (unsigned long long)(object)->field)

static int activation_policy_valid(
    const yvex_attention_activation_policy *policy,
    unsigned long long fp8_block_width, unsigned long long fp4_block_width,
    unsigned long long *expected, unsigned long long *actual)
{
    if (!policy) {
        *expected = 1ull;
        *actual = 0ull;
        return 0;
    }
    if (!policy->required) {
        *expected = 0ull;
        *actual = policy->quantization != YVEX_ATTENTION_QUANT_NONE ||
                  policy->block_width != 0ull ||
                  policy->pre_transform != YVEX_ATTENTION_TRANSFORM_NONE;
        return !*actual;
    }
    if (policy->block_axis != YVEX_ATTENTION_AXIS_FINAL_DIMENSION ||
        policy->scale_format != YVEX_ATTENTION_SCALE_UE8M0 ||
        policy->scale_dtype != YVEX_NATIVE_DTYPE_F8_E8M0 ||
        policy->tail_policy != YVEX_ATTENTION_TAIL_EXACT_OR_SHORT_FINAL_BLOCK ||
        policy->nonfinite_policy != YVEX_ATTENTION_NONFINITE_REFUSE ||
        !policy->fake_quant_inplace) {
        *expected = 1ull;
        *actual = 0ull;
        return 0;
    }
    if (policy->quantization == YVEX_ATTENTION_QUANT_FP8_E4M3_UE8M0_FAKE_DEQUANT) {
        *expected = fp8_block_width;
        *actual = policy->block_width;
        return policy->block_width == fp8_block_width &&
               policy->pre_transform == YVEX_ATTENTION_TRANSFORM_NONE;
    }
    if (policy->quantization == YVEX_ATTENTION_QUANT_FP4_E2M1_UE8M0_FAKE_DEQUANT) {
        *expected = fp4_block_width;
        *actual = policy->block_width;
        return policy->block_width == fp4_block_width &&
               policy->pre_transform == YVEX_ATTENTION_TRANSFORM_DAO_FHT_V1_1_0_POST2 &&
               policy->zero_pad_hadamard_to_power_of_two;
    }
    *expected = 1ull;
    *actual = 0ull;
    return 0;
}

int yvex_model_attention_numeric_validate(
    yvex_attention_compute_contract compute_contract,
    yvex_attention_compute_contract expected_compute_contract,
    const yvex_attention_activation_policy *const *activation_policies,
    unsigned long long activation_policy_count,
    const yvex_attention_topk_policy *topk_policy,
    unsigned long long fp8_block_width, unsigned long long fp4_block_width,
    unsigned int topk_policy_version, yvex_attention_numeric_mismatch *mismatch)
{
    unsigned long long index, expected = 0ull, actual = 0ull;
    yvex_attention_numeric_mismatch local = {0};

    if (!mismatch) mismatch = &local;
    memset(mismatch, 0, sizeof(*mismatch));
    if (compute_contract != expected_compute_contract) {
        mismatch->code = YVEX_ATTENTION_NUMERIC_MISMATCH_COMPUTE;
        mismatch->expected = expected_compute_contract;
        mismatch->actual = compute_contract;
        return 0;
    }
    if (!activation_policies && activation_policy_count) {
        mismatch->code = YVEX_ATTENTION_NUMERIC_MISMATCH_ACTIVATION;
        mismatch->expected = 1ull;
        return 0;
    }
    for (index = 0ull; index < activation_policy_count; ++index) {
        if (!activation_policy_valid(activation_policies[index], fp8_block_width,
                                     fp4_block_width, &expected, &actual)) {
            mismatch->code = YVEX_ATTENTION_NUMERIC_MISMATCH_ACTIVATION;
            mismatch->policy_index = index;
            mismatch->expected = expected;
            mismatch->actual = actual;
            return 0;
        }
    }
    if (topk_policy && topk_policy->required &&
        (topk_policy->version != topk_policy_version ||
         topk_policy->policy != YVEX_ATTENTION_TOPK_SCORE_DESC_ORDINAL_ASC_V1 ||
         topk_policy->k == 0ull || !topk_policy->reject_nonfinite ||
         !topk_policy->score_descending || !topk_policy->equal_score_ordinal_ascending ||
         !topk_policy->plus_zero_equals_minus_zero ||
        !topk_policy->duplicate_ordinal_refused || !topk_policy->output_ranked_order)) {
        mismatch->code = YVEX_ATTENTION_NUMERIC_MISMATCH_TOPK;
        mismatch->expected = 1ull;
        mismatch->actual = 0ull;
        return 0;
    }
    return 1;
}

int yvex_model_activation_identity_update(
    yvex_sha256 *hash, const yvex_attention_activation_policy *policy)
{
    return hash && policy && MODEL_IDENTITY_FIELD(policy, required) &&
           MODEL_IDENTITY_FIELD(policy, stage) &&
           MODEL_IDENTITY_FIELD(policy, quantization) &&
           MODEL_IDENTITY_FIELD(policy, block_axis) &&
           MODEL_IDENTITY_FIELD(policy, block_width) &&
           MODEL_IDENTITY_FIELD(policy, scale_format) &&
           MODEL_IDENTITY_FIELD(policy, scale_dtype) &&
           MODEL_IDENTITY_FIELD(policy, pre_transform) &&
           MODEL_IDENTITY_FIELD(policy, tail_policy) &&
           MODEL_IDENTITY_FIELD(policy, nonfinite_policy) &&
           MODEL_IDENTITY_FIELD(policy, fake_quant_inplace) &&
           MODEL_IDENTITY_FIELD(policy, zero_pad_hadamard_to_power_of_two);
}

int yvex_model_topk_identity_update(
    yvex_sha256 *hash, const yvex_attention_topk_policy *policy)
{
    return hash && policy && MODEL_IDENTITY_FIELD(policy, required) &&
           MODEL_IDENTITY_FIELD(policy, version) &&
           MODEL_IDENTITY_FIELD(policy, policy) && MODEL_IDENTITY_FIELD(policy, k) &&
           MODEL_IDENTITY_FIELD(policy, reject_nonfinite) &&
           MODEL_IDENTITY_FIELD(policy, score_descending) &&
           MODEL_IDENTITY_FIELD(policy, equal_score_ordinal_ascending) &&
           MODEL_IDENTITY_FIELD(policy, plus_zero_equals_minus_zero) &&
           MODEL_IDENTITY_FIELD(policy, duplicate_ordinal_refused) &&
           MODEL_IDENTITY_FIELD(policy, output_ranked_order);
}

int yvex_model_position_identity_update(
    yvex_sha256 *hash, const yvex_attention_position_policy *policy)
{
    return hash && policy && MODEL_IDENTITY_FIELD(policy, rope_dimension) &&
           MODEL_IDENTITY_FIELD(policy, theta) &&
           MODEL_IDENTITY_FIELD(policy, scaling_factor) &&
           MODEL_IDENTITY_FIELD(policy, original_context) &&
           MODEL_IDENTITY_FIELD(policy, beta_fast) &&
           MODEL_IDENTITY_FIELD(policy, beta_slow) &&
           MODEL_IDENTITY_FIELD(policy, maximum_context) &&
           MODEL_IDENTITY_FIELD(policy, partial_rope) &&
           MODEL_IDENTITY_FIELD(policy, inverse_output_rotation);
}

#undef MODEL_IDENTITY_FIELD

static int model_execution_refuse(yvex_error *err, yvex_status status,
                                  const char *reason)
{
    yvex_error_set(err, status, "model.execution.descriptor", reason);
    return status;
}

static unsigned long long model_execution_f64_bits(double value)
{
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (unsigned long long)bits;
}

static double model_execution_f64_from_bits(unsigned long long value)
{
    uint64_t bits = (uint64_t)value;
    double result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static int model_execution_descriptor_export(
    const yvex_model_execution_descriptor *descriptor,
    unsigned long long scalars[YVEX_MODEL_EXECUTION_SCALAR_COUNT],
    const char *identities[YVEX_MODEL_EXECUTION_IDENTITY_COUNT]);

int yvex_model_execution_descriptor_seal(
    const yvex_model_execution_descriptor_request *request,
    yvex_model_execution_descriptor *descriptor, yvex_error *err)
{
    const char *identities[YVEX_MODEL_EXECUTION_IDENTITY_COUNT];
    unsigned long long values[YVEX_MODEL_EXECUTION_SCALAR_COUNT];
    yvex_sha256 hash;
    unsigned long long attention_layers;
    size_t index, value_count = 0u;
    int routed_complete, dense_complete;

    if (descriptor) memset(descriptor, 0, sizeof(*descriptor));
    routed_complete = request && request->routed_experts && request->experts_per_row &&
                      request->experts_per_row <= request->routed_experts &&
                      request->routed_ffn_width &&
                      request->hash_router_layer_count <= request->layer_count &&
                      isfinite(request->routed_scaling_factor) &&
                      request->routed_scaling_factor > 0.0 &&
                      isfinite(request->activation_limit) && request->activation_limit > 0.0;
    dense_complete = request && !request->routed_experts && !request->experts_per_row &&
                     !request->shared_experts && !request->routed_ffn_width &&
                     !request->shared_ffn_width && !request->hash_router_layer_count &&
                     request->routed_scaling_factor == 0.0 &&
                     request->activation_limit == 0.0;
    if (!request || !descriptor ||
        request->schema_version != YVEX_MODEL_EXECUTION_DESCRIPTOR_SCHEMA_V1)
        return model_execution_refuse(
            err, YVEX_ERR_INVALID_ARG, "model execution descriptor schema is unsupported");
    if (!request->maximum_context || !request->original_context ||
        request->original_context > request->maximum_context ||
        request->rope_scaling > YVEX_MODEL_ROPE_SCALING_YARN ||
        !request->rope_theta || !request->rope_scaling_factor ||
        (request->rope_scaling == YVEX_MODEL_ROPE_SCALING_YARN &&
         (!request->compressed_rope_theta ||
          request->rope_beta_fast <= request->rope_beta_slow)) ||
        (request->rope_scaling != YVEX_MODEL_ROPE_SCALING_YARN &&
         request->compressed_rope_theta))
        return model_execution_refuse(
            err, YVEX_ERR_INVALID_ARG, "source-derived context or RoPE geometry is invalid");
    if (!request->layer_count || !request->hidden_width || !request->vocabulary_size ||
        !request->attention_heads || !request->kv_heads || !request->head_width)
        return model_execution_refuse(
            err, YVEX_ERR_INVALID_ARG, "source-derived model geometry is incomplete");
    if (!isfinite(request->normalization_epsilon) ||
        request->normalization_epsilon <= 0.0 ||
        (request->residual_streams &&
         (!request->mhc_sinkhorn_iterations || !isfinite(request->mhc_epsilon) ||
          request->mhc_epsilon <= 0.0)) ||
        (!request->residual_streams &&
         (request->mhc_sinkhorn_iterations || request->mhc_epsilon != 0.0)))
        return model_execution_refuse(
            err, YVEX_ERR_INVALID_ARG, "source-derived residual geometry is incomplete");
    if (!routed_complete && !dense_complete)
        return model_execution_refuse(
            err, YVEX_ERR_INVALID_ARG, "source-derived MoE geometry is incomplete");
    if (!request->output_input_width ||
        request->output_input_width != request->hidden_width ||
        request->output_vocabulary_size != request->vocabulary_size)
        return model_execution_refuse(
            err, YVEX_ERR_INVALID_ARG, "source-derived output geometry is inconsistent");
    if (!request->persistent_state_class_mask ||
        request->persistent_state_class_mask >=
            (1ull << (unsigned int)YVEX_MODEL_STATE_CLASS_COUNT) ||
        request->minimum_compression_ratio > request->maximum_compression_ratio)
        return model_execution_refuse(
            err, YVEX_ERR_INVALID_ARG, "source-derived persistent-state geometry is invalid");
    if (!yvex_core_u64_add(request->swa_layers, request->csa_layers,
                           &attention_layers) ||
        !yvex_core_u64_add(attention_layers, request->hca_layers,
                           &attention_layers) ||
        attention_layers != request->layer_count)
        return model_execution_refuse(
            err, YVEX_ERR_INVALID_ARG, "attention schedule does not cover every model layer");
    if (request->draft_layer_count &&
        (!request->proposal_width || request->proposal_width == ULLONG_MAX ||
         request->verification_width_maximum < request->proposal_width + 1ull ||
         !request->target_feature_count || !request->target_feature_width ||
         !request->markov_rank))
        return model_execution_refuse(
            err, YVEX_ERR_INVALID_ARG, "source-derived speculation geometry is incomplete");
    if (!request->draft_layer_count &&
        (request->proposal_width || request->verification_width_maximum ||
         request->target_feature_count || request->target_feature_width ||
         request->markov_rank || request->confidence_width || request->draft_noise_token_id))
        return model_execution_refuse(
            err, YVEX_ERR_INVALID_ARG, "non-speculative model carries draft-only geometry");
    for (index = 0u; index < YVEX_MODEL_EXECUTION_FEATURE_LAYER_CAP; ++index) {
        size_t prior;
        if (index >= request->target_feature_count) {
            if (request->target_feature_layers[index])
                return model_execution_refuse(
                    err, YVEX_ERR_INVALID_ARG,
                    "unused target feature layer slots must be canonical zero");
            continue;
        }
        if (request->target_feature_count > YVEX_MODEL_EXECUTION_FEATURE_LAYER_CAP ||
            request->target_feature_layers[index] >= request->layer_count)
            return model_execution_refuse(
                err, YVEX_ERR_BOUNDS, "target feature layer is outside model geometry");
        for (prior = 0u; prior < index; ++prior)
            if (request->target_feature_layers[prior] ==
                request->target_feature_layers[index])
                return model_execution_refuse(
                    err, YVEX_ERR_INVALID_ARG, "target feature layers must be unique");
    }

    identities[0] = request->logical_model_identity;
    identities[1] = request->source_model_identity;
    identities[2] = request->attention_schedule_identity;
    identities[3] = request->persistent_state_identity;
    for (index = 0u; index < sizeof(identities) / sizeof(identities[0]); ++index)
        if (!yvex_sha256_hex_valid(identities[index]))
            return model_execution_refuse(
                err, YVEX_ERR_FORMAT, "model execution identity input is malformed");

    descriptor->schema_version = request->schema_version;
#define COPY_MODEL_FIELD(name) descriptor->name = request->name
    COPY_MODEL_FIELD(maximum_context);
    COPY_MODEL_FIELD(original_context);
    COPY_MODEL_FIELD(rope_scaling);
    COPY_MODEL_FIELD(rope_theta);
    COPY_MODEL_FIELD(compressed_rope_theta);
    COPY_MODEL_FIELD(rope_scaling_factor);
    COPY_MODEL_FIELD(rope_beta_fast);
    COPY_MODEL_FIELD(rope_beta_slow);
    COPY_MODEL_FIELD(layer_count);
    COPY_MODEL_FIELD(hidden_width);
    COPY_MODEL_FIELD(vocabulary_size);
    COPY_MODEL_FIELD(attention_heads);
    COPY_MODEL_FIELD(kv_heads);
    COPY_MODEL_FIELD(head_width);
    COPY_MODEL_FIELD(swa_layers);
    COPY_MODEL_FIELD(csa_layers);
    COPY_MODEL_FIELD(hca_layers);
    COPY_MODEL_FIELD(sliding_window);
    COPY_MODEL_FIELD(minimum_compression_ratio);
    COPY_MODEL_FIELD(maximum_compression_ratio);
    COPY_MODEL_FIELD(index_heads);
    COPY_MODEL_FIELD(index_head_width);
    COPY_MODEL_FIELD(index_topk);
    COPY_MODEL_FIELD(residual_streams);
    COPY_MODEL_FIELD(mhc_sinkhorn_iterations);
    COPY_MODEL_FIELD(mhc_epsilon);
    COPY_MODEL_FIELD(normalization_epsilon);
    COPY_MODEL_FIELD(routed_experts);
    COPY_MODEL_FIELD(experts_per_row);
    COPY_MODEL_FIELD(shared_experts);
    COPY_MODEL_FIELD(routed_ffn_width);
    COPY_MODEL_FIELD(shared_ffn_width);
    COPY_MODEL_FIELD(hash_router_layer_count);
    COPY_MODEL_FIELD(routed_scaling_factor);
    COPY_MODEL_FIELD(activation_limit);
    COPY_MODEL_FIELD(output_input_width);
    COPY_MODEL_FIELD(output_vocabulary_size);
    COPY_MODEL_FIELD(proposal_width);
    COPY_MODEL_FIELD(verification_width_maximum);
    COPY_MODEL_FIELD(draft_layer_count);
    COPY_MODEL_FIELD(target_feature_count);
    memcpy(descriptor->target_feature_layers, request->target_feature_layers,
           sizeof(descriptor->target_feature_layers));
    COPY_MODEL_FIELD(target_feature_width);
    COPY_MODEL_FIELD(markov_rank);
    COPY_MODEL_FIELD(confidence_width);
    COPY_MODEL_FIELD(persistent_state_class_mask);
    COPY_MODEL_FIELD(bos_token_id);
    COPY_MODEL_FIELD(eos_token_id);
    COPY_MODEL_FIELD(draft_noise_token_id);
#undef COPY_MODEL_FIELD
    yvex_core_text_copy(descriptor->logical_model_identity,
                        sizeof(descriptor->logical_model_identity), identities[0]);
    yvex_core_text_copy(descriptor->source_model_identity,
                        sizeof(descriptor->source_model_identity), identities[1]);
    yvex_core_text_copy(descriptor->attention_schedule_identity,
                        sizeof(descriptor->attention_schedule_identity), identities[2]);
    yvex_core_text_copy(descriptor->persistent_state_identity,
                        sizeof(descriptor->persistent_state_identity), identities[3]);

    if (!model_execution_descriptor_export(descriptor, values, identities)) goto identity_failed;
    value_count = YVEX_MODEL_EXECUTION_SCALAR_COUNT;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.model-execution-descriptor.v1") ||
        !yvex_sha256_update_u64(&hash, descriptor->schema_version))
        goto identity_failed;
    for (index = 0u; index < value_count; ++index)
        if (!yvex_sha256_update_u64(&hash, values[index])) goto identity_failed;
    for (index = 0u; index < sizeof(identities) / sizeof(identities[0]); ++index)
        if (!yvex_sha256_update_text(&hash, identities[index])) goto identity_failed;
    {
        unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
        if (!yvex_sha256_final(&hash, digest)) goto identity_failed;
        yvex_sha256_hex(digest, descriptor->identity);
    }
    yvex_error_clear(err);
    return YVEX_OK;

identity_failed:
    memset(descriptor, 0, sizeof(*descriptor));
    return model_execution_refuse(
        err, YVEX_ERR_STATE, "model execution descriptor identity derivation failed");
}

static int model_execution_descriptor_export(
    const yvex_model_execution_descriptor *descriptor,
    unsigned long long scalars[YVEX_MODEL_EXECUTION_SCALAR_COUNT],
    const char *identities[YVEX_MODEL_EXECUTION_IDENTITY_COUNT])
{
    size_t index = 0u;

    if (!descriptor || !scalars || !identities ||
        descriptor->schema_version != YVEX_MODEL_EXECUTION_DESCRIPTOR_SCHEMA_V1)
        return 0;
#define EXPORT_MODEL_FIELD(name) scalars[index++] = descriptor->name
    EXPORT_MODEL_FIELD(maximum_context);
    EXPORT_MODEL_FIELD(original_context);
    EXPORT_MODEL_FIELD(rope_scaling);
    EXPORT_MODEL_FIELD(rope_theta);
    EXPORT_MODEL_FIELD(compressed_rope_theta);
    EXPORT_MODEL_FIELD(rope_scaling_factor);
    EXPORT_MODEL_FIELD(rope_beta_fast);
    EXPORT_MODEL_FIELD(rope_beta_slow);
    EXPORT_MODEL_FIELD(layer_count);
    EXPORT_MODEL_FIELD(hidden_width);
    EXPORT_MODEL_FIELD(vocabulary_size);
    EXPORT_MODEL_FIELD(attention_heads);
    EXPORT_MODEL_FIELD(kv_heads);
    EXPORT_MODEL_FIELD(head_width);
    EXPORT_MODEL_FIELD(swa_layers);
    EXPORT_MODEL_FIELD(csa_layers);
    EXPORT_MODEL_FIELD(hca_layers);
    EXPORT_MODEL_FIELD(sliding_window);
    EXPORT_MODEL_FIELD(minimum_compression_ratio);
    EXPORT_MODEL_FIELD(maximum_compression_ratio);
    EXPORT_MODEL_FIELD(index_heads);
    EXPORT_MODEL_FIELD(index_head_width);
    EXPORT_MODEL_FIELD(index_topk);
    EXPORT_MODEL_FIELD(residual_streams);
    EXPORT_MODEL_FIELD(mhc_sinkhorn_iterations);
    scalars[index++] = model_execution_f64_bits(descriptor->mhc_epsilon);
    scalars[index++] = model_execution_f64_bits(descriptor->normalization_epsilon);
    EXPORT_MODEL_FIELD(routed_experts);
    EXPORT_MODEL_FIELD(experts_per_row);
    EXPORT_MODEL_FIELD(shared_experts);
    EXPORT_MODEL_FIELD(routed_ffn_width);
    EXPORT_MODEL_FIELD(shared_ffn_width);
    EXPORT_MODEL_FIELD(hash_router_layer_count);
    scalars[index++] = model_execution_f64_bits(descriptor->routed_scaling_factor);
    scalars[index++] = model_execution_f64_bits(descriptor->activation_limit);
    EXPORT_MODEL_FIELD(output_input_width);
    EXPORT_MODEL_FIELD(output_vocabulary_size);
    EXPORT_MODEL_FIELD(proposal_width);
    EXPORT_MODEL_FIELD(verification_width_maximum);
    EXPORT_MODEL_FIELD(draft_layer_count);
    EXPORT_MODEL_FIELD(target_feature_count);
    {
        size_t layer;
        for (layer = 0u; layer < YVEX_MODEL_EXECUTION_FEATURE_LAYER_CAP; ++layer)
            scalars[index++] = descriptor->target_feature_layers[layer];
    }
    EXPORT_MODEL_FIELD(target_feature_width);
    EXPORT_MODEL_FIELD(markov_rank);
    EXPORT_MODEL_FIELD(confidence_width);
    EXPORT_MODEL_FIELD(persistent_state_class_mask);
    EXPORT_MODEL_FIELD(bos_token_id);
    EXPORT_MODEL_FIELD(eos_token_id);
    EXPORT_MODEL_FIELD(draft_noise_token_id);
#undef EXPORT_MODEL_FIELD
    identities[0] = descriptor->logical_model_identity;
    identities[1] = descriptor->source_model_identity;
    identities[2] = descriptor->attention_schedule_identity;
    identities[3] = descriptor->persistent_state_identity;
    return index == YVEX_MODEL_EXECUTION_SCALAR_COUNT;
}

static int model_execution_descriptor_import(
    const unsigned long long scalars[YVEX_MODEL_EXECUTION_SCALAR_COUNT],
    const char *const identities[YVEX_MODEL_EXECUTION_IDENTITY_COUNT],
    const char *expected_identity, yvex_model_execution_descriptor *descriptor,
    yvex_error *err)
{
    yvex_model_execution_descriptor_request request = {0};
    size_t index = 0u;
    int rc;

    if (!scalars || !identities || !descriptor ||
        !yvex_sha256_hex_valid(expected_identity))
        return model_execution_refuse(
            err, YVEX_ERR_INVALID_ARG, "serialized model execution facts are required");
    request.schema_version = YVEX_MODEL_EXECUTION_DESCRIPTOR_SCHEMA_V1;
    request.logical_model_identity = identities[0];
    request.source_model_identity = identities[1];
    request.attention_schedule_identity = identities[2];
    request.persistent_state_identity = identities[3];
#define IMPORT_MODEL_FIELD(name) request.name = scalars[index++]
    IMPORT_MODEL_FIELD(maximum_context);
    IMPORT_MODEL_FIELD(original_context);
    if (scalars[index] > (unsigned long long)YVEX_MODEL_ROPE_SCALING_YARN)
        return model_execution_refuse(
            err, YVEX_ERR_FORMAT, "serialized RoPE scaling policy is invalid");
    request.rope_scaling = (yvex_model_rope_scaling)scalars[index++];
    IMPORT_MODEL_FIELD(rope_theta);
    IMPORT_MODEL_FIELD(compressed_rope_theta);
    IMPORT_MODEL_FIELD(rope_scaling_factor);
    IMPORT_MODEL_FIELD(rope_beta_fast);
    IMPORT_MODEL_FIELD(rope_beta_slow);
    IMPORT_MODEL_FIELD(layer_count);
    IMPORT_MODEL_FIELD(hidden_width);
    IMPORT_MODEL_FIELD(vocabulary_size);
    IMPORT_MODEL_FIELD(attention_heads);
    IMPORT_MODEL_FIELD(kv_heads);
    IMPORT_MODEL_FIELD(head_width);
    IMPORT_MODEL_FIELD(swa_layers);
    IMPORT_MODEL_FIELD(csa_layers);
    IMPORT_MODEL_FIELD(hca_layers);
    IMPORT_MODEL_FIELD(sliding_window);
    IMPORT_MODEL_FIELD(minimum_compression_ratio);
    IMPORT_MODEL_FIELD(maximum_compression_ratio);
    IMPORT_MODEL_FIELD(index_heads);
    IMPORT_MODEL_FIELD(index_head_width);
    IMPORT_MODEL_FIELD(index_topk);
    IMPORT_MODEL_FIELD(residual_streams);
    IMPORT_MODEL_FIELD(mhc_sinkhorn_iterations);
    request.mhc_epsilon = model_execution_f64_from_bits(scalars[index++]);
    request.normalization_epsilon = model_execution_f64_from_bits(scalars[index++]);
    IMPORT_MODEL_FIELD(routed_experts);
    IMPORT_MODEL_FIELD(experts_per_row);
    IMPORT_MODEL_FIELD(shared_experts);
    IMPORT_MODEL_FIELD(routed_ffn_width);
    IMPORT_MODEL_FIELD(shared_ffn_width);
    IMPORT_MODEL_FIELD(hash_router_layer_count);
    request.routed_scaling_factor = model_execution_f64_from_bits(scalars[index++]);
    request.activation_limit = model_execution_f64_from_bits(scalars[index++]);
    IMPORT_MODEL_FIELD(output_input_width);
    IMPORT_MODEL_FIELD(output_vocabulary_size);
    IMPORT_MODEL_FIELD(proposal_width);
    IMPORT_MODEL_FIELD(verification_width_maximum);
    IMPORT_MODEL_FIELD(draft_layer_count);
    IMPORT_MODEL_FIELD(target_feature_count);
    {
        size_t layer;
        for (layer = 0u; layer < YVEX_MODEL_EXECUTION_FEATURE_LAYER_CAP; ++layer)
            request.target_feature_layers[layer] = scalars[index++];
    }
    IMPORT_MODEL_FIELD(target_feature_width);
    IMPORT_MODEL_FIELD(markov_rank);
    IMPORT_MODEL_FIELD(confidence_width);
    IMPORT_MODEL_FIELD(persistent_state_class_mask);
    IMPORT_MODEL_FIELD(bos_token_id);
    IMPORT_MODEL_FIELD(eos_token_id);
    IMPORT_MODEL_FIELD(draft_noise_token_id);
#undef IMPORT_MODEL_FIELD
    if (index != YVEX_MODEL_EXECUTION_SCALAR_COUNT)
        return model_execution_refuse(
            err, YVEX_ERR_STATE, "serialized model execution field count drifted");
    rc = yvex_model_execution_descriptor_seal(&request, descriptor, err);
    if (rc != YVEX_OK) return rc;
    if (strcmp(descriptor->identity, expected_identity) != 0) {
        memset(descriptor, 0, sizeof(*descriptor));
        return model_execution_refuse(
            err, YVEX_ERR_FORMAT, "serialized model execution identity disagrees");
    }
    return YVEX_OK;
}

static void model_execution_wire_put_u64(unsigned char *output, unsigned long long value)
{
    unsigned int index;
    for (index = 0u; index < 8u; ++index)
        output[index] = (unsigned char)(value >> (index * 8u));
}

static unsigned long long model_execution_wire_get_u64(const unsigned char *input)
{
    unsigned long long value = 0ull;
    unsigned int index;
    for (index = 0u; index < 8u; ++index)
        value |= (unsigned long long)input[index] << (index * 8u);
    return value;
}

int yvex_model_execution_descriptor_encode(
    const yvex_model_execution_descriptor *descriptor,
    unsigned char output[YVEX_MODEL_EXECUTION_WIRE_BYTES], yvex_error *err)
{
    unsigned long long scalars[YVEX_MODEL_EXECUTION_SCALAR_COUNT];
    const char *identities[YVEX_MODEL_EXECUTION_IDENTITY_COUNT];
    size_t offset = 0u;
    unsigned int index;
    if (!output || !descriptor || !yvex_sha256_hex_valid(descriptor->identity) ||
        !model_execution_descriptor_export(
                       descriptor, scalars, identities))
        return model_execution_refuse(
            err, YVEX_ERR_INVALID_ARG, "sealed model execution descriptor is required");
    model_execution_wire_put_u64(output, descriptor->schema_version);
    offset += 8u;
    for (index = 0u; index < YVEX_MODEL_EXECUTION_SCALAR_COUNT; ++index, offset += 8u)
        model_execution_wire_put_u64(output + offset, scalars[index]);
    for (index = 0u; index < YVEX_MODEL_EXECUTION_IDENTITY_COUNT; ++index) {
        memcpy(output + offset, identities[index], YVEX_SHA256_HEX_BYTES);
        offset += YVEX_SHA256_HEX_BYTES;
    }
    memcpy(output + offset, descriptor->identity, YVEX_SHA256_HEX_BYTES);
    offset += YVEX_SHA256_HEX_BYTES;
    if (offset != YVEX_MODEL_EXECUTION_WIRE_BYTES)
        return model_execution_refuse(
            err, YVEX_ERR_STATE, "model execution wire extent drifted");
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_model_execution_descriptor_decode(
    const unsigned char *bytes, size_t byte_count,
    yvex_model_execution_descriptor *descriptor, yvex_error *err)
{
    unsigned long long scalars[YVEX_MODEL_EXECUTION_SCALAR_COUNT];
    char storage[YVEX_MODEL_EXECUTION_IDENTITY_COUNT][YVEX_MODEL_EXECUTION_IDENTITY_CAP];
    const char *identities[YVEX_MODEL_EXECUTION_IDENTITY_COUNT];
    char expected[YVEX_MODEL_EXECUTION_IDENTITY_CAP];
    size_t offset = 0u;
    unsigned int index;
    if (!bytes || byte_count != YVEX_MODEL_EXECUTION_WIRE_BYTES || !descriptor ||
        model_execution_wire_get_u64(bytes) !=
            YVEX_MODEL_EXECUTION_DESCRIPTOR_SCHEMA_V1)
        return model_execution_refuse(
            err, YVEX_ERR_FORMAT, "model execution wire record is malformed");
    offset += 8u;
    for (index = 0u; index < YVEX_MODEL_EXECUTION_SCALAR_COUNT; ++index, offset += 8u)
        scalars[index] = model_execution_wire_get_u64(bytes + offset);
    for (index = 0u; index < YVEX_MODEL_EXECUTION_IDENTITY_COUNT; ++index) {
        memcpy(storage[index], bytes + offset, YVEX_SHA256_HEX_BYTES);
        storage[index][YVEX_SHA256_HEX_BYTES] = '\0';
        identities[index] = storage[index];
        offset += YVEX_SHA256_HEX_BYTES;
    }
    memcpy(expected, bytes + offset, YVEX_SHA256_HEX_BYTES);
    expected[YVEX_SHA256_HEX_BYTES] = '\0';
    offset += YVEX_SHA256_HEX_BYTES;
    if (offset != byte_count)
        return model_execution_refuse(
            err, YVEX_ERR_FORMAT, "model execution wire extent is noncanonical");
    return model_execution_descriptor_import(
        scalars, identities, expected, descriptor, err);
}

/* Dtype registry */

static const yvex_dtype_info dtype_table[] = {
    {YVEX_DTYPE_UNKNOWN, UINT_MAX},
    {YVEX_DTYPE_F32, YVEX_GGUF_QTYPE_F32},
    {YVEX_DTYPE_F16, YVEX_GGUF_QTYPE_F16},
    {YVEX_DTYPE_BF16, YVEX_GGUF_QTYPE_BF16},
    {YVEX_DTYPE_F64, YVEX_GGUF_QTYPE_F64},
    {YVEX_DTYPE_I8, YVEX_GGUF_QTYPE_I8},
    {YVEX_DTYPE_I16, YVEX_GGUF_QTYPE_I16},
    {YVEX_DTYPE_I32, YVEX_GGUF_QTYPE_I32},
    {YVEX_DTYPE_I64, YVEX_GGUF_QTYPE_I64},
    {YVEX_DTYPE_Q4_0, YVEX_GGUF_QTYPE_Q4_0},
    {YVEX_DTYPE_Q4_1, YVEX_GGUF_QTYPE_Q4_1},
    {YVEX_DTYPE_Q5_0, YVEX_GGUF_QTYPE_Q5_0},
    {YVEX_DTYPE_Q5_1, YVEX_GGUF_QTYPE_Q5_1},
    {YVEX_DTYPE_Q8_0, YVEX_GGUF_QTYPE_Q8_0},
    {YVEX_DTYPE_Q8_1, YVEX_GGUF_QTYPE_Q8_1},
    {YVEX_DTYPE_Q2_K, YVEX_GGUF_QTYPE_Q2_K},
    {YVEX_DTYPE_Q3_K, YVEX_GGUF_QTYPE_Q3_K},
    {YVEX_DTYPE_Q4_K, YVEX_GGUF_QTYPE_Q4_K},
    {YVEX_DTYPE_Q5_K, YVEX_GGUF_QTYPE_Q5_K},
    {YVEX_DTYPE_Q6_K, YVEX_GGUF_QTYPE_Q6_K},
    {YVEX_DTYPE_Q8_K, YVEX_GGUF_QTYPE_Q8_K},
    {YVEX_DTYPE_IQ2_XXS, YVEX_GGUF_QTYPE_IQ2_XXS},
    {YVEX_DTYPE_IQ2_XS, YVEX_GGUF_QTYPE_IQ2_XS},
    {YVEX_DTYPE_IQ3_XXS, YVEX_GGUF_QTYPE_IQ3_XXS},
    {YVEX_DTYPE_IQ1_S, YVEX_GGUF_QTYPE_IQ1_S},
    {YVEX_DTYPE_IQ4_NL, YVEX_GGUF_QTYPE_IQ4_NL},
    {YVEX_DTYPE_IQ3_S, YVEX_GGUF_QTYPE_IQ3_S},
    {YVEX_DTYPE_IQ2_S, YVEX_GGUF_QTYPE_IQ2_S},
    {YVEX_DTYPE_IQ4_XS, YVEX_GGUF_QTYPE_IQ4_XS},
    {YVEX_DTYPE_IQ1_M, YVEX_GGUF_QTYPE_IQ1_M},
    {YVEX_DTYPE_TQ1_0, YVEX_GGUF_QTYPE_TQ1_0},
    {YVEX_DTYPE_TQ2_0, YVEX_GGUF_QTYPE_TQ2_0},
    {YVEX_DTYPE_MXFP4, YVEX_GGUF_QTYPE_MXFP4},
};

static const unsigned long dtype_table_count = sizeof(dtype_table) / sizeof(dtype_table[0]);

typedef struct {
    const char *name;
    yvex_arch value;
} architecture_name;

static const architecture_name architecture_names[] = {
    {"llama", YVEX_ARCH_LLAMA}, {"qwen", YVEX_ARCH_QWEN},
    {"deepseek", YVEX_ARCH_DEEPSEEK}, {"gemma", YVEX_ARCH_GEMMA},
    {"phi", YVEX_ARCH_PHI}, {"kimi", YVEX_ARCH_KIMI}, {"glm", YVEX_ARCH_GLM}
};

const yvex_dtype_info *yvex_dtype_get_info(yvex_dtype dtype)
{
    unsigned long i;

    for (i = 0; i < dtype_table_count; ++i) {
        if (dtype_table[i].dtype == dtype) {
            return &dtype_table[i];
        }
    }

    return &dtype_table[0];
}

const yvex_dtype_info *yvex_dtype_from_ggml_type(unsigned int ggml_type)
{
    unsigned long i;

    for (i = 0; i < dtype_table_count; ++i) {
        if (dtype_table[i].ggml_type == ggml_type) {
            return &dtype_table[i];
        }
    }

    return &dtype_table[0];
}

const char *yvex_dtype_name(yvex_dtype dtype)
{
    const yvex_dtype_info *info = yvex_dtype_get_info(dtype);

    return info->dtype == YVEX_DTYPE_UNKNOWN
        ? "UNKNOWN"
        : yvex_gguf_qtype_name(info->ggml_type);
}

int yvex_dtype_is_quantized(yvex_dtype dtype)
{
    const yvex_dtype_info *info = yvex_dtype_get_info(dtype);
    const yvex_gguf_qtype_geometry *geometry =
        yvex_gguf_qtype_geometry_find(info->ggml_type);

    return geometry &&
        geometry->storage_class == YVEX_GGUF_QTYPE_STORAGE_BLOCK_QUANTIZED;
}

int yvex_dtype_storage_supported(yvex_dtype dtype)
{
    const yvex_dtype_info *info = yvex_dtype_get_info(dtype);

    return info->dtype != YVEX_DTYPE_UNKNOWN &&
        yvex_gguf_qtype_supported_for_storage(info->ggml_type, NULL);
}

static int dtype_storage_error_code(yvex_gguf_qtype_storage_status status)
{
    switch (status) {
    case YVEX_GGUF_QTYPE_STORAGE_OK:
        return YVEX_OK;
    case YVEX_GGUF_QTYPE_STORAGE_INVALID_ARGUMENT:
        return YVEX_ERR_INVALID_ARG;
    case YVEX_GGUF_QTYPE_STORAGE_INVALID_RANK:
    case YVEX_GGUF_QTYPE_STORAGE_INVALID_DIMENSION:
    case YVEX_GGUF_QTYPE_STORAGE_ROW_BLOCK_MISMATCH:
    case YVEX_GGUF_QTYPE_STORAGE_EXPECTED_ACTUAL_MISMATCH:
        return YVEX_ERR_FORMAT;
    case YVEX_GGUF_QTYPE_STORAGE_ELEMENT_COUNT_OVERFLOW:
    case YVEX_GGUF_QTYPE_STORAGE_ROW_BYTE_OVERFLOW:
    case YVEX_GGUF_QTYPE_STORAGE_ROW_COUNT_OVERFLOW:
    case YVEX_GGUF_QTYPE_STORAGE_TOTAL_BYTE_OVERFLOW:
        return YVEX_ERR_BOUNDS;
    case YVEX_GGUF_QTYPE_STORAGE_UNKNOWN_ID:
    case YVEX_GGUF_QTYPE_STORAGE_REMOVED_ID:
    case YVEX_GGUF_QTYPE_STORAGE_RESERVED_ID:
    case YVEX_GGUF_QTYPE_STORAGE_OUTSIDE_BASELINE:
    case YVEX_GGUF_QTYPE_STORAGE_GEOMETRY_UNAVAILABLE:
        return YVEX_ERR_UNSUPPORTED;
    }
    return YVEX_ERR_UNSUPPORTED;
}

int yvex_dtype_tensor_storage_bytes(yvex_dtype dtype,
                                    const unsigned long long *dims,
                                    unsigned int rank,
                                    unsigned long long *out,
                                    yvex_error *err)
{
    const yvex_dtype_info *info;
    yvex_gguf_qtype_storage_result result;
    yvex_gguf_qtype_storage_status status;
    int rc;

    if (!out || !dims) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG,
                       "yvex_dtype_tensor_storage_bytes",
                       "dims and out are required");
        return YVEX_ERR_INVALID_ARG;
    }
    *out = 0ull;
    info = yvex_dtype_get_info(dtype);
    if (info->dtype == YVEX_DTYPE_UNKNOWN) {
        yvex_error_set(err, YVEX_ERR_UNSUPPORTED,
                       "yvex_dtype_tensor_storage_bytes",
                       "unknown dtype has no GGUF storage identity");
        return YVEX_ERR_UNSUPPORTED;
    }
    status = yvex_gguf_qtype_tensor_storage(info->ggml_type, dims, rank, &result);
    rc = dtype_storage_error_code(status);
    if (rc != YVEX_OK) {
        yvex_error_setf(err, rc, "yvex_dtype_tensor_storage_bytes",
                        "%s: %s",
                        yvex_gguf_qtype_storage_status_name(status),
                        result.reason ? result.reason : "qtype storage refused");
        return rc;
    }
    *out = result.total_bytes;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_dtype_storage_bytes(yvex_dtype dtype,
                             unsigned long long row_element_count,
                             unsigned long long *out,
                             yvex_error *err)
{
    if (!out) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG,
                       "yvex_dtype_storage_bytes", "out is required");
        return YVEX_ERR_INVALID_ARG;
    }
    return yvex_dtype_tensor_storage_bytes(dtype, &row_element_count, 1u, out, err);
}

/* Model descriptors */

struct yvex_model_descriptor {
    yvex_arch arch;
    char *name;
    unsigned long long context_length;
    unsigned long long tensor_count;
    unsigned long long total_storage_bytes;
    unsigned long long unsupported_tensor_accounting_count;
    unsigned long long role_counts[YVEX_TENSOR_ROLE_COUNT];
};

static char *copy_bytes_string(const char *data, unsigned long long len)
{
    char *copy;

    if (!data) {
        return NULL;
    }
    if (len > (unsigned long long)(SIZE_MAX - 1)) {
        return NULL;
    }
    copy = (char *)malloc((size_t)len + 1u);
    if (!copy) {
        return NULL;
    }
    memcpy(copy, data, (size_t)len);
    copy[len] = '\0';
    return copy;
}

static yvex_arch arch_from_name(const char *name)
{
    size_t index;

    if (!name) return YVEX_ARCH_UNKNOWN;
    for (index = 0u; index < sizeof(architecture_names) / sizeof(architecture_names[0]);
         ++index) {
        if (strcmp(name, architecture_names[index].name) == 0)
            return architecture_names[index].value;
    }
    return YVEX_ARCH_UNKNOWN;
}

/*
 * Build a model descriptor from parsed GGUF metadata and tensor-table facts.
 *
 * Returns invalid-arg or allocation failures and releases partial descriptor ownership before
 * returning.
 */
int yvex_model_descriptor_from_gguf(yvex_model_descriptor **out,
                                    const yvex_gguf *gguf,
                                    const yvex_tensor_table *tensors,
                                    yvex_error *err)
{
    yvex_model_descriptor *model;
    const yvex_gguf_value *value;
    const char *text;
    unsigned long long len;
    unsigned long long i;

    if (!out) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "yvex_model_descriptor_from_gguf", "out is required");
        return YVEX_ERR_INVALID_ARG;
    }
    *out = NULL;

    if (!gguf || !tensors) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "yvex_model_descriptor_from_gguf", "gguf and tensors are required");
        return YVEX_ERR_INVALID_ARG;
    }

    model = (yvex_model_descriptor *)calloc(1, sizeof(*model));
    if (!model) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "yvex_model_descriptor_from_gguf", "failed to allocate model descriptor");
        return YVEX_ERR_NOMEM;
    }

    value = yvex_gguf_metadata_find(gguf, "general.architecture");
    if (value && yvex_gguf_value_as_string(value, &text, &len) == YVEX_OK) {
        char *arch_text = copy_bytes_string(text, len);
        model->arch = arch_from_name(arch_text);
        free(arch_text);
    }

    value = yvex_gguf_metadata_find(gguf, "general.name");
    if (value && yvex_gguf_value_as_string(value, &text, &len) == YVEX_OK) {
        model->name = copy_bytes_string(text, len);
        if (!model->name) {
            yvex_model_descriptor_close(model);
            yvex_error_set(err, YVEX_ERR_NOMEM, "yvex_model_descriptor_from_gguf", "failed to copy model name");
            return YVEX_ERR_NOMEM;
        }
    }

    value = yvex_gguf_metadata_find(gguf, "llama.context_length");
    if (value) {
        (void)yvex_gguf_value_as_u64(value, &model->context_length);
    }

    model->tensor_count = yvex_tensor_table_count(tensors);
    for (i = 0; i < model->tensor_count; ++i) {
        const yvex_tensor_info *tensor = yvex_tensor_table_at(tensors, i);
        if (!tensor) {
            continue;
        }
        if (tensor->storage_bytes == 0) {
            model->unsupported_tensor_accounting_count += 1;
        } else {
            model->total_storage_bytes += tensor->storage_bytes;
        }
        if (tensor->role < YVEX_TENSOR_ROLE_COUNT) {
            model->role_counts[tensor->role] += 1;
        }
    }

    *out = model;
    yvex_error_clear(err);
    return YVEX_OK;
}

void yvex_model_descriptor_close(yvex_model_descriptor *model)
{
    if (!model) {
        return;
    }
    free(model->name);
    free(model);
}

yvex_arch yvex_model_arch(const yvex_model_descriptor *model)
{
    return model ? model->arch : YVEX_ARCH_UNKNOWN;
}

const char *yvex_arch_name(yvex_arch arch)
{
    size_t index;

    for (index = 0u; index < sizeof(architecture_names) / sizeof(architecture_names[0]);
         ++index)
        if (architecture_names[index].value == arch) return architecture_names[index].name;
    return "unknown";
}

/* Return the immutable model descriptor name without transferring ownership. */
const char *yvex_model_name(const yvex_model_descriptor *model)
{
    if (!model || !model->name) {
        return "";
    }
    return model->name;
}

int yvex_model_context_open(const char *path_or_alias,
                            yvex_model_context *out,
                            yvex_error *err)
{
    yvex_model_ref ref;
    yvex_artifact_options artifact_options;
    yvex_artifact_integrity_options integrity_options;
    yvex_artifact_integrity_report integrity_report;
    int rc;

    if (!path_or_alias || !path_or_alias[0] || !out) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "model_context",
                       "path or alias and output context are required");
        return YVEX_ERR_INVALID_ARG;
    }
    memset(out, 0, sizeof(*out));
    memset(&ref, 0, sizeof(ref));
    memset(&artifact_options, 0, sizeof(artifact_options));
    memset(&integrity_options, 0, sizeof(integrity_options));
    memset(&integrity_report, 0, sizeof(integrity_report));

    rc = yvex_model_ref_resolve(&ref, path_or_alias, NULL, err);
    if (rc == YVEX_OK) {
        artifact_options.path = ref.path;
        artifact_options.readonly = 1;
        rc = yvex_artifact_open(&out->artifact, &artifact_options, err);
    }
    if (rc == YVEX_OK) {
        rc = yvex_gguf_open(&out->gguf, out->artifact, err);
    }
    if (rc == YVEX_OK) {
        rc = yvex_tensor_table_from_gguf(&out->table, out->gguf, err);
    }
    if (rc == YVEX_OK) {
        rc = yvex_model_descriptor_from_gguf(&out->model,
                                             out->gguf,
                                             out->table,
                                             err);
    }
    if (rc == YVEX_OK) {
        rc = yvex_artifact_integrity_validate(out->artifact,
                                              out->gguf,
                                              out->table,
                                              &integrity_options,
                                              &integrity_report,
                                              err);
    }
    yvex_model_ref_clear(&ref);
    if (rc != YVEX_OK) {
        yvex_model_context_close(out);
    }
    return rc;
}

int yvex_model_context_open_tokenizer(const char *path_or_alias,
                                      yvex_model_context *out,
                                      yvex_error *err)
{
    int rc = yvex_model_context_open(path_or_alias, out, err);

    if (rc == YVEX_OK) {
        rc = yvex_tokenizer_from_gguf(&out->tokenizer,
                                      out->gguf,
                                      out->model,
                                      err);
    }
    if (rc != YVEX_OK) {
        yvex_model_context_close(out);
    }
    return rc;
}

void yvex_model_context_close(yvex_model_context *context)
{
    if (!context) return;
    yvex_tokenizer_close(context->tokenizer);
    yvex_model_descriptor_close(context->model);
    yvex_tensor_table_close(context->table);
    yvex_gguf_close(context->gguf);
    yvex_artifact_close(context->artifact);
    memset(context, 0, sizeof(*context));
}

int yvex_model_context_vocab_size(const char *path_or_alias,
                                  unsigned long long *out_vocab_size,
                                  yvex_error *err)
{
    yvex_model_context context;
    const yvex_tensor_info *tensor;
    yvex_tokenizer *tokenizer = NULL;
    int rc;

    if (!path_or_alias || !out_vocab_size) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "model_vocab",
                       "path or alias and output are required");
        return YVEX_ERR_INVALID_ARG;
    }
    *out_vocab_size = 0ull;
    memset(&context, 0, sizeof(context));
    rc = yvex_model_context_open(path_or_alias, &context, err);
    if (rc != YVEX_OK) return rc;

    tensor = yvex_tensor_table_find(context.table, "token_embd.weight");
    if (tensor && tensor->rank == 2u && tensor->dims[1] > 0ull) {
        *out_vocab_size = tensor->dims[1];
        yvex_model_context_close(&context);
        yvex_error_clear(err);
        return YVEX_OK;
    }
    rc = yvex_tokenizer_from_gguf(&tokenizer,
                                  context.gguf,
                                  context.model,
                                  err);
    if (rc == YVEX_OK && yvex_tokenizer_vocab_size(tokenizer) > 0ull) {
        *out_vocab_size = yvex_tokenizer_vocab_size(tokenizer);
        yvex_tokenizer_close(tokenizer);
        yvex_model_context_close(&context);
        yvex_error_clear(err);
        return YVEX_OK;
    }
    yvex_tokenizer_close(tokenizer);
    yvex_model_context_close(&context);
    yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "model_vocab",
                   "tokenizer-metadata-missing");
    return YVEX_ERR_UNSUPPORTED;
}

unsigned long long yvex_model_context_length(const yvex_model_descriptor *model)
{
    return model ? model->context_length : 0;
}

unsigned long long yvex_model_tensor_count(const yvex_model_descriptor *model)
{
    return model ? model->tensor_count : 0;
}

unsigned long long yvex_model_total_storage_bytes(const yvex_model_descriptor *model)
{
    return model ? model->total_storage_bytes : 0;
}

unsigned long long yvex_model_unsupported_tensor_accounting_count(const yvex_model_descriptor *model)
{
    return model ? model->unsupported_tensor_accounting_count : 0;
}

unsigned long long yvex_model_role_count(const yvex_model_descriptor *model, yvex_tensor_role role)
{
    if (!model || role >= YVEX_TENSOR_ROLE_COUNT) {
        return 0;
    }
    return model->role_counts[role];
}

/* Tensor roles */

static const char *const tensor_role_names[YVEX_TENSOR_ROLE_COUNT] = {
    [YVEX_TENSOR_ROLE_UNKNOWN] = "unknown",
    [YVEX_TENSOR_ROLE_TOKEN_EMBEDDING] = "token_embedding",
    [YVEX_TENSOR_ROLE_OUTPUT_NORM] = "output_norm",
    [YVEX_TENSOR_ROLE_OUTPUT_HEAD] = "output_head",
    [YVEX_TENSOR_ROLE_ATTENTION_NORM] = "attention_norm",
    [YVEX_TENSOR_ROLE_ATTENTION_Q] = "attention_q",
    [YVEX_TENSOR_ROLE_ATTENTION_K] = "attention_k",
    [YVEX_TENSOR_ROLE_ATTENTION_V] = "attention_v",
    [YVEX_TENSOR_ROLE_ATTENTION_OUT] = "attention_out",
    [YVEX_TENSOR_ROLE_FFN_NORM] = "ffn_norm",
    [YVEX_TENSOR_ROLE_FFN_GATE] = "ffn_gate",
    [YVEX_TENSOR_ROLE_FFN_UP] = "ffn_up",
    [YVEX_TENSOR_ROLE_FFN_DOWN] = "ffn_down",
    [YVEX_TENSOR_ROLE_MOE_ROUTER] = "moe_router",
    [YVEX_TENSOR_ROLE_MOE_EXPERT_GATE] = "moe_expert_gate",
    [YVEX_TENSOR_ROLE_MOE_EXPERT_UP] = "moe_expert_up",
    [YVEX_TENSOR_ROLE_MOE_EXPERT_DOWN] = "moe_expert_down",
    [YVEX_TENSOR_ROLE_HC_HEAD_FUNCTION] = "hc_head_function",
    [YVEX_TENSOR_ROLE_HC_HEAD_BASE] = "hc_head_base",
    [YVEX_TENSOR_ROLE_HC_HEAD_SCALE] = "hc_head_scale",
    [YVEX_TENSOR_ROLE_ATTENTION_SINKS] = "attention_sinks",
    [YVEX_TENSOR_ROLE_ATTENTION_Q_A] = "attention_q_a",
    [YVEX_TENSOR_ROLE_ATTENTION_Q_B] = "attention_q_b",
    [YVEX_TENSOR_ROLE_ATTENTION_Q_A_NORM] = "attention_q_a_norm",
    [YVEX_TENSOR_ROLE_ATTENTION_KV] = "attention_kv",
    [YVEX_TENSOR_ROLE_ATTENTION_KV_NORM] = "attention_kv_norm",
    [YVEX_TENSOR_ROLE_ATTENTION_OUT_A] = "attention_out_a",
    [YVEX_TENSOR_ROLE_ATTENTION_OUT_B] = "attention_out_b",
    [YVEX_TENSOR_ROLE_HC_ATTENTION_FUNCTION] = "hc_attention_function",
    [YVEX_TENSOR_ROLE_HC_ATTENTION_BASE] = "hc_attention_base",
    [YVEX_TENSOR_ROLE_HC_ATTENTION_SCALE] = "hc_attention_scale",
    [YVEX_TENSOR_ROLE_HC_FFN_FUNCTION] = "hc_ffn_function",
    [YVEX_TENSOR_ROLE_HC_FFN_BASE] = "hc_ffn_base",
    [YVEX_TENSOR_ROLE_HC_FFN_SCALE] = "hc_ffn_scale",
    [YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_KV] = "attention_compressor_kv",
    [YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_GATE] = "attention_compressor_gate",
    [YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_APE] = "attention_compressor_ape",
    [YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_NORM] = "attention_compressor_norm",
    [YVEX_TENSOR_ROLE_INDEXER_PROJECTION] = "indexer_projection",
    [YVEX_TENSOR_ROLE_INDEXER_ATTENTION_Q_B] = "indexer_attention_q_b",
    [YVEX_TENSOR_ROLE_INDEXER_COMPRESSOR_KV] = "indexer_compressor_kv",
    [YVEX_TENSOR_ROLE_INDEXER_COMPRESSOR_GATE] = "indexer_compressor_gate",
    [YVEX_TENSOR_ROLE_INDEXER_COMPRESSOR_APE] = "indexer_compressor_ape",
    [YVEX_TENSOR_ROLE_INDEXER_COMPRESSOR_NORM] = "indexer_compressor_norm",
    [YVEX_TENSOR_ROLE_MOE_ROUTER_BIAS] = "moe_router_bias",
    [YVEX_TENSOR_ROLE_MOE_ROUTER_TABLE] = "moe_router_table",
    [YVEX_TENSOR_ROLE_MOE_SHARED_EXPERT_GATE] = "moe_shared_expert_gate",
    [YVEX_TENSOR_ROLE_MOE_SHARED_EXPERT_UP] = "moe_shared_expert_up",
    [YVEX_TENSOR_ROLE_MOE_SHARED_EXPERT_DOWN] = "moe_shared_expert_down",
    [YVEX_TENSOR_ROLE_DRAFT_FEATURE_PROJECTION] = "draft_feature_projection",
    [YVEX_TENSOR_ROLE_DRAFT_FEATURE_NORM] = "draft_feature_norm",
    [YVEX_TENSOR_ROLE_DRAFT_OUTPUT_NORM] = "draft_output_norm",
    [YVEX_TENSOR_ROLE_DRAFT_MARKOV_EMBEDDING] = "draft_markov_embedding",
    [YVEX_TENSOR_ROLE_DRAFT_MARKOV_OUTPUT] = "draft_markov_output",
    [YVEX_TENSOR_ROLE_DRAFT_CONFIDENCE] = "draft_confidence",
    [YVEX_TENSOR_ROLE_ATTENTION_Q_NORM] = "attention_q_norm",
    [YVEX_TENSOR_ROLE_ATTENTION_K_NORM] = "attention_k_norm",
    [YVEX_TENSOR_ROLE_SEQUENCE_MIXER_DECAY_LOG] = "sequence_mixer_decay_log",
    [YVEX_TENSOR_ROLE_SEQUENCE_MIXER_CONVOLUTION] = "sequence_mixer_convolution",
    [YVEX_TENSOR_ROLE_SEQUENCE_MIXER_TIME_BIAS] = "sequence_mixer_time_bias",
    [YVEX_TENSOR_ROLE_SEQUENCE_MIXER_DECAY_PROJECTION] =
        "sequence_mixer_decay_projection",
    [YVEX_TENSOR_ROLE_SEQUENCE_MIXER_BETA_PROJECTION] =
        "sequence_mixer_beta_projection",
    [YVEX_TENSOR_ROLE_SEQUENCE_MIXER_QKV_PROJECTION] =
        "sequence_mixer_qkv_projection",
    [YVEX_TENSOR_ROLE_SEQUENCE_MIXER_OUTPUT_GATE] = "sequence_mixer_output_gate",
    [YVEX_TENSOR_ROLE_SEQUENCE_MIXER_OUTPUT_NORM] = "sequence_mixer_output_norm",
    [YVEX_TENSOR_ROLE_SEQUENCE_MIXER_OUTPUT] = "sequence_mixer_output"
};

const char *yvex_tensor_role_name(yvex_tensor_role role)
{
    return role >= 0 && role < YVEX_TENSOR_ROLE_COUNT ? tensor_role_names[role] : "unknown";
}

static int ends_with(const char *text, const char *suffix)
{
    size_t text_len;
    size_t suffix_len;

    if (!text || !suffix) {
        return 0;
    }

    text_len = strlen(text);
    suffix_len = strlen(suffix);
    if (suffix_len > text_len) {
        return 0;
    }

    return strcmp(text + text_len - suffix_len, suffix) == 0;
}

static int contains(const char *text, const char *needle)
{
    return text && needle && strstr(text, needle) != NULL;
}

typedef enum {
    TENSOR_ROLE_EXACT = 0,
    TENSOR_ROLE_SUFFIX
} tensor_role_match;

typedef struct {
    const char *pattern;
    const char *required_fragment;
    yvex_tensor_role role;
    tensor_role_match match;
} tensor_role_rule;

static const tensor_role_rule tensor_role_rules[] = {
    {"token_embd.weight", NULL, YVEX_TENSOR_ROLE_TOKEN_EMBEDDING, TENSOR_ROLE_EXACT},
    {"model.embed_tokens.weight", NULL, YVEX_TENSOR_ROLE_TOKEN_EMBEDDING,
     TENSOR_ROLE_EXACT},
    {"tok_embeddings.weight", NULL, YVEX_TENSOR_ROLE_TOKEN_EMBEDDING, TENSOR_ROLE_EXACT},
    {"output_norm.weight", NULL, YVEX_TENSOR_ROLE_OUTPUT_NORM, TENSOR_ROLE_EXACT},
    {"model.norm.weight", NULL, YVEX_TENSOR_ROLE_OUTPUT_NORM, TENSOR_ROLE_EXACT},
    {"norm.weight", NULL, YVEX_TENSOR_ROLE_OUTPUT_NORM, TENSOR_ROLE_EXACT},
    {"output.weight", NULL, YVEX_TENSOR_ROLE_OUTPUT_HEAD, TENSOR_ROLE_EXACT},
    {"lm_head.weight", NULL, YVEX_TENSOR_ROLE_OUTPUT_HEAD, TENSOR_ROLE_EXACT},
    {".attn_norm.weight", NULL, YVEX_TENSOR_ROLE_ATTENTION_NORM, TENSOR_ROLE_SUFFIX},
    {".input_layernorm.weight", NULL, YVEX_TENSOR_ROLE_ATTENTION_NORM, TENSOR_ROLE_SUFFIX},
    {".attn_q.weight", NULL, YVEX_TENSOR_ROLE_ATTENTION_Q, TENSOR_ROLE_SUFFIX},
    {".self_attn.q_proj.weight", NULL, YVEX_TENSOR_ROLE_ATTENTION_Q, TENSOR_ROLE_SUFFIX},
    {".attn_k.weight", NULL, YVEX_TENSOR_ROLE_ATTENTION_K, TENSOR_ROLE_SUFFIX},
    {".self_attn.k_proj.weight", NULL, YVEX_TENSOR_ROLE_ATTENTION_K, TENSOR_ROLE_SUFFIX},
    {".attn_v.weight", NULL, YVEX_TENSOR_ROLE_ATTENTION_V, TENSOR_ROLE_SUFFIX},
    {".self_attn.v_proj.weight", NULL, YVEX_TENSOR_ROLE_ATTENTION_V, TENSOR_ROLE_SUFFIX},
    {".attn_output.weight", NULL, YVEX_TENSOR_ROLE_ATTENTION_OUT, TENSOR_ROLE_SUFFIX},
    {".self_attn.o_proj.weight", NULL, YVEX_TENSOR_ROLE_ATTENTION_OUT, TENSOR_ROLE_SUFFIX},
    {".ffn_norm.weight", NULL, YVEX_TENSOR_ROLE_FFN_NORM, TENSOR_ROLE_SUFFIX},
    {".post_attention_layernorm.weight", NULL, YVEX_TENSOR_ROLE_FFN_NORM,
     TENSOR_ROLE_SUFFIX},
    {".ffn_gate.weight", NULL, YVEX_TENSOR_ROLE_FFN_GATE, TENSOR_ROLE_SUFFIX},
    {".mlp.gate_proj.weight", NULL, YVEX_TENSOR_ROLE_FFN_GATE, TENSOR_ROLE_SUFFIX},
    {".ffn_up.weight", NULL, YVEX_TENSOR_ROLE_FFN_UP, TENSOR_ROLE_SUFFIX},
    {".mlp.up_proj.weight", NULL, YVEX_TENSOR_ROLE_FFN_UP, TENSOR_ROLE_SUFFIX},
    {".ffn_down.weight", NULL, YVEX_TENSOR_ROLE_FFN_DOWN, TENSOR_ROLE_SUFFIX},
    {".mlp.down_proj.weight", NULL, YVEX_TENSOR_ROLE_FFN_DOWN, TENSOR_ROLE_SUFFIX},
    {".ffn_gate_inp.weight", NULL, YVEX_TENSOR_ROLE_MOE_ROUTER, TENSOR_ROLE_SUFFIX},
    {".mlp.gate.weight", NULL, YVEX_TENSOR_ROLE_MOE_ROUTER, TENSOR_ROLE_SUFFIX},
    {".gate.weight", ".ffn.experts.", YVEX_TENSOR_ROLE_MOE_EXPERT_GATE,
     TENSOR_ROLE_SUFFIX},
    {".up.weight", ".ffn.experts.", YVEX_TENSOR_ROLE_MOE_EXPERT_UP, TENSOR_ROLE_SUFFIX},
    {".down.weight", ".ffn.experts.", YVEX_TENSOR_ROLE_MOE_EXPERT_DOWN,
     TENSOR_ROLE_SUFFIX}
};

yvex_tensor_role yvex_tensor_role_classify(const char *architecture,
                                           const char *tensor_name,
                                           unsigned int rank,
                                           const unsigned long long *dims,
                                           yvex_dtype dtype)
{
    size_t index;

    (void)architecture;
    (void)rank;
    (void)dims;
    (void)dtype;

    if (!tensor_name) return YVEX_TENSOR_ROLE_UNKNOWN;
    for (index = 0u; index < sizeof(tensor_role_rules) / sizeof(tensor_role_rules[0]);
         ++index) {
        const tensor_role_rule *rule = &tensor_role_rules[index];
        int matched = rule->match == TENSOR_ROLE_EXACT
                          ? strcmp(tensor_name, rule->pattern) == 0
                          : ends_with(tensor_name, rule->pattern);

        if (matched && (!rule->required_fragment ||
                        contains(tensor_name, rule->required_fragment)))
            return rule->role;
    }

    return YVEX_TENSOR_ROLE_UNKNOWN;
}

/* Tensor table */

struct yvex_tensor_table {
    yvex_tensor_info *items;
    unsigned long long count;
};

static int product_dims(const yvex_gguf_tensor_info *src, unsigned long long *out, yvex_error *err)
{
    unsigned int i;
    unsigned long long product = 1;

    for (i = 0; i < src->rank; ++i) {
        if (src->dims[i] == 0) {
            yvex_error_setf(err, YVEX_ERR_FORMAT, "yvex_tensor_table_from_gguf",
                            "tensor %s has zero dimension", src->name);
            return YVEX_ERR_FORMAT;
        }
        if (product > ULLONG_MAX / src->dims[i]) {
            yvex_error_setf(err, YVEX_ERR_BOUNDS, "yvex_tensor_table_from_gguf",
                            "element count overflow for tensor %s", src->name);
            return YVEX_ERR_BOUNDS;
        }
        product *= src->dims[i];
    }

    *out = product;
    return YVEX_OK;
}

int yvex_tensor_table_from_gguf(yvex_tensor_table **out,
                                const yvex_gguf *gguf,
                                yvex_error *err)
{
    yvex_tensor_table *table;
    unsigned long long count;
    unsigned long long i;

    if (!out) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "yvex_tensor_table_from_gguf", "out is required");
        return YVEX_ERR_INVALID_ARG;
    }
    *out = NULL;

    if (!gguf) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "yvex_tensor_table_from_gguf", "gguf is required");
        return YVEX_ERR_INVALID_ARG;
    }

    count = yvex_gguf_tensor_count(gguf);
    table = (yvex_tensor_table *)calloc(1, sizeof(*table));
    if (!table) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "yvex_tensor_table_from_gguf", "failed to allocate tensor table");
        return YVEX_ERR_NOMEM;
    }
    table->count = count;

    if (count > 0) {
        if (count > (unsigned long long)(SIZE_MAX / sizeof(*table->items))) {
            yvex_tensor_table_close(table);
            yvex_error_set(err, YVEX_ERR_NOMEM, "yvex_tensor_table_from_gguf", "tensor count too large");
            return YVEX_ERR_NOMEM;
        }
        table->items = (yvex_tensor_info *)calloc((size_t)count, sizeof(*table->items));
        if (!table->items) {
            yvex_tensor_table_close(table);
            yvex_error_set(err, YVEX_ERR_NOMEM, "yvex_tensor_table_from_gguf", "failed to allocate tensor rows");
            return YVEX_ERR_NOMEM;
        }
    }

    for (i = 0; i < count; ++i) {
        const yvex_gguf_tensor_info *src = yvex_gguf_tensor_at(gguf, i);
        yvex_tensor_info *dst = &table->items[i];
        const yvex_dtype_info *dtype_info;
        int rc;

        dst->name = yvex_core_strdup(src->name);
        if (!dst->name) {
            yvex_tensor_table_close(table);
            yvex_error_set(err, YVEX_ERR_NOMEM, "yvex_tensor_table_from_gguf", "failed to copy tensor name");
            return YVEX_ERR_NOMEM;
        }
        dst->rank = src->rank;
        memcpy(dst->dims, src->dims, sizeof(dst->dims));
        dst->ggml_type = src->ggml_type;
        dst->relative_offset = src->relative_offset;
        dst->absolute_offset = src->absolute_offset;

        dtype_info = yvex_dtype_from_ggml_type(src->ggml_type);
        dst->dtype = dtype_info->dtype;

        rc = product_dims(src, &dst->element_count, err);
        if (rc != YVEX_OK) {
            yvex_tensor_table_close(table);
            return rc;
        }

        rc = yvex_dtype_tensor_storage_bytes(dst->dtype,
                                             dst->dims,
                                             dst->rank,
                                             &dst->storage_bytes,
                                             err);
        if (rc == YVEX_ERR_UNSUPPORTED) {
            dst->storage_bytes = 0;
            yvex_error_clear(err);
        } else if (rc != YVEX_OK) {
            yvex_tensor_table_close(table);
            return rc;
        }

        dst->role = yvex_tensor_role_classify(NULL, dst->name, dst->rank, dst->dims, dst->dtype);
    }

    *out = table;
    yvex_error_clear(err);
    return YVEX_OK;
}

void yvex_tensor_table_close(yvex_tensor_table *table)
{
    unsigned long long i;

    if (!table) {
        return;
    }
    if (table->items) {
        for (i = 0; i < table->count; ++i) {
            free((char *)table->items[i].name);
            table->items[i].name = NULL;
        }
        free(table->items);
    }
    free(table);
}

unsigned long long yvex_tensor_table_count(const yvex_tensor_table *table)
{
    return table ? table->count : 0;
}

const yvex_tensor_info *yvex_tensor_table_at(const yvex_tensor_table *table,
                                             unsigned long long index)
{
    if (!table || index >= table->count) {
        return NULL;
    }
    return &table->items[index];
}

const yvex_tensor_info *yvex_tensor_table_find(const yvex_tensor_table *table,
                                               const char *name)
{
    unsigned long long i;

    if (!table || !name) {
        return NULL;
    }

    for (i = 0; i < table->count; ++i) {
        if (table->items[i].name && strcmp(table->items[i].name, name) == 0) {
            return &table->items[i];
        }
    }

    return NULL;
}
