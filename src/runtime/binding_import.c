/* Instantiate authenticated binding records without reopening family compilation. */
#include "src/runtime/private.h"

#include <stdlib.h>
#include <string.h>

static int binding_v14_hash_finish(yvex_sha256 *hash,
                                   char output[YVEX_SHA256_HEX_CAP])
{
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    if (!yvex_sha256_final(hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

static int binding_v14_decision_authentic(
    const yvex_runtime_binding_physical_decision_v14 *decision)
{
    yvex_sha256 hash;
    char identity[YVEX_SHA256_HEX_CAP];
    if (!decision || decision->schema_version != 4u ||
        decision->role <= YVEX_TENSOR_ROLE_UNKNOWN ||
        decision->role >= YVEX_TENSOR_ROLE_COUNT ||
        decision->scope > YVEX_TENSOR_SCOPE_DRAFT ||
        decision->consumer >= YVEX_EXECUTION_CONSUMER_COUNT ||
        decision->layout > 3u ||
        decision->placement > 4u || decision->sharing > YVEX_EXECUTION_SHARING_ALIAS ||
        decision->activation > YVEX_EXECUTION_ACTIVATION_DEVICE_ENCODED ||
        decision->required_backend > 2u || !decision->canonical_row_width ||
        !decision->canonical_row_count || !decision->encoded_bytes ||
        !decision->alignment || !decision->supported_width_mask ||
        decision->evidence > YVEX_EXECUTION_EVIDENCE_FORENSIC ||
        decision->fallback > YVEX_EXECUTION_CLASS_FORENSIC_REFERENCE ||
        !decision->kernel_family[0] ||
        ((decision->tensor_core_minimum != 0ull) !=
         (decision->tensor_core_kernel_family[0] != '\0')) ||
        (decision->worklist_width_mask &&
         (decision->consumer < YVEX_EXECUTION_CONSUMER_ROUTED_GATE_UP ||
          decision->consumer > YVEX_EXECUTION_CONSUMER_ROUTED_DOWN)) ||
        (decision->tensor_core_minimum &&
         (!decision->worklist_width_mask || decision->tensor_core_minimum >= 63ull ||
          !(decision->worklist_width_mask & (1ull << decision->tensor_core_minimum)) ||
          strcmp(decision->kernel_family, decision->tensor_core_kernel_family) == 0)) ||
        !yvex_sha256_hex_is_valid(decision->terminal_identity) ||
        !yvex_sha256_hex_is_valid(decision->decision_identity))
        return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.physical-execution.decision.v4") ||
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
        !yvex_sha256_update_u64(
            &hash, (unsigned long long)decision->derived_asset_required) ||
        !yvex_sha256_update_u64(&hash, decision->worklist_width_mask) ||
        !yvex_sha256_update_u64(&hash, decision->tensor_core_minimum) ||
        !yvex_sha256_update_text(&hash, decision->terminal_identity) ||
        !yvex_sha256_update_text(&hash, decision->kernel_family) ||
        !yvex_sha256_update_text(&hash, decision->tensor_core_kernel_family) ||
        !binding_v14_hash_finish(&hash, identity))
        return 0;
    return strcmp(identity, decision->decision_identity) == 0;
}

static int binding_v14_summary_authentic(
    const yvex_runtime_binding_physical_summary_v14 *summary,
    const yvex_runtime_binding_physical_decision_v14 *decisions,
    unsigned long long count)
{
    yvex_runtime_binding_physical_summary_v14 actual = {0};
    yvex_sha256 hash;
    char identity[YVEX_SHA256_HEX_CAP];
    unsigned long long index;
    if (!summary || !decisions || !count || summary->schema_version != 4u ||
        summary->decision_count != count ||
        !yvex_sha256_hex_is_valid(summary->physical_variant_identity) ||
        !yvex_sha256_hex_is_valid(summary->identity))
        return 0;
    for (index = 0ull; index < count; ++index) {
        const yvex_runtime_binding_physical_decision_v14 *decision = &decisions[index];
        if (!binding_v14_decision_authentic(decision) ||
            !yvex_core_u64_add(actual.encoded_bytes, decision->encoded_bytes,
                               &actual.encoded_bytes))
            return 0;
        actual.consumer_counts[decision->consumer]++;
        actual.layout_counts[decision->layout]++;
        actual.placement_counts[decision->placement]++;
    }
    if (actual.encoded_bytes != summary->encoded_bytes ||
        memcmp(actual.consumer_counts, summary->consumer_counts,
               sizeof(actual.consumer_counts)) != 0 ||
        memcmp(actual.layout_counts, summary->layout_counts,
               sizeof(actual.layout_counts)) != 0 ||
        memcmp(actual.placement_counts, summary->placement_counts,
               sizeof(actual.placement_counts)) != 0)
        return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.physical-execution.ir.v1") ||
        !yvex_sha256_update_u64(&hash, summary->schema_version) ||
        !yvex_sha256_update_text(&hash, summary->physical_variant_identity) ||
        !yvex_sha256_update_u64(&hash, summary->decision_count) ||
        !yvex_sha256_update_u64(&hash, summary->encoded_bytes))
        return 0;
    for (index = 0ull; index < count; ++index)
        if (!yvex_sha256_update_text(&hash, decisions[index].decision_identity)) return 0;
    return binding_v14_hash_finish(&hash, identity) &&
           strcmp(identity, summary->identity) == 0;
}

int yvex_runtime_private_binding_physical_v14_import(
    yvex_physical_execution_ir **out,
    const yvex_runtime_binding_physical_summary_v14 *summary,
    const yvex_runtime_binding_physical_decision_v14 *decisions,
    unsigned long long count, yvex_error *err)
{
    yvex_physical_execution_summary normalized = {0};
    yvex_physical_execution_decision *current = NULL;
    unsigned long long index;
    int rc = YVEX_ERR_FORMAT;
    if (out) *out = NULL;
    if (!out || !binding_v14_summary_authentic(summary, decisions, count)) return rc;
    current = calloc((size_t)count, sizeof(*current));
    if (!current) return YVEX_ERR_NOMEM;
    for (index = 0ull; index < count; ++index) {
        const yvex_runtime_binding_physical_decision_v14 *source = &decisions[index];
        yvex_physical_execution_decision *target = &current[index];
        if (source->derived_asset_required || source->layout == 3u)
            goto done;
        target->schema_version = YVEX_PHYSICAL_EXECUTION_SCHEMA_V5;
        target->terminal_tensor_id = source->terminal_tensor_id;
        target->role = (yvex_tensor_role)source->role;
        target->scope = (yvex_tensor_scope)source->scope;
        target->layer_index = source->layer_index;
        target->predictor_index = source->predictor_index;
        target->expert_count = source->expert_count;
        target->canonical_qtype = source->canonical_qtype;
        target->canonical_row_width = source->canonical_row_width;
        target->canonical_row_count = source->canonical_row_count;
        target->encoded_offset = source->encoded_offset;
        target->encoded_bytes = source->encoded_bytes;
        target->alignment = source->alignment;
        target->consumer = (yvex_execution_consumer_class)source->consumer;
        target->layout = (yvex_execution_layout_class)source->layout;
        target->sharing = (yvex_execution_sharing_class)source->sharing;
        yvex_core_text_copy(target->terminal_identity,
                            sizeof(target->terminal_identity),
                            source->terminal_identity);
    }
    normalized.schema_version = YVEX_PHYSICAL_EXECUTION_SCHEMA_V5;
    normalized.decision_count = count;
    yvex_core_text_copy(normalized.physical_variant_identity,
                        sizeof(normalized.physical_variant_identity),
                        summary->physical_variant_identity);
    rc = yvex_physical_execution_ir_import(out, &normalized, current, count, err);
done:
    free(current);
    return rc;
}

static int binding_boolean(int value)
{
    return value == 0 || value == 1;
}

static int binding_logits_policy_valid(const yvex_logits_family_policy *logits)
{
    return logits && logits->schema_version == YVEX_RUNTIME_LOGITS_SCHEMA_V1 &&
           binding_boolean(logits->separate_output_head) &&
           binding_boolean(logits->tied_output_head) &&
           binding_boolean(logits->output_head_bias) &&
           logits->separate_output_head != logits->tied_output_head;
}

static int binding_policy_not_applicable(
    const yvex_transformer_family_policy *transformer,
    const yvex_speculation_family_policy *speculation)
{
    unsigned long long index;

    if (!transformer || !speculation || transformer->schema_version ||
        transformer->initial_policy || transformer->final_policy ||
        transformer->residual_streams || transformer->hidden_width ||
        transformer->expanded_width || transformer->maximum_context ||
        transformer->sinkhorn_iterations || transformer->mhc_epsilon != 0.0 ||
        transformer->output_norm_epsilon != 0.0 ||
        transformer->attention_then_moe || transformer->deferred_ffn_post ||
        transformer->final_norm_after_head || speculation->schema_version ||
        speculation->block_size || speculation->noise_token_id ||
        speculation->target_feature_layer_count || speculation->target_feature_width ||
        speculation->concatenated_feature_width || speculation->draft_layer_count ||
        speculation->markov_rank || speculation->accepted_prefix_maximum ||
        speculation->feature_projection_role || speculation->feature_norm_role ||
        speculation->output_norm_role || speculation->markov_embedding_role ||
        speculation->markov_output_role || speculation->confidence_role ||
        speculation->parallel_block_backbone || speculation->sequential_markov ||
        speculation->confidence_available || speculation->shares_embedding ||
        speculation->shares_output_head ||
        speculation->target_verification_required || speculation->policy_identity[0])
        return 0;
    for (index = 0ull; index < YVEX_SPECULATION_MAX_FEATURE_LAYERS; ++index)
        if (speculation->target_feature_layers[index]) return 0;
    return 1;
}

int yvex_runtime_private_binding_policies_match_model(
    const yvex_model_execution_descriptor *model,
    const yvex_transformer_family_policy *transformer,
    const yvex_logits_family_policy *logits,
    const yvex_speculation_family_policy *speculation)
{
    unsigned long long expanded, features;
    if (!model || !transformer || !logits || !speculation ||
        !binding_logits_policy_valid(logits)) return 0;
    if (model->schema_version == YVEX_MODEL_EXECUTION_DESCRIPTOR_SCHEMA_V2)
        return binding_policy_not_applicable(
                   transformer, speculation);
    return model->schema_version == YVEX_MODEL_EXECUTION_DESCRIPTOR_SCHEMA_V1 &&
           transformer->schema_version == YVEX_TRANSFORMER_PLAN_SCHEMA_V2 &&
           (unsigned int)transformer->initial_policy < YVEX_TRANSFORMER_INITIAL_POLICY_COUNT &&
           (unsigned int)transformer->final_policy < YVEX_TRANSFORMER_FINAL_POLICY_COUNT &&
           transformer->residual_streams == model->residual_streams &&
           transformer->hidden_width == model->hidden_width &&
           yvex_core_u64_mul(model->residual_streams, model->hidden_width, &expanded) &&
           transformer->expanded_width == expanded &&
           transformer->maximum_context == model->maximum_context &&
           transformer->sinkhorn_iterations == model->mhc_sinkhorn_iterations &&
           transformer->mhc_epsilon == model->mhc_epsilon &&
           transformer->output_norm_epsilon == model->normalization_epsilon &&
           binding_boolean(transformer->attention_then_moe) &&
           binding_boolean(transformer->deferred_ffn_post) &&
           binding_boolean(transformer->final_norm_after_head) &&
           speculation->schema_version == YVEX_SPECULATION_FAMILY_POLICY_SCHEMA_V1 &&
           speculation->block_size == model->proposal_width &&
           speculation->noise_token_id == model->draft_noise_token_id &&
           speculation->target_feature_layer_count == model->target_feature_count &&
           memcmp(speculation->target_feature_layers, model->target_feature_layers,
                  sizeof(model->target_feature_layers)) == 0 &&
           speculation->target_feature_width == model->target_feature_width &&
           yvex_core_u64_mul(model->target_feature_count, model->target_feature_width,
                             &features) &&
           speculation->concatenated_feature_width == features &&
           speculation->draft_layer_count == model->draft_layer_count &&
           speculation->markov_rank == model->markov_rank &&
           speculation->accepted_prefix_maximum <= model->proposal_width &&
           (unsigned int)speculation->feature_projection_role < YVEX_TENSOR_ROLE_COUNT &&
           (unsigned int)speculation->feature_norm_role < YVEX_TENSOR_ROLE_COUNT &&
           (unsigned int)speculation->output_norm_role < YVEX_TENSOR_ROLE_COUNT &&
           (unsigned int)speculation->markov_embedding_role < YVEX_TENSOR_ROLE_COUNT &&
           (unsigned int)speculation->markov_output_role < YVEX_TENSOR_ROLE_COUNT &&
           (unsigned int)speculation->confidence_role < YVEX_TENSOR_ROLE_COUNT &&
           binding_boolean(speculation->parallel_block_backbone) &&
           binding_boolean(speculation->sequential_markov) &&
           binding_boolean(speculation->confidence_available) &&
           binding_boolean(speculation->shares_embedding) &&
           binding_boolean(speculation->shares_output_head) &&
           binding_boolean(speculation->target_verification_required) &&
           yvex_sha256_hex_is_valid(speculation->policy_identity);
}

static int binding_policies_valid(const yvex_runtime_binding *binding)
{
    return binding && yvex_runtime_private_binding_policies_match_model(
        &binding->descriptor.model_execution, &binding->transformer_policy,
        &binding->logits_policy, &binding->speculation_policy) &&
           binding->tokenizer_policy.family_adapter_id == binding->summary.family_adapter_id &&
           binding->tokenizer_policy.family_adapter_version ==
               binding->summary.family_adapter_version &&
           binding->tokenizer_policy.vocabulary_size ==
               binding->descriptor.model_execution.vocabulary_size &&
           yvex_tokenizer_family_policy_validate(&binding->tokenizer_policy, NULL) == YVEX_OK;
}

int yvex_runtime_private_binding_refuse(
    yvex_runtime_binding_failure *failure, yvex_runtime_binding_failure_code code,
    const char *field, const char *path, unsigned long long record,
    unsigned long long expected, unsigned long long actual, yvex_status status,
    const char *reason, yvex_error *err)
{
    if (failure) {
        memset(failure, 0, sizeof(*failure));
        failure->code = code;
        failure->record_index = record;
        failure->expected = expected;
        failure->actual = actual;
        failure->reason = reason;
        if (field)
            yvex_core_text_copy(failure->field, sizeof(failure->field), field);
        if (path)
            yvex_core_text_copy(failure->path, sizeof(failure->path), path);
    }
    yvex_error_set(err, status, "runtime.binding", reason);
    return status;
}

int yvex_runtime_binding_policies(
    const yvex_runtime_binding *binding,
    const yvex_transformer_family_policy **transformer,
    const yvex_logits_family_policy **logits,
    const yvex_speculation_family_policy **speculation)
{
    if (transformer) *transformer = NULL;
    if (logits) *logits = NULL;
    if (speculation) *speculation = NULL;
    if (!binding || (!transformer && !logits && !speculation)) return 0;
    if (transformer) *transformer = &binding->transformer_policy;
    if (logits) *logits = &binding->logits_policy;
    if (speculation) *speculation = &binding->speculation_policy;
    return 1;
}

const yvex_tokenizer_family_policy *yvex_runtime_binding_tokenizer_policy(
    const yvex_runtime_binding *binding)
{
    return binding ? &binding->tokenizer_policy : NULL;
}

void yvex_runtime_binding_close(yvex_runtime_binding *binding)
{
    if (!binding) return;
    yvex_compiled_model_plan_close(&binding->plan);
    yvex_physical_execution_ir_close(&binding->physical_execution);
    free(binding->materialized);
    free(binding->runtime);
    free(binding->layers);
    free(binding->draft_layers);
    free(binding);
}

static int compiled_plan_valid(
    const yvex_runtime_binding *binding)
{
    yvex_compiled_model_plan_admission admission;
    if (!binding) return 0;
    memset(&admission, 0, sizeof(admission));
    admission.family_adapter_id = binding->summary.family_adapter_id;
    admission.family_adapter_version = binding->summary.family_adapter_version;
    admission.tensor_count = binding->summary.tensor_count;
    admission.layer_count = binding->summary.layer_count;
    admission.draft_layer_count = binding->summary.draft_layer_count;
    admission.decoder_layer_count = binding->summary.decoder_layer_count;
    admission.recurrent_layer_count = binding->summary.recurrent_layer_count;
    admission.model_execution_identity =
        binding->descriptor.model_execution.identity;
    admission.semantic_maximum_context =
        binding->descriptor.model_execution.maximum_context;
    admission.capabilities = &binding->summary.capabilities;
    admission.artifact_identity = binding->admission.artifact_identity;
    admission.materialization_identity = binding->materialization.plan_identity;
    admission.runtime_descriptor_identity =
        binding->descriptor.runtime_descriptor_identity;
    admission.attention_plan_identity = binding->attention.attention_plan_identity;
    admission.draft_attention_plan_identity =
        binding->draft_attention.attention_plan_identity;
    admission.moe_plan_identity = binding->summary.moe_plan_identity;
    admission.draft_moe_plan_identity = binding->summary.draft_moe_plan_identity;
    admission.transformer_plan_identity =
        binding->summary.transformer_plan_identity;
    admission.draft_transformer_plan_identity =
        binding->summary.draft_transformer_plan_identity;
    admission.decoder_plan_identity = binding->summary.decoder_plan_identity;
    admission.output_head_plan_identity =
        binding->summary.output_head_plan_identity;
    return yvex_compiled_model_plan_admit(binding->plan, &admission);
}

int yvex_runtime_private_binding_admission_ready(
    const yvex_complete_artifact_admission *admission)
{
    return admission && admission->complete &&
           admission->materialization_input_ready && !admission->runtime_supported &&
           admission->artifact_identity_verified &&
           admission->file_snapshot.size == admission->file_bytes;
}

int yvex_runtime_private_binding_attention_ready(
    const yvex_attention_summary *attention)
{
    return attention && attention->history_contract_ready &&
           attention->state_delta_contract_ready &&
           attention->cpu_reference_ready && attention->cuda_execution_ready &&
           attention->full_execution_ready;
}

int yvex_runtime_private_binding_identity_chain_valid(
    const yvex_complete_artifact_admission *admission,
    const yvex_materialization_summary *materialization,
    const yvex_runtime_descriptor_summary *descriptor,
    const yvex_attention_summary *attention)
{
    return admission && materialization && descriptor && attention &&
           strcmp(admission->artifact_identity, materialization->artifact_identity) == 0 &&
           strcmp(materialization->artifact_identity, descriptor->artifact_identity) == 0 &&
           strcmp(materialization->plan_identity,
                  descriptor->materialization_plan_identity) == 0 &&
           strcmp(descriptor->artifact_identity, attention->artifact_identity) == 0 &&
           strcmp(descriptor->materialization_plan_identity,
                  attention->materialization_plan_identity) == 0 &&
           strcmp(descriptor->logical_model_identity,
                  attention->logical_model_identity) == 0 &&
           strcmp(descriptor->runtime_descriptor_identity,
                  attention->runtime_descriptor_identity) == 0 &&
           strcmp(descriptor->runtime_numeric_identity,
                  attention->runtime_numeric_identity) == 0;
}

int yvex_runtime_private_binding_decoder_matches(
    const yvex_decoder_plan_summary *decoder,
    const yvex_runtime_descriptor_summary *descriptor,
    const char *operator_graph_identity,
    const yvex_attention_summary *attention)
{
    const yvex_model_execution_descriptor *execution =
        descriptor ? &descriptor->model_execution : NULL;

    return decoder && descriptor && operator_graph_identity && attention && execution &&
           execution->schema_version == YVEX_MODEL_EXECUTION_DESCRIPTOR_SCHEMA_V2 &&
           decoder->layer_count == execution->layer_count &&
           decoder->attention_layer_count == attention->layer_count &&
           decoder->recurrent_layer_count == execution->sequence_mixer_layers &&
           decoder->maximum_context == execution->maximum_context &&
           strcmp(decoder->model_execution_identity, execution->identity) == 0 &&
           strcmp(decoder->operator_graph_identity, operator_graph_identity) == 0;
}

static int binding_attention_layers_valid(
    const yvex_runtime_binding *binding, const yvex_decoder_plan *decoder)
{
    unsigned long long attention;

    for (attention = 0ull; attention < binding->summary.layer_count; ++attention) {
        const yvex_attention_layer_plan *actual = &binding->layers[attention];
        unsigned long long expected_layer = attention;

        if (decoder) {
            const yvex_decoder_plan_summary *summary =
                yvex_decoder_plan_summary_get(decoder);
            unsigned long long layer;
            int found = 0;

            if (!summary) return 0;
            for (layer = 0ull; layer < summary->layer_count; ++layer) {
                const yvex_decoder_layer_plan *candidate =
                    yvex_decoder_plan_layer_at(decoder, layer);

                if (!candidate || candidate->mixer !=
                        YVEX_SEMANTIC_DECODER_MIXER_FULL_CAUSAL_ATTENTION ||
                    candidate->attention_ordinal != attention)
                    continue;
                if (found) return 0;
                expected_layer = candidate->layer_index;
                found = 1;
            }
            if (!found) return 0;
        }
        if (actual->ordinal != attention || actual->layer_index != expected_layer ||
            actual->tensor_scope != YVEX_TENSOR_SCOPE_MAIN_LAYER ||
            actual->predictor_index != YVEX_MATERIALIZATION_NO_INDEX)
            return 0;
    }
    return 1;
}

int yvex_runtime_private_binding_validate(
    const yvex_runtime_binding *binding, const char **field,
    yvex_runtime_binding_failure_code *code)
{
    char capability_identity[YVEX_SHA256_HEX_CAP];
    char semantic[YVEX_SHA256_HEX_CAP], executable[YVEX_SHA256_HEX_CAP];
    const yvex_physical_execution_summary *physical;
    const yvex_decoder_plan *decoder_plan;
    const yvex_decoder_plan_summary *decoder;
    const char *compatibility;
    unsigned long long index;

    if (!field || !code) return 0;
    *field = "canonical-body";
    *code = YVEX_RUNTIME_BINDING_FAILURE_FORMAT;
    if (!binding) return 0;
    decoder_plan = yvex_compiled_model_plan_decoder(binding->plan);
    decoder = yvex_decoder_plan_summary_get(decoder_plan);
    physical = yvex_physical_execution_ir_summary(binding->physical_execution);
    if (!physical || physical->decision_count != binding->summary.tensor_count ||
        strcmp(physical->physical_variant_identity,
               binding->admission.profile_identity) != 0) {
        *field = "physical-execution";
        *code = YVEX_RUNTIME_BINDING_FAILURE_IDENTITY;
        return 0;
    }
    for (index = 0ull; index < physical->decision_count; ++index) {
        const yvex_physical_execution_decision *decision =
            yvex_physical_execution_ir_decision_at(
                binding->physical_execution, index);

        if (!decision || decision->terminal_tensor_id >= binding->summary.tensor_count ||
            decision->canonical_qtype !=
                binding->runtime[decision->terminal_tensor_id].qtype) {
            *field = "physical-execution-package";
            *code = YVEX_RUNTIME_BINDING_FAILURE_IDENTITY;
            return 0;
        }
    }
    if (!yvex_runtime_capabilities_identity(&binding->summary.capabilities,
                                            capability_identity) ||
        strcmp(capability_identity,
               binding->summary.execution_capability_identity) != 0 ||
        !yvex_runtime_capabilities_contract_valid(&binding->summary.capabilities)) {
        *field = "execution-capabilities";
        return 0;
    }
    compatibility = yvex_artifact_physical_compatibility_mismatch(
        &binding->summary.physical_compatibility, &binding->admission,
        binding->summary.logical_transform_identity);
    if (compatibility) {
        *field = compatibility;
        *code = YVEX_RUNTIME_BINDING_FAILURE_COMPATIBILITY;
        return 0;
    }
    if (binding->summary.tensor_count != binding->admission.tensor_count ||
        binding->summary.tensor_count != binding->materialization.tensor_count ||
        binding->summary.tensor_count != binding->descriptor.tensor_count ||
        binding->summary.layer_count != binding->attention.layer_count ||
        binding->summary.draft_layer_count != binding->descriptor.draft_layer_count ||
        binding->summary.draft_layer_count != binding->draft_attention.layer_count ||
        (decoder &&
         (binding->summary.decoder_layer_count != decoder->layer_count ||
          binding->summary.recurrent_layer_count != decoder->recurrent_layer_count ||
          strcmp(binding->summary.decoder_plan_identity,
                 decoder->decoder_plan_identity) != 0 ||
          !yvex_runtime_private_binding_decoder_matches(
              decoder, &binding->descriptor,
              yvex_compiled_model_plan_operator_graph_identity(binding->plan),
              &binding->attention) || binding->summary.draft_layer_count)) ||
        (!decoder && (binding->summary.decoder_layer_count ||
                      binding->summary.recurrent_layer_count ||
                      binding->summary.decoder_plan_identity[0])) ||
        !binding->summary.tensor_count || !binding->summary.layer_count ||
        !yvex_runtime_private_binding_admission_ready(&binding->admission) ||
        !yvex_sha256_hex_is_valid(binding->admission.transform_identity) ||
        !yvex_sha256_hex_is_valid(binding->summary.logical_transform_identity) ||
        binding->materialization.committed || binding->materialization.cleanup_complete ||
        binding->materialization.status != YVEX_MATERIALIZATION_STATUS_PLANNED ||
        binding->materialization.access_calls ||
        binding->materialization.payload_bytes_accessed ||
        binding->materialization.full_walks ||
        binding->materialization.snapshot_drift_count ||
        binding->materialization.committed_bindings ||
        binding->materialization.aborted_bindings ||
        binding->descriptor.status != YVEX_RUNTIME_DESCRIPTOR_STATUS_READY ||
        (!decoder && binding->descriptor.model_execution.schema_version !=
                         YVEX_MODEL_EXECUTION_DESCRIPTOR_SCHEMA_V1) ||
        !yvex_sha256_hex_is_valid(binding->descriptor.model_execution.identity) ||
        strcmp(binding->descriptor.logical_model_identity,
               binding->descriptor.model_execution.logical_model_identity) != 0 ||
        binding->descriptor.layer_count != binding->descriptor.model_execution.layer_count ||
        binding->descriptor.draft_layer_count !=
            binding->descriptor.model_execution.draft_layer_count ||
        binding->descriptor.vocabulary_size !=
            binding->descriptor.model_execution.vocabulary_size ||
        !binding_policies_valid(binding) ||
        !compiled_plan_valid(binding) ||
        !yvex_runtime_private_binding_attention_ready(&binding->attention) ||
        binding->attention.tensor_scope != YVEX_TENSOR_SCOPE_MAIN_LAYER ||
        !binding->attention.required_binding_count ||
        binding->attention.missing_binding_count ||
        binding->attention.qtype_compute_refusal_count ||
        !yvex_runtime_private_binding_identity_chain_valid(
            &binding->admission, &binding->materialization,
            &binding->descriptor, &binding->attention) ||
        (binding->summary.draft_layer_count &&
         (!yvex_runtime_private_binding_attention_ready(&binding->draft_attention) ||
          binding->draft_attention.tensor_scope != YVEX_TENSOR_SCOPE_DRAFT ||
          !yvex_runtime_private_binding_identity_chain_valid(
              &binding->admission, &binding->materialization,
              &binding->descriptor, &binding->draft_attention))))
        return 0;
    if (!yvex_compiled_graph_identities(
            yvex_compiled_model_plan_operator_graph_identity(binding->plan),
            &binding->materialization, &binding->descriptor, &binding->attention,
            binding->summary.draft_layer_count ? &binding->draft_attention : NULL,
            semantic, executable)) {
        *field = "graph-identity-inputs";
        *code = YVEX_RUNTIME_BINDING_FAILURE_IDENTITY;
        return 0;
    }
    if (strcmp(binding->summary.semantic_graph_identity, semantic) != 0 ||
        strcmp(binding->summary.executable_graph_identity, executable) != 0) {
        *field = strcmp(binding->summary.semantic_graph_identity, semantic) != 0
                     ? "semantic-graph-identity" : "executable-graph-identity";
        *code = YVEX_RUNTIME_BINDING_FAILURE_IDENTITY;
        return 0;
    }
    for (index = 0ull; index < binding->summary.tensor_count; ++index)
        if (binding->materialized[index].tensor_id != index ||
            binding->runtime[index].tensor_id >= binding->summary.tensor_count)
            return 0;
    if (!binding_attention_layers_valid(binding, decoder_plan)) return 0;
    for (index = 0ull; index < binding->summary.draft_layer_count; ++index)
        if (binding->draft_layers[index].ordinal != index ||
            binding->draft_layers[index].tensor_scope != YVEX_TENSOR_SCOPE_DRAFT ||
            binding->draft_layers[index].predictor_index != index)
            return 0;
    return 1;
}

int yvex_runtime_binding_import_materialization(
    const yvex_runtime_binding *binding, const yvex_artifact *artifact,
    const yvex_materialization_options *options, yvex_materialization_plan **plan_out,
    yvex_materialization_session **session_out, yvex_runtime_binding_failure *failure,
    yvex_error *err)
{
    yvex_complete_artifact_admission admission;
    yvex_artifact_snapshot snapshot = {0};
    yvex_materialization_failure material_failure = {0};
    int rc;
    if (plan_out) *plan_out = NULL;
    if (session_out) *session_out = NULL;
    if (!binding || !artifact || !plan_out || !session_out)
        return yvex_runtime_private_binding_refuse(
            failure, YVEX_RUNTIME_BINDING_FAILURE_INVALID_ARGUMENT,
            "materialization-import", NULL, 0ull, 1ull, 0ull,
            YVEX_ERR_INVALID_ARG,
            "runtime binding materialization import arguments are required", err);
    if (yvex_artifact_snapshot_get(artifact, &snapshot, err) != YVEX_OK ||
        yvex_artifact_snapshot_validate(artifact, NULL, err) != YVEX_OK ||
        snapshot.size != binding->admission.file_bytes)
        return yvex_runtime_private_binding_refuse(
            failure, YVEX_RUNTIME_BINDING_FAILURE_ARTIFACT, "artifact-snapshot", NULL,
            0ull, binding->admission.file_bytes, snapshot.size, YVEX_ERR_STATE,
            "runtime binding artifact snapshot is stale or mismatched", err);
    admission = binding->admission;
    admission.file_snapshot = snapshot;
    yvex_core_text_copy(admission.artifact_path, sizeof(admission.artifact_path),
                        yvex_artifact_path(artifact));
    rc = yvex_materialization_plan_import(
        plan_out, &admission, &binding->materialization, binding->materialized,
        binding->summary.tensor_count, &material_failure, err);
    if (rc == YVEX_OK)
        rc = yvex_materialization_session_open(session_out, *plan_out, artifact, options,
                                               &material_failure, err);
    if (rc == YVEX_OK)
        rc = yvex_materialization_session_commit(*session_out, &material_failure, err);
    if (rc != YVEX_OK) {
        yvex_materialization_session_close(*session_out);
        yvex_materialization_plan_close(*plan_out);
        *session_out = NULL;
        *plan_out = NULL;
        return yvex_runtime_private_binding_refuse(
            failure, YVEX_RUNTIME_BINDING_FAILURE_MATERIALIZATION,
            yvex_materialization_failure_name(material_failure.code), NULL,
            material_failure.tensor_index, material_failure.expected,
            material_failure.actual, (yvex_status)rc,
            "runtime binding materialization import was refused", err);
    }
    if (failure) memset(failure, 0, sizeof(*failure));
    return YVEX_OK;
}

int yvex_runtime_binding_import_graph(
    const yvex_runtime_binding *binding, const yvex_materialization_session *session,
    yvex_runtime_descriptor **descriptor_out, yvex_attention_plan **attention_out,
    yvex_attention_plan **draft_attention_out,
    const yvex_physical_execution_ir **physical_execution_out,
    yvex_runtime_binding_failure *failure, yvex_error *err)
{
    yvex_runtime_descriptor_failure descriptor_failure = {0};
    yvex_attention_failure attention_failure = {0};
    yvex_runtime_descriptor *descriptor = NULL;
    yvex_attention_plan *attention = NULL, *draft_attention = NULL;
    int rc;
    if (descriptor_out) *descriptor_out = NULL;
    if (attention_out) *attention_out = NULL;
    if (draft_attention_out) *draft_attention_out = NULL;
    if (physical_execution_out) *physical_execution_out = NULL;
    if (!binding || !session || !descriptor_out || !attention_out ||
        !draft_attention_out || !physical_execution_out)
        return yvex_runtime_private_binding_refuse(
            failure, YVEX_RUNTIME_BINDING_FAILURE_INVALID_ARGUMENT,
            "runtime-graph-import", NULL, 0ull, 1ull, 0ull,
            YVEX_ERR_INVALID_ARG,
            "runtime binding graph import arguments are required", err);
    rc = yvex_runtime_descriptor_import(
        &descriptor, &binding->descriptor, binding->runtime,
        binding->summary.tensor_count, session, &descriptor_failure, err);
    if (rc != YVEX_OK)
        return yvex_runtime_private_binding_refuse(
            failure, YVEX_RUNTIME_BINDING_FAILURE_DESCRIPTOR,
            yvex_runtime_descriptor_failure_name(descriptor_failure.code), NULL,
            descriptor_failure.tensor_index, descriptor_failure.expected,
            descriptor_failure.actual, (yvex_status)rc,
            "runtime binding descriptor import was refused", err);
    rc = yvex_attention_plan_import(
        &attention, &binding->attention, binding->layers,
        binding->summary.layer_count, session, descriptor, &attention_failure, err);
    if (rc == YVEX_OK && binding->summary.draft_layer_count)
        rc = yvex_attention_plan_import(
            &draft_attention, &binding->draft_attention, binding->draft_layers,
            binding->summary.draft_layer_count, session, descriptor,
            &attention_failure, err);
    if (rc != YVEX_OK) {
        yvex_attention_plan_close(draft_attention);
        yvex_attention_plan_close(attention);
        yvex_runtime_descriptor_close(descriptor);
        return yvex_runtime_private_binding_refuse(
            failure, YVEX_RUNTIME_BINDING_FAILURE_ATTENTION,
            attention_failure.reason ? attention_failure.reason : "attention", NULL,
            attention_failure.layer_index, attention_failure.expected,
            attention_failure.actual, (yvex_status)rc,
            "runtime binding attention import was refused", err);
    }
    *descriptor_out = descriptor;
    *attention_out = attention;
    *draft_attention_out = draft_attention;
    *physical_execution_out = binding->physical_execution;
    if (failure) memset(failure, 0, sizeof(*failure));
    return YVEX_OK;
}
