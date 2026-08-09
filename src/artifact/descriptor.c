/*
 * Project complete artifact admission into immutable descriptor facts.
 *
 * An executable descriptor requires complete canonical admission evidence. Descriptor facts do not
 * bind memory or execute graphs.
 */
#include <yvex/internal/artifact.h>
#include <yvex/internal/core.h>
#include <yvex/internal/families/deepseek_v4.h>

#include <stdio.h>
#include <string.h>

static void refuse_missing_gguf(yvex_artifact_descriptor_fact *fact) {
    if (!fact)
        return;
    memset(fact, 0, sizeof(*fact));
    fact->status = YVEX_ARTIFACT_DESCRIPTOR_REFUSED;
    fact->format = "gguf";
    fact->reason = "artifact descriptor support requires complete-artifact admission";
    fact->next_row = "V010.ARTIFACT.MATERIALIZE.0";
}

int yvex_artifact_descriptor_from_admission(const yvex_complete_artifact_admission *admission,
                                            yvex_artifact_descriptor_fact *fact) {
    if (!fact)
        return 0;
    memset(fact, 0, sizeof(*fact));
    if (!admission || !admission->complete ||
        admission->artifact_class != YVEX_ARTIFACT_CLASS_COMPLETE_YVEX ||
        !admission->materialization_input_ready || !admission->artifact_identity[0]) {
        refuse_missing_gguf(fact);
        return 0;
    }
    fact->status = YVEX_ARTIFACT_DESCRIPTOR_COMPLETE_ADMITTED;
    fact->format = "gguf";
    fact->reason = "complete YVEX artifact passed canonical admission";
    fact->next_row = "V010.ARTIFACT.MATERIALIZE.0";
    fact->artifact_identity = admission->artifact_identity;
    fact->tensor_count = admission->tensor_count;
    fact->materialization_input_ready = 1;
    fact->runtime_supported = 0;
    return 1;
}

static int runtime_descriptor_refuse(yvex_runtime_descriptor_failure *failure,
                                     yvex_runtime_descriptor_failure_code code,
                                     const char *name, unsigned long long index,
                                     unsigned long long expected, unsigned long long actual,
                                     const char *reason, yvex_error *err, yvex_status status)
{
    if (failure) {
        memset(failure, 0, sizeof(*failure));
        failure->code = code;
        failure->tensor_index = index;
        failure->expected = expected;
        failure->actual = actual;
        failure->reason = reason;
        if (name) yvex_core_text_copy(failure->tensor_name, sizeof(failure->tensor_name), name);
    }
    yvex_error_set(err, status, "yvex_runtime_descriptor_build_deepseek", reason);
    return status;
}

static void runtime_descriptor_hash_activation(
    yvex_sha256 *hash, const yvex_attention_activation_policy *policy)
{
#define HASH(MEMBER) yvex_sha256_update_u64(hash, (unsigned long long)policy->MEMBER)
    HASH(required); HASH(stage); HASH(quantization); HASH(block_axis); HASH(block_width);
    HASH(scale_format); HASH(scale_dtype); HASH(pre_transform); HASH(tail_policy);
    HASH(nonfinite_policy); HASH(fake_quant_inplace); HASH(zero_pad_hadamard_to_power_of_two);
#undef HASH
}

static void runtime_descriptor_hash_topk(yvex_sha256 *hash,
                                         const yvex_attention_topk_policy *policy)
{
#define HASH(MEMBER) yvex_sha256_update_u64(hash, (unsigned long long)policy->MEMBER)
    HASH(required); HASH(version); HASH(policy); HASH(k); HASH(reject_nonfinite);
    HASH(score_descending); HASH(equal_score_ordinal_ascending); HASH(plus_zero_equals_minus_zero);
    HASH(duplicate_ordinal_refused); HASH(output_ranked_order);
#undef HASH
}

/*
 * Derive complete family facts for the family-neutral descriptor builder.
 *
 * Sealed DeepSeek architecture and caller-owned fact/identity storage. Fills deterministic values
 * and hashes no pointers or local paths.
 */
static int runtime_descriptor_family_facts(
    const yvex_deepseek_v4_ir *ir, yvex_runtime_descriptor_family_facts *facts,
    yvex_model_execution_descriptor *model_execution,
    char logical[YVEX_SHA256_HEX_CAP], char numeric[YVEX_SHA256_HEX_CAP],
    yvex_runtime_descriptor_failure *failure, yvex_error *err)
{
    const yvex_model_family_api *api = yvex_model_register_deepseek_v4();
    const yvex_deepseek_v4_model_spec *model = api->ir.model(ir);
    const yvex_deepseek_v4_layer_spec *first = api->ir.layer_at(ir, 0ull);
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_sha256 hash;
    unsigned long long count, index;

    if (!model || !first || model->runtime_numeric_schema_version != 2u ||
        model->runtime_compute_policy_count != 1ull ||
        !model->runtime_activation_policy_count || !model->hadamard_revision[0] ||
        !api->transform.architecture_identity(ir, logical))
        return runtime_descriptor_refuse(
            failure, YVEX_RUNTIME_DESCRIPTOR_FAILURE_ARCHITECTURE, NULL,
            YVEX_MATERIALIZATION_NO_INDEX, 1ull, 0ull,
            "DeepSeek runtime numeric authority is incomplete", err, YVEX_ERR_FORMAT);
    yvex_sha256_init(&hash);
    yvex_sha256_update_text(&hash, "yvex.runtime.numeric.deepseek-v4.v2");
    yvex_sha256_update_text(&hash, model->hadamard_revision);
    yvex_sha256_update_text(&hash, model->sglang_revision);
    yvex_sha256_update_u64(&hash, model->runtime_numeric_schema_version);
    yvex_sha256_update_u64(&hash, model->runtime_compute_policy_count);
    yvex_sha256_update_u64(&hash, model->runtime_activation_policy_count);
    yvex_sha256_update_u64(&hash, model->runtime_sparse_topk_policy_count);
    count = api->ir.layer_count(ir);
    yvex_sha256_update_u64(&hash, count);
    for (index = 0ull; index < count; ++index) {
        const yvex_deepseek_v4_layer_spec *layer = api->ir.layer_at(ir, index);
        if (!layer || layer->moe.routed_experts != first->moe.routed_experts ||
            layer->moe.experts_per_token != first->moe.experts_per_token)
            return runtime_descriptor_refuse(
                failure, YVEX_RUNTIME_DESCRIPTOR_FAILURE_ARCHITECTURE, NULL, index,
                1ull, 0ull, "DeepSeek runtime numeric layer is missing", err,
                YVEX_ERR_FORMAT);
        yvex_sha256_update_u64(&hash, layer->layer_index);
        yvex_sha256_update_u64(&hash, (unsigned long long)layer->attention_class);
        yvex_sha256_update_u64(&hash, (unsigned long long)layer->compute_contract);
        runtime_descriptor_hash_activation(&hash, &layer->attention_kv_activation);
        runtime_descriptor_hash_activation(&hash, &layer->compressor_activation);
        runtime_descriptor_hash_activation(&hash, &layer->compressor_rotated_activation);
        runtime_descriptor_hash_activation(&hash, &layer->indexer_query_activation);
        runtime_descriptor_hash_topk(&hash, &layer->sparse_topk);
    }
    if (!yvex_sha256_final(&hash, digest))
        return runtime_descriptor_refuse(
            failure, YVEX_RUNTIME_DESCRIPTOR_FAILURE_ARCHITECTURE, NULL,
            YVEX_MATERIALIZATION_NO_INDEX, 1ull, 0ull,
            "DeepSeek runtime numeric identity failed", err, YVEX_ERR_STATE);
    yvex_sha256_hex(digest, numeric);
    if (!api->ir.execution_descriptor ||
        api->ir.execution_descriptor(ir, logical, model_execution, err) != YVEX_OK)
        return runtime_descriptor_refuse(
            failure, YVEX_RUNTIME_DESCRIPTOR_FAILURE_ARCHITECTURE, NULL,
            YVEX_MATERIALIZATION_NO_INDEX, 1ull, 0ull,
            "DeepSeek model execution descriptor is incomplete", err, YVEX_ERR_FORMAT);
    *facts = (yvex_runtime_descriptor_family_facts){
        logical, numeric, model->hadamard_revision, model->runtime_numeric_schema_version,
        model->runtime_compute_policy_count, model->runtime_activation_policy_count,
        model->runtime_sparse_topk_policy_count, model->main_layer_count,
        model->auxiliary_layer_count, first->moe.routed_experts,
        first->moe.experts_per_token, model->vocabulary_size, model_execution};
    return YVEX_OK;
}

static const yvex_runtime_tensor_binding *runtime_descriptor_find_name(
    const yvex_runtime_descriptor *descriptor, unsigned long long count, const char *name)
{
    unsigned long long index;
    for (index = 0ull; index < count; ++index) {
        const yvex_runtime_tensor_binding *row =
            yvex_runtime_descriptor_tensor_at(descriptor, index);
        if (row && row->binding && strcmp(row->binding->name, name) == 0) return row;
    }
    return NULL;
}

int yvex_runtime_descriptor_build_deepseek(
    yvex_runtime_descriptor **out, const yvex_complete_artifact_admission *admission,
    const yvex_materialization_session *session, const yvex_deepseek_gguf_map *map,
    const yvex_deepseek_v4_ir *ir, yvex_runtime_descriptor_failure *failure,
    yvex_error *err)
{
    const yvex_model_family_api *api = yvex_model_register_deepseek_v4();
    const yvex_deepseek_gguf_map_summary *map_summary;
    yvex_runtime_descriptor_family_facts facts;
    yvex_model_execution_descriptor model_execution;
    yvex_runtime_descriptor *descriptor = NULL;
    char logical[YVEX_SHA256_HEX_CAP], numeric[YVEX_SHA256_HEX_CAP];
    unsigned long long index;
    int rc;

    if (out) *out = NULL;
    if (!out || !map || !ir)
        return runtime_descriptor_refuse(
            failure, YVEX_RUNTIME_DESCRIPTOR_FAILURE_INVALID_ARGUMENT, NULL,
            YVEX_MATERIALIZATION_NO_INDEX, 1ull, 0ull,
            "DeepSeek runtime descriptor requires map and architecture", err,
            YVEX_ERR_INVALID_ARG);
    map_summary = api->lowering.summary(map);
    if (!map_summary || !map_summary->complete)
        return runtime_descriptor_refuse(
            failure, YVEX_RUNTIME_DESCRIPTOR_FAILURE_ARCHITECTURE, NULL,
            YVEX_MATERIALIZATION_NO_INDEX, 1ull, 0ull,
            "DeepSeek GGUF map is incomplete", err, YVEX_ERR_FORMAT);
    rc = runtime_descriptor_family_facts(
        ir, &facts, &model_execution, logical, numeric, failure, err);
    if (rc != YVEX_OK) return rc;
    rc = yvex_runtime_descriptor_build(&descriptor, admission, session, &facts, failure, err);
    if (rc != YVEX_OK) return rc;
    if (map_summary->descriptor_count !=
        yvex_runtime_descriptor_summary_get(descriptor)->tensor_count) {
        rc = runtime_descriptor_refuse(
            failure, YVEX_RUNTIME_DESCRIPTOR_FAILURE_MATERIALIZATION, NULL,
            YVEX_MATERIALIZATION_NO_INDEX, map_summary->descriptor_count,
            yvex_runtime_descriptor_summary_get(descriptor)->tensor_count,
            "DeepSeek runtime descriptor count differs from map", err, YVEX_ERR_FORMAT);
        goto fail;
    }
    for (index = 0ull; index < map_summary->descriptor_count; ++index) {
        const yvex_deepseek_gguf_descriptor *map_row = api->lowering.at(map, index);
        const yvex_runtime_tensor_binding *row =
            map_row ? runtime_descriptor_find_name(descriptor, map_summary->descriptor_count,
                                                   map_row->emitted_name) : NULL;
        if (!map_row || !row || row->role != map_row->role || row->scope != map_row->scope ||
            row->layer_index != map_row->layer_index ||
            row->predictor_index != map_row->predictor_index) {
            rc = runtime_descriptor_refuse(
                failure, map_row && !row ? YVEX_RUNTIME_DESCRIPTOR_FAILURE_MISSING_BINDING
                                         : YVEX_RUNTIME_DESCRIPTOR_FAILURE_ARCHITECTURE,
                map_row ? map_row->emitted_name : NULL, index, 1ull, 0ull,
                "DeepSeek runtime binding does not match the lowering map", err,
                YVEX_ERR_FORMAT);
            goto fail;
        }
    }
    *out = descriptor;
    yvex_error_clear(err);
    return YVEX_OK;
fail:
    yvex_runtime_descriptor_close(descriptor);
    return rc;
}
static int deepseek_terminal_find(const void *context, const char *emitted_name,
                                  yvex_materialization_terminal *out)
{
    const yvex_model_family_lowering_api *lowering = yvex_model_deepseek_lowering_api();
    const yvex_deepseek_gguf_map *map = (const yvex_deepseek_gguf_map *)context;
    const yvex_deepseek_gguf_descriptor *descriptor;
    const yvex_deepseek_gguf_descriptor *first;

    if (!map || !emitted_name || !out || !lowering) return 0;
    descriptor = lowering->find_emitted(map, emitted_name);
    first = lowering->at(map, 0ull);
    if (!descriptor || !first) return 0;
    memset(out, 0, sizeof(*out));
    out->descriptor_index = (unsigned long long)(descriptor - first);
    out->role = descriptor->role;
    out->collection = descriptor->collection;
    out->scope = descriptor->scope;
    out->layer_index = descriptor->layer_index;
    out->predictor_index = descriptor->predictor_index;
    out->expert_count = descriptor->expert_count;
    return 1;
}

int yvex_deepseek_materialization_projection(const yvex_deepseek_gguf_map *map,
                                             yvex_materialization_projection *out,
                                             yvex_error *err)
{
    const yvex_model_family_lowering_api *lowering = yvex_model_deepseek_lowering_api();
    const yvex_deepseek_gguf_map_summary *summary = map && lowering ? lowering->summary(map) : NULL;

    if (!out || !summary || !summary->complete || !summary->mapping_identity) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "artifact.deepseek.materialization",
                       "complete family lowering is required for terminal projection");
        return YVEX_ERR_INVALID_ARG;
    }
    memset(out, 0, sizeof(*out));
    out->schema_version = YVEX_MATERIALIZATION_PROJECTION_SCHEMA_VERSION;
    out->mapping_identity = summary->mapping_identity;
    out->descriptor_count = summary->descriptor_count;
    out->context = map;
    out->find = deepseek_terminal_find;
    out->complete = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}
