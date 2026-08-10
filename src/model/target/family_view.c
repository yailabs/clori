/*
 * Project family-owned inspection objects into the bounded model-target view model.
 *
 * The report retains its source owner for lifecycle cleanup, while CLI rendering sees only this
 * copied representation. This prevents presentation code from borrowing family object layouts.
 */
#include <yvex/internal/model_target.h>

#include <yvex/internal/compilation.h>
#include <yvex/internal/core.h>
#include <yvex/internal/families/deepseek_v4.h>
#include <yvex/internal/families/minimax_h3.h>

#include <string.h>

static const char *attention_name(yvex_attention_class value)
{
    static const char *const names[] = {"swa", "csa", "hca"};

    return value <= YVEX_ATTENTION_CLASS_HCA ? names[value] : "unknown";
}

static int projection_refuse(yvex_error *err, const char *reason)
{
    yvex_error_set(err, YVEX_ERR_BOUNDS, "model_target_family_projection", reason);
    return YVEX_ERR_BOUNDS;
}

static int project_map(yvex_model_target_report *report, yvex_error *err)
{
    const yvex_model_family_api *family = yvex_model_register_deepseek_v4();
    const yvex_deepseek_gguf_map_summary *source = family->lowering.summary(
        (const yvex_deepseek_gguf_map *)report->family_lowering);
    yvex_model_target_map_projection *out = &report->detail.map;
    unsigned int index;

    if (!source || YVEX_TENSOR_COLLECTION_COUNT > YVEX_MODEL_TARGET_COLLECTION_CAP)
        return projection_refuse(err, "tensor-map projection exceeds its bounded view");
    memset(out, 0, sizeof(*out));
    out->source_contributions = source->source_contribution_count;
    out->descriptors = source->descriptor_count;
    out->trunk_descriptors = source->trunk_descriptor_count;
    out->draft_descriptors = source->draft_descriptor_count;
    out->pinned_standard_names = source->pinned_standard_count;
    out->semantic_standard_names = source->semantic_standard_count;
    out->extension_names = source->extension_count;
    out->metadata = source->metadata_count;
    out->header_scans = source->header_scan_count;
    out->payload_bytes = source->payload_bytes_read;
    out->source_identity = source->source_identity;
    out->coverage_identity = source->coverage_identity;
    out->mapping_identity = source->mapping_identity;
    out->collection_count = YVEX_TENSOR_COLLECTION_COUNT;
    for (index = 0u; index < out->collection_count; ++index) {
        out->collections[index].count = source->collection_counts[index];
    }
    report->detail_kind = YVEX_MODEL_TARGET_DETAIL_TENSOR_MAP;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int project_coverage(yvex_model_target_report *report, yvex_error *err)
{
    const yvex_model_family_api *family = yvex_model_register_deepseek_v4();
    const yvex_deepseek_tensor_coverage_summary *source = family->coverage.summary(
        (const yvex_deepseek_tensor_coverage *)report->family_coverage);
    yvex_model_target_coverage_projection *out = &report->detail.coverage;
    unsigned int index;

    if (!source || YVEX_TENSOR_COLLECTION_COUNT > YVEX_MODEL_TARGET_COLLECTION_CAP)
        return projection_refuse(err, "tensor-coverage projection exceeds its bounded view");
    memset(out, 0, sizeof(*out));
    out->source_tensors = source->source_tensor_count;
    out->required_tensors = source->required_tensor_count;
    out->matched_tensors = source->matched_tensor_count;
    out->missing_tensors = source->missing_count;
    out->ambiguous_tensors = source->ambiguous_count;
    out->unexpected_tensors = source->unexpected_count;
    out->header_scans = source->header_scan_count;
    out->payload_bytes = source->payload_bytes_read;
    out->source_lookups = source->source_lookup_count;
    out->source_collisions = source->source_collision_count;
    out->source_maximum_probe = source->source_maximum_probe;
    out->source_identity = source->source_identity;
    out->coverage_identity = source->coverage_identity;
    out->collection_count = YVEX_TENSOR_COLLECTION_COUNT;
    for (index = 0u; index < out->collection_count; ++index) {
        yvex_core_text_copy(out->collections[index].name,
                            sizeof(out->collections[index].name),
                            family->coverage.collection_name((yvex_tensor_collection)index));
        out->collections[index].count = source->collection_counts[index];
    }
    report->detail_kind = YVEX_MODEL_TARGET_DETAIL_TENSOR_COVERAGE;
    yvex_error_clear(err);
    return YVEX_OK;
}

static void project_architecture_model(
    yvex_model_target_architecture_projection *out,
    const yvex_deepseek_v4_model_spec *source)
{
#define COPY_TEXT(member) \
    yvex_core_text_copy(out->member, sizeof(out->member), source->member)
    COPY_TEXT(target_id);
    COPY_TEXT(family);
    COPY_TEXT(architecture);
    COPY_TEXT(repository);
    COPY_TEXT(revision);
    COPY_TEXT(verification_stage);
    COPY_TEXT(paper_revision);
    COPY_TEXT(sglang_revision);
    COPY_TEXT(vllm_revision);
#undef COPY_TEXT
    yvex_core_text_copy(out->tokenizer_class, sizeof(out->tokenizer_class),
                        source->tokenizer.tokenizer_class);
    yvex_core_text_copy(out->tokenizer_model_type, sizeof(out->tokenizer_model_type),
                        source->tokenizer.model_type);
    out->hidden_size = source->hidden_size;
    out->vocabulary_size = source->vocabulary_size;
    out->maximum_context = source->maximum_context;
    out->target_layers = source->main_layer_count;
    out->draft_layers = source->auxiliary_layer_count;
    out->swa_layers = source->swa_layer_count;
    out->csa_layers = source->csa_layer_count;
    out->hca_layers = source->hca_layer_count;
    out->hash_router_layers = source->hash_router_layer_count;
    out->learned_router_layers = source->learned_router_layer_count;
    out->dspark_block_size = source->dspark.block_size;
    out->dspark_noise_token_id = source->dspark.noise_token_id;
    out->dspark_markov_rank = source->dspark.markov_rank;
    out->dspark_confidence_available = source->dspark.confidence_available;
    out->mhc_residual_streams = source->final_mhc.residual_streams;
    out->mhc_expanded_width = source->final_mhc.expanded_width;
    out->mhc_mixing_rows = source->final_mhc.mixing_rows;
    out->mhc_mixing_columns = source->final_mhc.mixing_columns;
    out->mhc_sinkhorn_iterations = source->final_mhc.sinkhorn_iterations;
    out->final_mhc_post_required = source->final_mhc_post_required;
    out->final_mhc_head_required = source->final_mhc_head_required;
    out->final_norm_after_mhc_head = source->final_norm_after_mhc_head;
    out->tokenizer_vocabulary_size = source->tokenizer.vocabulary_size;
    out->tokenizer_base_vocab_entries = source->tokenizer.base_vocab_entries;
    out->tokenizer_added_token_entries = source->tokenizer.added_token_entries;
    out->bos_token_id = source->tokenizer.bos_token_id;
    out->eos_token_id = source->tokenizer.eos_token_id;
    out->output_head_required = source->output.required;
    out->output_head_tied = source->output.tied_to_embedding;
    out->source_quant_block_rows = source->source_constraint.quant_block_rows;
    out->source_quant_block_columns = source->source_constraint.quant_block_columns;
    out->source_header_scans = source->source_header_scan_count;
    out->source_header_tensors = source->source_header_tensor_count;
    out->source_payload_bytes = source->source_payload_bytes_read;
}

static int project_architecture(yvex_model_target_report *report, yvex_error *err)
{
    const yvex_model_family_api *family = yvex_model_register_deepseek_v4();
    const yvex_deepseek_v4_ir *ir = report->family_architecture;
    const yvex_deepseek_v4_model_spec *model = family->ir.model(ir);
    yvex_model_target_architecture_projection *out = &report->detail.architecture;
    unsigned long long index;

    if (!model || family->ir.layer_count(ir) > YVEX_MODEL_TARGET_LAYER_CAP ||
        family->ir.auxiliary_count(ir) > YVEX_MODEL_TARGET_LAYER_CAP ||
        model->dspark.target_layer_count > YVEX_MODEL_TARGET_FEATURE_CAP)
        return projection_refuse(err, "architecture projection exceeds its bounded view");
    memset(out, 0, sizeof(*out));
    project_architecture_model(out, model);
    yvex_core_text_copy(out->source_weight_dtype, sizeof(out->source_weight_dtype),
                        family->ir.source_weight_dtype_name(model->source_constraint.weight_dtype));
    yvex_core_text_copy(out->source_expert_dtype, sizeof(out->source_expert_dtype),
                        family->ir.source_expert_dtype_name(model->source_constraint.expert_dtype));
    yvex_core_text_copy(out->source_quantization, sizeof(out->source_quantization),
                        family->ir.source_quantization_name(model->source_constraint.quantization));
    out->dspark_feature_layer_count = (unsigned int)model->dspark.target_layer_count;
    memcpy(out->dspark_feature_layers, model->dspark.target_layer_ids,
           out->dspark_feature_layer_count * sizeof(out->dspark_feature_layers[0]));
    out->layer_count = (unsigned int)family->ir.layer_count(ir);
    for (index = 0u; index < out->layer_count; ++index) {
        const yvex_deepseek_v4_layer_spec *source = family->ir.layer_at(ir, index);
        yvex_model_target_layer_projection *layer = &out->layers[index];

        layer->index = source->layer_index;
        layer->compression_ratio = source->compression_ratio;
        yvex_core_text_copy(layer->attention, sizeof(layer->attention),
                            attention_name(source->attention_class));
        yvex_core_text_copy(layer->kv, sizeof(layer->kv), family->ir.kv_name(source->kv.class_id));
        yvex_core_text_copy(layer->router, sizeof(layer->router),
                            family->ir.router_name(source->moe.router_class));
        yvex_core_text_copy(layer->mhc_entry, sizeof(layer->mhc_entry),
                            source->mhc.entry == YVEX_DEEPSEEK_V4_MHC_STANDALONE_PRE
                                ? "standalone-pre" : "fused-prior-post-pre");
        if (!index) {
            out->query_heads = source->query_heads;
            out->kv_heads = source->kv_heads;
            out->head_dimension = source->head_dimension;
            out->rope_head_dimension = source->rope_head_dimension;
            out->routed_experts = source->moe.routed_experts;
            out->experts_per_token = source->moe.experts_per_token;
            out->shared_experts = source->moe.shared_experts;
        }
    }
    out->draft_count = (unsigned int)family->ir.auxiliary_count(ir);
    for (index = 0u; index < out->draft_count; ++index) {
        const yvex_deepseek_v4_auxiliary_spec *source = family->ir.auxiliary_at(ir, index);
        yvex_model_target_draft_projection *draft = &out->drafts[index];

        draft->predictor_index = source->predictor_index;
        draft->layer_index = source->layer.layer_index;
        draft->compression_ratio = source->layer.compression_ratio;
        yvex_core_text_copy(draft->attention, sizeof(draft->attention),
                            attention_name(source->layer.attention_class));
        yvex_core_text_copy(draft->router, sizeof(draft->router),
                            family->ir.router_name(source->layer.moe.router_class));
        draft->feature_projection = source->has_feature_projection;
        draft->markov = source->has_markov_head;
        draft->confidence = source->has_confidence_head;
        draft->shared_head = source->shares_output_head;
    }
    report->detail_kind = YVEX_MODEL_TARGET_DETAIL_MODEL_ARCHITECTURE;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int project_composite(yvex_model_target_report *report, yvex_error *err)
{
    const yvex_minimax_h3_api *api = yvex_model_register_minimax_h3();
    const yvex_minimax_h3_target *target = report->family_architecture;
    const yvex_minimax_h3_summary *source = api->summary(target);
    const yvex_transform_ir_summary *transform = yvex_transform_ir_summary_get(
        (const yvex_transform_ir *)report->family_transformation);
    yvex_model_target_composite_projection *out = &report->detail.composite;
    unsigned long long index;

    if (!source || !api->architecture(target) || !transform ||
        source->component_count > YVEX_MODEL_TARGET_COMPONENT_CAP ||
        source->phase_edge_count > YVEX_MODEL_TARGET_EDGE_CAP)
        return projection_refuse(err, "composite architecture exceeds its bounded view");
    memset(out, 0, sizeof(*out));
#define COPY_TEXT(destination, value) \
    yvex_core_text_copy((destination), sizeof(destination), (value))
    COPY_TEXT(out->repository, YVEX_MINIMAX_H3_REPOSITORY);
    COPY_TEXT(out->revision, YVEX_MINIMAX_H3_REVISION);
    COPY_TEXT(out->subtree, YVEX_MINIMAX_H3_SUBTREE);
    COPY_TEXT(out->source_snapshot_identity, source->source_snapshot_identity);
    COPY_TEXT(out->component_manifest_identity, source->component_manifest_identity);
    COPY_TEXT(out->phase_dag_identity, source->phase_dag_identity);
    COPY_TEXT(out->architecture_identity, source->architecture_identity);
    COPY_TEXT(out->role_map_identity, source->role_map_identity);
    COPY_TEXT(out->transformation_identity, transform->transform_identity);
    COPY_TEXT(out->unresolved_requirements_identity, source->unresolved_requirements_identity);
#undef COPY_TEXT
    out->components = source->component_count;
    out->weighted_components = source->weighted_component_count;
    out->phase_edges = source->phase_edge_count;
    out->shards = source->shard_count;
    out->tensors = source->tensor_count;
    out->elements = source->element_count;
    out->payload_bytes = source->payload_bytes;
    out->payload_execution_bytes = transform->payload_bytes_read;
    out->component_count = (unsigned int)source->component_count;
    for (index = 0u; index < out->component_count; ++index) {
        const yvex_minimax_h3_component *component = api->component_at(target, index);
        yvex_model_target_component_projection *row = &out->component[index];

        if (!component) return projection_refuse(err, "composite component projection is absent");
        yvex_core_text_copy(row->canonical_id, sizeof(row->canonical_id), component->canonical_id);
        yvex_core_text_copy(row->identity, sizeof(row->identity), component->identity);
        row->shards = component->shard_count;
        row->tensors = component->tensor_count;
        row->phase = (unsigned int)component->phase;
        row->weighted = component->weighted;
        row->release_after_phase = component->release_after_phase;
    }
    out->edge_count = (unsigned int)source->phase_edge_count;
    for (index = 0u; index < out->edge_count; ++index) {
        const yvex_minimax_h3_phase_edge *edge = api->phase_edge_at(index);
        yvex_model_target_edge_projection *row = &out->edge[index];

        if (!edge) return projection_refuse(err, "composite phase-edge projection is absent");
        row->source_phase = (unsigned int)edge->source_phase;
        row->destination_phase = (unsigned int)edge->destination_phase;
        row->data_classes = edge->data_classes;
        row->lifetime = (unsigned int)edge->lifetime;
    }
    report->detail_kind = YVEX_MODEL_TARGET_DETAIL_COMPOSITE_ARCHITECTURE;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_model_target_report_project_family_detail(
    yvex_model_target_report *report, yvex_error *err)
{
    if (!report) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "model_target_family_projection",
                       "report is required");
        return YVEX_ERR_INVALID_ARG;
    }
    report->detail_kind = YVEX_MODEL_TARGET_DETAIL_NONE;
    memset(&report->detail, 0, sizeof(report->detail));
    if (report->family_lowering) return project_map(report, err);
    if (report->family_coverage) return project_coverage(report, err);
    if (report->family_architecture_kind == YVEX_MODEL_TARGET_FAMILY_ARCHITECTURE_DEEPSEEK)
        return project_architecture(report, err);
    if (report->family_architecture_kind == YVEX_MODEL_TARGET_FAMILY_ARCHITECTURE_MINIMAX_H3)
        return project_composite(report, err);
    yvex_error_clear(err);
    return YVEX_OK;
}

void yvex_model_target_report_close_family_detail(yvex_model_target_report *report)
{
    if (!report) return;
    yvex_model_register_deepseek_v4()->lowering.close(
        (yvex_deepseek_gguf_map *)report->family_lowering);
    yvex_model_register_deepseek_v4()->coverage.close(
        (yvex_deepseek_tensor_coverage *)report->family_coverage);
    yvex_transform_ir_release((yvex_transform_ir **)&report->family_transformation);
    if (report->family_architecture_kind == YVEX_MODEL_TARGET_FAMILY_ARCHITECTURE_DEEPSEEK) {
        yvex_model_register_deepseek_v4()->ir.close(
            (yvex_deepseek_v4_ir *)report->family_architecture);
    } else if (report->family_architecture_kind ==
               YVEX_MODEL_TARGET_FAMILY_ARCHITECTURE_MINIMAX_H3) {
        yvex_minimax_h3_target *target = report->family_architecture;

        yvex_model_register_minimax_h3()->close(&target);
    }
}
