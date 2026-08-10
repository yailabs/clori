/*
 * Provide normal/table/audit/help rendering for typed model-target reports.
 *
 * This typed renderer writes only through src/cli/io helpers and renders typed report rows
 * supplied by model-target report builders. Model-target rendering serializes existing report-only
 * facts and does not prove quantization, artifact emission, runtime execution, generation,
 * evaluation, benchmark, throughput, or release readiness.
 */
#include "src/cli/render/private.h"

#include "src/cli/io/private.h"

#define TARGET_PTR(type_name, key_name, member, default_text) \
    {key_name, YVEX_CLI_FIELD_TEXT, offsetof(type_name, member), default_text}
#define TARGET_TEXT(type_name, key_name, member, default_text) \
    {key_name, YVEX_CLI_FIELD_TEXT_ARRAY, offsetof(type_name, member), default_text}
#define TARGET_U64(type_name, key_name, member) \
    {key_name, YVEX_CLI_FIELD_U64, offsetof(type_name, member), NULL}
#define TARGET_BOOL(type_name, key_name, member) \
    {key_name, YVEX_CLI_FIELD_BOOL, offsetof(type_name, member), NULL}
#define TARGET_HEX(type_name, key_name, member) \
    {key_name, YVEX_CLI_FIELD_HEX64, offsetof(type_name, member), NULL}

static const yvex_cli_field_spec map_report_head[] = {
    TARGET_PTR(yvex_model_target_report, "mapping_status", status, "unknown"),
    TARGET_TEXT(yvex_model_target_report, "target_id", target_id, "unknown"),
};
static const yvex_cli_field_spec map_summary_head[] = {
    TARGET_U64(yvex_model_target_map_projection, "source_contribution_count",
               source_contributions),
    TARGET_U64(yvex_model_target_map_projection, "descriptor_count", descriptors),
    TARGET_U64(yvex_model_target_map_projection, "trunk_descriptor_count", trunk_descriptors),
    TARGET_U64(yvex_model_target_map_projection, "draft_descriptor_count", draft_descriptors),
    TARGET_U64(yvex_model_target_map_projection, "pinned_standard_name_count",
               pinned_standard_names),
    TARGET_U64(yvex_model_target_map_projection, "semantic_standard_name_count",
               semantic_standard_names),
    TARGET_U64(yvex_model_target_map_projection, "extension_name_count", extension_names),
};
static const yvex_cli_field_spec map_summary_tail[] = {
    TARGET_U64(yvex_model_target_map_projection, "metadata_count", metadata),
    TARGET_U64(yvex_model_target_map_projection, "header_scan_count", header_scans),
    TARGET_U64(yvex_model_target_map_projection, "payload_bytes_read", payload_bytes),
    TARGET_HEX(yvex_model_target_map_projection, "source_identity", source_identity),
    TARGET_HEX(yvex_model_target_map_projection, "coverage_identity", coverage_identity),
    TARGET_HEX(yvex_model_target_map_projection, "mapping_identity", mapping_identity),
};
static const yvex_cli_field_spec map_report_tail[] = {
    TARGET_TEXT(yvex_model_target_report, "runtime_status", runtime_status, "unsupported"),
    TARGET_TEXT(yvex_model_target_report, "generation_status", generation_status, "unsupported"),
    TARGET_TEXT(yvex_model_target_report, "next", next_row, "unknown"),
    TARGET_TEXT(yvex_model_target_report, "boundary", boundary, "unknown"),
};

static const yvex_cli_field_spec coverage_report_head[] = {
    TARGET_PTR(yvex_model_target_report, "tensor_coverage_status", status, "unknown"),
    TARGET_TEXT(yvex_model_target_report, "target_id", target_id, "unknown"),
};
static const yvex_cli_field_spec coverage_summary_head[] = {
    TARGET_U64(yvex_model_target_coverage_projection, "source_tensor_count", source_tensors),
    TARGET_U64(yvex_model_target_coverage_projection, "required_tensor_count", required_tensors),
    TARGET_U64(yvex_model_target_coverage_projection, "matched_tensor_count", matched_tensors),
    TARGET_U64(yvex_model_target_coverage_projection, "missing_tensor_count", missing_tensors),
    TARGET_U64(yvex_model_target_coverage_projection, "ambiguous_tensor_count", ambiguous_tensors),
    TARGET_U64(yvex_model_target_coverage_projection, "unexpected_tensor_count", unexpected_tensors),
};
static const yvex_cli_field_spec coverage_summary_tail[] = {
    TARGET_U64(yvex_model_target_coverage_projection, "header_scan_count", header_scans),
    TARGET_U64(yvex_model_target_coverage_projection, "payload_bytes_read", payload_bytes),
    TARGET_U64(yvex_model_target_coverage_projection, "source_lookup_count", source_lookups),
    TARGET_U64(yvex_model_target_coverage_projection, "source_collision_count", source_collisions),
    TARGET_U64(yvex_model_target_coverage_projection, "source_maximum_probe", source_maximum_probe),
    TARGET_HEX(yvex_model_target_coverage_projection, "source_identity", source_identity),
    TARGET_HEX(yvex_model_target_coverage_projection, "coverage_identity", coverage_identity),
};
static const yvex_cli_field_spec coverage_report_tail[] = {
    TARGET_TEXT(yvex_model_target_report, "next_required_row", next_row, "unknown"),
    TARGET_TEXT(yvex_model_target_report, "boundary", boundary, "unknown"),
};

static const yvex_cli_field_spec architecture_model_fields[] = {
    TARGET_TEXT(yvex_model_target_architecture_projection, "target_id", target_id, "unknown"),
    TARGET_TEXT(yvex_model_target_architecture_projection, "family", family, "unknown"),
    TARGET_TEXT(yvex_model_target_architecture_projection, "architecture", architecture, "unknown"),
    TARGET_TEXT(yvex_model_target_architecture_projection, "repository", repository, "unknown"),
    TARGET_TEXT(yvex_model_target_architecture_projection, "revision", revision, "unknown"),
    TARGET_TEXT(yvex_model_target_architecture_projection, "verification_stage", verification_stage, "unknown"),
    TARGET_TEXT(yvex_model_target_architecture_projection, "paper_revision", paper_revision, "unknown"),
    TARGET_TEXT(yvex_model_target_architecture_projection, "sglang_revision", sglang_revision, "unknown"),
    TARGET_TEXT(yvex_model_target_architecture_projection, "vllm_revision", vllm_revision, "unknown"),
    TARGET_U64(yvex_model_target_architecture_projection, "hidden_size", hidden_size),
    TARGET_U64(yvex_model_target_architecture_projection, "vocabulary_size", vocabulary_size),
    TARGET_U64(yvex_model_target_architecture_projection, "maximum_context", maximum_context),
    TARGET_U64(yvex_model_target_architecture_projection, "main_layer_count", target_layers),
    TARGET_U64(yvex_model_target_architecture_projection, "auxiliary_layer_count", draft_layers),
    TARGET_U64(yvex_model_target_architecture_projection, "swa_layer_count", swa_layers),
    TARGET_U64(yvex_model_target_architecture_projection, "csa_layer_count", csa_layers),
    TARGET_U64(yvex_model_target_architecture_projection, "hca_layer_count", hca_layers),
    TARGET_U64(yvex_model_target_architecture_projection, "hash_router_layer_count", hash_router_layers),
    TARGET_U64(yvex_model_target_architecture_projection, "learned_router_layer_count", learned_router_layers),
    TARGET_U64(yvex_model_target_architecture_projection, "dspark_block_size", dspark_block_size),
    TARGET_U64(yvex_model_target_architecture_projection, "dspark_noise_token_id", dspark_noise_token_id),
    TARGET_U64(yvex_model_target_architecture_projection, "dspark_draft_layer_count", draft_layers),
    TARGET_U64(yvex_model_target_architecture_projection, "dspark_markov_rank", dspark_markov_rank),
    TARGET_BOOL(yvex_model_target_architecture_projection, "dspark_confidence_available", dspark_confidence_available),
    TARGET_U64(yvex_model_target_architecture_projection, "mhc_residual_streams", mhc_residual_streams),
    TARGET_U64(yvex_model_target_architecture_projection, "mhc_expanded_width", mhc_expanded_width),
    TARGET_U64(yvex_model_target_architecture_projection, "mhc_mixing_rows", mhc_mixing_rows),
    TARGET_U64(yvex_model_target_architecture_projection, "mhc_mixing_columns", mhc_mixing_columns),
    TARGET_U64(yvex_model_target_architecture_projection, "mhc_sinkhorn_iterations", mhc_sinkhorn_iterations),
    TARGET_BOOL(yvex_model_target_architecture_projection, "final_mhc_post_required", final_mhc_post_required),
    TARGET_BOOL(yvex_model_target_architecture_projection, "final_mhc_head_required", final_mhc_head_required),
    TARGET_BOOL(yvex_model_target_architecture_projection, "final_norm_after_mhc_head", final_norm_after_mhc_head),
    TARGET_TEXT(yvex_model_target_architecture_projection, "tokenizer_class", tokenizer_class, "unknown"),
    TARGET_TEXT(yvex_model_target_architecture_projection, "tokenizer_model_type", tokenizer_model_type, "unknown"),
    TARGET_U64(yvex_model_target_architecture_projection, "tokenizer_vocabulary_size", tokenizer_vocabulary_size),
    TARGET_U64(yvex_model_target_architecture_projection, "tokenizer_base_vocab_entries", tokenizer_base_vocab_entries),
    TARGET_U64(yvex_model_target_architecture_projection, "tokenizer_added_token_entries",
               tokenizer_added_token_entries),
    TARGET_U64(yvex_model_target_architecture_projection, "bos_token_id", bos_token_id),
    TARGET_U64(yvex_model_target_architecture_projection, "eos_token_id", eos_token_id),
    TARGET_BOOL(yvex_model_target_architecture_projection, "output_head_required", output_head_required),
    TARGET_BOOL(yvex_model_target_architecture_projection, "output_head_tied", output_head_tied),
};
static const yvex_cli_field_spec architecture_source_fields[] = {
    TARGET_U64(yvex_model_target_architecture_projection, "source_header_scan_count", source_header_scans),
    TARGET_U64(yvex_model_target_architecture_projection, "source_header_tensor_count", source_header_tensors),
    TARGET_U64(yvex_model_target_architecture_projection, "source_payload_bytes_read", source_payload_bytes),
};
static const yvex_cli_field_spec architecture_report_tail[] = {
    TARGET_TEXT(yvex_model_target_report, "next_required_row", next_row, "unknown"),
    TARGET_TEXT(yvex_model_target_report, "boundary", boundary, "unknown"),
};
#undef TARGET_HEX
#undef TARGET_BOOL
#undef TARGET_U64
#undef TARGET_TEXT
#undef TARGET_PTR

static int model_target_render_rows(FILE *fp,
                                    const yvex_model_target_text_value *rows,
                                    unsigned long count)
{
    unsigned long i;
    int rc = 0;

    for (i = 0; i < count; ++i) {
        rc = yvex_cli_out_writef(fp, "%s\n", rows[i].value);
        if (rc < 0) {
            return rc;
        }
    }
    return rc;
}

static int model_target_render_help_rows(FILE *fp,
                                         yvex_model_target_command_kind kind)
{
    if (kind == YVEX_MODEL_TARGET_COMMAND_DECISION) {
        return yvex_cli_out_writef(
            fp, "usage: yvex inspect target decision --release v0.1.0 [options]\n");
    }
    if (kind == YVEX_MODEL_TARGET_COMMAND_CANDIDATE) {
        return yvex_cli_out_writef(
            fp, "usage: yvex inspect target candidate --release v0.1.0 [options]\n");
    }
    if (kind == YVEX_MODEL_TARGET_COMMAND_DENSE_CANDIDATE) {
        return yvex_cli_out_writef(
            fp, "usage: yvex inspect target dense-candidate --release v0.1.0 [options]\n");
    }
    if (kind == YVEX_MODEL_TARGET_COMMAND_QWEN_METAL) {
        return yvex_cli_out_writef(
            fp, "usage: yvex inspect target qwen-metal --release v0.1.0 [options]\n");
    }
    return yvex_cli_out_writef(
        fp,
        "usage: yvex inspect target <action> [TARGET]\nusage: yvex inspect target classes\n       yvex "
            "inspect target list\n       yvex inspect target candidate --release v0.1.0 [options]\n       "
            "yvex inspect target dense-candidate --release v0.1.0 [options]\n       yvex inspect target "
            "qwen-metal --release v0.1.0 [options]\n       yvex inspect target decision --release v0.1.0 "
            "[options]\n       yvex inspect target class-profile TARGET\n       yvex inspect target "
            "tensor-collection TARGET\n       yvex inspect target tensor-map TARGET\n       yvex inspect target "
            "missing-roles TARGET --gate v0.1.0\n       "
            "yvex inspect target inspect TARGET [--paths] [--models-root DIR]\n--paths           show expected "
            "operator-local source, artifact, report, reference, and registry paths\n--models-root DIR "
            "override configured operator model root for this command only\noption_classes: selector, path, "
            "diagnostic, transitional-layout\n");
}

static int model_target_render_table_rows(FILE *fp,
                                          const yvex_model_target_report *report)
{
    unsigned long r;
    int rc = 0;

    for (r = 0; r < report->table_row_count; ++r) {
        const yvex_model_target_table_row *row = &report->table_rows[r];
        unsigned int c;

        for (c = 0; c < row->column_count; ++c) {
            rc = yvex_cli_out_writef(fp, "%s%s",
                                     c == 0 ? "" : "  ",
                                     row->columns[c]);
            if (rc < 0) {
                return rc;
            }
        }
        rc = yvex_cli_out_writef(fp, "\n");
        if (rc < 0) {
            return rc;
        }
    }
    return rc;
}

static int model_target_render_tensor_map(
    FILE *fp,
    yvex_model_target_render_mode mode,
    const yvex_model_target_report *report)
{
    const yvex_model_target_map_projection *summary = &report->detail.map;
    int rc = 0;

    if (mode == YVEX_MODEL_TARGET_OUTPUT_JSON) {
        return yvex_cli_out_writef(
            fp,
            "{\"status\":\"%s\",\"target_id\":\"%s\",\"source_contributions\":%llu,\"descriptors\":%llu,"
                "\"trunk_descriptors\":%llu,\"draft_descriptors\":%llu,\"pinned_standard_names\":%llu,"
                "\"extension_names\":%llu,\"metadata\":%llu,\"header_scans\":%llu,\"payload_bytes_read\":%llu,"
                "\"mapping_identity\":\"%016llx\",\"artifact\":\"not-produced\",\"runtime\":\"unsupported\","
                "\"generation\":\"unsupported\",\"next\":\"%s\"}\n",
            report->status, report->target_id,
            summary->source_contributions, summary->descriptors,
            summary->trunk_descriptors, summary->draft_descriptors,
            summary->pinned_standard_names, summary->extension_names,
            summary->metadata, summary->header_scans,
            summary->payload_bytes, summary->mapping_identity,
            report->next_row);
    }
    if (mode == YVEX_MODEL_TARGET_OUTPUT_TABLE) {
        rc |= yvex_cli_out_writef(
            fp, "TARGET  STATUS  SOURCES  GGUF  TRUNK  DRAFT  METADATA  PAYLOAD  NEXT\n");
        rc |= yvex_cli_out_writef(
            fp, "%s  %s  %llu  %llu  %llu  %llu  %llu  %llu  %s\n",
            report->target_id, report->status,
            summary->source_contributions, summary->descriptors,
            summary->trunk_descriptors, summary->draft_descriptors,
            summary->metadata, summary->payload_bytes,
            report->next_row);
        return rc < 0 ? rc : 0;
    }
    if (mode == YVEX_MODEL_TARGET_OUTPUT_AUDIT) {
        unsigned int collection;

        rc |= yvex_cli_out_fields(fp, report, map_report_head,
                                  sizeof(map_report_head) / sizeof(map_report_head[0]));
        rc |= yvex_cli_out_fields(fp, summary, map_summary_head,
                                  sizeof(map_summary_head) / sizeof(map_summary_head[0]));
        for (collection = 0u;
             collection < summary->collection_count;
             ++collection) {
            rc |= yvex_cli_out_writef(
                fp, "collection_%u_count: %llu\n", collection,
                summary->collections[collection].count);
        }
        rc |= yvex_cli_out_fields(fp, summary, map_summary_tail,
                                  sizeof(map_summary_tail) / sizeof(map_summary_tail[0]));
        rc |= yvex_cli_out_writef(fp, "artifact_status: not-produced\n");
        rc |= yvex_cli_out_fields(fp, report, map_report_tail,
                                  sizeof(map_report_tail) / sizeof(map_report_tail[0]));
        return rc < 0 ? rc : 0;
    }
    rc |= yvex_cli_out_writef(fp, "deepseek-gguf-map: %s [%s]\n",
                              report->target_id, report->status);
    rc |= yvex_cli_out_writef(
        fp, "plan: sources=%llu gguf=%llu trunk=%llu draft=%llu metadata=%llu\n",
        summary->source_contributions, summary->descriptors,
        summary->trunk_descriptors, summary->draft_descriptors,
        summary->metadata);
    rc |= yvex_cli_out_writef(
        fp, "evidence: header-scans=%llu payload-bytes=%llu identity=%016llx\n",
        summary->header_scans, summary->payload_bytes,
        summary->mapping_identity);
    rc |= yvex_cli_out_writef(fp, "next: %s\n", report->next_row);
    rc |= yvex_cli_out_writef(fp, "boundary: %s\n", report->boundary);
    return rc < 0 ? rc : 0;
}

static int model_target_render_tensor_coverage(
    FILE *fp,
    yvex_model_target_render_mode mode,
    const yvex_model_target_report *report)
{
    const yvex_model_target_coverage_projection *summary = &report->detail.coverage;
    int rc = 0;

    if (mode == YVEX_MODEL_TARGET_OUTPUT_JSON) {
        return yvex_cli_out_writef(
            fp,
            "{\"status\":\"%s\",\"target_id\":\"%s\",\"source_tensors\":%llu,\"required_tensors\":%llu,"
                "\"matched_tensors\":%llu,\"missing\":%llu,\"ambiguous\":%llu,\"unexpected\":%llu,"
                "\"header_scans\":%llu,\"payload_bytes_read\":%llu,\"coverage_identity\":\"%016llx\","
                "\"mapping\":\"blocked\",\"runtime\":\"unsupported\",\"generation\":\"unsupported\",\"next\":"
                "\"%s\"}\n",
            report->status, report->target_id, summary->source_tensors,
            summary->required_tensors, summary->matched_tensors,
            summary->missing_tensors, summary->ambiguous_tensors,
            summary->unexpected_tensors,
            summary->header_scans, summary->payload_bytes,
            summary->coverage_identity, report->next_row);
    }
    if (mode == YVEX_MODEL_TARGET_OUTPUT_TABLE) {
        rc |= yvex_cli_out_writef(
            fp, "TARGET  STATUS  SOURCE  REQUIRED  MATCHED  MISSING  UNEXPECTED  NEXT\n");
        rc |= yvex_cli_out_writef(
            fp, "%s  %s  %llu  %llu  %llu  %llu  %llu  %s\n",
            report->target_id, report->status, summary->source_tensors,
            summary->required_tensors, summary->matched_tensors,
            summary->missing_tensors, summary->unexpected_tensors,
            report->next_row);
        return rc < 0 ? rc : 0;
    }
    if (mode == YVEX_MODEL_TARGET_OUTPUT_AUDIT) {
        unsigned int collection;

        rc |= yvex_cli_out_fields(fp, report, coverage_report_head,
                                  sizeof(coverage_report_head) / sizeof(coverage_report_head[0]));
        rc |= yvex_cli_out_fields(fp, summary, coverage_summary_head,
                                  sizeof(coverage_summary_head) / sizeof(coverage_summary_head[0]));
        for (collection = 0u;
             collection < summary->collection_count;
             ++collection) {
            rc |= yvex_cli_out_writef(
                fp, "collection_%s_count: %llu\n",
                summary->collections[collection].name,
                summary->collections[collection].count);
        }
        rc |= yvex_cli_out_fields(fp, summary, coverage_summary_tail,
                                  sizeof(coverage_summary_tail) /
                                      sizeof(coverage_summary_tail[0]));
        rc |= yvex_cli_out_writef(fp, "mapping: blocked\n");
        rc |= yvex_cli_out_writef(fp, "runtime_execution: unsupported\n");
        rc |= yvex_cli_out_writef(fp, "generation: unsupported\n");
        rc |= yvex_cli_out_fields(fp, report, coverage_report_tail,
                                  sizeof(coverage_report_tail) /
                                      sizeof(coverage_report_tail[0]));
        return rc < 0 ? rc : 0;
    }
    rc |= yvex_cli_out_writef(fp, "tensor-coverage: deepseek-v4-flash-dspark\n");
    rc |= yvex_cli_out_writef(fp, "target: %s\n", report->target_id);
    rc |= yvex_cli_out_writef(fp, "status: %s\n", report->status);
    rc |= yvex_cli_out_writef(
        fp, "coverage: source=%llu required=%llu matched=%llu missing=%llu unexpected=%llu\n",
        summary->source_tensors, summary->required_tensors,
        summary->matched_tensors, summary->missing_tensors,
        summary->unexpected_tensors);
    rc |= yvex_cli_out_writef(fp, "identity: %016llx\n",
                              summary->coverage_identity);
    rc |= yvex_cli_out_writef(fp, "next: %s\n", report->next_row);
    rc |= yvex_cli_out_writef(fp, "boundary: %s\n", report->boundary);
    return rc < 0 ? rc : 0;
}

static int model_target_render_deepseek_normal(
    FILE *fp,
    const yvex_model_target_report *report,
    const yvex_model_target_architecture_projection *model)
{
    int rc = 0;

    rc |= yvex_cli_out_writef(fp, "model-class: deepseek-v4-flash-dspark\n");
    rc |= yvex_cli_out_writef(fp, "target: %s\n", model->target_id);
    rc |= yvex_cli_out_writef(fp, "status: %s\n", report->status);
    rc |= yvex_cli_out_writef(
        fp, "topology: target-layers=%llu draft-layers=%llu hidden=%llu vocab=%llu context=%llu\n",
        model->target_layers, model->draft_layers,
        model->hidden_size, model->vocabulary_size, model->maximum_context);
    rc |= yvex_cli_out_writef(
        fp, "attention: swa=%llu csa=%llu hca=%llu heads=%llu kv_heads=%llu head_dim=%llu rope_dim=%llu\n",
        model->swa_layers, model->csa_layers,
        model->hca_layers, model->query_heads, model->kv_heads,
        model->head_dimension, model->rope_head_dimension);
    rc |= yvex_cli_out_writef(
        fp, "routing: hash=%llu learned=%llu experts=%llu topk=%llu shared=%llu\n",
        model->hash_router_layers, model->learned_router_layers,
        model->routed_experts, model->experts_per_token,
        model->shared_experts);
    rc |= yvex_cli_out_writef(
        fp, "mhc: streams=%llu expanded=%llu mixing_rows=%llu sinkhorn=%llu\n",
        model->mhc_residual_streams, model->mhc_expanded_width,
        model->mhc_mixing_rows, model->mhc_sinkhorn_iterations);
    rc |= yvex_cli_out_writef(
        fp, "dspark: block=%llu noise=%llu taps=%llu,%llu,%llu markov-rank=%llu confidence=%s\n",
        model->dspark_block_size, model->dspark_noise_token_id,
        model->dspark_feature_layers[0], model->dspark_feature_layers[1],
        model->dspark_feature_layers[2], model->dspark_markov_rank,
        model->dspark_confidence_available ? "available" : "unavailable");
    rc |= yvex_cli_out_writef(fp, "next: %s\n", report->next_row);
    rc |= yvex_cli_out_writef(fp, "boundary: %s\n", report->boundary);
    return rc < 0 ? rc : 0;
}

static int model_target_render_deepseek_table(
    FILE *fp,
    const yvex_model_target_report *report,
    const yvex_model_target_architecture_projection *model)
{
    int rc = 0;

    rc |= yvex_cli_out_writef(
        fp, "TARGET  STATUS  TARGET_LAYERS  DRAFT_LAYERS  SWA  CSA  HCA  HASH  LEARNED  NEXT\n");
    rc |= yvex_cli_out_writef(
        fp, "%s  %s  %llu  %llu  %llu  %llu  %llu  %llu  %llu  %s\n",
        model->target_id, report->status, model->target_layers,
        model->draft_layers, model->swa_layers,
        model->csa_layers, model->hca_layers,
        model->hash_router_layers, model->learned_router_layers,
        report->next_row);
    return rc < 0 ? rc : 0;
}

static int model_target_render_deepseek_audit(
    FILE *fp,
    const yvex_model_target_report *report,
    const yvex_model_target_architecture_projection *model)
{
    unsigned long long i;
    int rc = 0;

    rc |= yvex_cli_out_writef(fp, "architecture_ir_status: %s\n", report->status);
    rc |= yvex_cli_out_fields(fp, model, architecture_model_fields,
                              sizeof(architecture_model_fields) /
                                  sizeof(architecture_model_fields[0]));
    rc |= yvex_cli_out_writef(
        fp, "source_weight_dtype: %s\n",
        model->source_weight_dtype);
    rc |= yvex_cli_out_writef(
        fp, "source_expert_dtype: %s\n",
        model->source_expert_dtype);
    rc |= yvex_cli_out_writef(
        fp, "source_quantization: %s block=%llux%llu\n",
        model->source_quantization,
        model->source_quant_block_rows,
        model->source_quant_block_columns);
    rc |= yvex_cli_out_fields(fp, model, architecture_source_fields,
                              sizeof(architecture_source_fields) /
                                  sizeof(architecture_source_fields[0]));
    for (i = 0u; i < model->layer_count; ++i) {
        const yvex_model_target_layer_projection *layer = &model->layers[i];
        rc |= yvex_cli_out_writef(
            fp,
            "layer_%llu: attention=%s ratio=%llu kv=%s router=%s mhc_entry=%s\n",
            layer->index,
            layer->attention,
            layer->compression_ratio,
            layer->kv, layer->router, layer->mhc_entry);
    }
    for (i = 0u; i < model->draft_count; ++i) {
        const yvex_model_target_draft_projection *aux = &model->drafts[i];
        rc |= yvex_cli_out_writef(
            fp,
            "draft_%llu: layer=%llu attention=%s ratio=%llu router=%s "
            "feature_projection=%s markov=%s confidence=%s shared_head=%s\n",
            aux->predictor_index, aux->layer_index,
            aux->attention, aux->compression_ratio, aux->router,
            aux->feature_projection ? "true" : "false",
            aux->markov ? "true" : "false",
            aux->confidence ? "true" : "false",
            aux->shared_head ? "true" : "false");
    }
    rc |= yvex_cli_out_writef(fp, "runtime_execution: unsupported\n");
    rc |= yvex_cli_out_writef(fp, "generation: unsupported\n");
    rc |= yvex_cli_out_fields(fp, report, architecture_report_tail,
                              sizeof(architecture_report_tail) /
                                  sizeof(architecture_report_tail[0]));
    return rc < 0 ? rc : 0;
}

static int model_target_render_deepseek_json(
    FILE *fp,
    const yvex_model_target_report *report,
    const yvex_model_target_architecture_projection *model)
{
    return yvex_cli_out_writef(
        fp,
        "{\"status\":\"%s\",\"target_id\":\"%s\",\"repository\":\"%s\",\"revision\":\"%s\",\"layers\":%llu,"
            "\"draft_layers\":%llu,\"hidden_size\":%llu,\"vocabulary_size\":%llu,\"context\":%llu,\"attention\":"
            "{\"swa\":%llu,\"csa\":%llu,\"hca\":%llu},\"routing\":{\"hash\":%llu,\"learned\":%llu},"
            "\"mhc_streams\":%llu,\"payload_bytes_read\":%llu,\"runtime\":\"unsupported\",\"generation\":"
            "\"unsupported\",\"next\":\"%s\"}\n",
        report->status, model->target_id, model->repository, model->revision,
        model->target_layers, model->draft_layers,
        model->hidden_size, model->vocabulary_size, model->maximum_context,
        model->swa_layers, model->csa_layers,
        model->hca_layers, model->hash_router_layers,
        model->learned_router_layers,
        model->mhc_residual_streams, model->source_payload_bytes,
        report->next_row);
}

static int model_target_render_deepseek_ir(
    FILE *fp,
    yvex_model_target_render_mode mode,
    const yvex_model_target_report *report)
{
    const yvex_model_target_architecture_projection *model =
        &report->detail.architecture;

    if (mode == YVEX_MODEL_TARGET_OUTPUT_TABLE) {
        return model_target_render_deepseek_table(fp, report, model);
    }
    if (mode == YVEX_MODEL_TARGET_OUTPUT_AUDIT) {
        return model_target_render_deepseek_audit(fp, report, model);
    }
    if (mode == YVEX_MODEL_TARGET_OUTPUT_JSON) {
        return model_target_render_deepseek_json(fp, report, model);
    }
    return model_target_render_deepseek_normal(fp, report, model);
}

static int model_target_render_composite(
    FILE *fp,
    yvex_model_target_render_mode mode,
    const yvex_model_target_report *report)
{
    const yvex_model_target_composite_projection *summary = &report->detail.composite;
    int rc = 0;

    if (mode == YVEX_MODEL_TARGET_OUTPUT_JSON) {
        return yvex_cli_out_writef(
            fp,
            "{\"status\":\"%s\",\"target_id\":\"%s\",\"family\":\"minimax-h3\","
            "\"repository\":\"%s\",\"revision\":\"%s\",\"subtree\":\"%s\","
            "\"source_verified\":true,\"components\":%llu,\"weighted_components\":%llu,"
            "\"phase_edges\":%llu,"
            "\"shards\":%llu,\"tensors\":%llu,\"elements\":%llu,\"payload_bytes\":%llu,"
            "\"source_snapshot_identity\":\"%s\",\"component_manifest_identity\":\"%s\","
            "\"phase_dag_identity\":\"%s\","
            "\"architecture_identity\":\"%s\",\"role_map_identity\":\"%s\","
            "\"transformation_ir_identity\":\"%s\",\"derivation_identity\":\"%s\","
            "\"payload_execution_bytes\":%llu,\"evidence_stage\":"
            "\"source-verified-architecture-and-transformation-ir\","
            "\"artifact\":\"not-produced\",\"runtime\":\"unsupported\","
            "\"generation\":\"unsupported\",\"next\":\"%s\"}\n",
            report->status, report->target_id, summary->repository,
            summary->revision, summary->subtree,
            summary->components, summary->weighted_components,
            summary->phase_edges,
            summary->shards, summary->tensors, summary->elements,
            summary->payload_bytes, summary->source_snapshot_identity,
            summary->component_manifest_identity, summary->phase_dag_identity,
            summary->architecture_identity,
            summary->role_map_identity, summary->transformation_identity,
            report->family_derivation_identity, summary->payload_execution_bytes,
            report->next_row);
    }
    if (mode == YVEX_MODEL_TARGET_OUTPUT_TABLE) {
        rc |= yvex_cli_out_writef(
            fp, "TARGET  STATUS  COMPONENTS  SHARDS  TENSORS  PAYLOAD  EXECUTION  NEXT\n");
        rc |= yvex_cli_out_writef(
            fp, "%s  %s  %llu  %llu  %llu  %llu  unsupported  %s\n",
            report->target_id, report->status, summary->components,
            summary->shards, summary->tensors, summary->payload_bytes,
            report->next_row);
        return rc < 0 ? rc : 0;
    }
    if (mode == YVEX_MODEL_TARGET_OUTPUT_AUDIT) {
        unsigned long long index;

        rc |= yvex_cli_out_writef(fp, "status: %s\n", report->status);
        rc |= yvex_cli_out_writef(fp, "target_id: %s\nfamily: minimax-h3\n", report->target_id);
        rc |= yvex_cli_out_writef(fp, "repository: %s\nrevision: %s\nsubtree: %s\n",
                                  summary->repository, summary->revision,
                                  summary->subtree);
        rc |= yvex_cli_out_writef(
            fp, "source_verified: true\ncomponents: %llu\nweighted_components: %llu\n"
                "shards: %llu\ntensors: %llu\nelements: %llu\npayload_bytes: %llu\n",
            summary->components, summary->weighted_components,
            summary->shards, summary->tensors, summary->elements,
            summary->payload_bytes);
        for (index = 0u; index < summary->component_count; ++index) {
            const yvex_model_target_component_projection *component =
                &summary->component[index];
            rc |= yvex_cli_out_writef(
                fp, "component_%llu: id=%s weighted=%s shards=%llu tensors=%llu "
                    "phase=%u release_after_phase=%s identity=%s\n",
                index, component->canonical_id, component->weighted ? "true" : "false",
                component->shards, component->tensors, component->phase,
                component->release_after_phase ? "true" : "false",
                component->identity);
        }
        for (index = 0u; index < summary->edge_count; ++index) {
            const yvex_model_target_edge_projection *edge = &summary->edge[index];
            rc |= yvex_cli_out_writef(
                fp, "phase_edge_%llu: from=%u to=%u data=%u lifetime=%u\n",
                index, edge->source_phase, edge->destination_phase,
                edge->data_classes, edge->lifetime);
        }
        rc |= yvex_cli_out_writef(fp, "source_snapshot_identity: %s\n",
                                  summary->source_snapshot_identity);
        rc |= yvex_cli_out_writef(fp, "component_manifest_identity: %s\n",
                                  summary->component_manifest_identity);
        rc |= yvex_cli_out_writef(fp, "phase_dag_identity: %s\n",
                                  summary->phase_dag_identity);
        rc |= yvex_cli_out_writef(fp, "architecture_identity: %s\n",
                                  summary->architecture_identity);
        rc |= yvex_cli_out_writef(fp, "role_map_identity: %s\n",
                                  summary->role_map_identity);
        rc |= yvex_cli_out_writef(fp, "transformation_ir_identity: %s\n",
                                  summary->transformation_identity);
        rc |= yvex_cli_out_writef(fp, "derivation_identity: %s\n",
                                  report->family_derivation_identity);
        rc |= yvex_cli_out_writef(
            fp, "unresolved_requirements_identity: %s\npayload_execution_bytes: %llu\n",
            summary->unresolved_requirements_identity,
            summary->payload_execution_bytes);
        rc |= yvex_cli_out_writef(
            fp, "unknowns: mm-rope,masks,conditioning-placement,latent-rng,solver,"
                "timestep-schedule,iteration-count,update-equation,guidance,output-geometry,"
                "codec-container,synchronization,adaln-cache-validity,quantization-validity\n");
        rc |= yvex_cli_out_writef(
            fp, "artifact_status: not-produced\nruntime_execution: unsupported\n"
                "generation: unsupported\nbenchmark_status: not-measured\nnext: %s\nboundary: %s\n",
            report->next_row, report->boundary);
        return rc < 0 ? rc : 0;
    }
    rc |= yvex_cli_out_writef(fp, "minimax-h3: %s [%s]\n", report->target_id, report->status);
    rc |= yvex_cli_out_writef(
        fp, "source: %s@%s %s verified\ncoverage: components=%llu shards=%llu "
            "tensors=%llu payload=%llu\n",
        summary->repository, summary->revision,
        summary->subtree, summary->components,
        summary->shards, summary->tensors, summary->payload_bytes);
    rc |= yvex_cli_out_writef(
        fp, "identity: architecture=%s phases=%s roles=%s transform=%s\n",
        summary->architecture_identity, summary->phase_dag_identity,
        summary->role_map_identity,
        summary->transformation_identity);
    rc |= yvex_cli_out_writef(
        fp, "stage: source/architecture/role-map/Transformation-IR; "
            "artifact/runtime/generation unsupported\nnext: %s\n",
        report->next_row);
    return rc < 0 ? rc : 0;
}

int yvex_model_target_render(FILE *fp,
                             yvex_model_target_render_mode mode,
                             const yvex_model_target_report *report)
{
    int rc;

    if (!report) {
        return 0;
    }
    if (report->help_requested ||
        report->kind == YVEX_MODEL_TARGET_COMMAND_HELP) {
        rc = model_target_render_help_rows(fp, report->kind);
        if (rc < 0) return rc;
    }
    if (report->detail_kind == YVEX_MODEL_TARGET_DETAIL_TENSOR_MAP)
        return model_target_render_tensor_map(fp, mode, report);
    if (report->detail_kind == YVEX_MODEL_TARGET_DETAIL_TENSOR_COVERAGE)
        return model_target_render_tensor_coverage(fp, mode, report);
    if (report->detail_kind == YVEX_MODEL_TARGET_DETAIL_COMPOSITE_ARCHITECTURE)
        return model_target_render_composite(fp, mode, report);
    if (report->detail_kind == YVEX_MODEL_TARGET_DETAIL_MODEL_ARCHITECTURE)
        return model_target_render_deepseek_ir(fp, mode, report);
    if (mode == YVEX_MODEL_TARGET_OUTPUT_TABLE && report->table_row_count > 0) {
        rc = model_target_render_table_rows(fp, report);
        if (rc < 0) {
            return rc;
        }
    }
    return model_target_render_rows(fp, report->rows, report->row_count);
}

int yvex_model_target_render_errors(FILE *fp,
                                    const yvex_model_target_report *report)
{
    if (!report) {
        return 0;
    }
    return model_target_render_rows(fp, report->error_rows,
                                    report->error_row_count);
}

int yvex_model_target_render_help(FILE *fp)
{
    yvex_model_target_report report;
    yvex_error err;
    int rc;

    yvex_error_clear(&err);
    rc = yvex_model_target_help_report_build(&report, &err);
    if (rc != YVEX_OK) {
        yvex_cli_out_writef(fp, "model-target help unavailable\n");
        return rc;
    }
    (void)yvex_model_target_render(fp, YVEX_MODEL_TARGET_OUTPUT_NORMAL, &report);
    yvex_model_target_report_close(&report);
    return 0;
}
