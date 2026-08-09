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
#include <yvex/internal/families/deepseek_v4.h>
#include <yvex/internal/families/minimax_h3.h>
#include <yvex/internal/compilation.h>
#include <yvex/internal/model.h>

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

static const char *attention_class_name(yvex_attention_class class_id)
{
    static const char *const names[] = {"swa", "csa", "hca"};

    return class_id >= YVEX_ATTENTION_CLASS_SWA && class_id <= YVEX_ATTENTION_CLASS_HCA
               ? names[class_id]
               : "unknown";
}

static const yvex_cli_field_spec map_report_head[] = {
    TARGET_PTR(yvex_model_target_report, "mapping_status", status, "unknown"),
    TARGET_TEXT(yvex_model_target_report, "target_id", target_id, "unknown"),
};
static const yvex_cli_field_spec map_summary_head[] = {
    TARGET_U64(yvex_deepseek_gguf_map_summary, "source_contribution_count",
               source_contribution_count),
    TARGET_U64(yvex_deepseek_gguf_map_summary, "descriptor_count", descriptor_count),
    TARGET_U64(yvex_deepseek_gguf_map_summary, "trunk_descriptor_count",
               trunk_descriptor_count),
    TARGET_U64(yvex_deepseek_gguf_map_summary, "draft_descriptor_count", draft_descriptor_count),
    TARGET_U64(yvex_deepseek_gguf_map_summary, "pinned_standard_name_count",
               pinned_standard_count),
    TARGET_U64(yvex_deepseek_gguf_map_summary, "semantic_standard_name_count",
               semantic_standard_count),
    TARGET_U64(yvex_deepseek_gguf_map_summary, "extension_name_count", extension_count),
};
static const yvex_cli_field_spec map_summary_tail[] = {
    TARGET_U64(yvex_deepseek_gguf_map_summary, "metadata_count", metadata_count),
    TARGET_U64(yvex_deepseek_gguf_map_summary, "header_scan_count", header_scan_count),
    TARGET_U64(yvex_deepseek_gguf_map_summary, "payload_bytes_read", payload_bytes_read),
    TARGET_HEX(yvex_deepseek_gguf_map_summary, "source_identity", source_identity),
    TARGET_HEX(yvex_deepseek_gguf_map_summary, "coverage_identity", coverage_identity),
    TARGET_HEX(yvex_deepseek_gguf_map_summary, "mapping_identity", mapping_identity),
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
    TARGET_U64(yvex_deepseek_tensor_coverage_summary, "source_tensor_count", source_tensor_count),
    TARGET_U64(yvex_deepseek_tensor_coverage_summary, "required_tensor_count",
               required_tensor_count),
    TARGET_U64(yvex_deepseek_tensor_coverage_summary, "matched_tensor_count", matched_tensor_count),
    TARGET_U64(yvex_deepseek_tensor_coverage_summary, "missing_tensor_count", missing_count),
    TARGET_U64(yvex_deepseek_tensor_coverage_summary, "ambiguous_tensor_count", ambiguous_count),
    TARGET_U64(yvex_deepseek_tensor_coverage_summary, "unexpected_tensor_count", unexpected_count),
};
static const yvex_cli_field_spec coverage_summary_tail[] = {
    TARGET_U64(yvex_deepseek_tensor_coverage_summary, "header_scan_count", header_scan_count),
    TARGET_U64(yvex_deepseek_tensor_coverage_summary, "payload_bytes_read", payload_bytes_read),
    TARGET_U64(yvex_deepseek_tensor_coverage_summary, "source_lookup_count", source_lookup_count),
    TARGET_U64(yvex_deepseek_tensor_coverage_summary, "source_collision_count",
               source_collision_count),
    TARGET_U64(yvex_deepseek_tensor_coverage_summary, "source_maximum_probe", source_maximum_probe),
    TARGET_HEX(yvex_deepseek_tensor_coverage_summary, "source_identity", source_identity),
    TARGET_HEX(yvex_deepseek_tensor_coverage_summary, "coverage_identity", coverage_identity),
};
static const yvex_cli_field_spec coverage_report_tail[] = {
    TARGET_TEXT(yvex_model_target_report, "next_required_row", next_row, "unknown"),
    TARGET_TEXT(yvex_model_target_report, "boundary", boundary, "unknown"),
};

#define MODEL_NESTED(key_name, nested_type, outer_member, nested_member, kind_name) \
    {key_name, kind_name, offsetof(yvex_deepseek_v4_model_spec, outer_member) + \
                         offsetof(nested_type, nested_member), NULL}
static const yvex_cli_field_spec architecture_model_fields[] = {
    TARGET_TEXT(yvex_deepseek_v4_model_spec, "target_id", target_id, "unknown"),
    TARGET_TEXT(yvex_deepseek_v4_model_spec, "family", family, "unknown"),
    TARGET_TEXT(yvex_deepseek_v4_model_spec, "architecture", architecture, "unknown"),
    TARGET_TEXT(yvex_deepseek_v4_model_spec, "repository", repository, "unknown"),
    TARGET_TEXT(yvex_deepseek_v4_model_spec, "revision", revision, "unknown"),
    TARGET_TEXT(yvex_deepseek_v4_model_spec, "verification_stage", verification_stage, "unknown"),
    TARGET_TEXT(yvex_deepseek_v4_model_spec, "paper_revision", paper_revision, "unknown"),
    TARGET_TEXT(yvex_deepseek_v4_model_spec, "sglang_revision", sglang_revision, "unknown"),
    TARGET_TEXT(yvex_deepseek_v4_model_spec, "vllm_revision", vllm_revision, "unknown"),
    TARGET_U64(yvex_deepseek_v4_model_spec, "hidden_size", hidden_size),
    TARGET_U64(yvex_deepseek_v4_model_spec, "vocabulary_size", vocabulary_size),
    TARGET_U64(yvex_deepseek_v4_model_spec, "maximum_context", maximum_context),
    TARGET_U64(yvex_deepseek_v4_model_spec, "main_layer_count", main_layer_count),
    TARGET_U64(yvex_deepseek_v4_model_spec, "auxiliary_layer_count", auxiliary_layer_count),
    TARGET_U64(yvex_deepseek_v4_model_spec, "swa_layer_count", swa_layer_count),
    TARGET_U64(yvex_deepseek_v4_model_spec, "csa_layer_count", csa_layer_count),
    TARGET_U64(yvex_deepseek_v4_model_spec, "hca_layer_count", hca_layer_count),
    TARGET_U64(yvex_deepseek_v4_model_spec, "hash_router_layer_count", hash_router_layer_count),
    TARGET_U64(yvex_deepseek_v4_model_spec, "learned_router_layer_count", learned_router_layer_count),
    MODEL_NESTED("dspark_block_size", yvex_deepseek_v4_dspark_spec, dspark,
                 block_size, YVEX_CLI_FIELD_U64),
    MODEL_NESTED("dspark_noise_token_id", yvex_deepseek_v4_dspark_spec, dspark,
                 noise_token_id, YVEX_CLI_FIELD_U64),
    MODEL_NESTED("dspark_draft_layer_count", yvex_deepseek_v4_dspark_spec, dspark,
                 draft_layer_count, YVEX_CLI_FIELD_U64),
    MODEL_NESTED("dspark_markov_rank", yvex_deepseek_v4_dspark_spec, dspark,
                 markov_rank, YVEX_CLI_FIELD_U64),
    MODEL_NESTED("dspark_confidence_available", yvex_deepseek_v4_dspark_spec, dspark,
                 confidence_available, YVEX_CLI_FIELD_BOOL),
    MODEL_NESTED("mhc_residual_streams", yvex_deepseek_v4_mhc_spec, final_mhc,
                 residual_streams, YVEX_CLI_FIELD_U64),
    MODEL_NESTED("mhc_expanded_width", yvex_deepseek_v4_mhc_spec, final_mhc,
                 expanded_width, YVEX_CLI_FIELD_U64),
    MODEL_NESTED("mhc_mixing_rows", yvex_deepseek_v4_mhc_spec, final_mhc,
                 mixing_rows, YVEX_CLI_FIELD_U64),
    MODEL_NESTED("mhc_mixing_columns", yvex_deepseek_v4_mhc_spec, final_mhc,
                 mixing_columns, YVEX_CLI_FIELD_U64),
    MODEL_NESTED("mhc_sinkhorn_iterations", yvex_deepseek_v4_mhc_spec, final_mhc,
                 sinkhorn_iterations, YVEX_CLI_FIELD_U64),
    TARGET_BOOL(yvex_deepseek_v4_model_spec, "final_mhc_post_required", final_mhc_post_required),
    TARGET_BOOL(yvex_deepseek_v4_model_spec, "final_mhc_head_required", final_mhc_head_required),
    TARGET_BOOL(yvex_deepseek_v4_model_spec, "final_norm_after_mhc_head", final_norm_after_mhc_head),
    MODEL_NESTED("tokenizer_class", yvex_deepseek_v4_tokenizer_spec, tokenizer,
                 tokenizer_class, YVEX_CLI_FIELD_TEXT_ARRAY),
    MODEL_NESTED("tokenizer_model_type", yvex_deepseek_v4_tokenizer_spec, tokenizer,
                 model_type, YVEX_CLI_FIELD_TEXT_ARRAY),
    MODEL_NESTED("tokenizer_vocabulary_size", yvex_deepseek_v4_tokenizer_spec, tokenizer,
                 vocabulary_size, YVEX_CLI_FIELD_U64),
    MODEL_NESTED("tokenizer_base_vocab_entries", yvex_deepseek_v4_tokenizer_spec, tokenizer,
                 base_vocab_entries, YVEX_CLI_FIELD_U64),
    MODEL_NESTED("tokenizer_added_token_entries", yvex_deepseek_v4_tokenizer_spec, tokenizer,
                 added_token_entries, YVEX_CLI_FIELD_U64),
    MODEL_NESTED("bos_token_id", yvex_deepseek_v4_tokenizer_spec, tokenizer,
                 bos_token_id, YVEX_CLI_FIELD_U64),
    MODEL_NESTED("eos_token_id", yvex_deepseek_v4_tokenizer_spec, tokenizer,
                 eos_token_id, YVEX_CLI_FIELD_U64),
    MODEL_NESTED("output_head_required", yvex_deepseek_v4_output_spec, output,
                 required, YVEX_CLI_FIELD_BOOL),
    MODEL_NESTED("output_head_tied", yvex_deepseek_v4_output_spec, output,
                 tied_to_embedding, YVEX_CLI_FIELD_BOOL),
};
static const yvex_cli_field_spec architecture_source_fields[] = {
    TARGET_U64(yvex_deepseek_v4_model_spec, "source_header_scan_count", source_header_scan_count),
    TARGET_U64(yvex_deepseek_v4_model_spec, "source_header_tensor_count", source_header_tensor_count),
    TARGET_U64(yvex_deepseek_v4_model_spec, "source_payload_bytes_read", source_payload_bytes_read),
};
static const yvex_cli_field_spec architecture_report_tail[] = {
    TARGET_TEXT(yvex_model_target_report, "next_required_row", next_row, "unknown"),
    TARGET_TEXT(yvex_model_target_report, "boundary", boundary, "unknown"),
};
#undef MODEL_NESTED
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

static int model_target_render_deepseek_map(
    FILE *fp,
    yvex_model_target_render_mode mode,
    const yvex_model_target_report *report)
{
    const yvex_deepseek_gguf_map *map =
        (const yvex_deepseek_gguf_map *)report->family_lowering;
    const yvex_deepseek_gguf_map_summary *summary =
        yvex_model_register_deepseek_v4()->lowering.summary(map);
    int rc = 0;

    if (!summary) return 0;
    if (mode == YVEX_MODEL_TARGET_OUTPUT_JSON) {
        return yvex_cli_out_writef(
            fp,
            "{\"status\":\"%s\",\"target_id\":\"%s\",\"source_contributions\":%llu,\"descriptors\":%llu,"
                "\"trunk_descriptors\":%llu,\"draft_descriptors\":%llu,\"pinned_standard_names\":%llu,"
                "\"extension_names\":%llu,\"metadata\":%llu,\"header_scans\":%llu,\"payload_bytes_read\":%llu,"
                "\"mapping_identity\":\"%016llx\",\"artifact\":\"not-produced\",\"runtime\":\"unsupported\","
                "\"generation\":\"unsupported\",\"next\":\"%s\"}\n",
            report->status, report->target_id,
            summary->source_contribution_count, summary->descriptor_count,
            summary->trunk_descriptor_count, summary->draft_descriptor_count,
            summary->pinned_standard_count, summary->extension_count,
            summary->metadata_count, summary->header_scan_count,
            summary->payload_bytes_read, summary->mapping_identity,
            report->next_row);
    }
    if (mode == YVEX_MODEL_TARGET_OUTPUT_TABLE) {
        rc |= yvex_cli_out_writef(
            fp, "TARGET  STATUS  SOURCES  GGUF  TRUNK  DRAFT  METADATA  PAYLOAD  NEXT\n");
        rc |= yvex_cli_out_writef(
            fp, "%s  %s  %llu  %llu  %llu  %llu  %llu  %llu  %s\n",
            report->target_id, report->status,
            summary->source_contribution_count, summary->descriptor_count,
            summary->trunk_descriptor_count, summary->draft_descriptor_count,
            summary->metadata_count, summary->payload_bytes_read,
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
             collection < YVEX_TENSOR_COLLECTION_COUNT;
             ++collection) {
            rc |= yvex_cli_out_writef(
                fp, "collection_%u_count: %llu\n", collection,
                summary->collection_counts[collection]);
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
        summary->source_contribution_count, summary->descriptor_count,
        summary->trunk_descriptor_count, summary->draft_descriptor_count,
        summary->metadata_count);
    rc |= yvex_cli_out_writef(
        fp, "evidence: header-scans=%llu payload-bytes=%llu identity=%016llx\n",
        summary->header_scan_count, summary->payload_bytes_read,
        summary->mapping_identity);
    rc |= yvex_cli_out_writef(fp, "next: %s\n", report->next_row);
    rc |= yvex_cli_out_writef(fp, "boundary: %s\n", report->boundary);
    return rc < 0 ? rc : 0;
}

static int model_target_render_deepseek_coverage(
    FILE *fp,
    yvex_model_target_render_mode mode,
    const yvex_model_target_report *report)
{
    const yvex_deepseek_tensor_coverage *coverage =
        (const yvex_deepseek_tensor_coverage *)report->family_coverage;
    const yvex_deepseek_tensor_coverage_summary *summary =
        yvex_model_register_deepseek_v4()->coverage.summary(coverage);
    int rc = 0;

    if (!summary) return 0;
    if (mode == YVEX_MODEL_TARGET_OUTPUT_JSON) {
        return yvex_cli_out_writef(
            fp,
            "{\"status\":\"%s\",\"target_id\":\"%s\",\"source_tensors\":%llu,\"required_tensors\":%llu,"
                "\"matched_tensors\":%llu,\"missing\":%llu,\"ambiguous\":%llu,\"unexpected\":%llu,"
                "\"header_scans\":%llu,\"payload_bytes_read\":%llu,\"coverage_identity\":\"%016llx\","
                "\"mapping\":\"blocked\",\"runtime\":\"unsupported\",\"generation\":\"unsupported\",\"next\":"
                "\"%s\"}\n",
            report->status, report->target_id, summary->source_tensor_count,
            summary->required_tensor_count, summary->matched_tensor_count,
            summary->missing_count, summary->ambiguous_count,
            summary->unexpected_count,
            summary->header_scan_count, summary->payload_bytes_read,
            summary->coverage_identity, report->next_row);
    }
    if (mode == YVEX_MODEL_TARGET_OUTPUT_TABLE) {
        rc |= yvex_cli_out_writef(
            fp, "TARGET  STATUS  SOURCE  REQUIRED  MATCHED  MISSING  UNEXPECTED  NEXT\n");
        rc |= yvex_cli_out_writef(
            fp, "%s  %s  %llu  %llu  %llu  %llu  %llu  %s\n",
            report->target_id, report->status, summary->source_tensor_count,
            summary->required_tensor_count, summary->matched_tensor_count,
            summary->missing_count, summary->unexpected_count,
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
             collection < YVEX_TENSOR_COLLECTION_COUNT;
             ++collection) {
            rc |= yvex_cli_out_writef(
                fp, "collection_%s_count: %llu\n",
                yvex_model_register_deepseek_v4()->coverage.collection_name(
                    (yvex_tensor_collection)collection),
                summary->collection_counts[collection]);
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
        summary->source_tensor_count, summary->required_tensor_count,
        summary->matched_tensor_count, summary->missing_count,
        summary->unexpected_count);
    rc |= yvex_cli_out_writef(fp, "identity: %016llx\n",
                              summary->coverage_identity);
    rc |= yvex_cli_out_writef(fp, "next: %s\n", report->next_row);
    rc |= yvex_cli_out_writef(fp, "boundary: %s\n", report->boundary);
    return rc < 0 ? rc : 0;
}

static int model_target_render_deepseek_normal(
    FILE *fp,
    const yvex_model_target_report *report,
    const yvex_deepseek_v4_ir *ir,
    const yvex_deepseek_v4_model_spec *model)
{
    const yvex_deepseek_v4_layer_spec *first =
        yvex_model_register_deepseek_v4()->ir.layer_at(ir, 0u);
    int rc = 0;

    rc |= yvex_cli_out_writef(fp, "model-class: deepseek-v4-flash-dspark\n");
    rc |= yvex_cli_out_writef(fp, "target: %s\n", model->target_id);
    rc |= yvex_cli_out_writef(fp, "status: %s\n", report->status);
    rc |= yvex_cli_out_writef(
        fp, "topology: target-layers=%llu draft-layers=%llu hidden=%llu vocab=%llu context=%llu\n",
        model->main_layer_count, model->auxiliary_layer_count,
        model->hidden_size, model->vocabulary_size, model->maximum_context);
    rc |= yvex_cli_out_writef(
        fp, "attention: swa=%llu csa=%llu hca=%llu heads=%llu kv_heads=%llu head_dim=%llu rope_dim=%llu\n",
        model->swa_layer_count, model->csa_layer_count,
        model->hca_layer_count, first->query_heads, first->kv_heads,
        first->head_dimension, first->rope_head_dimension);
    rc |= yvex_cli_out_writef(
        fp, "routing: hash=%llu learned=%llu experts=%llu topk=%llu shared=%llu\n",
        model->hash_router_layer_count, model->learned_router_layer_count,
        first->moe.routed_experts, first->moe.experts_per_token,
        first->moe.shared_experts);
    rc |= yvex_cli_out_writef(
        fp, "mhc: streams=%llu expanded=%llu mixing_rows=%llu sinkhorn=%llu\n",
        model->final_mhc.residual_streams, model->final_mhc.expanded_width,
        model->final_mhc.mixing_rows, model->final_mhc.sinkhorn_iterations);
    rc |= yvex_cli_out_writef(
        fp, "dspark: block=%llu noise=%llu taps=%llu,%llu,%llu markov-rank=%llu confidence=%s\n",
        model->dspark.block_size, model->dspark.noise_token_id,
        model->dspark.target_layer_ids[0], model->dspark.target_layer_ids[1],
        model->dspark.target_layer_ids[2], model->dspark.markov_rank,
        model->dspark.confidence_available ? "available" : "unavailable");
    rc |= yvex_cli_out_writef(fp, "next: %s\n", report->next_row);
    rc |= yvex_cli_out_writef(fp, "boundary: %s\n", report->boundary);
    return rc < 0 ? rc : 0;
}

static int model_target_render_deepseek_table(
    FILE *fp,
    const yvex_model_target_report *report,
    const yvex_deepseek_v4_model_spec *model)
{
    int rc = 0;

    rc |= yvex_cli_out_writef(
        fp, "TARGET  STATUS  TARGET_LAYERS  DRAFT_LAYERS  SWA  CSA  HCA  HASH  LEARNED  NEXT\n");
    rc |= yvex_cli_out_writef(
        fp, "%s  %s  %llu  %llu  %llu  %llu  %llu  %llu  %llu  %s\n",
        model->target_id, report->status, model->main_layer_count,
        model->auxiliary_layer_count, model->swa_layer_count,
        model->csa_layer_count, model->hca_layer_count,
        model->hash_router_layer_count, model->learned_router_layer_count,
        report->next_row);
    return rc < 0 ? rc : 0;
}

static int model_target_render_deepseek_audit(
    FILE *fp,
    const yvex_model_target_report *report,
    const yvex_deepseek_v4_ir *ir,
    const yvex_deepseek_v4_model_spec *model)
{
    unsigned long long i;
    int rc = 0;

    rc |= yvex_cli_out_writef(fp, "architecture_ir_status: %s\n", report->status);
    rc |= yvex_cli_out_fields(fp, model, architecture_model_fields,
                              sizeof(architecture_model_fields) /
                                  sizeof(architecture_model_fields[0]));
    rc |= yvex_cli_out_writef(
        fp, "source_weight_dtype: %s\n",
        yvex_model_register_deepseek_v4()->ir.source_weight_dtype_name(
            model->source_constraint.weight_dtype));
    rc |= yvex_cli_out_writef(
        fp, "source_expert_dtype: %s\n",
        yvex_model_register_deepseek_v4()->ir.source_expert_dtype_name(
            model->source_constraint.expert_dtype));
    rc |= yvex_cli_out_writef(
        fp, "source_quantization: %s block=%llux%llu\n",
        yvex_model_register_deepseek_v4()->ir.source_quantization_name(
            model->source_constraint.quantization),
        model->source_constraint.quant_block_rows,
        model->source_constraint.quant_block_columns);
    rc |= yvex_cli_out_fields(fp, model, architecture_source_fields,
                              sizeof(architecture_source_fields) /
                                  sizeof(architecture_source_fields[0]));
    for (i = 0u; i < yvex_model_register_deepseek_v4()->ir.layer_count(ir); ++i) {
        const yvex_deepseek_v4_layer_spec *layer =
            yvex_model_register_deepseek_v4()->ir.layer_at(ir, i);
        rc |= yvex_cli_out_writef(
            fp,
            "layer_%llu: attention=%s ratio=%llu kv=%s router=%s mhc_entry=%s\n",
            layer->layer_index,
            attention_class_name(layer->attention_class),
            layer->compression_ratio,
            yvex_model_register_deepseek_v4()->ir.kv_name(layer->kv.class_id),
            yvex_model_register_deepseek_v4()->ir.router_name(layer->moe.router_class),
            layer->mhc.entry == YVEX_DEEPSEEK_V4_MHC_STANDALONE_PRE
                ? "standalone-pre"
                : "fused-prior-post-pre");
    }
    for (i = 0u; i < yvex_model_register_deepseek_v4()->ir.auxiliary_count(ir); ++i) {
        const yvex_deepseek_v4_auxiliary_spec *aux =
            yvex_model_register_deepseek_v4()->ir.auxiliary_at(ir, i);
        rc |= yvex_cli_out_writef(
            fp,
            "draft_%llu: layer=%llu attention=%s ratio=%llu router=%s "
            "feature_projection=%s markov=%s confidence=%s shared_head=%s\n",
            aux->predictor_index, aux->layer.layer_index,
            attention_class_name(aux->layer.attention_class),
            aux->layer.compression_ratio,
            yvex_model_register_deepseek_v4()->ir.router_name(aux->layer.moe.router_class),
            aux->has_feature_projection ? "true" : "false",
            aux->has_markov_head ? "true" : "false",
            aux->has_confidence_head ? "true" : "false",
            aux->shares_output_head ? "true" : "false");
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
    const yvex_deepseek_v4_model_spec *model)
{
    return yvex_cli_out_writef(
        fp,
        "{\"status\":\"%s\",\"target_id\":\"%s\",\"repository\":\"%s\",\"revision\":\"%s\",\"layers\":%llu,"
            "\"draft_layers\":%llu,\"hidden_size\":%llu,\"vocabulary_size\":%llu,\"context\":%llu,\"attention\":"
            "{\"swa\":%llu,\"csa\":%llu,\"hca\":%llu},\"routing\":{\"hash\":%llu,\"learned\":%llu},"
            "\"mhc_streams\":%llu,\"payload_bytes_read\":%llu,\"runtime\":\"unsupported\",\"generation\":"
            "\"unsupported\",\"next\":\"%s\"}\n",
        report->status, model->target_id, model->repository, model->revision,
        model->main_layer_count, model->auxiliary_layer_count,
        model->hidden_size, model->vocabulary_size, model->maximum_context,
        model->swa_layer_count, model->csa_layer_count,
        model->hca_layer_count, model->hash_router_layer_count,
        model->learned_router_layer_count,
        model->final_mhc.residual_streams, model->source_payload_bytes_read,
        report->next_row);
}

static int model_target_render_deepseek_ir(
    FILE *fp,
    yvex_model_target_render_mode mode,
    const yvex_model_target_report *report)
{
    const yvex_deepseek_v4_ir *ir =
        (const yvex_deepseek_v4_ir *)report->family_architecture;
    const yvex_deepseek_v4_model_spec *model =
        yvex_model_register_deepseek_v4()->ir.model(ir);

    if (!model) return 0;
    if (mode == YVEX_MODEL_TARGET_OUTPUT_TABLE) {
        return model_target_render_deepseek_table(fp, report, model);
    }
    if (mode == YVEX_MODEL_TARGET_OUTPUT_AUDIT) {
        return model_target_render_deepseek_audit(
            fp, report, ir, model);
    }
    if (mode == YVEX_MODEL_TARGET_OUTPUT_JSON) {
        return model_target_render_deepseek_json(fp, report, model);
    }
    return model_target_render_deepseek_normal(
        fp, report, ir, model);
}

static int model_target_render_minimax(
    FILE *fp,
    yvex_model_target_render_mode mode,
    const yvex_model_target_report *report)
{
    const yvex_minimax_h3_api *api = yvex_model_register_minimax_h3();
    const yvex_minimax_h3_target *target =
        (const yvex_minimax_h3_target *)report->family_architecture;
    const yvex_minimax_h3_summary *summary = api->summary(target);
    const yvex_minimax_h3_architecture *architecture = api->architecture(target);
    const yvex_transform_ir_summary *transformation =
        yvex_transform_ir_summary_get(
            (const yvex_transform_ir *)report->family_transformation);
    int rc = 0;

    if (!summary || !architecture || !transformation) return 0;
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
            report->status, report->target_id, YVEX_MINIMAX_H3_REPOSITORY,
            YVEX_MINIMAX_H3_REVISION, YVEX_MINIMAX_H3_SUBTREE,
            summary->component_count, summary->weighted_component_count,
            summary->phase_edge_count,
            summary->shard_count, summary->tensor_count, summary->element_count,
            summary->payload_bytes, summary->source_snapshot_identity,
            summary->component_manifest_identity, summary->phase_dag_identity,
            summary->architecture_identity,
            summary->role_map_identity, transformation->transform_identity,
            report->family_derivation_identity, transformation->payload_bytes_read,
            report->next_row);
    }
    if (mode == YVEX_MODEL_TARGET_OUTPUT_TABLE) {
        rc |= yvex_cli_out_writef(
            fp, "TARGET  STATUS  COMPONENTS  SHARDS  TENSORS  PAYLOAD  EXECUTION  NEXT\n");
        rc |= yvex_cli_out_writef(
            fp, "%s  %s  %llu  %llu  %llu  %llu  unsupported  %s\n",
            report->target_id, report->status, summary->component_count,
            summary->shard_count, summary->tensor_count, summary->payload_bytes,
            report->next_row);
        return rc < 0 ? rc : 0;
    }
    if (mode == YVEX_MODEL_TARGET_OUTPUT_AUDIT) {
        unsigned long long index;

        rc |= yvex_cli_out_writef(fp, "status: %s\n", report->status);
        rc |= yvex_cli_out_writef(fp, "target_id: %s\nfamily: minimax-h3\n", report->target_id);
        rc |= yvex_cli_out_writef(fp, "repository: %s\nrevision: %s\nsubtree: %s\n",
                                  YVEX_MINIMAX_H3_REPOSITORY,
                                  YVEX_MINIMAX_H3_REVISION,
                                  YVEX_MINIMAX_H3_SUBTREE);
        rc |= yvex_cli_out_writef(
            fp, "source_verified: true\ncomponents: %llu\nweighted_components: %llu\n"
                "shards: %llu\ntensors: %llu\nelements: %llu\npayload_bytes: %llu\n",
            summary->component_count, summary->weighted_component_count,
            summary->shard_count, summary->tensor_count, summary->element_count,
            summary->payload_bytes);
        for (index = 0u; index < summary->component_count; ++index) {
            const yvex_minimax_h3_component *component = api->component_at(target, index);
            if (component) {
                rc |= yvex_cli_out_writef(
                    fp, "component_%llu: id=%s weighted=%s shards=%llu tensors=%llu "
                        "phase=%u release_after_phase=%s identity=%s\n",
                    index, component->canonical_id, component->weighted ? "true" : "false",
                    component->shard_count, component->tensor_count,
                    (unsigned int)component->phase,
                    component->release_after_phase ? "true" : "false",
                    component->identity);
            }
        }
        for (index = 0u; index < summary->phase_edge_count; ++index) {
            const yvex_minimax_h3_phase_edge *edge = api->phase_edge_at(index);
            if (edge) {
                rc |= yvex_cli_out_writef(
                    fp, "phase_edge_%llu: from=%u to=%u data=%u lifetime=%u\n",
                    index, (unsigned int)edge->source_phase,
                    (unsigned int)edge->destination_phase, edge->data_classes,
                    (unsigned int)edge->lifetime);
            }
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
                                  transformation->transform_identity);
        rc |= yvex_cli_out_writef(fp, "derivation_identity: %s\n",
                                  report->family_derivation_identity);
        rc |= yvex_cli_out_writef(
            fp, "unresolved_requirements_identity: %s\npayload_execution_bytes: %llu\n",
            summary->unresolved_requirements_identity,
            transformation->payload_bytes_read);
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
        YVEX_MINIMAX_H3_REPOSITORY, YVEX_MINIMAX_H3_REVISION,
        YVEX_MINIMAX_H3_SUBTREE, summary->component_count,
        summary->shard_count, summary->tensor_count, summary->payload_bytes);
    rc |= yvex_cli_out_writef(
        fp, "identity: architecture=%s phases=%s roles=%s transform=%s\n",
        summary->architecture_identity, summary->phase_dag_identity,
        summary->role_map_identity,
        transformation->transform_identity);
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
    if (report->family_lowering) {
        return model_target_render_deepseek_map(fp, mode, report);
    }
    if (report->family_coverage) {
        return model_target_render_deepseek_coverage(fp, mode, report);
    }
    if (report->family_architecture) {
        if (report->family_architecture_kind ==
            YVEX_MODEL_TARGET_FAMILY_ARCHITECTURE_MINIMAX_H3)
            return model_target_render_minimax(fp, mode, report);
        return model_target_render_deepseek_ir(fp, mode, report);
    }
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
