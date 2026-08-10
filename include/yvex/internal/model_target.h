/*
 * Target discovery and reporting exchange architecture signatures, role coverage, and typed
 * blockers through this ABI. Report presence alone does not establish implementation support.
 */
#ifndef INCLUDE_YVEX_INTERNAL_MODEL_TARGET_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_MODEL_TARGET_H_INCLUDED

#include <stddef.h>
#include <yvex/core.h>
#include <yvex/internal/source.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Target report rows. */
#define YVEX_MODEL_TARGET_TEXT_CAP 512u
#define YVEX_MODEL_TARGET_ROW_CAP 384u
#define YVEX_MODEL_TARGET_TABLE_COL_CAP 8u
#define YVEX_MODEL_TARGET_TABLE_ROW_CAP 128u
typedef enum {
    YVEX_MODEL_TARGET_OUTPUT_NORMAL = 0,
    YVEX_MODEL_TARGET_OUTPUT_TABLE,
    YVEX_MODEL_TARGET_OUTPUT_AUDIT,
    YVEX_MODEL_TARGET_OUTPUT_JSON
} yvex_model_target_render_mode;
typedef enum {
    YVEX_MODEL_TARGET_COMMAND_HELP = 0,
    YVEX_MODEL_TARGET_COMMAND_CLASSES,
    YVEX_MODEL_TARGET_COMMAND_LIST,
    YVEX_MODEL_TARGET_COMMAND_DECISION,
    YVEX_MODEL_TARGET_COMMAND_CANDIDATE,
    YVEX_MODEL_TARGET_COMMAND_DENSE_CANDIDATE,
    YVEX_MODEL_TARGET_COMMAND_QWEN_METAL,
    YVEX_MODEL_TARGET_COMMAND_CLASS_PROFILE,
    YVEX_MODEL_TARGET_COMMAND_TENSOR_COLLECTION,
    YVEX_MODEL_TARGET_COMMAND_TENSOR_MAP,
    YVEX_MODEL_TARGET_COMMAND_TOKENIZER_MAP,
    YVEX_MODEL_TARGET_COMMAND_MISSING_ROLES,
    YVEX_MODEL_TARGET_COMMAND_QUANT_POLICY,
    YVEX_MODEL_TARGET_COMMAND_INSPECT,
    YVEX_MODEL_TARGET_COMMAND_UNKNOWN
} yvex_model_target_command_kind;
typedef enum {
    YVEX_MODEL_TARGET_FAMILY_ARCHITECTURE_NONE = 0,
    YVEX_MODEL_TARGET_FAMILY_ARCHITECTURE_DEEPSEEK,
    YVEX_MODEL_TARGET_FAMILY_ARCHITECTURE_MINIMAX_H3
} yvex_model_target_family_architecture_kind;
typedef struct {
    char value[YVEX_MODEL_TARGET_TEXT_CAP];
} yvex_model_target_text_value;
typedef struct {
    unsigned int column_count;
    char columns[YVEX_MODEL_TARGET_TABLE_COL_CAP][YVEX_MODEL_TARGET_TEXT_CAP];
} yvex_model_target_table_row;
/* Family projections expose report facts without lending concrete family representations. */
#define YVEX_MODEL_TARGET_COLLECTION_CAP YVEX_MODEL_TARGET_TABLE_ROW_CAP
#define YVEX_MODEL_TARGET_LAYER_CAP YVEX_MODEL_TARGET_TABLE_ROW_CAP
#define YVEX_MODEL_TARGET_COMPONENT_CAP YVEX_MODEL_TARGET_TABLE_ROW_CAP
#define YVEX_MODEL_TARGET_EDGE_CAP YVEX_MODEL_TARGET_TABLE_ROW_CAP
#define YVEX_MODEL_TARGET_FEATURE_CAP YVEX_MODEL_TARGET_TABLE_ROW_CAP
typedef enum {
    YVEX_MODEL_TARGET_DETAIL_NONE = 0,
    YVEX_MODEL_TARGET_DETAIL_TENSOR_MAP,
    YVEX_MODEL_TARGET_DETAIL_TENSOR_COVERAGE,
    YVEX_MODEL_TARGET_DETAIL_MODEL_ARCHITECTURE,
    YVEX_MODEL_TARGET_DETAIL_COMPOSITE_ARCHITECTURE
} yvex_model_target_detail_kind;
typedef struct {
    char name[32];
    unsigned long long count;
} yvex_model_target_named_count;
typedef struct {
    unsigned long long source_contributions, descriptors, trunk_descriptors;
    unsigned long long draft_descriptors, pinned_standard_names, semantic_standard_names;
    unsigned long long extension_names, metadata, header_scans, payload_bytes;
    unsigned long long source_identity, coverage_identity, mapping_identity;
    unsigned int collection_count;
    yvex_model_target_named_count collections[YVEX_MODEL_TARGET_COLLECTION_CAP];
} yvex_model_target_map_projection;
typedef struct {
    unsigned long long source_tensors, required_tensors, matched_tensors;
    unsigned long long missing_tensors, ambiguous_tensors, unexpected_tensors;
    unsigned long long header_scans, payload_bytes, source_lookups;
    unsigned long long source_collisions, source_maximum_probe;
    unsigned long long source_identity, coverage_identity;
    unsigned int collection_count;
    yvex_model_target_named_count collections[YVEX_MODEL_TARGET_COLLECTION_CAP];
} yvex_model_target_coverage_projection;
typedef struct {
    unsigned long long index, compression_ratio;
    char attention[16], kv[32], router[32], mhc_entry[32];
} yvex_model_target_layer_projection;
typedef struct {
    unsigned long long predictor_index, layer_index, compression_ratio;
    char attention[16], router[32];
    int feature_projection, markov, confidence, shared_head;
} yvex_model_target_draft_projection;
typedef struct {
    char target_id[128], family[64], architecture[64], repository[128], revision[64];
    char verification_stage[64], paper_revision[64], sglang_revision[64], vllm_revision[64];
    char source_weight_dtype[32], source_expert_dtype[32], source_quantization[32];
    char tokenizer_class[64], tokenizer_model_type[64];
    unsigned long long hidden_size, vocabulary_size, maximum_context;
    unsigned long long target_layers, draft_layers, swa_layers, csa_layers, hca_layers;
    unsigned long long hash_router_layers, learned_router_layers;
    unsigned long long query_heads, kv_heads, head_dimension, rope_head_dimension;
    unsigned long long routed_experts, experts_per_token, shared_experts;
    unsigned long long dspark_block_size, dspark_noise_token_id, dspark_markov_rank;
    unsigned long long dspark_feature_layers[YVEX_MODEL_TARGET_FEATURE_CAP];
    unsigned int dspark_feature_layer_count;
    int dspark_confidence_available;
    unsigned long long mhc_residual_streams, mhc_expanded_width, mhc_mixing_rows;
    unsigned long long mhc_mixing_columns, mhc_sinkhorn_iterations;
    int final_mhc_post_required, final_mhc_head_required, final_norm_after_mhc_head;
    unsigned long long tokenizer_vocabulary_size, tokenizer_base_vocab_entries;
    unsigned long long tokenizer_added_token_entries, bos_token_id, eos_token_id;
    int output_head_required, output_head_tied;
    unsigned long long source_quant_block_rows, source_quant_block_columns;
    unsigned long long source_header_scans, source_header_tensors, source_payload_bytes;
    unsigned int layer_count, draft_count;
    yvex_model_target_layer_projection layers[YVEX_MODEL_TARGET_LAYER_CAP];
    yvex_model_target_draft_projection drafts[YVEX_MODEL_TARGET_LAYER_CAP];
} yvex_model_target_architecture_projection;
typedef struct {
    char canonical_id[64], identity[65];
    unsigned long long shards, tensors;
    unsigned int phase;
    int weighted, release_after_phase;
} yvex_model_target_component_projection;
typedef struct {
    unsigned int source_phase, destination_phase, data_classes, lifetime;
} yvex_model_target_edge_projection;
typedef struct {
    char repository[128], revision[64], subtree[128];
    char source_snapshot_identity[65], component_manifest_identity[65];
    char phase_dag_identity[65], architecture_identity[65], role_map_identity[65];
    char transformation_identity[65], unresolved_requirements_identity[65];
    unsigned long long components, weighted_components, phase_edges, shards;
    unsigned long long tensors, elements, payload_bytes, payload_execution_bytes;
    unsigned int component_count, edge_count;
    yvex_model_target_component_projection component[YVEX_MODEL_TARGET_COMPONENT_CAP];
    yvex_model_target_edge_projection edge[YVEX_MODEL_TARGET_EDGE_CAP];
} yvex_model_target_composite_projection;
typedef union {
    yvex_model_target_map_projection map;
    yvex_model_target_coverage_projection coverage;
    yvex_model_target_architecture_projection architecture;
    yvex_model_target_composite_projection composite;
} yvex_model_target_detail;
typedef struct yvex_model_target_request {
    yvex_model_target_command_kind kind;
    yvex_model_target_render_mode mode;
    int help_requested;
    char target_id[128];
    char release[64];
    char family[32];
    char models_root[512];
    char source_path[512];
    char role[64];
    char gate[64];
    char candidate_kind[64];
    char output_contract[32];
    int include_hardware;
    int include_backend;
    int include_source;
    int include_blockers;
    int include_next;
    int include_examples;
    int include_candidates;
    int include_pressure_targets;
    int include_critical_path;
    int include_requirements;
    int include_paths;
    int output_json;
    int strict;
    int write_sidecar;
    char sidecar_path[512];
} yvex_model_target_request;
typedef struct yvex_model_target_report {
    yvex_model_target_command_kind kind;
    yvex_model_target_render_mode mode;
    int help_requested;
    const char *status;
    char target_id[128];
    char family[32];
    char model[128];
    char target_class[128];
    char stage[128];
    char eligibility[128];
    char source_status[128];
    char artifact_status[128];
    char tensor_map_status[128];
    char qtype_policy_status[128];
    char runtime_status[128];
    char generation_status[128];
    char benchmark_status[128];
    char next_row[128];
    char boundary[256];
    char reason[256];
    yvex_model_target_text_value rows[YVEX_MODEL_TARGET_ROW_CAP];
    unsigned long row_count;
    yvex_model_target_text_value error_rows[64];
    unsigned long error_row_count;
    yvex_model_target_table_row table_rows[YVEX_MODEL_TARGET_TABLE_ROW_CAP];
    unsigned long table_row_count;
    yvex_model_target_detail_kind detail_kind;
    yvex_model_target_detail detail;
    void *family_architecture;
    yvex_model_target_family_architecture_kind family_architecture_kind;
    void *family_transformation;
    char family_derivation_identity[65];
    void *family_lowering;
    int exit_code;
} yvex_model_target_report;
typedef struct {
    char source_path[512];
    int source_present;
    unsigned long long tensors;
    unsigned long long embed;
    unsigned long long attn;
    unsigned long long mlp;
    unsigned long long norm;
    unsigned long long head;
    unsigned long long moe;
    unsigned long long layers;
} yvex_model_target_source_scan;
typedef struct {
    int source_requested;
    int source_directory_present;
    int header_present;
    int metadata_present;
    int attention_k_present;
    int output_head_present;
    int output_head_ambiguous;
    unsigned long f32_count;
    unsigned long f16_count;
    unsigned long bf16_count;
    unsigned long other_count;
    unsigned long tensor_count;
} yvex_model_target_source_profile;
typedef enum {
    YVEX_MODEL_TARGET_ROW_LITERAL = 0,
    YVEX_MODEL_TARGET_ROW_STRING,
    YVEX_MODEL_TARGET_ROW_ULONG,
    YVEX_MODEL_TARGET_ROW_U64,
    YVEX_MODEL_TARGET_ROW_INT
} yvex_model_target_row_kind;
typedef struct {
    yvex_model_target_row_kind kind;
    const char *format;
    size_t value_offset;
} yvex_model_target_row_spec;
typedef struct {
    const char *status;
    const char *target_id;
    const char *family;
    const char *model;
    const char *target_class;
    const char *stage;
    const char *eligibility;
    const char *source_status;
    const char *artifact_status;
    const char *tensor_map_status;
    const char *qtype_policy_status;
    const char *runtime_status;
    const char *generation_status;
    const char *benchmark_status;
    const char *next_row;
    const char *boundary;
    const char *reason;
} yvex_model_target_report_profile;
typedef struct {
    yvex_model_target_command_kind expected_kind;
    const char *kind_failure_status;
    const char *kind_failure_message;
    const char *required_target_operation;
    int reject_json;
} yvex_model_target_request_rules;
void yvex_model_target_report_prepare(
    yvex_model_target_report *report,
    const yvex_model_target_request *request,
    const yvex_model_target_report_profile *profile);
int yvex_model_target_report_add_row(yvex_model_target_report *report,
                                     const char *fmt,
                                     ...);
int yvex_model_target_report_project_family_detail(
    yvex_model_target_report *report, yvex_error *err);
void yvex_model_target_report_close_family_detail(
    yvex_model_target_report *report);
int yvex_model_target_family_class_profile_build(
    const yvex_model_target_request *request,
    yvex_model_target_report *report,
    int *handled,
    yvex_error *err);
int yvex_model_target_family_mapping_report_build(
    const yvex_model_target_request *request,
    yvex_model_target_report *report,
    yvex_error *err);
void yvex_model_target_report_add_rows(yvex_model_target_report *report,
                                       const char *const *rows,
                                       size_t row_count);
void yvex_model_target_report_project_rows(
    yvex_model_target_report *report,
    const yvex_model_target_row_spec *rows,
    size_t row_count,
    const void *facts);
int yvex_model_target_probe_source_path(
    const yvex_model_target_request *request,
    const char *family,
    const char *leaf,
    char *out,
    size_t cap);
int yvex_model_target_probe_directory(const char *path);
int yvex_model_target_probe_file(const char *path);
int yvex_model_target_probe_read(const char *path, char *out, size_t cap);
int yvex_model_target_probe_header(const char *path, char **out);
void yvex_model_target_scan_source(
    const yvex_model_target_request *request,
    const char *family,
    yvex_model_target_source_scan *scan);
void yvex_model_target_probe_source_profile(
    const yvex_model_target_request *request,
    const char *family,
    yvex_model_target_source_profile *profile);
int yvex_model_target_report_release_coverage(
    const yvex_model_target_request *request,
    yvex_model_target_report *report,
    const char *operation,
    const char *error_where,
    const char *success_status,
    const char *success_boundary,
    yvex_error *err);
int yvex_model_target_validate_request_shape(
    const yvex_model_target_request *request,
    yvex_model_target_report *report,
    const yvex_model_target_request_rules *rules,
    const char *release);
int yvex_model_target_validate_supported(
    const yvex_model_target_request *request,
    yvex_model_target_report *report,
    const char *operation,
    int contract_refusal_row);
int yvex_model_target_report_add_error(yvex_model_target_report *report,
                                       const char *fmt,
                                       ...);
int yvex_model_target_report_add_table_row(yvex_model_target_report *report,
                                           unsigned int column_count,
                                           const char *c0,
                                           const char *c1,
                                           const char *c2,
                                           const char *c3,
                                           const char *c4,
                                           const char *c5,
                                           const char *c6,
                                           const char *c7);
int yvex_model_target_report_build(const yvex_model_target_request *request,
                                   yvex_model_target_report *report,
                                   yvex_error *err);
int yvex_model_target_help_report_build(yvex_model_target_report *report,
                                        yvex_error *err);
void yvex_model_target_report_close(yvex_model_target_report *report);

/* Target catalog. */
typedef struct {
    const char *class_id;
    const char *capability_claim;
    const char *runtime_execution;
    const char *generation;
    const char *description;
} yvex_model_target_class_record;
typedef struct {
    const char *target_id;
    const char *family;
    const char *model;
    const char *target_class;
    const char *source_artifact_class;
    const char *target_artifact_class;
    const char *pressure_purpose;
    const char *tensor_set;
    const char *local_path_class;
    const char *source_footprint_class;
    const char *runtime_boundary;
    const char *runtime_execution;
    const char *generation;
    const char *external_reference;
} yvex_model_target_record;
int yvex_model_target_release_source_paths(
    const yvex_model_target_request *request,
    char *models_root,
    size_t models_root_cap,
    char *source_path,
    size_t source_path_cap);
const yvex_model_target_record *yvex_model_target_find(const char *target_id);
int yvex_model_target_catalog_report_build(
    const yvex_model_target_request *request,
    yvex_model_target_report *report,
    yvex_error *err);
int yvex_model_target_catalog_help_report_build(
    yvex_model_target_report *report,
    yvex_error *err);

const char *yvex_model_target_family_key(const char *target_id);
int yvex_model_target_supported_source_target(const char *target_id);
void yvex_model_target_report_common_tail(yvex_model_target_report *report);
void yvex_model_target_report_add_output_contract(yvex_model_target_report *report,
                                                  const char *report_name,
                                                  const char *mode);

/* Role-mapping admission. */
int yvex_mapping_gate_report_build(
    const yvex_model_target_request *request,
    yvex_model_target_report *report,
    yvex_error *err);

/* Missing-role diagnostics. */
int yvex_missing_role_report_build(
    const yvex_model_target_request *request,
    yvex_model_target_report *report,
    yvex_error *err);

/* Output-head mapping. */
int yvex_output_head_map_report_build(
    const yvex_model_target_request *request,
    yvex_model_target_report *report,
    yvex_error *err);

/* Qtype policy. */
int yvex_qtype_policy_report_build(
    const yvex_model_target_request *request,
    yvex_model_target_report *report,
    yvex_error *err);

/* Qtype support by tensor role. */
int yvex_qtype_role_support_report_build(
    const yvex_model_target_request *request,
    yvex_model_target_report *report,
    yvex_error *err);

/* Sidecar publication. */
typedef enum {
    YVEX_MODEL_TARGET_SIDECAR_TENSOR_MAP = 0,
    YVEX_MODEL_TARGET_SIDECAR_OUTPUT_HEAD = 1,
    YVEX_MODEL_TARGET_SIDECAR_TOKENIZER = 2
} yvex_model_target_sidecar_kind;
int yvex_model_target_write_sidecar(yvex_model_target_sidecar_kind kind,
                                    const char *path,
                                    const char *target_id,
                                    const char *family,
                                    const char *status,
                                    const char *coverage);

/* Tensor collections. */
int yvex_tensor_collection_report_build(
    const yvex_model_target_request *request,
    yvex_model_target_report *report,
    yvex_error *err);

/* Tensor naming. */
int yvex_tensor_naming_report_build(
    const yvex_model_target_request *request,
    yvex_model_target_report *report,
    yvex_error *err);

/* Tokenizer mapping. */
int yvex_tokenizer_map_report_build(
    const yvex_model_target_request *request,
    yvex_model_target_report *report,
    yvex_error *err);

#ifdef __cplusplus
}
#endif

#endif /* INCLUDE_YVEX_INTERNAL_MODEL_TARGET_H_INCLUDED */
