/* GGUF readers and writers expose container structure without promoting it to artifact admission.
 * Callers retain ownership of model semantics; this ABI owns format geometry and serialized
 * representation only. */
#ifndef YVEX_GGUF_H
#define YVEX_GGUF_H

#include <stddef.h>
#include <yvex/core.h>
#include <yvex/model.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct yvex_artifact yvex_artifact;

typedef enum {
    YVEX_GGUF_NAME_PINNED_STANDARD = 0,
    YVEX_GGUF_NAME_SEMANTIC_STANDARD,
    YVEX_GGUF_NAME_YVEX_EXTENSION
} yvex_gguf_name_provenance;

/* Container reader. */
#define YVEX_GGUF_MAGIC 0x46554747u
#define YVEX_GGUF_MAX_DIMS 4u

typedef struct {
    unsigned int version;
    unsigned long long metadata_count;
    unsigned long long tensor_count;
} yvex_gguf_header;

typedef struct {
    int is_gguf;
    yvex_gguf_header header;
} yvex_gguf_probe;

typedef struct yvex_gguf yvex_gguf;
typedef struct yvex_gguf_value yvex_gguf_value;

typedef enum {
    YVEX_GGUF_VALUE_UINT8 = 0,
    YVEX_GGUF_VALUE_INT8 = 1,
    YVEX_GGUF_VALUE_UINT16 = 2,
    YVEX_GGUF_VALUE_INT16 = 3,
    YVEX_GGUF_VALUE_UINT32 = 4,
    YVEX_GGUF_VALUE_INT32 = 5,
    YVEX_GGUF_VALUE_FLOAT32 = 6,
    YVEX_GGUF_VALUE_BOOL = 7,
    YVEX_GGUF_VALUE_STRING = 8,
    YVEX_GGUF_VALUE_ARRAY = 9,
    YVEX_GGUF_VALUE_UINT64 = 10,
    YVEX_GGUF_VALUE_INT64 = 11,
    YVEX_GGUF_VALUE_FLOAT64 = 12,
    YVEX_GGUF_VALUE_INVALID = 13
} yvex_gguf_value_type;

typedef enum {
    YVEX_GGUF_PARSE_SECTION_NONE = 0,
    YVEX_GGUF_PARSE_SECTION_FILE = 1,
    YVEX_GGUF_PARSE_SECTION_CONTAINER = 2,
    YVEX_GGUF_PARSE_SECTION_METADATA = 3,
    YVEX_GGUF_PARSE_SECTION_TENSOR_INFO = 4,
    YVEX_GGUF_PARSE_SECTION_QTYPE = 5,
    YVEX_GGUF_PARSE_SECTION_RANGE = 6,
    YVEX_GGUF_PARSE_SECTION_RESOURCE = 7
} yvex_gguf_parse_section;

typedef enum {
    YVEX_GGUF_PARSE_OK = 0,
    YVEX_GGUF_PARSE_INVALID_ARGUMENT = 1,
    YVEX_GGUF_PARSE_FILE_UNREADABLE = 2,
    YVEX_GGUF_PARSE_SHORT_READ = 3,
    YVEX_GGUF_PARSE_INVALID_MAGIC = 4,
    YVEX_GGUF_PARSE_UNSUPPORTED_VERSION = 5,
    YVEX_GGUF_PARSE_INVALID_COUNT = 6,
    YVEX_GGUF_PARSE_RESOURCE_LIMIT = 7,
    YVEX_GGUF_PARSE_MALFORMED_KEY = 8,
    YVEX_GGUF_PARSE_DUPLICATE_METADATA_KEY = 9,
    YVEX_GGUF_PARSE_UNSUPPORTED_METADATA_TYPE = 10,
    YVEX_GGUF_PARSE_MALFORMED_VALUE = 11,
    YVEX_GGUF_PARSE_MALFORMED_STRING = 12,
    YVEX_GGUF_PARSE_MALFORMED_ARRAY = 13,
    YVEX_GGUF_PARSE_INVALID_ALIGNMENT = 14,
    YVEX_GGUF_PARSE_MALFORMED_TENSOR_NAME = 15,
    YVEX_GGUF_PARSE_DUPLICATE_TENSOR_NAME = 16,
    YVEX_GGUF_PARSE_INVALID_RANK = 17,
    YVEX_GGUF_PARSE_INVALID_DIMENSION = 18,
    YVEX_GGUF_PARSE_REFUSED_QTYPE = 19,
    YVEX_GGUF_PARSE_OFFSET_OVERFLOW = 20,
    YVEX_GGUF_PARSE_INCOMPLETE_DIRECTORY = 21,
    YVEX_GGUF_PARSE_ALLOCATION_FAILURE = 22,
    YVEX_GGUF_PARSE_EMPTY_METADATA_KEY = 23,
    YVEX_GGUF_PARSE_EMPTY_TENSOR_NAME = 24,
    YVEX_GGUF_PARSE_ELEMENT_COUNT_OVERFLOW = 25,
    YVEX_GGUF_PARSE_ROW_COUNT_OVERFLOW = 26,
    YVEX_GGUF_PARSE_ROW_BYTE_OVERFLOW = 27,
    YVEX_GGUF_PARSE_TOTAL_BYTE_OVERFLOW = 28
} yvex_gguf_parse_code;

typedef struct {
    yvex_gguf_parse_code code;
    yvex_gguf_parse_section section;
    unsigned long long byte_offset;
    unsigned long long record_index;
    const char *reason;
} yvex_gguf_parse_result;

typedef struct {
    unsigned long long max_metadata_entries;
    unsigned long long max_tensor_entries;
    unsigned long long max_array_entries;
    unsigned long long max_total_array_entries;
    unsigned long long max_string_bytes;
    unsigned long long max_total_string_bytes;
    unsigned long long max_owned_bytes;
    unsigned long long max_structural_bytes;
    unsigned int max_array_depth;
} yvex_gguf_reader_options;

typedef struct {
    unsigned long long file_size;
    unsigned long long structural_bytes_read;
    unsigned long long payload_bytes_read;
    unsigned long long read_calls;
    unsigned long long owned_bytes;
    unsigned long long total_string_bytes;
    unsigned long long total_array_entries;
    unsigned long long directory_end_offset;
} yvex_gguf_reader_stats;

typedef struct {
    const unsigned char *data;
    unsigned long long len;
} yvex_gguf_bytes;

typedef struct {
    yvex_gguf_value_type element_type;
    unsigned long long count;
} yvex_gguf_array_info;

typedef struct {
    const char *name;
    unsigned long long name_len;
    unsigned int rank;
    unsigned long long dims[YVEX_GGUF_MAX_DIMS];
    unsigned int ggml_type;
    const char *ggml_type_name;
    unsigned long long relative_offset;
    unsigned long long absolute_offset;
    unsigned long long storage_bytes;
    unsigned long long absolute_end_offset;
    int range_addressable;
} yvex_gguf_tensor_info;

int yvex_gguf_probe_file(const yvex_artifact *artifact, yvex_gguf_probe *out, yvex_error *err);
int yvex_gguf_read_header(const yvex_artifact *artifact, yvex_gguf_header *out, yvex_error *err);

void yvex_gguf_reader_options_default(yvex_gguf_reader_options *options);
/*
 * Opens one immutable structural view. The view owns decoded metadata, names,
 * tensors, and indexes; it does not borrow artifact storage after return.
 * Every accessor result remains valid until yvex_gguf_close. */
int yvex_gguf_open_ex(yvex_gguf **out,
                      const yvex_artifact *artifact,
                      const yvex_gguf_reader_options *options,
                      yvex_gguf_parse_result *result,
                      yvex_error *err);
int yvex_gguf_open(yvex_gguf **out, const yvex_artifact *artifact, yvex_error *err);
/* Safe for null and for every successfully returned parsed view. */
void yvex_gguf_close(yvex_gguf *gguf);

const yvex_gguf_header *yvex_gguf_header_view(const yvex_gguf *gguf);
const char *yvex_gguf_value_type_name(yvex_gguf_value_type type);

unsigned long long yvex_gguf_metadata_count(const yvex_gguf *gguf);
const char *yvex_gguf_metadata_key(const yvex_gguf *gguf, unsigned long long index);
const yvex_gguf_value *yvex_gguf_metadata_value(const yvex_gguf *gguf, unsigned long long index);
const yvex_gguf_value *yvex_gguf_metadata_find(const yvex_gguf *gguf, const char *key);

yvex_gguf_value_type yvex_gguf_value_type_of(const yvex_gguf_value *value);
int yvex_gguf_value_as_u64(const yvex_gguf_value *value, unsigned long long *out);
int yvex_gguf_value_as_i64(const yvex_gguf_value *value, long long *out);
int yvex_gguf_value_as_f64(const yvex_gguf_value *value, double *out);
int yvex_gguf_value_as_bool(const yvex_gguf_value *value, int *out);
int yvex_gguf_value_as_string(const yvex_gguf_value *value, const char **data, unsigned long long *len);
int yvex_gguf_value_array_info(const yvex_gguf_value *value, yvex_gguf_array_info *out);
const yvex_gguf_value *yvex_gguf_value_array_at(const yvex_gguf_value *value, unsigned long long index);

unsigned long long yvex_gguf_tensor_count(const yvex_gguf *gguf);
const yvex_gguf_tensor_info *yvex_gguf_tensor_at(const yvex_gguf *gguf, unsigned long long index);
const yvex_gguf_tensor_info *yvex_gguf_tensor_find(const yvex_gguf *gguf, const char *name);
unsigned long long yvex_gguf_tensor_data_offset(const yvex_gguf *gguf);
unsigned int yvex_gguf_alignment(const yvex_gguf *gguf);
unsigned long long yvex_gguf_file_size(const yvex_gguf *gguf);
const yvex_gguf_reader_stats *yvex_gguf_reader_stats_view(const yvex_gguf *gguf);

/* Controlled emission. */
typedef struct yvex_gguf_emit_plan yvex_gguf_emit_plan;

typedef enum {
    YVEX_GGUF_EMIT_STATUS_UNKNOWN = 0,
    YVEX_GGUF_EMIT_STATUS_PLANNED,
    YVEX_GGUF_EMIT_STATUS_WRITTEN,
    YVEX_GGUF_EMIT_STATUS_FAILED
} yvex_gguf_emit_status;

typedef struct {
    const char *out_path;
    const char *template_path;
    const char *native_source_dir;
    const char *tensor_name;
    const char *target_name;
    const char *target_qtype;
    const char *model_name;
    const char *architecture;
    int transpose_2d;
    int overwrite;
} yvex_gguf_emit_options;

typedef struct {
    yvex_gguf_emit_status status;
    const char *out_path;
    const char *template_path;
    const char *model_name;
    const char *architecture;
    unsigned long long metadata_count;
    unsigned long long tensor_count;
    unsigned long long bytes_written;
    unsigned long long tensor_payload_bytes;
    unsigned long long alignment;
    int roundtrip_validated;
} yvex_gguf_emit_summary;

int yvex_gguf_emit_controlled(const yvex_gguf_emit_options *options,
                              yvex_gguf_emit_summary *summary_out,
                              yvex_error *err);

const char *yvex_gguf_emit_status_name(yvex_gguf_emit_status status);

/* Layout admission. */
#define YVEX_GGUF_LAYOUT_TENSOR_NAME_CAP 128u
#define YVEX_GGUF_LAYOUT_REASON_CAP 256u

typedef enum {
    YVEX_GGUF_LAYOUT_OK = 0,
    YVEX_GGUF_LAYOUT_INVALID_ARGUMENT = 1,
    YVEX_GGUF_LAYOUT_INVALID_ALIGNMENT_TYPE = 2,
    YVEX_GGUF_LAYOUT_INVALID_ALIGNMENT_VALUE = 3,
    YVEX_GGUF_LAYOUT_ALIGNMENT_NOT_POWER_OF_TWO = 4,
    YVEX_GGUF_LAYOUT_ALIGNMENT_OVERFLOW = 5,
    YVEX_GGUF_LAYOUT_STORAGE_REFUSED = 6,
    YVEX_GGUF_LAYOUT_FIRST_OFFSET_NOT_ZERO = 7,
    YVEX_GGUF_LAYOUT_OFFSET_REVERSED = 8,
    YVEX_GGUF_LAYOUT_UNEXPECTED_GAP = 9,
    YVEX_GGUF_LAYOUT_DUPLICATE_START = 10,
    YVEX_GGUF_LAYOUT_COMPLETE_OVERLAP = 11,
    YVEX_GGUF_LAYOUT_PARTIAL_OVERLAP = 12,
    YVEX_GGUF_LAYOUT_PADDED_OVERLAP = 13,
    YVEX_GGUF_LAYOUT_RAW_END_OVERFLOW = 14,
    YVEX_GGUF_LAYOUT_PADDED_END_OVERFLOW = 15,
    YVEX_GGUF_LAYOUT_AGGREGATE_OVERFLOW = 16,
    YVEX_GGUF_LAYOUT_TENSOR_PAYLOAD_TRUNCATED = 17,
    YVEX_GGUF_LAYOUT_PADDING_TRUNCATED = 18,
    YVEX_GGUF_LAYOUT_DIRECTORY_PADDING_NONZERO = 19,
    YVEX_GGUF_LAYOUT_TENSOR_PADDING_NONZERO = 20,
    YVEX_GGUF_LAYOUT_NONCANONICAL_TAIL = 21,
    YVEX_GGUF_LAYOUT_FILE_IDENTITY_DRIFT = 22,
    YVEX_GGUF_LAYOUT_IO_FAILURE = 23,
    YVEX_GGUF_LAYOUT_VIEW_FILE_MISMATCH = 24,
    YVEX_GGUF_LAYOUT_TENSOR_INFO_MISSING = 25,
    YVEX_GGUF_LAYOUT_TENSOR_RANGE_MISMATCH = 26
} yvex_gguf_layout_code;

typedef struct {
    yvex_gguf_layout_code code;
    int accepted;
    char reason[YVEX_GGUF_LAYOUT_REASON_CAP];
    unsigned int alignment;
    unsigned long long tensor_count;
    unsigned long long tensors_validated;
    unsigned long long tensor_index;
    char tensor_name[YVEX_GGUF_LAYOUT_TENSOR_NAME_CAP];
    unsigned long long expected_relative_offset;
    unsigned long long declared_relative_offset;
    unsigned long long tensor_raw_size;
    unsigned long long tensor_padded_size;
    unsigned long long tensor_raw_end;
    unsigned long long tensor_padded_end;
    unsigned long long failure_absolute_offset;
    unsigned long long directory_end;
    unsigned long long directory_padding_bytes;
    unsigned long long tensor_data_offset;
    unsigned long long raw_tensor_bytes;
    unsigned long long inter_tensor_padding_bytes;
    unsigned long long final_tensor_padding_bytes;
    unsigned long long total_padding_bytes;
    unsigned long long data_section_span;
    unsigned long long required_file_end;
    unsigned long long actual_file_size;
    unsigned long long trailing_bytes;
    unsigned long long structural_bytes_read;
    unsigned long long padding_bytes_read;
    unsigned long long tensor_payload_bytes_read;
    unsigned long long padding_read_calls;
} yvex_gguf_layout_result;

const char *yvex_gguf_layout_code_name(yvex_gguf_layout_code code);

yvex_gguf_layout_code yvex_gguf_layout_interval_measure(
    unsigned long long relative_offset,
    unsigned long long raw_size,
    unsigned int alignment,
    unsigned long long *raw_end,
    unsigned long long *padded_end);
yvex_gguf_layout_code yvex_gguf_layout_sum_checked(
    unsigned long long current,
    unsigned long long addition,
    unsigned long long *out);

/*
 * Validates one parsed view against the exact opened artifact snapshot. The
 * result owns all diagnostic text and remains valid independently of both
 * borrowed inputs. Tensor payload bytes are never read. */
int yvex_gguf_layout_validate(const yvex_artifact *artifact,
                              const yvex_gguf *gguf,
                              yvex_gguf_layout_result *out,
                              yvex_error *err);

/* Template inspection. */
typedef struct yvex_gguf_template yvex_gguf_template;

typedef enum {
    YVEX_GGUF_TEMPLATE_STATUS_UNKNOWN = 0,
    YVEX_GGUF_TEMPLATE_STATUS_VALID,
    YVEX_GGUF_TEMPLATE_STATUS_PARTIAL,
    YVEX_GGUF_TEMPLATE_STATUS_INVALID
} yvex_gguf_template_status;

typedef enum {
    YVEX_GGUF_TEMPLATE_ISSUE_NONE = 0,
    YVEX_GGUF_TEMPLATE_ISSUE_MISSING_ARCHITECTURE,
    YVEX_GGUF_TEMPLATE_ISSUE_MISSING_MODEL_NAME,
    YVEX_GGUF_TEMPLATE_ISSUE_MISSING_TOKENIZER,
    YVEX_GGUF_TEMPLATE_ISSUE_EMPTY_TENSOR_DIRECTORY,
    YVEX_GGUF_TEMPLATE_ISSUE_UNKNOWN_TENSOR_ROLE,
    YVEX_GGUF_TEMPLATE_ISSUE_NATIVE_MISSING_TENSOR,
    YVEX_GGUF_TEMPLATE_ISSUE_NATIVE_SHAPE_MISMATCH,
    YVEX_GGUF_TEMPLATE_ISSUE_UNSUPPORTED_DTYPE,
    YVEX_GGUF_TEMPLATE_ISSUE_FORMAT
} yvex_gguf_template_issue_kind;

typedef struct {
    yvex_gguf_template_issue_kind kind;
    const char *tensor_name;
    const char *message;
} yvex_gguf_template_issue;

typedef struct {
    yvex_gguf_template_status status;
    const char *architecture;
    const char *model_name;
    unsigned long long metadata_count;
    unsigned long long tensor_count;
    unsigned long long known_role_count;
    unsigned long long unknown_role_count;
    unsigned long long tokenizer_metadata_count;
    unsigned long long issue_count;
    unsigned long long native_tensor_count;
    unsigned long long matched_exact;
    unsigned long long missing_in_native;
    unsigned long long shape_mismatch;
    int has_architecture;
    int has_tokenizer;
    int has_tensor_directory;
} yvex_gguf_template_summary;

typedef struct {
    const char *template_path;
    const char *native_source_dir;
    int compare_native;
    int require_tokenizer;
    int require_all_template_tensors_in_native;
} yvex_gguf_template_options;

int yvex_gguf_template_open(yvex_gguf_template **out,
                            const yvex_gguf_template_options *options,
                            yvex_error *err);

void yvex_gguf_template_close(yvex_gguf_template *tmpl);

int yvex_gguf_template_get_summary(const yvex_gguf_template *tmpl,
                                   yvex_gguf_template_summary *out,
                                   yvex_error *err);

unsigned long long yvex_gguf_template_issue_count(const yvex_gguf_template *tmpl);

const yvex_gguf_template_issue *yvex_gguf_template_issue_at(const yvex_gguf_template *tmpl,
                                                            unsigned long long index);

const char *yvex_gguf_template_status_name(yvex_gguf_template_status status);
const char *yvex_gguf_template_issue_kind_name(yvex_gguf_template_issue_kind kind);

/* Conversion planning. */
typedef enum {
    YVEX_CONVERSION_STATUS_UNKNOWN = 0,
    YVEX_CONVERSION_STATUS_PLANNED,
    YVEX_CONVERSION_STATUS_EMITTED,
    YVEX_CONVERSION_STATUS_PARTIAL,
    YVEX_CONVERSION_STATUS_FAILED
} yvex_conversion_status;

typedef enum {
    YVEX_CONVERT_TENSOR_STATUS_UNKNOWN = 0,
    YVEX_CONVERT_TENSOR_STATUS_READY,
    YVEX_CONVERT_TENSOR_STATUS_EMITTED,
    YVEX_CONVERT_TENSOR_STATUS_SKIPPED,
    YVEX_CONVERT_TENSOR_STATUS_UNSUPPORTED_QTYPE,
    YVEX_CONVERT_TENSOR_STATUS_UNMAPPED,
    YVEX_CONVERT_TENSOR_STATUS_FAILED
} yvex_convert_tensor_status;

typedef enum {
    YVEX_CONVERT_TRANSFORM_NONE = 0,
    YVEX_CONVERT_TRANSFORM_TRANSPOSE_2D,
    YVEX_CONVERT_TRANSFORM_DTYPE_CAST,
    YVEX_CONVERT_TRANSFORM_QUANTIZE,
    YVEX_CONVERT_TRANSFORM_UNSUPPORTED
} yvex_convert_transform_kind;

typedef struct {
    const char *architecture;
    const char *source_manifest_path;
    const char *native_source_dir;
    const char *template_path;
    const char *quant_policy_path;
    const char *imatrix_manifest_path;
    const char *out_path;
    const char *tensor_name;
    const char *target_qtype;
    unsigned long long limit_tensors;
    int plan_only;
    int overwrite;
    int allow_unsupported_qtype;
    int require_all;
} yvex_conversion_options;

typedef struct {
    yvex_conversion_status status;
    const char *architecture;
    const char *out_path;
    unsigned long long native_tensor_count;
    unsigned long long planned_tensor_count;
    unsigned long long emitted_tensor_count;
    unsigned long long skipped_tensor_count;
    unsigned long long unsupported_qtype_count;
    unsigned long long unmapped_tensor_count;
    unsigned long long bytes_read;
    unsigned long long bytes_written;
    int roundtrip_validated;
    int execution_ready;
} yvex_conversion_summary;

int yvex_conversion_plan_write_json(const yvex_conversion_options *options,
                                    const char *plan_out_path,
                                    yvex_conversion_summary *summary_out,
                                    yvex_error *err);

int yvex_conversion_emit_gguf(const yvex_conversion_options *options,
                              yvex_conversion_summary *summary_out,
                              yvex_error *err);

int yvex_conversion_suggest_artifact_name(char *out,
                                          unsigned long long out_size,
                                          const char *family,
                                          const char *model,
                                          const char *scope,
                                          const char *artifact_class,
                                          const char *qprofile,
                                          const char *calibration,
                                          const char *producer,
                                          const char *schema,
                                          yvex_error *err);

const char *yvex_convert_tensor_status_name(yvex_convert_tensor_status status);

/* Tensor name mapping. */
#define YVEX_WEIGHT_MAPPING_MAX_DIMS 8u

typedef enum {
    YVEX_WEIGHT_MAPPING_STATUS_UNKNOWN = 0,
    YVEX_WEIGHT_MAPPING_STATUS_MAPPED,
    YVEX_WEIGHT_MAPPING_STATUS_UNMAPPED,
    YVEX_WEIGHT_MAPPING_STATUS_AMBIGUOUS,
    YVEX_WEIGHT_MAPPING_STATUS_SHAPE_MISMATCH,
    YVEX_WEIGHT_MAPPING_STATUS_UNSUPPORTED_ARCH
} yvex_weight_mapping_status;

typedef enum {
    YVEX_WEIGHT_MAPPING_ISSUE_NONE = 0,
    YVEX_WEIGHT_MAPPING_ISSUE_UNKNOWN_NATIVE_NAME,
    YVEX_WEIGHT_MAPPING_ISSUE_UNKNOWN_TEMPLATE_NAME,
    YVEX_WEIGHT_MAPPING_ISSUE_ROLE_UNSUPPORTED,
    YVEX_WEIGHT_MAPPING_ISSUE_SHAPE_MISMATCH,
    YVEX_WEIGHT_MAPPING_ISSUE_ARCH_UNSUPPORTED,
    YVEX_WEIGHT_MAPPING_ISSUE_MOE_EXPERT_UNPARSED
} yvex_weight_mapping_issue_kind;

typedef struct {
    const char *native_name;
    const char *target_name;
    const char *architecture;
    yvex_tensor_role role;
    yvex_weight_mapping_status status;
    yvex_weight_mapping_issue_kind issue;
    unsigned int native_rank;
    unsigned long long native_dims[YVEX_WEIGHT_MAPPING_MAX_DIMS];
    unsigned int target_rank;
    unsigned long long target_dims[YVEX_WEIGHT_MAPPING_MAX_DIMS];
    int requires_transpose;
} yvex_weight_mapping_info;

typedef struct yvex_weight_mapping_table yvex_weight_mapping_table;

typedef struct {
    const char *architecture;
    const char *native_source_dir;
    const char *template_path;
    int compare_template;
    int require_all_native_mapped;
    int require_all_template_matched;
} yvex_weight_mapping_options;

int yvex_weight_mapping_table_build(yvex_weight_mapping_table **out,
                                    const yvex_weight_mapping_options *options,
                                    yvex_error *err);

void yvex_weight_mapping_table_close(yvex_weight_mapping_table *table);

unsigned long long yvex_weight_mapping_table_count(const yvex_weight_mapping_table *table);

const yvex_weight_mapping_info *yvex_weight_mapping_table_at(const yvex_weight_mapping_table *table,
                                                             unsigned long long index);

const yvex_weight_mapping_info *yvex_weight_mapping_table_find_native(const yvex_weight_mapping_table *table,
                                                                      const char *native_name);

const char *yvex_weight_mapping_status_name(yvex_weight_mapping_status status);
const char *yvex_weight_mapping_issue_kind_name(yvex_weight_mapping_issue_kind issue);

#ifdef __cplusplus
}
#endif

#endif /* YVEX_GGUF_H */
