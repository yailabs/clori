/* Artifact handles borrow no caller storage and remain bound to the file snapshot admitted at
 * open. Byte access, structural completeness, and runtime support are deliberately separate
 * contracts. */
#ifndef YVEX_ARTIFACT_H
#define YVEX_ARTIFACT_H

#include <stddef.h>
#include <yvex/core.h>
#include <yvex/gguf.h>
#include <yvex/model.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Artifact snapshots. */
#define YVEX_ARTIFACT_PATH_CAP 4096

typedef struct yvex_artifact yvex_artifact;

typedef struct {
    const char *path;
    /* Read-only access is the only admitted mode. A false value is refused. */
    int readonly;
    /* Whole-file mapping is opt-in and intended only for explicit payload users. */
    int map;
} yvex_artifact_options;

typedef struct {
    unsigned long long device;
    unsigned long long inode;
    unsigned long long size;
    long long mtime_seconds;
    long long mtime_nanoseconds;
    long long ctime_seconds;
    long long ctime_nanoseconds;
} yvex_artifact_snapshot;

int yvex_artifact_open(yvex_artifact **out, const yvex_artifact_options *options, yvex_error *err);
void yvex_artifact_close(yvex_artifact *artifact);

const char *yvex_artifact_path(const yvex_artifact *artifact);
unsigned long long yvex_artifact_size(const yvex_artifact *artifact);
int yvex_artifact_is_mapped(const yvex_artifact *artifact);
/* The returned mapping is borrowed until yvex_artifact_close, or null if map=0. */
const unsigned char *yvex_artifact_data(const yvex_artifact *artifact);
/* Reads exactly len bytes without changing shared file position or mapping the file. */
int yvex_artifact_read_at(const yvex_artifact *artifact,
                          unsigned long long offset,
                          void *dst,
                          size_t len,
                          yvex_error *err);
/* Copies the immutable identity captured when the file handle was opened. */
int yvex_artifact_snapshot_get(const yvex_artifact *artifact,
                               yvex_artifact_snapshot *out,
                               yvex_error *err);
/* Fails if either the open file or its path no longer names that snapshot. */
int yvex_artifact_snapshot_validate(const yvex_artifact *artifact,
                                    yvex_artifact_snapshot *current,
                                    yvex_error *err);

int yvex_range_check(unsigned long long file_size,
                     unsigned long long offset,
                     unsigned long long len,
                     yvex_error *err);

/* Artifact identity. */
#define YVEX_SHA256_HEX_CAP 65u

typedef struct {
    char path[YVEX_ARTIFACT_PATH_CAP];
    unsigned long long file_size;
    char sha256[YVEX_SHA256_HEX_CAP];
} yvex_artifact_file_identity;

int yvex_artifact_compute_sha256(const char *path,
                                 char out_hex[YVEX_SHA256_HEX_CAP],
                                 yvex_error *err);

int yvex_artifact_identity_read(const char *path,
                                yvex_artifact_file_identity *out,
                                yvex_error *err);

/* Hashes the exact already-opened artifact snapshot through positioned reads. */
int yvex_artifact_identity_read_open(const yvex_artifact *artifact,
                                     yvex_artifact_file_identity *out,
                                     yvex_error *err);

int yvex_artifact_sha256_hex_bytes(const unsigned char *data,
                                   unsigned long long len,
                                   char out_hex[YVEX_SHA256_HEX_CAP],
                                   yvex_error *err);

int yvex_sha256_hex_is_valid(const char *hex);

/* Artifact integrity. */
#define YVEX_INTEGRITY_CODE_CAP 64u
#define YVEX_INTEGRITY_TENSOR_CAP 128u
#define YVEX_INTEGRITY_REASON_CAP 256u
#define YVEX_INTEGRITY_FORMAT_CAP 16u
#define YVEX_INTEGRITY_ARCH_CAP 64u
#define YVEX_INTEGRITY_MAX_ISSUES 32u
#define YVEX_INTEGRITY_SHA256_CAP 65u
#define YVEX_INTEGRITY_DIGEST_STATUS_CAP 24u

typedef enum {
    YVEX_INTEGRITY_SEVERITY_ERROR = 0,
    YVEX_INTEGRITY_SEVERITY_WARNING = 1
} yvex_integrity_severity;

typedef struct {
    yvex_integrity_severity severity;
    char code[YVEX_INTEGRITY_CODE_CAP];
    char tensor[YVEX_INTEGRITY_TENSOR_CAP];
    char reason[YVEX_INTEGRITY_REASON_CAP];
    unsigned long long relative_offset;
    unsigned long long absolute_offset;
    unsigned long long tensor_bytes;
    unsigned long long file_size;
    int has_range;
} yvex_integrity_issue;

typedef struct {
    char tensor_name[YVEX_INTEGRITY_TENSOR_CAP];
    yvex_dtype dtype;
    unsigned int rank;
    unsigned long long dims[YVEX_TENSOR_MAX_DIMS];
    unsigned long long element_count;
    unsigned long long storage_unit_bytes;
    unsigned long long storage_byte_count;
    int dtype_known;
    int byte_count_computable;
    int storage_supported;
    int compute_supported_for_selected_embedding;
    int compute_supported_for_fixture_embedding;
    int shape_valid;
    int dtype_valid;
    int byte_count_valid;
} yvex_tensor_shape_accounting;

typedef struct {
    char tensor_name[YVEX_INTEGRITY_TENSOR_CAP];
    yvex_dtype dtype;
    unsigned int rank;
    unsigned long long dims[YVEX_TENSOR_MAX_DIMS];
    unsigned long long element_count;
    unsigned long long dtype_size;
    unsigned long long tensor_bytes;
    unsigned long long tensor_relative_offset;
    unsigned long long tensor_data_offset;
    unsigned long long tensor_absolute_offset;
    unsigned long long tensor_end_offset;
    unsigned long long file_size;
    unsigned long long alignment;
    int aligned;
    int range_valid;
} yvex_tensor_range;

typedef struct {
    unsigned int token_id;
    unsigned long long slice_bytes;
    unsigned long long slice_relative_offset;
    unsigned long long slice_absolute_offset;
    unsigned long long slice_end_offset;
    int range_valid;
} yvex_tensor_slice_range;

typedef struct {
    char tensor_name[YVEX_INTEGRITY_TENSOR_CAP];
    unsigned int token_id;
    unsigned long long hidden_size;
    unsigned long long vocab_size;
    unsigned long long output_count;
    unsigned long long output_bytes;
    unsigned long long slice_bytes;
    int shape_valid;
} yvex_selected_embedding_shape;

typedef struct {
    int require_token_embedding;
    unsigned int token_id;
    const char *expect_sha256;
    const char *registered_sha256;
} yvex_artifact_integrity_options;

typedef struct {
    int checked;
    int passed;
    char path[YVEX_ARTIFACT_PATH_CAP];
    char format[YVEX_INTEGRITY_FORMAT_CAP];
    unsigned long long file_size;
    unsigned int version;
    char architecture[YVEX_INTEGRITY_ARCH_CAP];
    int identity_checked;
    char sha256[YVEX_INTEGRITY_SHA256_CAP];
    char registered_sha256[YVEX_INTEGRITY_SHA256_CAP];
    char expected_sha256[YVEX_INTEGRITY_SHA256_CAP];
    char digest_status[YVEX_INTEGRITY_DIGEST_STATUS_CAP];
    unsigned long long tensor_count;
    int layout_checked;
    yvex_gguf_layout_result layout;
    unsigned long long known_tensor_bytes;
    unsigned long long tensor_ranges_checked;
    unsigned long long tensor_ranges_valid;
    unsigned long long tensor_ranges_invalid;
    unsigned long long tensor_shapes_checked;
    unsigned long long tensor_shapes_valid;
    unsigned long long tensor_shapes_invalid;
    unsigned long long tensor_dtypes_checked;
    unsigned long long tensor_dtypes_valid;
    unsigned long long tensor_dtypes_invalid;
    unsigned long long tensor_byte_counts_checked;
    unsigned long long tensor_byte_counts_invalid;
    char selected_embedding_shape[YVEX_INTEGRITY_DIGEST_STATUS_CAP];
    unsigned long long selected_embedding_hidden_size;
    unsigned long long selected_embedding_vocab_size;
    unsigned long long selected_embedding_output_count;
    unsigned long long selected_embedding_output_bytes;
    unsigned long long selected_embedding_slice_bytes;
    unsigned int error_count;
    unsigned int warning_count;
    unsigned int issue_count;
    yvex_integrity_issue issues[YVEX_INTEGRITY_MAX_ISSUES];
} yvex_artifact_integrity_report;

int yvex_artifact_integrity_check_path(const char *path,
                                       const yvex_artifact_integrity_options *options,
                                       yvex_artifact_integrity_report *out,
                                       yvex_error *err);

int yvex_artifact_integrity_validate(const yvex_artifact *artifact,
                                     const yvex_gguf *gguf,
                                     const yvex_tensor_table *tensors,
                                     const yvex_artifact_integrity_options *options,
                                     yvex_artifact_integrity_report *out,
                                     yvex_error *err);

int yvex_tensor_range_validate(const yvex_artifact *artifact,
                               const yvex_gguf *gguf,
                               const yvex_tensor_info *tensor,
                               yvex_tensor_range *out,
                               yvex_error *err);

int yvex_tensor_shape_accounting_validate(
    const yvex_tensor_info *tensor,
    yvex_tensor_shape_accounting *out,
    yvex_error *err);

int yvex_selected_embedding_shape_validate(
    const yvex_tensor_info *tensor,
    unsigned int token_id,
    yvex_selected_embedding_shape *out,
    yvex_error *err);

int yvex_tensor_embedding_slice_range_validate(
    const yvex_tensor_range *range,
    unsigned int token_id,
    yvex_tensor_slice_range *out,
    yvex_error *err);

const yvex_integrity_issue *yvex_artifact_integrity_issue_at(
    const yvex_artifact_integrity_report *report,
    unsigned int index);

/* Artifact naming. */
int yvex_artifact_name_suggest(char *out,
                               size_t out_size,
                               const char *family,
                               const char *model,
                               const char *scope,
                               const char *artifact_class,
                               const char *qprofile,
                               const char *calibration,
                               const char *producer,
                               const char *schema,
                               yvex_error *err);

/* Materialization admission. */
typedef enum {
    YVEX_MATERIALIZE_GATE_UNKNOWN = 0,
    YVEX_MATERIALIZE_GATE_PASS,
    YVEX_MATERIALIZE_GATE_PARTIAL,
    YVEX_MATERIALIZE_GATE_FAIL,
    YVEX_MATERIALIZE_GATE_BLOCKED
} yvex_materialize_gate_status;

typedef enum {
    YVEX_MATERIALIZE_SCOPE_UNKNOWN = 0,
    YVEX_MATERIALIZE_SCOPE_SELECTED_TENSOR,
    YVEX_MATERIALIZE_SCOPE_PARTIAL_MODEL,
    YVEX_MATERIALIZE_SCOPE_FULL_MODEL
} yvex_materialize_scope;

typedef enum {
    YVEX_MATERIALIZE_BACKEND_NOT_TESTED = 0,
    YVEX_MATERIALIZE_BACKEND_PASS,
    YVEX_MATERIALIZE_BACKEND_FAIL,
    YVEX_MATERIALIZE_BACKEND_UNAVAILABLE
} yvex_materialize_backend_status;

typedef enum {
    YVEX_MATERIALIZE_FAILURE_NONE = 0,
    YVEX_MATERIALIZE_FAILURE_MISSING_FILE,
    YVEX_MATERIALIZE_FAILURE_HASH_MISMATCH,
    YVEX_MATERIALIZE_FAILURE_GGUF_PARSE,
    YVEX_MATERIALIZE_FAILURE_TENSOR_SPEC_MISMATCH,
    YVEX_MATERIALIZE_FAILURE_UNSUPPORTED_DTYPE,
    YVEX_MATERIALIZE_FAILURE_UNSUPPORTED_QTYPE,
    YVEX_MATERIALIZE_FAILURE_BACKEND_UNAVAILABLE,
    YVEX_MATERIALIZE_FAILURE_BACKEND_ALLOC,
    YVEX_MATERIALIZE_FAILURE_BACKEND_COPY,
    YVEX_MATERIALIZE_FAILURE_OOM,
    YVEX_MATERIALIZE_FAILURE_UNKNOWN
} yvex_materialize_failure_class;

typedef struct {
    const char *name;
    const char *dtype;
    unsigned int rank;
    unsigned long long dims[4];
    unsigned long long bytes;
} yvex_materialize_expected_tensor;

typedef struct {
    const char *model_path;
    const char *label;
    const char *family;
    const char *sha256;
    const char *metadata_status;
    yvex_materialize_scope scope;
    const yvex_materialize_expected_tensor *expected_tensors;
    unsigned long long expected_tensor_count;
    int check_cpu;
    int check_cuda;
    int require_cpu;
    int require_cuda;
    unsigned int repeat_count;
    int check_cleanup;
    int json;
} yvex_materialize_gate_options;

typedef struct {
    yvex_materialize_gate_status status;
    yvex_materialize_scope scope;
    yvex_materialize_failure_class failure_class;
    const char *label;
    const char *family;
    const char *model_path;
    const char *expected_sha256;
    char actual_sha256[65];
    const char *digest_status;
    const char *identity_status;
    const char *metadata_status;
    const char *materialization_gate;
    const char *materialization_phase;
    const char *integrity_status;
    const char *shape_status;
    const char *range_status;
    const char *backend_status;
    const char *cleanup_status;
    int allocation_attempted;
    int transfer_attempted;
    int cleanup_attempted;
    unsigned long long file_bytes;
    unsigned long long tensor_count;
    unsigned long long expected_tensor_matches;
    unsigned long long expected_tensor_mismatches;
    unsigned long long bytes_materialized_cpu;
    unsigned long long bytes_materialized_cuda;
    yvex_materialize_backend_status cpu_status;
    yvex_materialize_backend_status cuda_status;
    unsigned int repeat_count;
    int cleanup_verified;
    unsigned long long bytes_planned;
    unsigned long long bytes_allocated;
    unsigned long long bytes_transferred;
    int execution_ready;
} yvex_materialize_gate_summary;

int yvex_materialize_gate_check(const yvex_materialize_gate_options *options,
                                yvex_materialize_gate_summary *summary_out,
                                yvex_error *err);

const char *yvex_materialize_gate_status_name(yvex_materialize_gate_status status);
const char *yvex_materialize_scope_name(yvex_materialize_scope scope);
const char *yvex_materialize_backend_status_name(yvex_materialize_backend_status status);
const char *yvex_materialize_failure_class_name(yvex_materialize_failure_class failure);

/* Model support admission. */
typedef enum {
    YVEX_MODEL_SUPPORT_NONE = 0,
    YVEX_MODEL_SUPPORT_DESCRIPTOR_ONLY,
    YVEX_MODEL_SUPPORT_SELECTED_TENSOR_MATERIALIZED,
    YVEX_MODEL_SUPPORT_FULL_WEIGHTS_MATERIALIZED,
    YVEX_MODEL_SUPPORT_PARTIAL_GRAPH_EXECUTABLE,
    YVEX_MODEL_SUPPORT_PREFILL_READY,
    YVEX_MODEL_SUPPORT_DECODE_READY,
    YVEX_MODEL_SUPPORT_GENERATION_READY
} yvex_model_support_level;

typedef enum {
    YVEX_MODEL_GATE_UNKNOWN = 0,
    YVEX_MODEL_GATE_PASS,
    YVEX_MODEL_GATE_PARTIAL,
    YVEX_MODEL_GATE_FAIL,
    YVEX_MODEL_GATE_BLOCKED
} yvex_model_gate_status;

typedef enum {
    YVEX_MODEL_GATE_BACKEND_NOT_TESTED = 0,
    YVEX_MODEL_GATE_BACKEND_PASS,
    YVEX_MODEL_GATE_BACKEND_FAIL,
    YVEX_MODEL_GATE_BACKEND_UNAVAILABLE
} yvex_model_gate_backend_status;

typedef struct {
    const char *name;
    const char *dtype;
    unsigned int rank;
    unsigned long long dims[4];
    unsigned long long bytes;
} yvex_model_gate_expected_tensor;

typedef struct {
    const char *model_path;
    const char *model_label;
    const char *family;
    const char *artifact_sha256;
    const yvex_model_gate_expected_tensor *expected_tensors;
    unsigned long long expected_tensor_count;
    int check_cpu;
    int check_cuda;
    int require_cpu;
    int require_cuda;
} yvex_model_gate_options;

typedef struct {
    yvex_model_gate_status status;
    yvex_model_support_level support_level;
    const char *model_path;
    const char *model_label;
    const char *family;
    const char *expected_sha256;
    char actual_sha256[65];
    const char *digest_status;
    const char *identity_status;
    unsigned long long file_bytes;
    unsigned long long tensor_count;
    unsigned long long expected_tensor_matches;
    unsigned long long expected_tensor_mismatches;
    yvex_model_gate_backend_status cpu_status;
    yvex_model_gate_backend_status cuda_status;
    int execution_ready;
} yvex_model_gate_summary;

int yvex_model_gate_check(const yvex_model_gate_options *options,
                          yvex_model_gate_summary *summary_out,
                          yvex_error *err);

const char *yvex_model_gate_status_name(yvex_model_gate_status status);
const char *yvex_model_support_level_name(yvex_model_support_level level);
const char *yvex_model_gate_backend_status_name(yvex_model_gate_backend_status status);

#ifdef __cplusplus
}
#endif

#endif /* YVEX_ARTIFACT_H */
