/*
 * GGUF planners and sinks exchange immutable layout facts and transactional delivery state here.
 * A completed write still requires independent artifact admission.
 */
#ifndef INCLUDE_YVEX_INTERNAL_GGUF_WRITER_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_GGUF_WRITER_H_INCLUDED

#include <stddef.h>
#include <yvex/artifact.h>
#include <yvex/internal/quant_numeric.h>
#include <yvex/internal/source.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct yvex_gguf_tokenizer_metadata yvex_gguf_tokenizer_metadata;
typedef struct yvex_model_family_api yvex_model_family_api;

/* Immutable writer plans. */
#define YVEX_GGUF_WRITER_SCHEMA_VERSION 1u
#define YVEX_GGUF_WRITER_IDENTITY_CAP 65u
#define YVEX_GGUF_WRITER_NAME_CAP 192u
typedef enum {
    YVEX_GGUF_WRITER_OK = 0,
    YVEX_GGUF_WRITER_INVALID_ARGUMENT,
    YVEX_GGUF_WRITER_UNSEALED_PLAN,
    YVEX_GGUF_WRITER_IDENTITY_MISMATCH,
    YVEX_GGUF_WRITER_METADATA_INCOMPLETE,
    YVEX_GGUF_WRITER_DUPLICATE_METADATA,
    YVEX_GGUF_WRITER_UNSUPPORTED_METADATA,
    YVEX_GGUF_WRITER_DUPLICATE_TENSOR,
    YVEX_GGUF_WRITER_TENSOR_DIVERGENCE,
    YVEX_GGUF_WRITER_QTYPE_GEOMETRY,
    YVEX_GGUF_WRITER_ARITHMETIC_OVERFLOW,
    YVEX_GGUF_WRITER_RESOURCE_LIMIT,
    YVEX_GGUF_WRITER_ALLOCATION,
    YVEX_GGUF_WRITER_SERIALIZATION,
    YVEX_GGUF_WRITER_LIFECYCLE
} yvex_gguf_writer_code;
typedef struct yvex_gguf_writer_failure {
    yvex_gguf_writer_code code;
    unsigned long long metadata_index;
    unsigned long long tensor_index;
    unsigned long long expected;
    unsigned long long actual;
    char name[YVEX_GGUF_WRITER_NAME_CAP];
} yvex_gguf_writer_failure;
typedef struct {
    char name[YVEX_GGUF_WRITER_NAME_CAP];
    unsigned int rank;
    unsigned long long dims[YVEX_GGUF_QTYPE_MAX_DIMS];
    unsigned int qtype;
    unsigned long long relative_offset;
    unsigned long long absolute_offset;
    unsigned long long raw_bytes;
    unsigned long long padded_bytes;
    unsigned long long absolute_end;
    unsigned long long padded_end;
} yvex_gguf_writer_tensor;
typedef struct {
    unsigned int schema_version;
    unsigned int gguf_version;
    unsigned int alignment;
    unsigned long long metadata_count;
    unsigned long long tensor_count;
    unsigned long long structural_bytes;
    unsigned long long pre_data_padding_bytes;
    unsigned long long tensor_payload_bytes;
    unsigned long long tensor_padding_bytes;
    unsigned long long data_section_bytes;
    unsigned long long final_file_bytes;
    unsigned long long source_snapshot_identity;
    char payload_identity[YVEX_GGUF_WRITER_IDENTITY_CAP];
    char transform_identity[YVEX_GGUF_WRITER_IDENTITY_CAP];
    unsigned long long mapping_identity;
    char profile_name[64];
    char profile_identity[YVEX_GGUF_WRITER_IDENTITY_CAP];
    char payload_plan_identity[YVEX_GGUF_WRITER_IDENTITY_CAP];
    char required_execution_identity[YVEX_GGUF_WRITER_IDENTITY_CAP];
    char writer_plan_identity[YVEX_GGUF_WRITER_IDENTITY_CAP];
    unsigned long long qtype_tensor_counts[
        YVEX_GGUF_QTYPE_ABI_UPSTREAM_MAX_ID + 1u];
    unsigned long long qtype_payload_bytes[
        YVEX_GGUF_QTYPE_ABI_UPSTREAM_MAX_ID + 1u];
    unsigned long long tokenizer_token_count;
    unsigned long long tokenizer_merge_count;
    unsigned long long tokenizer_embedded_bytes;
    unsigned long long owned_bytes;
    unsigned long long payload_bytes_read;
    int complete;
} yvex_gguf_writer_plan_summary;
typedef struct yvex_gguf_writer_plan_options {
    unsigned int alignment;
    size_t maximum_owned_bytes;
    const char *required_execution_identity;
} yvex_gguf_writer_plan_options;
typedef struct yvex_gguf_writer_plan yvex_gguf_writer_plan;
typedef struct {
    const char *name;
} yvex_gguf_writer_proof_tensor;
typedef enum {
    YVEX_GGUF_WRITER_INPUT_INVALID = 0,
    YVEX_GGUF_WRITER_INPUT_COMPLETE_ARTIFACT,
    YVEX_GGUF_WRITER_INPUT_LOGICAL_COMPONENT,
    YVEX_GGUF_WRITER_INPUT_TENSOR_PROOF
} yvex_gguf_writer_input_class;
typedef struct {
    const yvex_model_family_api *family_adapter;
    const void *lowering;
    const yvex_source_verification *verification;
} yvex_gguf_writer_complete_input;
typedef struct {
    const yvex_gguf_writer_proof_tensor *tensors;
    unsigned long long tensor_count;
} yvex_gguf_writer_proof_input;
typedef struct {
    const char *architecture;
    const char *target_id;
    const char *component_id;
    const char *source_snapshot_identity;
    unsigned long long source_snapshot_key;
    const char *component_identity;
    const char *component_manifest_identity;
    const char *architecture_identity;
    const char *role_map_identity;
} yvex_gguf_writer_component_input;
typedef union {
    yvex_gguf_writer_complete_input complete;
    yvex_gguf_writer_component_input component;
    yvex_gguf_writer_proof_input tensor_proof;
} yvex_gguf_writer_input;
typedef struct {
    yvex_gguf_writer_input_class input_class;
    const yvex_quant_plan *quant_plan;
    const yvex_gguf_writer_plan_options *options;
    yvex_gguf_writer_input input;
} yvex_gguf_writer_plan_request;
void yvex_gguf_writer_plan_options_default(
    yvex_gguf_writer_plan_options *options);
int yvex_gguf_writer_plan_build(
    yvex_gguf_writer_plan **out,
    const yvex_gguf_writer_plan_request *request,
    yvex_gguf_writer_failure *failure,
    yvex_error *err);
void yvex_gguf_writer_plan_release(yvex_gguf_writer_plan **plan);
const yvex_gguf_writer_plan_summary *yvex_gguf_writer_plan_summary_get(
    const yvex_gguf_writer_plan *plan);
const yvex_gguf_writer_tensor *yvex_gguf_writer_plan_tensor_at(
    const yvex_gguf_writer_plan *plan, unsigned long long ordinal);
const unsigned char *yvex_gguf_writer_plan_prefix(
    const yvex_gguf_writer_plan *plan, size_t *byte_count);

/* Transactional file sink. */
typedef enum {
    YVEX_GGUF_FILE_OK = 0,
    YVEX_GGUF_FILE_INVALID_ARGUMENT,
    YVEX_GGUF_FILE_UNSAFE_DESTINATION,
    YVEX_GGUF_FILE_DESTINATION_EXISTS,
    YVEX_GGUF_FILE_DIRECTORY_OPEN,
    YVEX_GGUF_FILE_TEMP_CREATE,
    YVEX_GGUF_FILE_INSUFFICIENT_SPACE,
    YVEX_GGUF_FILE_PREALLOCATE,
    YVEX_GGUF_FILE_WRITE,
    YVEX_GGUF_FILE_TERMINAL_PROTOCOL,
    YVEX_GGUF_FILE_TERMINAL_ABORT,
    YVEX_GGUF_FILE_INCOMPLETE,
    YVEX_GGUF_FILE_EXECUTION_IDENTITY,
    YVEX_GGUF_FILE_FLUSH,
    YVEX_GGUF_FILE_SNAPSHOT_DRIFT,
    YVEX_GGUF_FILE_VALIDATION_REQUIRED,
    YVEX_GGUF_FILE_PUBLICATION,
    YVEX_GGUF_FILE_DIRECTORY_FLUSH,
    YVEX_GGUF_FILE_CLEANUP,
    YVEX_GGUF_FILE_ALLOCATION,
    YVEX_GGUF_FILE_CANCELLED
} yvex_gguf_file_code;
typedef struct {
    yvex_gguf_file_code code;
    int system_error;
    unsigned long long terminal_ordinal;
    unsigned long long expected;
    unsigned long long actual;
    unsigned long long file_offset;
} yvex_gguf_file_failure;
typedef struct {
    const char *destination_path;
    unsigned long long safety_margin_bytes;
    unsigned long long injected_write_failure_call;
    unsigned long long injected_write_eintr_call;
    size_t injected_write_max_bytes;
    int inject_temp_create_failure;
    int inject_preallocate_failure;
    int inject_fsync_failure;
    int inject_publish_failure;
    int inject_directory_fsync_failure;
} yvex_gguf_file_sink_options;
typedef struct {
    unsigned long long planned_file_bytes;
    unsigned long long available_bytes;
    unsigned long long required_safe_bytes;
    unsigned long long prefix_bytes_written;
    unsigned long long encoded_bytes_written;
    unsigned long long physical_write_bytes;
    unsigned long long source_ranges_committed;
    unsigned long long source_bytes_committed;
    unsigned long long write_calls;
    unsigned long long committed_terminals;
    unsigned long long aborted_terminals;
    unsigned long long output_chunks;
    unsigned long long peak_owned_bytes;
    unsigned long long file_device;
    unsigned long long file_inode;
    unsigned long long file_size;
    long long validated_mtime_seconds;
    long long validated_mtime_nanoseconds;
    long long validated_ctime_seconds;
    long long validated_ctime_nanoseconds;
    long long published_mtime_seconds;
    long long published_mtime_nanoseconds;
    long long published_ctime_seconds;
    long long published_ctime_nanoseconds;
    char execution_identity[YVEX_QUANT_PLAN_IDENTITY_CAP];
    char payload_byte_identity[YVEX_QUANT_PLAN_IDENTITY_CAP];
    char temporary_path[YVEX_ARTIFACT_PATH_CAP];
    char published_path[YVEX_ARTIFACT_PATH_CAP];
    int preallocated;
    int finalized;
    int published;
} yvex_gguf_file_sink_summary;
typedef struct yvex_gguf_file_sink yvex_gguf_file_sink;
typedef struct yvex_gguf_roundtrip_summary yvex_gguf_roundtrip_summary;
void yvex_gguf_file_sink_options_default(
    yvex_gguf_file_sink_options *options);
int yvex_gguf_file_sink_create(
    yvex_gguf_file_sink **out,
    const yvex_gguf_writer_plan *writer_plan,
    const yvex_quant_plan *quant_plan,
    const yvex_gguf_file_sink_options *options,
    yvex_gguf_file_failure *failure,
    yvex_error *err);
void yvex_gguf_file_sink_adapter(
    yvex_gguf_file_sink *sink,
    yvex_quant_output_sink *out);
int yvex_gguf_file_sink_finalize(
    yvex_gguf_file_sink *sink,
    yvex_gguf_file_sink_summary *out,
    yvex_gguf_file_failure *failure,
    yvex_error *err);
int yvex_gguf_file_sink_publish(
    yvex_gguf_file_sink *sink,
    const yvex_gguf_roundtrip_summary *roundtrip,
    yvex_gguf_file_sink_summary *out,
    yvex_gguf_file_failure *failure,
    yvex_error *err);
int yvex_gguf_file_sink_withdraw(
    yvex_gguf_file_sink *sink,
    yvex_gguf_file_failure *failure,
    yvex_error *err);
yvex_quant_digest_sink *yvex_gguf_file_sink_digest(
    yvex_gguf_file_sink *sink);
const char *yvex_gguf_file_sink_temporary_path(
    const yvex_gguf_file_sink *sink);
int yvex_gguf_file_sink_summary_get(
    yvex_gguf_file_sink *sink,
    yvex_gguf_file_sink_summary *out);
void yvex_gguf_file_sink_release(yvex_gguf_file_sink **sink);

#ifdef __cplusplus
}
#endif

#endif /* INCLUDE_YVEX_INTERNAL_GGUF_WRITER_H_INCLUDED */
