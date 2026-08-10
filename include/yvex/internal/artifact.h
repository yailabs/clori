/*
 * Artifact owners share identity, integrity, compatibility, and materialization lifecycles through
 * this ABI. Each stage preserves its own admission boundary.
 */
#ifndef INCLUDE_YVEX_INTERNAL_ARTIFACT_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_ARTIFACT_H_INCLUDED

#include <stddef.h>
#include <yvex/artifact.h>
#include <yvex/gguf.h>
#include <yvex/internal/core.h>
#include <yvex/internal/gguf.h>
#include <yvex/internal/gguf_writer.h>
#include <yvex/internal/model.h>
#include <yvex/model.h>
#include <yvex/qtype.h>
#include <yvex/registry.h>
#ifdef __cplusplus
extern "C" {
#endif
int yvex_artifact_snapshot_equal(const yvex_artifact_snapshot *left,
                                 const yvex_artifact_snapshot *right);
int yvex_artifact_cache_release(const yvex_artifact *artifact,
                                unsigned long long offset,
                                unsigned long long byte_count,
                                yvex_error *err);
/* Identity. */
typedef struct {
    yvex_sha256 hash;
    unsigned long long bytes;
    int active;
} yvex_artifact_identity_stream;
void yvex_artifact_identity_stream_init(yvex_artifact_identity_stream *stream);
int yvex_artifact_identity_stream_update(yvex_artifact_identity_stream *stream,
                                         const unsigned char *bytes, size_t byte_count,
                                         yvex_error *err);
int yvex_artifact_identity_stream_final(yvex_artifact_identity_stream *stream,
                                        unsigned long long expected_bytes,
                                        char out_hex[YVEX_SHA256_HEX_CAP], yvex_error *err);
int yvex_artifact_identity_read_open_progress(
    const yvex_artifact *artifact, yvex_artifact_file_identity *out,
    int (*progress)(void *context, unsigned long long completed,
                    unsigned long long total),
    void *progress_context, yvex_error *err);
typedef struct {
    unsigned long long tensor_count;
    unsigned long long payload_bytes_read;
    char payload_byte_identity[YVEX_SHA256_HEX_CAP];
    int complete;
} yvex_artifact_payload_identity;
int yvex_artifact_payload_identity_compute(const yvex_artifact *artifact, const yvex_gguf *gguf,
                                           size_t buffer_bytes,
                                           yvex_artifact_payload_identity *out,
                                           yvex_error *err);
/* Roundtrip Gate. */
#define YVEX_GGUF_OFFICIAL_READER_REVISION "af97976c7810cdabb1863172f31c432dab767de7"
typedef enum {
    YVEX_ARTIFACT_CLASS_REFUSED = 0,
    YVEX_ARTIFACT_CLASS_TENSOR_PROOF,
    YVEX_ARTIFACT_CLASS_EXTERNAL_UNADMITTED,
    YVEX_ARTIFACT_CLASS_COMPLETE_YVEX,
    YVEX_ARTIFACT_CLASS_COMPONENT_YVEX
} yvex_artifact_class;
typedef enum {
    YVEX_ARTIFACT_ADMISSION_OK = 0,
    YVEX_ARTIFACT_ADMISSION_INVALID_ARGUMENT,
    YVEX_ARTIFACT_ADMISSION_WRITER_INCOMPLETE,
    YVEX_ARTIFACT_ADMISSION_EMISSION_INCOMPLETE,
    YVEX_ARTIFACT_ADMISSION_NATIVE_ROUNDTRIP,
    YVEX_ARTIFACT_ADMISSION_OFFICIAL_READER,
    YVEX_ARTIFACT_ADMISSION_IDENTITY_MISMATCH,
    YVEX_ARTIFACT_ADMISSION_TOKENIZER_INCOMPLETE,
    YVEX_ARTIFACT_ADMISSION_TENSOR_COVERAGE,
    YVEX_ARTIFACT_ADMISSION_FILE_OPEN,
    YVEX_ARTIFACT_ADMISSION_FILE_DRIFT
} yvex_artifact_admission_code;
typedef struct {
    char revision[41];
    unsigned long long metadata_count;
    unsigned long long tensor_count;
    unsigned long long file_bytes;
    unsigned long long file_device;
    unsigned long long file_inode;
    long long file_mtime_seconds;
    long long file_mtime_nanoseconds;
    long long file_ctime_seconds;
    long long file_ctime_nanoseconds;
    int accepted;
} yvex_artifact_official_reader_fact;
typedef struct yvex_artifact_admission_failure {
    yvex_artifact_admission_code code;
    unsigned long long expected;
    unsigned long long actual;
    char field[96];
} yvex_artifact_admission_failure;
typedef struct {
    const char *artifact_path;
    const yvex_gguf_writer_plan *writer_plan;
    const yvex_gguf_file_sink_summary *emission;
    const yvex_gguf_roundtrip_summary *native_roundtrip;
    const yvex_artifact_official_reader_fact *official_reader;
} yvex_artifact_admission_request;
typedef struct yvex_complete_artifact_admission {
    yvex_artifact_class artifact_class;
    unsigned long long metadata_count;
    unsigned long long tensor_count;
    unsigned long long payload_bytes;
    unsigned long long file_bytes;
    unsigned long long source_snapshot_identity;
    unsigned long long mapping_identity;
    yvex_artifact_snapshot file_snapshot;
    char artifact_path[YVEX_ARTIFACT_PATH_CAP];
    char payload_identity[YVEX_GGUF_WRITER_IDENTITY_CAP];
    char transform_identity[YVEX_GGUF_WRITER_IDENTITY_CAP];
    char profile_identity[YVEX_GGUF_WRITER_IDENTITY_CAP];
    char profile_name[64];
    char quant_execution_identity[YVEX_GGUF_WRITER_IDENTITY_CAP];
    char payload_plan_identity[YVEX_GGUF_WRITER_IDENTITY_CAP];
    char payload_byte_identity[YVEX_GGUF_WRITER_IDENTITY_CAP];
    char writer_plan_identity[YVEX_GGUF_WRITER_IDENTITY_CAP];
    char artifact_identity[YVEX_SHA256_HEX_CAP];
    char admission_identity[YVEX_SHA256_HEX_CAP];
    char official_reader_revision[41];
    char logical_target[128];
    char logical_component[64];
    char logical_component_identity[YVEX_SHA256_HEX_CAP];
    int tokenizer_complete;
    int native_reader_accepted;
    int official_reader_accepted;
    int payload_integrity_accepted;
    int materialization_input_ready;
    int runtime_supported;
    unsigned long long artifact_bytes_hashed;
    int artifact_identity_verified;
    int complete;
} yvex_complete_artifact_admission;

/* Physical compatibility. */
#define YVEX_ARTIFACT_PHYSICAL_COMPATIBILITY_SCHEMA_VERSION 1u
typedef enum {
    YVEX_ARTIFACT_COMPATIBILITY_OK = 0,
    YVEX_ARTIFACT_COMPATIBILITY_INVALID_ARGUMENT,
    YVEX_ARTIFACT_COMPATIBILITY_WRITER_PLAN,
    YVEX_ARTIFACT_COMPATIBILITY_ADMISSION,
    YVEX_ARTIFACT_COMPATIBILITY_SNAPSHOT,
    YVEX_ARTIFACT_COMPATIBILITY_IDENTITY,
    YVEX_ARTIFACT_COMPATIBILITY_LAYOUT,
    YVEX_ARTIFACT_COMPATIBILITY_TENSOR_COUNT,
    YVEX_ARTIFACT_COMPATIBILITY_TENSOR_NAME,
    YVEX_ARTIFACT_COMPATIBILITY_TENSOR_RANK,
    YVEX_ARTIFACT_COMPATIBILITY_TENSOR_DIMENSION,
    YVEX_ARTIFACT_COMPATIBILITY_TENSOR_QTYPE,
    YVEX_ARTIFACT_COMPATIBILITY_TENSOR_BYTES,
    YVEX_ARTIFACT_COMPATIBILITY_TENSOR_OFFSET,
    YVEX_ARTIFACT_COMPATIBILITY_TENSOR_RANGE
} yvex_artifact_compatibility_code;
typedef struct {
    yvex_artifact_compatibility_code code;
    unsigned long long tensor_index;
    unsigned int dimension;
    unsigned long long expected;
    unsigned long long actual;
    char field[64];
    char tensor_name[YVEX_GGUF_WRITER_NAME_CAP];
} yvex_artifact_compatibility_failure;
typedef struct yvex_artifact_physical_compatibility {
    unsigned int schema_version;
    unsigned long long source_snapshot_identity;
    unsigned long long mapping_identity;
    unsigned long long tensor_count;
    unsigned long long tensors_compared;
    unsigned long long payload_bytes;
    unsigned long long payload_bytes_read;
    char writer_plan_identity[YVEX_GGUF_WRITER_IDENTITY_CAP];
    char admitted_writer_plan_identity[YVEX_GGUF_WRITER_IDENTITY_CAP];
    char artifact_identity[YVEX_SHA256_HEX_CAP];
    char payload_identity[YVEX_GGUF_WRITER_IDENTITY_CAP];
    char writer_transform_identity[YVEX_GGUF_WRITER_IDENTITY_CAP];
    char admitted_transform_identity[YVEX_GGUF_WRITER_IDENTITY_CAP];
    char writer_profile_identity[YVEX_GGUF_WRITER_IDENTITY_CAP];
    char admitted_profile_identity[YVEX_GGUF_WRITER_IDENTITY_CAP];
    char quant_execution_identity[YVEX_GGUF_WRITER_IDENTITY_CAP];
    char payload_plan_identity[YVEX_GGUF_WRITER_IDENTITY_CAP];
    char payload_byte_identity[YVEX_GGUF_WRITER_IDENTITY_CAP];
    int physical_payload_compatible;
    int artifact_rebuild_required;
    int materialization_rebuild_required;
    int tensor_inventory_equal;
    int qtype_equal;
    int layout_equal;
    int offset_equal;
    int payload_digest_equal;
} yvex_artifact_physical_compatibility;

int yvex_artifact_physical_compatibility_validate(
    const yvex_gguf_writer_plan *writer_plan, const yvex_complete_artifact_admission *admission,
    const yvex_artifact *artifact, const yvex_gguf *gguf, yvex_artifact_physical_compatibility *out,
    yvex_artifact_compatibility_failure *failure, yvex_error *err);
const char *yvex_artifact_physical_compatibility_mismatch(
    const yvex_artifact_physical_compatibility *proof,
    const yvex_complete_artifact_admission *admission,
    const char *logical_transform_identity);
typedef enum {
    YVEX_ARTIFACT_DESCRIPTOR_REFUSED = 0,
    YVEX_ARTIFACT_DESCRIPTOR_REPORT_ONLY = 1,
    YVEX_ARTIFACT_DESCRIPTOR_COMPLETE_ADMITTED = 2
} yvex_artifact_descriptor_status;
typedef struct {
    yvex_artifact_descriptor_status status;
    const char *format;
    const char *reason;
    const char *next_row;
    const char *artifact_identity;
    unsigned long long tensor_count;
    int materialization_input_ready;
    int runtime_supported;
} yvex_artifact_descriptor_fact;
int yvex_complete_artifact_admit(const yvex_artifact_admission_request *request,
                                 yvex_complete_artifact_admission *out,
                                 yvex_artifact_admission_failure *failure, yvex_error *err);
typedef struct { const char *key, *value; } yvex_artifact_component_metadata;
typedef struct { unsigned int qtype; unsigned long long tensors; } yvex_artifact_component_storage;
typedef struct {
    const yvex_complete_artifact_admission *catalog;
    const yvex_artifact_component_metadata *metadata;
    const yvex_artifact_component_storage *storage;
    unsigned long long metadata_count, storage_count, elements, alignment;
} yvex_artifact_catalog_contract;
/* Reconcile a family-provided exact catalog row with one unchanged opened artifact. */
int yvex_artifact_admit_catalog(
    const yvex_artifact *artifact, const yvex_gguf *gguf, const yvex_tensor_table *tensors,
    const yvex_artifact_catalog_contract *contract, yvex_complete_artifact_admission *out,
    yvex_artifact_admission_failure *failure, yvex_error *err);
int yvex_artifact_admission_identity_verify(
    const yvex_artifact *artifact, yvex_complete_artifact_admission *admission,
    int (*progress)(void *context, unsigned long long completed,
                    unsigned long long total), void *progress_context,
    yvex_artifact_admission_failure *failure, yvex_error *err);
const char *yvex_artifact_admission_code_name(yvex_artifact_admission_code code);
int yvex_artifact_descriptor_from_admission(const yvex_complete_artifact_admission *admission,
                                            yvex_artifact_descriptor_fact *fact);
/* Materialize. */
#define YVEX_MATERIALIZATION_IDENTITY_CAP 65u
#define YVEX_MATERIALIZATION_NAME_CAP 192u
#define YVEX_MATERIALIZATION_QTYPE_CAP 43u
#define YVEX_MATERIALIZATION_NO_INDEX (~0ull)
typedef struct {
    const char *status;
    const char *reason;
    const char *next_row;
} yvex_artifact_materialize_fact;
typedef enum {
    YVEX_MATERIALIZATION_STATUS_REFUSED = 0,
    YVEX_MATERIALIZATION_STATUS_PLANNED,
    YVEX_MATERIALIZATION_STATUS_COMMITTED,
    YVEX_MATERIALIZATION_STATUS_ABORTED
} yvex_materialization_status;
typedef enum {
    YVEX_MATERIALIZATION_PLACEMENT_FILE_BACKED = 0,
    YVEX_MATERIALIZATION_PLACEMENT_STAGED_CACHE,
    YVEX_MATERIALIZATION_PLACEMENT_BACKEND_RESIDENT_CANDIDATE
} yvex_materialization_placement;
typedef enum {
    YVEX_MATERIALIZATION_ACCESS_FILE_RANGE = 0,
    YVEX_MATERIALIZATION_ACCESS_STAGED_SUBVIEW,
    YVEX_MATERIALIZATION_ACCESS_BACKEND_CANDIDATE_FILE_RANGE
} yvex_materialization_access_mode;
typedef enum {
    YVEX_MATERIALIZATION_FAILURE_NONE = 0,
    YVEX_MATERIALIZATION_FAILURE_INVALID_ARGUMENT,
    YVEX_MATERIALIZATION_FAILURE_ADMISSION,
    YVEX_MATERIALIZATION_FAILURE_SNAPSHOT_DRIFT,
    YVEX_MATERIALIZATION_FAILURE_LAYOUT,
    YVEX_MATERIALIZATION_FAILURE_TENSOR_COUNT,
    YVEX_MATERIALIZATION_FAILURE_TENSOR_RECORD,
    YVEX_MATERIALIZATION_FAILURE_DUPLICATE_TENSOR,
    YVEX_MATERIALIZATION_FAILURE_QTYPE,
    YVEX_MATERIALIZATION_FAILURE_RANGE,
    YVEX_MATERIALIZATION_FAILURE_BUDGET,
    YVEX_MATERIALIZATION_FAILURE_ALLOCATION,
    YVEX_MATERIALIZATION_FAILURE_READ,
    YVEX_MATERIALIZATION_FAILURE_CANCELLED,
    YVEX_MATERIALIZATION_FAILURE_LIFECYCLE,
    YVEX_MATERIALIZATION_FAILURE_EXPERT_SUBVIEW
} yvex_materialization_failure_code;
typedef struct yvex_materialization_failure {
    yvex_materialization_failure_code code;
    unsigned long long tensor_index;
    unsigned long long expected;
    unsigned long long actual;
    unsigned long long offset;
    char tensor_name[YVEX_MATERIALIZATION_NAME_CAP];
    const char *reason;
} yvex_materialization_failure;
typedef struct yvex_materialization_options {
    unsigned long long max_chunk_bytes;
    unsigned long long cache_budget_bytes;
    unsigned long long backend_resident_budget_bytes;
    unsigned long long future_graph_scratch_reserve_bytes;
    unsigned long long future_kv_reserve_bytes;
    int require_complete_admission, require_terminal_projection;
    int release_artifact_cache_after_read, cancel_after_first_chunk;
} yvex_materialization_options;
typedef struct {
    unsigned long long descriptor_index;
    yvex_tensor_role role;
    yvex_tensor_collection collection;
    yvex_tensor_scope scope;
    unsigned long long layer_index, predictor_index, expert_count;
} yvex_materialization_terminal;
typedef int (*yvex_materialization_terminal_find_fn)(
    const void *context, const char *emitted_name, yvex_materialization_terminal *out);
typedef struct yvex_materialization_projection {
    unsigned int schema_version;
    unsigned long long mapping_identity, descriptor_count;
    const void *context;
    yvex_materialization_terminal_find_fn find;
    int complete;
} yvex_materialization_projection;
#define YVEX_MATERIALIZATION_PROJECTION_SCHEMA_VERSION 1u
struct yvex_artifact_lowering_map;
int yvex_materialization_project_artifact_lowering(
    const struct yvex_artifact_lowering_map *map,
    yvex_materialization_projection *out, yvex_error *err);
typedef struct {
    unsigned long long tensor_id;
    unsigned long long descriptor_index;
    char name[YVEX_MATERIALIZATION_NAME_CAP];
    yvex_tensor_role role;
    yvex_tensor_collection collection;
    yvex_tensor_scope scope;
    unsigned long long layer_index;
    unsigned long long predictor_index;
    unsigned long long expert_count;
    unsigned int rank;
    unsigned long long dims[YVEX_TENSOR_MAX_DIMS];
    unsigned int qtype;
    yvex_gguf_qtype_storage_class storage_class;
    unsigned long long row_width;
    unsigned long long row_count;
    unsigned long long block_size;
    unsigned long long bytes_per_block;
    unsigned long long encoded_bytes;
    unsigned long long absolute_offset;
    unsigned long long absolute_end_offset;
    unsigned int alignment;
    yvex_materialization_placement placement;
    yvex_materialization_access_mode access_mode;
    int backend_compatible;
} yvex_materialized_tensor_binding;
typedef struct {
    unsigned long long expert_index;
    unsigned long long expert_count;
    unsigned long long absolute_offset;
    unsigned long long encoded_bytes;
    unsigned long long block_size;
    int block_aligned;
} yvex_materialized_expert_subview;
typedef struct {
    yvex_materialization_status status;
    char artifact_identity[YVEX_SHA256_HEX_CAP];
    char plan_identity[YVEX_MATERIALIZATION_IDENTITY_CAP];
    unsigned long long tensor_count;
    unsigned long long payload_bytes;
    unsigned long long file_bytes;
    unsigned long long qtype_tensor_counts[YVEX_MATERIALIZATION_QTYPE_CAP];
    unsigned long long qtype_bytes[YVEX_MATERIALIZATION_QTYPE_CAP];
    unsigned long long file_backed_tensors;
    unsigned long long file_backed_bytes;
    unsigned long long staged_cache_tensors;
    unsigned long long staged_cache_bytes;
    unsigned long long backend_candidate_tensors;
    unsigned long long backend_candidate_bytes;
    unsigned long long mapped_virtual_bytes;
    unsigned long long file_backed_bytes_owned;
    unsigned long long process_resident_bytes;
    unsigned long long pageable_host_bytes;
    unsigned long long pinned_host_bytes;
    unsigned long long backend_allocated_bytes;
    unsigned long long staging_bytes;
    unsigned long long cache_bytes;
    unsigned long long graph_scratch_reserved_bytes;
    unsigned long long kv_reserved_bytes;
    unsigned long long peak_executor_owned_bytes;
    unsigned long long access_calls;
    unsigned long long payload_bytes_accessed;
    unsigned long long full_walks;
    unsigned long long snapshot_drift_count;
    unsigned long long committed_bindings;
    unsigned long long aborted_bindings;
    unsigned long long expert_subview_count;
    int committed;
    int cleanup_complete;
    int execution_ready;
} yvex_materialization_summary;
typedef struct yvex_materialization_plan yvex_materialization_plan;
typedef struct yvex_materialization_session yvex_materialization_session;

/* Immutable admitted model-to-runtime descriptor projection. */
#define YVEX_RUNTIME_DESCRIPTOR_IDENTITY_CAP 65u
#define YVEX_RUNTIME_DESCRIPTOR_QTYPE_CAP 43u
typedef struct {
    const char *status, *artifact_status, *reason, *next_row;
} yvex_runtime_descriptor_fact;
typedef enum {
    YVEX_RUNTIME_DESCRIPTOR_STATUS_REFUSED = 0,
    YVEX_RUNTIME_DESCRIPTOR_STATUS_READY
} yvex_runtime_descriptor_status;
typedef enum {
    YVEX_RUNTIME_DESCRIPTOR_FAILURE_NONE = 0,
    YVEX_RUNTIME_DESCRIPTOR_FAILURE_INVALID_ARGUMENT,
    YVEX_RUNTIME_DESCRIPTOR_FAILURE_ADMISSION,
    YVEX_RUNTIME_DESCRIPTOR_FAILURE_MATERIALIZATION,
    YVEX_RUNTIME_DESCRIPTOR_FAILURE_DUPLICATE_BINDING,
    YVEX_RUNTIME_DESCRIPTOR_FAILURE_MISSING_BINDING,
    YVEX_RUNTIME_DESCRIPTOR_FAILURE_ARCHITECTURE,
    YVEX_RUNTIME_DESCRIPTOR_FAILURE_QTYPE,
    YVEX_RUNTIME_DESCRIPTOR_FAILURE_ALLOCATION
} yvex_runtime_descriptor_failure_code;
typedef struct yvex_runtime_descriptor_failure {
    yvex_runtime_descriptor_failure_code code;
    unsigned long long tensor_index, expected, actual;
    char tensor_name[YVEX_MATERIALIZATION_NAME_CAP];
    const char *reason;
} yvex_runtime_descriptor_failure;
typedef struct {
    unsigned long long tensor_id, descriptor_index;
    const yvex_materialized_tensor_binding *binding;
    yvex_tensor_role role;
    yvex_tensor_collection collection;
    yvex_tensor_scope scope;
    unsigned long long layer_index, predictor_index;
    unsigned int qtype;
    yvex_materialization_placement placement;
    yvex_materialization_access_mode access_mode;
} yvex_runtime_tensor_binding;
typedef struct yvex_runtime_descriptor_summary {
    yvex_runtime_descriptor_status status;
    char artifact_identity[YVEX_SHA256_HEX_CAP];
    char materialization_plan_identity[YVEX_MATERIALIZATION_IDENTITY_CAP];
    char logical_model_identity[YVEX_RUNTIME_DESCRIPTOR_IDENTITY_CAP];
    char runtime_descriptor_identity[YVEX_RUNTIME_DESCRIPTOR_IDENTITY_CAP];
    char runtime_numeric_identity[YVEX_RUNTIME_DESCRIPTOR_IDENTITY_CAP];
    char runtime_hadamard_revision[128];
    unsigned int runtime_numeric_schema_version;
    unsigned long long runtime_compute_policy_count, runtime_activation_policy_count;
    unsigned long long runtime_sparse_topk_policy_count, tensor_count, payload_bytes;
    unsigned long long qtype_tensor_counts[YVEX_RUNTIME_DESCRIPTOR_QTYPE_CAP];
    unsigned long long qtype_bytes[YVEX_RUNTIME_DESCRIPTOR_QTYPE_CAP];
    unsigned long long role_counts[YVEX_TENSOR_ROLE_COUNT];
    unsigned long long global_bindings, main_layer_bindings, draft_bindings;
    unsigned long long routed_expert_bindings, expert_subview_count;
    unsigned long long missing_required_bindings, duplicate_bindings, unexpected_bindings;
    unsigned long long layer_count, draft_layer_count, routed_experts, experts_per_token;
    unsigned long long vocabulary_size;
    yvex_model_execution_descriptor model_execution;
    int tokenizer_metadata_available, graph_execution_ready, generation_ready;
} yvex_runtime_descriptor_summary;
typedef struct yvex_runtime_descriptor yvex_runtime_descriptor;
typedef struct {
    const char *logical_model_identity;
    const char *runtime_numeric_identity;
    const char *runtime_hadamard_revision;
    unsigned int runtime_numeric_schema_version;
    unsigned long long runtime_compute_policy_count;
    unsigned long long runtime_activation_policy_count;
    unsigned long long runtime_sparse_topk_policy_count;
    unsigned long long layer_count, draft_layer_count;
    unsigned long long routed_experts, experts_per_token, vocabulary_size;
    const yvex_model_execution_descriptor *model_execution;
} yvex_runtime_descriptor_family_facts;
const char *yvex_runtime_descriptor_failure_name(yvex_runtime_descriptor_failure_code code);
int yvex_runtime_descriptor_build(yvex_runtime_descriptor **out,
                                  const yvex_complete_artifact_admission *admission,
                                  const yvex_materialization_session *session,
                                  const yvex_runtime_descriptor_family_facts *family,
                                  yvex_runtime_descriptor_failure *failure, yvex_error *err);
int yvex_runtime_descriptor_build_projected(
    yvex_runtime_descriptor **out,
    const yvex_complete_artifact_admission *admission,
    const yvex_materialization_session *session,
    const yvex_runtime_descriptor_family_facts *family,
    const yvex_materialization_projection *projection,
    yvex_runtime_descriptor_failure *failure, yvex_error *err);
void yvex_runtime_descriptor_close(yvex_runtime_descriptor *descriptor);
const yvex_runtime_descriptor_summary *
yvex_runtime_descriptor_summary_get(const yvex_runtime_descriptor *descriptor);
const yvex_runtime_tensor_binding *
yvex_runtime_descriptor_tensor_at(const yvex_runtime_descriptor *descriptor,
                                  unsigned long long index);
int yvex_runtime_descriptor_import(
    yvex_runtime_descriptor **out, const yvex_runtime_descriptor_summary *summary,
    const yvex_runtime_tensor_binding *bindings, unsigned long long binding_count,
    const yvex_materialization_session *session, yvex_runtime_descriptor_failure *failure,
    yvex_error *err);
const yvex_runtime_tensor_binding *
yvex_runtime_descriptor_find_role(const yvex_runtime_descriptor *descriptor, yvex_tensor_role role,
                                  yvex_tensor_scope scope, unsigned long long layer_index,
                                  unsigned long long predictor_index);
typedef enum {
    YVEX_MATERIALIZATION_READ_MISS = 0,
    YVEX_MATERIALIZATION_READ_HIT = 1,
    YVEX_MATERIALIZATION_READ_INVALID = -1
} yvex_materialization_read_result;
typedef int (*yvex_materialization_read_resolve_fn)(
    const void *context, const yvex_materialized_tensor_binding *binding,
    const unsigned char **data, unsigned long long *bytes);
typedef int (*yvex_materialization_read_note_fn)(const void *context,
                                                 unsigned long long bytes);
typedef void (*yvex_materialization_read_detach_fn)(const void *context);
typedef struct {
    const void *context;
    yvex_materialization_read_resolve_fn resolve;
    yvex_materialization_read_note_fn note_access;
    yvex_materialization_read_detach_fn detached;
} yvex_materialization_read_provider;
typedef struct {
    unsigned long long artifact_read_calls, artifact_bytes_read;
    unsigned long long resident_read_calls, resident_bytes_read;
} yvex_materialization_access_summary;
typedef void (*yvex_materialization_progress_fn)(void *context,
                                                 const yvex_materialization_summary *summary,
                                                 const yvex_materialized_tensor_binding *binding);
void yvex_materialization_options_default(yvex_materialization_options *options);
const char *yvex_materialization_failure_name(yvex_materialization_failure_code code);
void yvex_materialization_plan_close(yvex_materialization_plan *plan);
const yvex_materialization_summary *
yvex_materialization_plan_summary(const yvex_materialization_plan *plan);
int yvex_materialization_plan_build(
    yvex_materialization_plan **out, const yvex_complete_artifact_admission *admission,
    const yvex_artifact *artifact, const yvex_gguf *gguf, const yvex_tensor_table *tensors,
    const yvex_materialization_projection *projection,
    const yvex_materialization_options *options, yvex_materialization_failure *failure,
    yvex_error *err);
int yvex_materialization_plan_import(
    yvex_materialization_plan **out, const yvex_complete_artifact_admission *admission,
    const yvex_materialization_summary *summary,
    const yvex_materialized_tensor_binding *bindings, unsigned long long binding_count,
    yvex_materialization_failure *failure, yvex_error *err);
int yvex_materialization_session_open(yvex_materialization_session **out,
                                      const yvex_materialization_plan *plan,
                                      const yvex_artifact *artifact,
                                      const yvex_materialization_options *options,
                                      yvex_materialization_failure *failure, yvex_error *err);
int yvex_materialization_session_commit(yvex_materialization_session *session,
                                        yvex_materialization_failure *failure, yvex_error *err);
void yvex_materialization_session_close(yvex_materialization_session *session);
const yvex_materialization_summary *
yvex_materialization_session_summary(const yvex_materialization_session *session);
const yvex_materialized_tensor_binding *
yvex_materialization_session_tensor_at(const yvex_materialization_session *session,
                                       unsigned long long index);
int yvex_materialization_session_read(yvex_materialization_session *session,
                                      const yvex_materialized_tensor_binding *binding,
                                      unsigned long long binding_offset, void *dst, size_t len,
                                      yvex_materialization_failure *failure, yvex_error *err);
int yvex_materialization_session_borrow(
    yvex_materialization_session *session, const yvex_materialized_tensor_binding *binding,
    unsigned long long binding_offset, size_t len, const unsigned char **data,
    yvex_materialization_failure *failure, yvex_error *err);
int yvex_materialization_session_attach_read_provider(
    yvex_materialization_session *session, const yvex_materialization_read_provider *provider,
    yvex_materialization_failure *failure, yvex_error *err);
int yvex_materialization_session_detach_read_provider(
    yvex_materialization_session *session, const void *context,
    yvex_materialization_failure *failure, yvex_error *err);
int yvex_materialization_session_access_summary(
    const yvex_materialization_session *session, yvex_materialization_access_summary *out,
    yvex_error *err);
int yvex_materialization_session_walk_payload(yvex_materialization_session *session,
                                              yvex_materialization_progress_fn progress,
                                              void *progress_context,
                                              yvex_materialization_failure *failure,
                                              yvex_error *err);
int yvex_materialization_session_expert_subview(const yvex_materialization_session *session,
                                                const yvex_materialized_tensor_binding *binding,
                                                unsigned long long expert_index,
                                                yvex_materialized_expert_subview *out,
                                                yvex_materialization_failure *failure,
                                                yvex_error *err);
#ifdef __cplusplus
}
#endif

#endif /* INCLUDE_YVEX_INTERNAL_ARTIFACT_H_INCLUDED */
