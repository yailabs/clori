/* Immutable model resources and isolated mutable sequence state remain explicit in this runtime ABI. */
#ifndef INCLUDE_YVEX_INTERNAL_RUNTIME_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_RUNTIME_H_INCLUDED
#include <string.h>
#include <yvex/artifact.h>
#include <yvex/backend.h>
#include <yvex/core.h>
#include <yvex/gguf.h>
#include <yvex/graph.h>
#include <yvex/internal/artifact.h>
#include <yvex/internal/benchmark.h>
#include <yvex/internal/execution.h>
#include <yvex/internal/graph.h>
#include <yvex/internal/graph_state.h>
#include <yvex/model.h>
#include <yvex/registry.h>
#include <yvex/tokenizer.h>
#ifdef __cplusplus
extern "C" {
#endif
static inline void yvex_runtime_identity_copy(char destination[YVEX_SHA256_HEX_CAP], const char *source)
{
    size_t length = source ? strnlen(source, YVEX_SHA256_HEX_CAP - 1u) : 0u;
    memset(destination, 0, YVEX_SHA256_HEX_CAP);
    if (length) memcpy(destination, source, length);
}
#define YVEX_RUNTIME_REASON_CAP 256u
#define YVEX_RUNTIME_BINDING_SCHEMA_V7 7u
#define YVEX_RUNTIME_BINDING_SCHEMA_V8 8u
#define YVEX_RUNTIME_BINDING_SCHEMA_CURRENT YVEX_RUNTIME_BINDING_SCHEMA_V8
#define YVEX_RUNTIME_BINDING_SUFFIX ".yvex-runtime-binding"
typedef enum {
    YVEX_RUNTIME_BINDING_FAILURE_NONE = 0, YVEX_RUNTIME_BINDING_FAILURE_INVALID_ARGUMENT,
    YVEX_RUNTIME_BINDING_FAILURE_UNSEALED_INPUT, YVEX_RUNTIME_BINDING_FAILURE_IDENTITY,
    YVEX_RUNTIME_BINDING_FAILURE_BOUNDS, YVEX_RUNTIME_BINDING_FAILURE_ALLOCATION,
    YVEX_RUNTIME_BINDING_FAILURE_DIRECTORY, YVEX_RUNTIME_BINDING_FAILURE_CREATE,
    YVEX_RUNTIME_BINDING_FAILURE_WRITE, YVEX_RUNTIME_BINDING_FAILURE_SYNC,
    YVEX_RUNTIME_BINDING_FAILURE_CONFLICT, YVEX_RUNTIME_BINDING_FAILURE_PUBLISH,
    YVEX_RUNTIME_BINDING_FAILURE_OPEN, YVEX_RUNTIME_BINDING_FAILURE_FORMAT,
    YVEX_RUNTIME_BINDING_FAILURE_SCHEMA, YVEX_RUNTIME_BINDING_FAILURE_TRUNCATED,
    YVEX_RUNTIME_BINDING_FAILURE_TRAILING_DATA, YVEX_RUNTIME_BINDING_FAILURE_ARTIFACT,
    YVEX_RUNTIME_BINDING_FAILURE_MATERIALIZATION, YVEX_RUNTIME_BINDING_FAILURE_DESCRIPTOR,
    YVEX_RUNTIME_BINDING_FAILURE_ATTENTION, YVEX_RUNTIME_BINDING_FAILURE_COMPATIBILITY
} yvex_runtime_binding_failure_code;
typedef struct yvex_runtime_binding_failure {
    yvex_runtime_binding_failure_code code;
    unsigned long long record_index, expected, actual;
    char field[64], path[YVEX_PATH_CAP];
    const char *reason;
} yvex_runtime_binding_failure;
#define YVEX_RUNTIME_MODEL_SCHEMA_V1 1u
#define YVEX_RUNTIME_FAMILY_ADAPTER_SCHEMA_V3 3u
#define YVEX_RUNTIME_EXECUTION_DESCRIPTOR_SCHEMA_V2 2u
typedef enum {
    YVEX_SEQUENCE_MIXER_SOFTMAX_ATTENTION = 0, YVEX_SEQUENCE_MIXER_RECURRENT_LINEAR, YVEX_SEQUENCE_MIXER_STATE_SPACE
} yvex_sequence_mixer_family;
typedef enum {
    YVEX_SEQUENCE_MIXER_DENSE_MHA = 0, YVEX_SEQUENCE_MIXER_GROUPED_QUERY,
    YVEX_SEQUENCE_MIXER_MULTI_QUERY, YVEX_SEQUENCE_MIXER_SLIDING_WINDOW,
    YVEX_SEQUENCE_MIXER_LATENT_ATTENTION, YVEX_SEQUENCE_MIXER_SPARSE_ATTENTION,
    YVEX_SEQUENCE_MIXER_COMPRESSED_SPARSE, YVEX_SEQUENCE_MIXER_HIERARCHICAL_COMPRESSED,
    YVEX_SEQUENCE_MIXER_CROSS_ATTENTION, YVEX_SEQUENCE_MIXER_DELTANET,
    YVEX_SEQUENCE_MIXER_GATED_DELTANET, YVEX_SEQUENCE_MIXER_KIMI_DELTA, YVEX_SEQUENCE_MIXER_MAMBA
} yvex_sequence_mixer_semantics;
typedef enum {
    YVEX_RUNTIME_MIXER_UNSUPPORTED = 0, YVEX_RUNTIME_MIXER_SUPPORTED,
    YVEX_RUNTIME_MIXER_NOT_IMPLEMENTED, YVEX_RUNTIME_MIXER_NOT_ADMITTED
} yvex_runtime_mixer_capability_state;
typedef struct {
    yvex_sequence_mixer_family family;
    yvex_sequence_mixer_semantics semantics;
    yvex_runtime_mixer_capability_state state;
    const char *reason;
} yvex_runtime_mixer_capability;
typedef enum {
    YVEX_RUNTIME_MODE_EAGER = 0, YVEX_RUNTIME_MODE_PIECEWISE,
    YVEX_RUNTIME_MODE_FULL, YVEX_RUNTIME_MODE_AUTO
} yvex_runtime_execution_mode;
typedef enum {
    YVEX_RUNTIME_SCOPE_ATTENTION_CORE = 0, YVEX_RUNTIME_SCOPE_ATTENTION_ENVELOPE,
    YVEX_RUNTIME_SCOPE_RELEASE_ATTENTION_SET
} yvex_runtime_execution_scope;
typedef enum {
    YVEX_RUNTIME_TRACE_NONE = 0, YVEX_RUNTIME_TRACE_SUMMARY,
    YVEX_RUNTIME_TRACE_STAGES, YVEX_RUNTIME_TRACE_FULL
} yvex_runtime_trace_policy;
typedef enum {
    YVEX_RUNTIME_LIFECYCLE_ARTIFACT_OPEN = 0, YVEX_RUNTIME_LIFECYCLE_ARTIFACT_HASH,
    YVEX_RUNTIME_LIFECYCLE_ARTIFACT_ADMISSION, YVEX_RUNTIME_LIFECYCLE_BINDING_OPEN,
    YVEX_RUNTIME_LIFECYCLE_MATERIALIZATION_OPEN, YVEX_RUNTIME_LIFECYCLE_MODEL_SEAL,
    YVEX_RUNTIME_LIFECYCLE_RESIDENCY, YVEX_RUNTIME_LIFECYCLE_BACKEND_OPEN,
    YVEX_RUNTIME_LIFECYCLE_WORKSPACE_PREPARE, YVEX_RUNTIME_LIFECYCLE_GRAPH_WARMUP,
    YVEX_RUNTIME_LIFECYCLE_GRAPH_CAPTURE, YVEX_RUNTIME_LIFECYCLE_GRAPH_INSTANTIATE,
    YVEX_RUNTIME_LIFECYCLE_EXECUTION, YVEX_RUNTIME_LIFECYCLE_PUBLICATION,
    YVEX_RUNTIME_LIFECYCLE_CLEANUP, YVEX_RUNTIME_LIFECYCLE_COUNT
} yvex_runtime_lifecycle_phase;
typedef int (*yvex_runtime_progress_callback)(void *, yvex_runtime_lifecycle_phase, unsigned long long,
                                              unsigned long long);
typedef struct {
    int attention_semantics_ready, attention_core_ready, attention_envelope_ready;
    int cpu_prefill_eager_ready, cpu_decode_eager_ready, cuda_prefill_eager_ready,
        cuda_decode_eager_ready;
    int cuda_eager_implemented, cuda_piecewise_graph_implemented, cuda_full_graph_implemented;
    int cuda_prefill_piecewise_graph_ready, cuda_decode_piecewise_graph_ready;
    int cuda_prefill_full_graph_ready, cuda_decode_full_graph_ready;
    int attention_weight_residency_ready, attention_workspace_ready;
    int attention_state_delta_ready, attention_operator_ready, attention_trace_ready;
    int attention_profile_ready, attention_benchmark_ready, mixed_attention_ready;
    int speculative_attention_ready, persistent_kv_ready;
    int moe_plan_ready, moe_router_ready, moe_routed_expert_ready, moe_shared_expert_ready, moe_block_ready;
    int transformer_ready, output_head_binding_ready, output_head_projection_ready;
    int logits_cpu_ready, logits_cuda_ready, logits_prefill_ready, logits_decode_ready;
    int logits_full_vocabulary_ready, logits_hidden_contract_ready, logits_partial_progress_ready, logits_ready;
    int generation_ready;
} yvex_runtime_capabilities;
#define YVEX_RUNTIME_EXECUTION_CAPABILITY_SCHEMA_V2 2u
int yvex_runtime_capabilities_identity(const yvex_runtime_capabilities *facts,
                                       char output[YVEX_SHA256_HEX_CAP]);
int yvex_runtime_capabilities_admitted_by(const yvex_runtime_capabilities *facts,
                                          const yvex_runtime_capabilities *maximum);
int yvex_runtime_capabilities_contract_valid(const yvex_runtime_capabilities *facts);
const struct yvex_runtime_family_adapter *yvex_runtime_family_at(unsigned long long index);
struct yvex_model_family_api;
struct yvex_transformer_family_policy;
struct yvex_logits_family_policy;
struct yvex_speculation_family_policy;
typedef struct yvex_runtime_family_adapter {
    unsigned int schema_version;
    unsigned long long adapter_id, adapter_version;
    const char *target_id, *family_name, *operator_family_key;
    const char *operator_artifact_filename, *logical_transform_identity;
    yvex_sequence_mixer_family mixer_family;
    int (*mixer_capability)(yvex_sequence_mixer_semantics, yvex_runtime_mixer_capability *);
    const yvex_graph_family_api *(*graph)(void);
    int (*execution_capabilities)(yvex_runtime_capabilities *out);
    int (*transformer_policy)(const yvex_runtime_descriptor_summary *, struct yvex_transformer_family_policy *);
    int (*logits_policy)(struct yvex_logits_family_policy *out);
    int (*speculation_policy)(const yvex_runtime_descriptor_summary *, struct yvex_speculation_family_policy *);
} yvex_runtime_family_adapter;
typedef struct {
    const char *directory;
    const yvex_complete_artifact_admission *admission;
    const yvex_artifact_physical_compatibility *physical_compatibility;
    const yvex_materialization_session *materialization;
    const yvex_runtime_descriptor *runtime_descriptor;
    const yvex_attention_plan *attention_plan;
    const yvex_attention_plan *draft_attention_plan;
    unsigned long long family_adapter_id, family_adapter_version;
    const char *artifact_format;
    unsigned int artifact_format_version;
    const char *logical_transform_identity;
    yvex_runtime_capabilities capabilities;
} yvex_runtime_binding_prepare_request;
typedef struct yvex_runtime_binding_summary {
    unsigned int schema_version;
    unsigned long long family_adapter_id, family_adapter_version;
    unsigned long long tensor_count, layer_count, draft_layer_count, file_bytes;
    unsigned long long source_snapshot_identity, mapping_identity;
    unsigned int artifact_format_version;
    char artifact_format[16];
    char identity[YVEX_SHA256_HEX_CAP];
    char execution_capability_identity[YVEX_SHA256_HEX_CAP];
    char payload_identity[YVEX_SHA256_HEX_CAP];
    char artifact_transform_identity[YVEX_SHA256_HEX_CAP], logical_transform_identity[YVEX_SHA256_HEX_CAP];
    char profile_identity[YVEX_SHA256_HEX_CAP], quant_execution_identity[YVEX_SHA256_HEX_CAP];
    char artifact_identity[YVEX_SHA256_HEX_CAP];
    char materialization_identity[YVEX_SHA256_HEX_CAP], logical_model_identity[YVEX_SHA256_HEX_CAP];
    char runtime_numeric_identity[YVEX_SHA256_HEX_CAP];
    char runtime_descriptor_identity[YVEX_SHA256_HEX_CAP], model_execution_identity[YVEX_SHA256_HEX_CAP];
    char attention_plan_identity[YVEX_SHA256_HEX_CAP], moe_plan_identity[YVEX_SHA256_HEX_CAP];
    char draft_attention_plan_identity[YVEX_SHA256_HEX_CAP];
    char draft_moe_plan_identity[YVEX_SHA256_HEX_CAP];
    char semantic_graph_identity[YVEX_SHA256_HEX_CAP], executable_graph_identity[YVEX_SHA256_HEX_CAP];
    yvex_artifact_physical_compatibility physical_compatibility;
    yvex_runtime_capabilities capabilities;
} yvex_runtime_binding_summary;
typedef struct {
    char path[YVEX_PATH_CAP];
    yvex_runtime_binding_summary summary;
    int published;
} yvex_runtime_binding_prepare_result;
typedef struct yvex_runtime_binding yvex_runtime_binding;
int yvex_runtime_binding_prepare(const yvex_runtime_binding_prepare_request *request,
                                 yvex_runtime_binding_prepare_result *result,
                                 yvex_runtime_binding_failure *failure, yvex_error *err);
int yvex_runtime_binding_open(yvex_runtime_binding **out, const char *path,
    yvex_runtime_binding_summary *summary,
    yvex_complete_artifact_admission *admission,
    yvex_runtime_binding_failure *failure, yvex_error *err);
void yvex_runtime_binding_close(yvex_runtime_binding *binding);
int yvex_runtime_binding_import_materialization(
    const yvex_runtime_binding *binding, const yvex_artifact *artifact,
    const yvex_materialization_options *options, yvex_materialization_plan **plan_out,
    yvex_materialization_session **session_out, yvex_runtime_binding_failure *failure,
    yvex_error *err);
int yvex_runtime_binding_import_graph(
    const yvex_runtime_binding *binding, const yvex_materialization_session *session,
    yvex_runtime_descriptor **descriptor_out, yvex_attention_plan **attention_out,
    yvex_attention_plan **draft_attention_out,
    yvex_runtime_binding_failure *failure, yvex_error *err);
typedef enum {
    YVEX_RUNTIME_MODEL_FAILURE_NONE = 0, YVEX_RUNTIME_MODEL_FAILURE_INVALID_ARGUMENT,
    YVEX_RUNTIME_MODEL_FAILURE_ADAPTER, YVEX_RUNTIME_MODEL_FAILURE_BINDING,
    YVEX_RUNTIME_MODEL_FAILURE_ARTIFACT, YVEX_RUNTIME_MODEL_FAILURE_IDENTITY,
    YVEX_RUNTIME_MODEL_FAILURE_MATERIALIZATION, YVEX_RUNTIME_MODEL_FAILURE_DESCRIPTOR,
    YVEX_RUNTIME_MODEL_FAILURE_GRAPH, YVEX_RUNTIME_MODEL_FAILURE_BACKEND,
    YVEX_RUNTIME_MODEL_FAILURE_DRIFT, YVEX_RUNTIME_MODEL_FAILURE_BUSY,
    YVEX_RUNTIME_MODEL_FAILURE_CANCELLED,
    YVEX_RUNTIME_MODEL_FAILURE_ALLOCATION, YVEX_RUNTIME_MODEL_FAILURE_CLEANUP
} yvex_runtime_model_failure_code;
typedef struct yvex_runtime_model_failure {
    yvex_runtime_model_failure_code code;
    unsigned long long expected, actual;
    char field[64];
    const char *reason;
} yvex_runtime_model_failure;
typedef struct {
    const char *artifact_path, *runtime_binding_path, *target_id;
    yvex_backend_kind residency_backend;
    unsigned long long maximum_host_bytes, maximum_device_bytes;
    yvex_runtime_progress_callback progress;
    void *progress_context;
} yvex_runtime_model_open_request;
typedef struct {
    unsigned int schema_version;
    int sealed, valid;
    char runtime_model_identity[YVEX_SHA256_HEX_CAP];
    char runtime_binding_identity[YVEX_SHA256_HEX_CAP];
    char artifact_identity[YVEX_SHA256_HEX_CAP];
    char materialization_identity[YVEX_SHA256_HEX_CAP];
    char runtime_descriptor_identity[YVEX_SHA256_HEX_CAP];
    char runtime_numeric_identity[YVEX_SHA256_HEX_CAP];
    char semantic_graph_identity[YVEX_SHA256_HEX_CAP];
    char executable_graph_identity[YVEX_SHA256_HEX_CAP];
    char physical_execution_identity[YVEX_SHA256_HEX_CAP];
    unsigned long long artifact_hash_passes, artifact_bytes_hashed;
    unsigned long long gguf_directory_parses, runtime_binding_parses;
    unsigned long long runtime_model_builds, runtime_descriptor_builds;
    unsigned long long semantic_graph_builds, executable_graph_builds;
    unsigned long long drift_checks, invalidation_count;
    unsigned long long tensor_count, attention_layer_count, draft_attention_layer_count;
    unsigned long long attention_binding_count, draft_attention_binding_count;
    unsigned long long physical_execution_decision_count;
    double lifecycle_seconds[YVEX_RUNTIME_LIFECYCLE_COUNT], total_seconds;
    yvex_runtime_capabilities capabilities;
} yvex_runtime_model_summary;
typedef struct yvex_runtime_model yvex_runtime_model;
typedef struct yvex_runtime_execution_session yvex_runtime_execution_session;
typedef struct yvex_runtime_cleanup_lease yvex_runtime_cleanup_lease;
#define YVEX_RUNTIME_RESIDENCY_SCHEMA_V4 4u
typedef enum {
    YVEX_RUNTIME_RESIDENCY_FAILURE_NONE = 0,
    YVEX_RUNTIME_RESIDENCY_FAILURE_INVALID_ARGUMENT,
    YVEX_RUNTIME_RESIDENCY_FAILURE_MODEL,
    YVEX_RUNTIME_RESIDENCY_FAILURE_PLAN,
    YVEX_RUNTIME_RESIDENCY_FAILURE_MISSING_BINDING,
    YVEX_RUNTIME_RESIDENCY_FAILURE_DUPLICATE_BINDING,
    YVEX_RUNTIME_RESIDENCY_FAILURE_GEOMETRY,
    YVEX_RUNTIME_RESIDENCY_FAILURE_BUDGET,
    YVEX_RUNTIME_RESIDENCY_FAILURE_ALLOCATION,
    YVEX_RUNTIME_RESIDENCY_FAILURE_READ,
    YVEX_RUNTIME_RESIDENCY_FAILURE_ATTACH,
    YVEX_RUNTIME_RESIDENCY_FAILURE_LIFECYCLE
} yvex_runtime_residency_failure_code;
typedef struct {
    yvex_runtime_residency_failure_code code;
    unsigned long long tensor_id, layer_index, expected, actual;
    yvex_tensor_role role;
    const char *reason;
} yvex_runtime_residency_failure;
typedef struct {
    unsigned long long maximum_host_bytes;
} yvex_runtime_residency_options;
typedef struct {
    unsigned int schema_version;
    int sealed, attached, host_ready, host_locked, cuda_ready, invalidated;
    int model_complete, core_complete, envelope_complete, output_head_complete;
    unsigned long long generation, expected_model_binding_count, model_binding_count, expected_core_binding_count;
    unsigned long long expected_envelope_binding_count, core_binding_count, envelope_binding_count, binding_count;
    unsigned long long expected_output_head_binding_count, output_head_binding_count, output_head_encoded_bytes;
    unsigned long long accelerator_encoded_bytes, encoded_bytes, host_resident_bytes, device_resident_bytes;
    unsigned long long cuda_addressable_bytes, cuda_upload_bytes, cuda_upload_count, cuda_host_registration_count;
    unsigned long long cuda_managed_bytes, cuda_managed_allocation_count;
    unsigned long long cold_artifact_read_calls, cold_artifact_bytes_read;
    unsigned long long resident_read_calls, resident_bytes_read;
    unsigned long long qtype_binding_counts[YVEX_RUNTIME_DESCRIPTOR_QTYPE_CAP];
    unsigned long long qtype_bytes[YVEX_RUNTIME_DESCRIPTOR_QTYPE_CAP];
    char payload_digest[YVEX_SHA256_HEX_CAP], residency_identity[YVEX_SHA256_HEX_CAP];
    char output_head_residency_identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_residency_summary;
typedef struct yvex_runtime_residency yvex_runtime_residency;
typedef struct yvex_runtime_state_residency yvex_runtime_state_residency;
typedef struct yvex_runtime_component_session yvex_runtime_component_session;
typedef struct {
    int sealed, cuda_ready, paged, invalidated;
    unsigned long long layer_count, host_bytes, device_bytes, virtual_device_bytes;
    unsigned long long page_granularity, page_commit_count, page_release_count, upload_bytes, upload_count;
    unsigned long long copy_bytes, copy_count, device_stage_bytes, device_stage_count;
    unsigned long long generation, staged_layer_count, commit_count, abort_count;
    char layout_identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_state_residency_summary;
typedef struct {
    const yvex_runtime_residency *residency;
    const yvex_runtime_binding_summary *binding;
    const yvex_runtime_family_adapter *adapter;
    const yvex_attention_plan *attention, *draft_attention;
    const yvex_runtime_descriptor *descriptor;
    const yvex_physical_execution_ir *physical_execution;
    const yvex_tokenizer *tokenizer;
    yvex_materialization_session *materialization;
} yvex_runtime_model_view;
int yvex_runtime_residency_prepare(yvex_runtime_residency **out, yvex_runtime_model *model,
    const yvex_runtime_residency_options *options, yvex_runtime_residency_failure *failure, yvex_error *err);
int yvex_runtime_component_residency_prepare(yvex_runtime_residency **, yvex_materialization_session *,
    const char *, const yvex_runtime_residency_options *, yvex_runtime_residency_failure *, yvex_error *);
int yvex_runtime_residency_close(yvex_runtime_residency **residency, yvex_error *err);
int yvex_runtime_residency_snapshot(const yvex_runtime_residency *residency, yvex_runtime_residency_summary *summary,
    const unsigned char **arena, unsigned long long *arena_bytes, yvex_error *err);
int yvex_runtime_residency_binding_view(const yvex_runtime_residency *,
    const yvex_materialized_tensor_binding *, const unsigned char **, unsigned long long *, yvex_error *);
/* A non-null backend is consumed as an already-open CUDA owner before a large arena is registered. */
int yvex_runtime_residency_cuda_session_attach(yvex_runtime_residency *residency, yvex_backend **backend,
    unsigned long long maximum_device_bytes, int *uploaded, yvex_runtime_residency_summary *summary, yvex_error *err);
int yvex_runtime_component_session_open(yvex_runtime_component_session **,
    const yvex_complete_artifact_admission *, const yvex_artifact *, const yvex_gguf *,
    const yvex_tensor_table *, yvex_backend_kind, unsigned long long, unsigned long long, yvex_error *);
int yvex_runtime_component_session_close(yvex_runtime_component_session **, yvex_error *);
yvex_materialization_session *yvex_runtime_component_session_materialization(const yvex_runtime_component_session *);
const yvex_runtime_residency *yvex_runtime_component_session_residency(const yvex_runtime_component_session *);
yvex_backend *yvex_runtime_component_session_backend(const yvex_runtime_component_session *);
const yvex_runtime_residency_summary *yvex_runtime_component_session_summary(const yvex_runtime_component_session *);
int yvex_runtime_residency_invalidate(yvex_runtime_residency *residency, yvex_error *err);
int yvex_runtime_state_residency_prepare(yvex_runtime_state_residency **out, yvex_backend *backend,
    const yvex_graph_attention_capacity_plan *capacity, const yvex_attention_state_provider *provider,
    unsigned long long prior_host_bytes, unsigned long long maximum_host_bytes, unsigned long long prior_device_bytes,
    unsigned long long maximum_device_bytes, yvex_error *err);
typedef enum { YVEX_RUNTIME_STATE_BEGIN = 0, YVEX_RUNTIME_STATE_STAGE } yvex_runtime_state_action;
int yvex_runtime_state_residency_transition(yvex_runtime_state_residency *,
    const yvex_attention_state_provider *, const yvex_attention_publication *,
    unsigned long long, unsigned long long, yvex_runtime_state_action, yvex_error *);
int yvex_runtime_state_residency_publish(yvex_runtime_state_residency *residency, yvex_error *err);
void yvex_runtime_state_residency_commit(yvex_runtime_state_residency *residency);
void yvex_runtime_state_residency_abort(yvex_runtime_state_residency *residency);
int yvex_runtime_state_residency_reset(yvex_runtime_state_residency *residency, yvex_error *err);
int yvex_runtime_state_residency_invalidate(yvex_runtime_state_residency *residency, yvex_error *err);
int yvex_runtime_state_residency_close(yvex_runtime_state_residency **residency, yvex_error *err);
int yvex_runtime_state_residency_summary_copy(const yvex_runtime_state_residency *residency,
                                              yvex_runtime_state_residency_summary *out, yvex_error *err);
const yvex_runtime_family_adapter *yvex_runtime_family_adapter_find(const char *target_id);
/* A cleanup failure may publish an unpublished model in out; close retries exact ownership. */
int yvex_runtime_model_open(yvex_runtime_model **out, const yvex_runtime_model_open_request *request,
                            yvex_runtime_model_failure *failure, yvex_error *err);
int yvex_runtime_model_validate(yvex_runtime_model *model, yvex_runtime_model_failure *failure, yvex_error *err);
int yvex_runtime_model_summary_copy(const yvex_runtime_model *model, yvex_runtime_model_summary *out, yvex_error *err);
void yvex_runtime_model_close(yvex_runtime_model **model);
const yvex_runtime_model_view *yvex_runtime_model_view_get(const yvex_runtime_model *model);
typedef struct {
    yvex_backend_kind backend;
    unsigned long long maximum_host_bytes, maximum_device_bytes;
    const yvex_attention_state_provider_factory *attention_state_factory;
} yvex_runtime_session_open_request;
typedef struct {
    int open, busy, cancelled, invalidated;
    yvex_backend_kind backend;
    yvex_runtime_capabilities capabilities;
    unsigned long long execution_count, failure_count, cancellation_count;
    unsigned long long warm_artifact_hash_passes, warm_weight_artifact_reads;
    unsigned long long warm_weight_upload_bytes, warm_host_allocations;
    unsigned long long warm_device_allocations, warm_device_frees;
    unsigned long long peak_host_bytes, peak_device_bytes;
    unsigned long long resident_binding_count, resident_encoded_bytes;
    unsigned long long host_resident_bytes, device_resident_bytes;
    unsigned long long upload_bytes, upload_count, residency_generation, workspace_generation;
    unsigned long long workspace_bytes, device_workspace_bytes, workspace_peak_bytes;
    unsigned long long workspace_allocation_count, host_workspace_bytes, host_workspace_peak_bytes;
    unsigned long long workspace_capacity_failure_count;
    int host_workspace_owned, host_workspace_pinned;
    int device_index, compute_capability_major, compute_capability_minor;
    unsigned long long total_device_bytes, sustainable_read_bytes_per_second, sustainable_copy_bytes_per_second;
    unsigned long long sustainable_coherent_host_bytes_per_second;
    char device_name[128], bandwidth_evidence_identity[YVEX_SHA256_HEX_CAP];
    char residency_identity[YVEX_SHA256_HEX_CAP], workspace_identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_session_summary;
typedef struct {
    const yvex_runtime_model *model;
    yvex_backend *backend;
    const yvex_attention_state_provider *attention_state_provider;
    const yvex_attention_state_provider *draft_attention_state_provider;
    yvex_attention_workspace *attention_workspace;
    yvex_runtime_state_residency *state_residency;
    yvex_runtime_state_residency *draft_state_residency;
} yvex_runtime_session_view;
/* A cleanup failure may retain an unpublished closing session in out; retry close discharges it. */
int yvex_runtime_session_open(yvex_runtime_execution_session **out, yvex_runtime_model *model,
    const yvex_runtime_session_open_request *request, yvex_runtime_model_failure *failure,
    yvex_error *err);
int yvex_runtime_session_prepare_attention_workspace(yvex_runtime_execution_session *session,
    yvex_runtime_execution_mode mode, yvex_runtime_execution_scope scope,
    yvex_attention_evidence_level evidence_level,
    const yvex_graph_attention_capacity_plan *capacity,
    unsigned long long minimum_bytes,
    yvex_runtime_model_failure *failure, yvex_error *err);
int yvex_runtime_session_summary_copy(const yvex_runtime_execution_session *session,
                                      yvex_runtime_session_summary *out, yvex_error *err);
int yvex_runtime_session_close(yvex_runtime_execution_session **session, yvex_error *err);
const yvex_runtime_session_view *yvex_runtime_session_view_get(const yvex_runtime_execution_session *session);
int yvex_runtime_device_view_bind(yvex_execution_device_view *out, yvex_execution_device_value_kind kind,
    yvex_runtime_model *model, yvex_runtime_execution_session *session,
    const yvex_attention_state_provider *provider,
    const yvex_compiled_execution_profile *profile, const yvex_device_tensor *tensor,
    unsigned long long offset, unsigned long long rows, unsigned long long columns,
    yvex_error *err);
int yvex_runtime_cleanup_lease_acquire(
    yvex_runtime_cleanup_lease **out, const yvex_runtime_model_open_request *model_request,
    const yvex_runtime_session_open_request *session_request,
    yvex_runtime_model **borrowed_model, yvex_runtime_execution_session **borrowed_session,
    yvex_runtime_model_failure *failure, yvex_error *err);
int yvex_runtime_cleanup_lease_session_open(
    yvex_runtime_cleanup_lease *lease, const yvex_runtime_session_open_request *request,
    yvex_runtime_execution_session **borrowed_session,
    yvex_runtime_model_failure *failure, yvex_error *err);
int yvex_runtime_cleanup_lease_close(yvex_runtime_cleanup_lease **lease, yvex_error *err);
typedef int (*yvex_runtime_cleanup_release_fn)(void **context, yvex_error *err);
int yvex_runtime_cleanup_lease_adopt(yvex_runtime_cleanup_lease *lease, void *context,
    yvex_runtime_cleanup_release_fn release, yvex_error *err);
/* Immutable compatibility facts hashed before one attention dispatch. */
typedef struct {
    unsigned int schema_version;
    const char *runtime_model_identity, *runtime_binding_identity;
    const char *artifact_identity, *runtime_numeric_identity;
    const char *runtime_descriptor_identity, *semantic_graph_identity;
    const char *executable_graph_identity, *selected_mode, *capture_bucket;
    const char *residency_identity, *workspace_identity, *capacity_plan_identity;
    const char *state_layout_identity;
    unsigned long long family_adapter_id, family_adapter_version;
    unsigned int probe, probe_scope, operation_scope, phase, backend, requested_mode;
    int compare_backends;
    unsigned long long token_count, request_count, start_position;
    unsigned long long layer_start, layer_count, selection_key, binding_count;
    unsigned long long state_component_entries[YVEX_ATTENTION_STATE_BINDING_COUNT];
    unsigned long long state_component_capacities[YVEX_ATTENTION_STATE_BINDING_COUNT];
    unsigned long long maximum_compression_ratio, maximum_topk_capacity;
    unsigned int trace_policy;
    unsigned long long maximum_host_bytes, maximum_device_bytes;
    unsigned long long residency_generation, resident_binding_count;
    unsigned long long resident_encoded_bytes, workspace_bytes, workspace_generation;
    unsigned long long prepared_state_layers, state_allocated_bytes, state_generation;
    unsigned long long qtype_binding_counts[YVEX_RUNTIME_DESCRIPTOR_QTYPE_CAP];
    unsigned long long qtype_bytes[YVEX_RUNTIME_DESCRIPTOR_QTYPE_CAP];
    unsigned int device_kind;
    int device_index, compute_capability_major, compute_capability_minor;
    unsigned long long total_device_bytes;
} yvex_runtime_execution_descriptor_facts;
int yvex_runtime_execution_descriptor_identity_compute(
    const yvex_runtime_execution_descriptor_facts *facts,
    char output[YVEX_SHA256_HEX_CAP], yvex_error *err);
int yvex_runtime_workspace_identity_compute(
    const char *runtime_model_identity, yvex_backend_kind backend,
    unsigned long long maximum_host_bytes, unsigned long long maximum_device_bytes,
    unsigned long long workspace_bytes, unsigned long long host_workspace_bytes,
    const char *capacity_identity, char output[YVEX_SHA256_HEX_CAP], yvex_error *err);
/* Operator reachability over the common runtime model/session. */
#define YVEX_GRAPH_ATTENTION_TEXT_CAP 128u
#define YVEX_GRAPH_ATTENTION_REASON_CAP 256u
#define YVEX_RUNTIME_CAPTURE_BUCKET_CAP 64u
typedef enum {
    YVEX_RUNTIME_OPERATOR_EXECUTE = 0, YVEX_RUNTIME_OPERATOR_PLAN,
    YVEX_RUNTIME_OPERATOR_STATE_INSPECT, YVEX_RUNTIME_OPERATOR_STATE_VALIDATE,
    YVEX_RUNTIME_OPERATOR_STATE_EXERCISE,
    YVEX_RUNTIME_OPERATOR_CAPTURE, YVEX_RUNTIME_OPERATOR_REPLAY, YVEX_RUNTIME_OPERATOR_GRAPH_LIST,
    YVEX_RUNTIME_OPERATOR_GRAPH_INSPECT, YVEX_RUNTIME_OPERATOR_GRAPH_WARMUP,
    YVEX_RUNTIME_OPERATOR_GRAPH_UPDATE, YVEX_RUNTIME_OPERATOR_GRAPH_INVALIDATE,
    YVEX_RUNTIME_OPERATOR_GRAPH_RELEASE, YVEX_RUNTIME_OPERATOR_TRACE, YVEX_RUNTIME_OPERATOR_PROFILE,
    YVEX_RUNTIME_OPERATOR_BENCHMARK, YVEX_RUNTIME_OPERATOR_QUALIFY,
    YVEX_RUNTIME_OPERATOR_CAPABILITIES,
    YVEX_RUNTIME_OPERATOR_RESIDENCY_INSPECT
} yvex_runtime_operator_action;
typedef struct {
    const char *target, *artifact_path;
    const char *runtime_binding_path, *activation_input_path, *capture_bucket, *attention_class;
    yvex_backend_kind backend;
    yvex_attention_probe_kind probe;
    yvex_attention_probe_scope scope;
    yvex_execution_phase phase;
    yvex_runtime_execution_mode mode;
    yvex_runtime_execution_scope operation_scope;
    yvex_runtime_trace_policy trace_policy;
    yvex_runtime_operator_action operator_action;
    unsigned long long token_count, chunk_tokens, context_capacity, warmup, repeat;
    unsigned long long layer_start, layer_count, history_tokens;
    unsigned long long maximum_host_bytes, maximum_device_bytes, selection_key;
    int compare_backends, require_mode, select_layer, select_selection_key;
    yvex_runtime_progress_callback progress;
    void *progress_context;
    int (*cancel_requested)(void *context);
    void *cancel_context;
} yvex_graph_attention_operator_request;
typedef enum {
    YVEX_RUNTIME_QUALITY_SOFTWARE = 0,
    YVEX_RUNTIME_QUALITY_NUMERICAL,
    YVEX_RUNTIME_QUALITY_QUALIFICATION,
    YVEX_RUNTIME_QUALITY_COMPONENT_BENCHMARK,
    YVEX_RUNTIME_QUALITY_CORRECTNESS,
    YVEX_RUNTIME_QUALITY_STRUCTURAL,
    YVEX_RUNTIME_QUALITY_PERFORMANCE,
    YVEX_RUNTIME_QUALITY_STATUS_COUNT
} yvex_runtime_quality_status;
int yvex_graph_attention_operator_selection_validate(
    const yvex_graph_attention_operator_request *request, yvex_error *err);
typedef struct yvex_graph_attention_operator_result {
    int completed;
    char status[32], command[YVEX_GRAPH_ATTENTION_TEXT_CAP];
    char target[YVEX_GRAPH_ATTENTION_TEXT_CAP], family[32], backend[32], scope[32];
    char operation_scope[32], phase[32], trace_policy[16];
    char requested_mode[16], selected_mode[16], selection_reason[96];
    char capture_bucket[YVEX_RUNTIME_CAPTURE_BUCKET_CAP], input_class[64];
    char execution_class[32], weights_class[64];
    char artifact_path[YVEX_PATH_CAP], runtime_binding_path[YVEX_PATH_CAP];
    char source_snapshot_identity[17];
    char cuda_driver[32], cuda_build_identity[YVEX_SHA256_HEX_CAP];
    char payload_identity[YVEX_SHA256_HEX_CAP], artifact_identity[YVEX_SHA256_HEX_CAP];
    char artifact_transform_identity[YVEX_SHA256_HEX_CAP], logical_transform_identity[YVEX_SHA256_HEX_CAP];
    char materialization_identity[YVEX_SHA256_HEX_CAP], logical_model_identity[YVEX_SHA256_HEX_CAP];
    char runtime_binding_identity[YVEX_SHA256_HEX_CAP], runtime_model_identity[YVEX_SHA256_HEX_CAP];
    char runtime_numeric_identity[YVEX_SHA256_HEX_CAP], runtime_descriptor_identity[YVEX_SHA256_HEX_CAP];
    char attention_plan_identity[YVEX_SHA256_HEX_CAP];
    char semantic_graph_identity[YVEX_SHA256_HEX_CAP], executable_graph_identity[YVEX_SHA256_HEX_CAP];
    char execution_descriptor_identity[YVEX_SHA256_HEX_CAP];
    char cuda_launch_graph_identity[YVEX_SHA256_HEX_CAP], cuda_graph_exec_identity[YVEX_SHA256_HEX_CAP];
    char cuda_graph_registry_scope[32], cuda_graph_entry_compatibility_identity[160];
    char residency_identity[YVEX_SHA256_HEX_CAP], workspace_identity[YVEX_SHA256_HEX_CAP];
    char state_layout_identity[YVEX_SHA256_HEX_CAP], state_content_identity[YVEX_SHA256_HEX_CAP],
         state_residency_identity[YVEX_SHA256_HEX_CAP];
    char execution_evidence_digest[YVEX_SHA256_HEX_CAP], execution_identity[YVEX_SHA256_HEX_CAP];
    char qualification_identity[YVEX_SHA256_HEX_CAP], quality_matrix_identity[YVEX_SHA256_HEX_CAP];
    char activation_input_identity[YVEX_SHA256_HEX_CAP], quality_status[YVEX_RUNTIME_QUALITY_STATUS_COUNT][24];
    char benchmark_scope[32], attention_class[16];
    char current_writer_plan_identity[YVEX_SHA256_HEX_CAP];
    char payload_plan_identity[YVEX_SHA256_HEX_CAP], payload_byte_identity[YVEX_SHA256_HEX_CAP];
    char reason[YVEX_GRAPH_ATTENTION_REASON_CAP], first_failing_stage[32];
    unsigned long long main_layers_total, bindings_total;
    unsigned long long repeat_count, warmup_count, benchmark_sample_count;
    unsigned long long requested_token_count, requested_history_tokens, requested_layer_start, requested_layer_count;
    unsigned long long artifact_hash_passes, warm_artifact_hash_passes;
    unsigned long long runtime_source_headers_read, runtime_source_payload_bytes_read;
    unsigned long long runtime_transform_plans_built, runtime_quant_plans_built;
    unsigned long long runtime_writer_plans_built, runtime_model_builds;
    unsigned long long runtime_descriptor_builds, semantic_graph_builds, executable_graph_builds;
    unsigned long long resident_binding_count, resident_encoded_bytes, host_resident_bytes, device_resident_bytes;
    unsigned long long workspace_bytes, pinned_host_bytes, pinned_host_peak_bytes, upload_bytes, upload_count;
    unsigned long long warm_weight_artifact_reads, warm_weight_upload_bytes;
    unsigned long long warm_h2d_bytes, warm_d2h_bytes;
    unsigned long long warm_host_allocations, warm_device_allocations, warm_device_frees;
    unsigned long long cuda_graph_count, cuda_graph_piece_count, cuda_graph_capture_count, cuda_graph_instantiate_count;
    unsigned long long cuda_graph_replay_count, cuda_graph_launch_count, cuda_graph_node_count;
    unsigned long long cuda_graph_kernel_node_count, cuda_graph_memcpy_node_count,
                       cuda_graph_memset_node_count, cuda_graph_invalidation_count;
    unsigned long long cuda_graph_update_count, cuda_graph_update_pending_count;
    unsigned long long cuda_graph_registry_count, cuda_graph_registry_index,
                       cuda_graph_registry_affected_count;
    unsigned long long cuda_graph_capture_elapsed_ns, cuda_graph_instantiate_elapsed_ns,
                       cuda_graph_last_update_elapsed_ns, cuda_graph_last_replay_elapsed_ns;
    unsigned long long execution_dispatch_count, trace_stage_count, trace_value_count;
    unsigned long long state_layer_count, state_prepared_layer_count, state_allocated_bytes;
    unsigned long long state_capacity, state_committed_sequence_length, state_next_position;
    unsigned long long state_generation, state_residency_generation, state_device_bytes;
    unsigned long long state_upload_bytes, state_upload_count;
    unsigned long long state_commit_count, state_abort_count, state_cancellation_count, state_reset_count;
    unsigned long long prefill_chunk_count, committed_prefix;
    int cuda_graph_entry_state, cuda_graph_entry_reason, cuda_graph_entry_capture_mode;
    int cuda_graph_entry_uploaded, cuda_graph_entry_update_requested, pinned_host_residency;
    int physical_payload_compatible, artifact_rebuild_required, materialization_rebuild_required;
    int tensor_inventory_equal, qtype_equal, layout_equal, offset_equal, payload_digest_equal;
    yvex_attention_probe_result probe;
    yvex_runtime_capabilities capabilities;
    int attention_cuda_execution_ready;
    int state_sealed, state_persistent, state_position_consistent, state_cuda_ready;
    int state_transaction_active, state_validation_passed, state_read_after_write_verified,
        state_clear_reuse_verified;
    int production_api_available, internal_live_runner_available, operator_command_available;
    int end_user_generation_available, model_behavior_evaluation_available;
    int model_quality_evaluation_available;
    int agent_runtime_available, agent_evaluation_available, release_qualification_available;
    int benchmark_correctness_precondition_passed, benchmark_runtime_precondition_passed;
    int activation_prefill_ready, prefill_persistent_state_ready, full_model_prefill_ready;
    unsigned long long artifact_bytes_hashed;
    double lifecycle_seconds[YVEX_RUNTIME_LIFECYCLE_COUNT], total_seconds, benchmark_first_execution_seconds;
    double benchmark_host_seconds[YVEX_RUNTIME_BENCHMARK_STATISTIC_COUNT];
    int benchmark_device_timing_available, artifact_identity_verified;
    double benchmark_device_seconds[YVEX_RUNTIME_BENCHMARK_STATISTIC_COUNT];
    yvex_runtime_benchmark_operator_summary benchmark;
    char failure_code[32], failure_where[YVEX_ERROR_WHERE_CAP];
} yvex_graph_attention_operator_result;
int yvex_graph_attention_operator_execute(const yvex_graph_attention_operator_request *request,
    yvex_graph_attention_operator_result *result, yvex_runtime_cleanup_lease **retained_cleanup,
    yvex_error *err);
#ifdef __cplusplus
}
#endif
#endif /* INCLUDE_YVEX_INTERNAL_RUNTIME_H_INCLUDED */
