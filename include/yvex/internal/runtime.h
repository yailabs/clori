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
#include <yvex/internal/compiler.h>
#include <yvex/internal/deployment.h>
#include <yvex/internal/device_view.h>
#include <yvex/internal/engine_resource.h>
#include <yvex/internal/execution.h>
#include <yvex/internal/execution_batch.h>
#include <yvex/internal/graph.h>
#include <yvex/internal/graph_state.h>
#include <yvex/internal/sequence_state.h>
#include <yvex/internal/tokenizer.h>
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
#define YVEX_RUNTIME_EXECUTION_PROFILE_SCHEMA_V1 1u
typedef struct {
    unsigned int schema_version;
    unsigned long long engine_generation;
    const char *engine_specialization_identity;
    const char *kernel_bundle_identity;
    const char *workload_profile_identity;
    yvex_execution_generation_mode generation_mode;
    yvex_execution_evidence_profile evidence;
    yvex_execution_class execution_class;
    yvex_execution_resolution attention_resolution;
    yvex_execution_resolution moe_resolution;
    yvex_execution_resolution sampling_resolution;
} yvex_runtime_execution_profile_request;
typedef struct yvex_runtime_execution_profile {
    unsigned int schema_version;
    unsigned long long engine_generation;
    yvex_execution_generation_mode generation_mode;
    yvex_execution_evidence_profile evidence;
    yvex_execution_class execution_class;
    yvex_execution_resolution resolution;
    yvex_execution_resolution attention_resolution;
    yvex_execution_resolution moe_resolution;
    yvex_execution_resolution sampling_resolution;
    char engine_specialization_identity[YVEX_SHA256_HEX_CAP];
    char kernel_bundle_identity[YVEX_SHA256_HEX_CAP];
    char workload_profile_identity[YVEX_SHA256_HEX_CAP];
    char identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_execution_profile;
int yvex_runtime_execution_profile_seal(
    const yvex_runtime_execution_profile_request *request,
    yvex_runtime_execution_profile *profile, yvex_error *err);
#define YVEX_RUNTIME_REASON_CAP 256u
#define YVEX_RUNTIME_BINDING_SCHEMA_CURRENT 16u
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
#define YVEX_MODEL_ENGINE_SCHEMA_V1 1u
#define YVEX_RUNTIME_EXECUTION_DESCRIPTOR_SCHEMA_V2 2u
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
typedef struct yvex_runtime_binding_prepare_request {
    const char *directory;
    const yvex_complete_artifact_admission *admission;
    const yvex_artifact_physical_compatibility *physical_compatibility;
    const yvex_materialization_session *materialization;
    const yvex_runtime_descriptor *runtime_descriptor;
    const yvex_operator_graph_ir *operator_graph;
    const yvex_physical_execution_ir *physical_execution;
    const yvex_compiled_model_plan *compiled_plan;
    const yvex_attention_plan *attention_plan;
    const yvex_attention_plan *draft_attention_plan;
    unsigned long long family_adapter_id, family_adapter_version;
    const char *artifact_format;
    unsigned int artifact_format_version;
    const char *logical_transform_identity;
    yvex_runtime_capabilities capabilities;
    yvex_transformer_family_policy transformer_policy;
    yvex_logits_family_policy logits_policy;
    yvex_speculation_family_policy speculation_policy;
    yvex_tokenizer_family_policy tokenizer_policy;
} yvex_runtime_binding_prepare_request;
typedef struct yvex_runtime_binding_summary {
    unsigned int schema_version;
    unsigned long long family_adapter_id, family_adapter_version;
    unsigned long long tensor_count, layer_count, draft_layer_count, file_bytes;
    unsigned long long decoder_layer_count, recurrent_layer_count;
    unsigned long long semantic_maximum_context;
    unsigned long long physical_execution_decision_count, source_snapshot_identity, mapping_identity;
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
    char physical_execution_identity[YVEX_SHA256_HEX_CAP];
    char attention_plan_identity[YVEX_SHA256_HEX_CAP], moe_plan_identity[YVEX_SHA256_HEX_CAP];
    char draft_attention_plan_identity[YVEX_SHA256_HEX_CAP];
    char draft_moe_plan_identity[YVEX_SHA256_HEX_CAP];
    char transformer_plan_identity[YVEX_SHA256_HEX_CAP];
    char draft_transformer_plan_identity[YVEX_SHA256_HEX_CAP];
    char decoder_plan_identity[YVEX_SHA256_HEX_CAP];
    char output_head_plan_identity[YVEX_SHA256_HEX_CAP];
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
int yvex_runtime_binding_compile_publish(
    const yvex_family_compiler_adapter *adapter,
    const struct yvex_compilation_runtime_binding_request *request,
    char path[YVEX_PATH_CAP], int *published, yvex_error *err);
int yvex_runtime_binding_open(yvex_runtime_binding **out, const char *path,
    yvex_runtime_binding_summary *summary, yvex_complete_artifact_admission *admission,
    yvex_runtime_binding_failure *failure, yvex_error *err);
int yvex_runtime_binding_open_compatible(
    yvex_runtime_binding **out, const char *path,
    unsigned long long family_adapter_id, unsigned long long family_adapter_version,
    const char *logical_transform_identity, yvex_runtime_binding_summary *summary,
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
    yvex_attention_plan **draft_attention_out, const yvex_physical_execution_ir **physical_execution_out,
    yvex_runtime_binding_failure *failure, yvex_error *err);
int yvex_runtime_binding_policies(
    const yvex_runtime_binding *binding,
    const yvex_transformer_family_policy **transformer,
    const yvex_logits_family_policy **logits,
    const yvex_speculation_family_policy **speculation);
const yvex_tokenizer_family_policy *yvex_runtime_binding_tokenizer_policy(
    const yvex_runtime_binding *binding);
typedef enum {
    YVEX_MODEL_ENGINE_FAILURE_NONE = 0, YVEX_MODEL_ENGINE_FAILURE_INVALID_ARGUMENT,
    YVEX_MODEL_ENGINE_FAILURE_ADAPTER, YVEX_MODEL_ENGINE_FAILURE_BINDING,
    YVEX_MODEL_ENGINE_FAILURE_ARTIFACT, YVEX_MODEL_ENGINE_FAILURE_IDENTITY,
    YVEX_MODEL_ENGINE_FAILURE_MATERIALIZATION, YVEX_MODEL_ENGINE_FAILURE_DESCRIPTOR,
    YVEX_MODEL_ENGINE_FAILURE_GRAPH, YVEX_MODEL_ENGINE_FAILURE_BACKEND,
    YVEX_MODEL_ENGINE_FAILURE_DRIFT, YVEX_MODEL_ENGINE_FAILURE_BUSY,
    YVEX_MODEL_ENGINE_FAILURE_CANCELLED, YVEX_MODEL_ENGINE_FAILURE_ALLOCATION,
    YVEX_MODEL_ENGINE_FAILURE_CLEANUP
} yvex_model_engine_failure_code;
typedef enum {
    YVEX_RUNTIME_FAILURE_ORIGIN_NONE = 0,
    YVEX_RUNTIME_FAILURE_ORIGIN_EXTERNAL_REQUEST,
    YVEX_RUNTIME_FAILURE_ORIGIN_INTEGRITY,
    YVEX_RUNTIME_FAILURE_ORIGIN_CAPABILITY,
    YVEX_RUNTIME_FAILURE_ORIGIN_RESOURCE,
    YVEX_RUNTIME_FAILURE_ORIGIN_BACKEND,
    YVEX_RUNTIME_FAILURE_ORIGIN_SEQUENCE,
    YVEX_RUNTIME_FAILURE_ORIGIN_ENGINE,
    YVEX_RUNTIME_FAILURE_ORIGIN_INTERNAL
} yvex_runtime_failure_origin;
typedef enum {
    YVEX_RUNTIME_RECOVERY_NONE = 0,
    YVEX_RUNTIME_RECOVERY_REFUSE_REQUEST,
    YVEX_RUNTIME_RECOVERY_ABORT_TRANSACTION,
    YVEX_RUNTIME_RECOVERY_RETRY_EQUIVALENT,
    YVEX_RUNTIME_RECOVERY_PREPARE_OR_EVICT,
    YVEX_RUNTIME_RECOVERY_INVALIDATE_SEQUENCE,
    YVEX_RUNTIME_RECOVERY_DRAIN_ENGINE,
    YVEX_RUNTIME_RECOVERY_REFUSE_ENGINE_OPEN,
    YVEX_RUNTIME_RECOVERY_INTERNAL_INVARIANT
} yvex_runtime_recovery_action;
typedef struct yvex_model_engine_failure {
    yvex_model_engine_failure_code code;
    yvex_runtime_failure_origin origin;
    yvex_runtime_recovery_action recovery;
    unsigned long long expected, actual;
    char field[64];
    const char *reason;
} yvex_model_engine_failure;
struct yvex_runtime_generation_options;
typedef struct {
    const char *artifact_path, *runtime_binding_path, *target_id, *artifact_reopen_cache_root;
    const char *expected_logical_transform_identity;
    unsigned long long expected_family_adapter_id, expected_family_adapter_version;
    const struct yvex_runtime_generation_options *startup_generation;
    yvex_backend_kind residency_backend;
    unsigned long long maximum_host_bytes, maximum_device_bytes;
    yvex_runtime_progress_callback progress;
    void *progress_context;
} yvex_model_engine_open_request;
typedef struct {
    int sealed, valid;
    unsigned long long engine_generation; /* Process-local; never persisted or hashed. */
    char runtime_model_identity[YVEX_SHA256_HEX_CAP], runtime_binding_identity[YVEX_SHA256_HEX_CAP];
    char artifact_identity[YVEX_SHA256_HEX_CAP];
    char materialization_identity[YVEX_SHA256_HEX_CAP];
    char runtime_descriptor_identity[YVEX_SHA256_HEX_CAP];
    char runtime_numeric_identity[YVEX_SHA256_HEX_CAP];
    char semantic_graph_identity[YVEX_SHA256_HEX_CAP];
    char executable_graph_identity[YVEX_SHA256_HEX_CAP];
    char physical_execution_identity[YVEX_SHA256_HEX_CAP];
    unsigned long long artifact_hash_passes, artifact_verified_reopen_passes;
    unsigned long long artifact_reopen_cache_failures, artifact_bytes_hashed;
    unsigned long long gguf_directory_parses, runtime_binding_parses;
    unsigned long long drift_checks, invalidation_count;
    unsigned long long tensor_count, attention_layer_count, draft_attention_layer_count;
    unsigned long long attention_binding_count, draft_attention_binding_count;
    unsigned long long physical_execution_decision_count;
    unsigned long long engine_specialization_count;
    unsigned long long engine_resource_count, engine_resource_generation;
    unsigned long long mapped_package_bytes, prepared_bytes;
    unsigned long long resident_host_bytes, resident_device_bytes;
    double lifecycle_seconds[YVEX_RUNTIME_LIFECYCLE_COUNT], total_seconds;
    yvex_runtime_capabilities capabilities;
} yvex_model_engine_summary;
typedef struct yvex_model_engine yvex_model_engine;
typedef struct yvex_runtime_execution_session yvex_runtime_execution_session;
typedef struct yvex_runtime_cleanup_lease yvex_runtime_cleanup_lease;
enum { YVEX_RUNTIME_RESIDENCY_SCHEMA_V7 = 7u };
typedef enum {
    YVEX_RUNTIME_WEIGHT_PLACEMENT_HOST_LOCKED = 0,
    YVEX_RUNTIME_WEIGHT_PLACEMENT_CUDA_MANAGED,
    YVEX_RUNTIME_WEIGHT_PLACEMENT_ARTIFACT_MAPPED
} yvex_runtime_weight_placement;
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
    yvex_runtime_weight_placement placement;
} yvex_runtime_residency_options;
typedef struct {
    unsigned int schema_version;
    yvex_runtime_weight_placement placement;
    int sealed, attached, host_ready, host_locked, cuda_ready, invalidated;
    int model_complete, core_complete, envelope_complete, output_head_complete;
    unsigned long long generation, expected_model_binding_count, model_binding_count, expected_core_binding_count;
    unsigned long long expected_envelope_binding_count, core_binding_count, envelope_binding_count, binding_count;
    unsigned long long expected_output_head_binding_count, output_head_binding_count, output_head_encoded_bytes;
    unsigned long long accelerator_encoded_bytes, encoded_bytes;
    unsigned long long host_resident_bytes, device_resident_bytes;
    unsigned long long mapped_package_bytes, prepared_bytes;
    unsigned long long cuda_addressable_bytes, cuda_upload_bytes, cuda_upload_count, cuda_host_registration_count;
    unsigned long long cuda_pageable_map_bytes, cuda_pageable_map_count, cuda_managed_bytes,
        cuda_managed_allocation_count;
    unsigned long long cuda_managed_prefetch_bytes, cuda_managed_prefetch_count;
    unsigned long long cuda_pageable_prefetch_bytes, cuda_pageable_prefetch_count;
    unsigned long long cold_artifact_read_calls, cold_artifact_bytes_read, resident_read_calls, resident_bytes_read;
    unsigned long long qtype_binding_counts[YVEX_RUNTIME_DESCRIPTOR_QTYPE_CAP];
    unsigned long long qtype_bytes[YVEX_RUNTIME_DESCRIPTOR_QTYPE_CAP];
    char payload_digest[YVEX_SHA256_HEX_CAP], residency_identity[YVEX_SHA256_HEX_CAP];
    char output_head_residency_identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_residency_summary;
typedef struct yvex_runtime_residency yvex_runtime_residency;
typedef struct yvex_runtime_state_residency yvex_runtime_state_residency;
typedef struct yvex_runtime_component_session yvex_runtime_component_session;
typedef struct yvex_decoder_plan yvex_decoder_plan;
typedef struct yvex_moe_plan yvex_moe_plan;
typedef struct yvex_transformer_plan yvex_transformer_plan;
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
    const yvex_engine_resource_catalog *resources;
    const yvex_runtime_binding *compiled_binding;
    const yvex_compiled_model_plan *compiled_plan;
    const yvex_runtime_binding_summary *binding;
    const yvex_graph_execution_api *graph;
    const char *target_id;
    const yvex_attention_plan *attention, *draft_attention;
    const yvex_decoder_plan *decoder;
    const yvex_moe_plan *moe, *draft_moe;
    const yvex_transformer_plan *transformer, *draft_transformer;
    const yvex_runtime_logits_plan_summary *output_head;
    const yvex_runtime_descriptor *descriptor;
    const yvex_physical_execution_ir *physical_execution;
    const yvex_tokenizer *tokenizer;
    yvex_materialization_session *materialization;
} yvex_model_engine_view;
int yvex_runtime_residency_prepare(yvex_runtime_residency **out, yvex_model_engine *model,
    const yvex_runtime_residency_options *options, yvex_runtime_residency_failure *failure, yvex_error *err);
int yvex_runtime_component_residency_prepare(yvex_runtime_residency **out,
    yvex_materialization_session *materialization, const char *component_identity,
    const yvex_runtime_residency_options *options, yvex_runtime_residency_failure *failure, yvex_error *err);
typedef struct yvex_component_binding {
    unsigned int schema_version;
    unsigned long long binding_id, binding_version;
    const char *target_id, *component_id, *admission_component;
    yvex_backend_kind backend;
    int (*plan)(const yvex_component_plan_request *, yvex_component_plan *,
                yvex_component_failure *, yvex_error *);
    int (*admit)(const char *, const yvex_artifact *, const yvex_gguf *,
                 const yvex_tensor_table *, const yvex_artifact_admission_options *,
                 yvex_complete_artifact_admission *, yvex_artifact_admission_evidence *,
                 yvex_artifact_admission_failure *, yvex_error *);
    int (*execute)(yvex_materialization_session *, const yvex_component_execution_request *,
                   yvex_component_execution_result *, yvex_component_failure *, yvex_error *);
} yvex_component_binding;
const yvex_component_binding *yvex_component_binding_at(unsigned long long index);
typedef struct {
    int (*plan_build)(const yvex_component_plan_request *, yvex_component_plan *,
                      yvex_component_failure *, yvex_error *);
    int (*plan_validate)(const yvex_component_plan *, yvex_component_failure *,
                         yvex_error *);
    int (*execute)(const yvex_artifact *, const yvex_gguf *, const yvex_tensor_table *,
                   const yvex_component_execution_request *,
                   yvex_component_execution_result *, yvex_component_failure *,
                   yvex_error *);
} yvex_runtime_component_api;
const yvex_runtime_component_api *yvex_runtime_component_api_get(void);
int yvex_runtime_residency_close(yvex_runtime_residency **residency, yvex_error *err);
int yvex_runtime_residency_snapshot(const yvex_runtime_residency *residency, yvex_runtime_residency_summary *summary,
    const unsigned char **arena, unsigned long long *arena_bytes, yvex_error *err);
int yvex_runtime_residency_binding_view(const yvex_runtime_residency *,
    const yvex_materialized_tensor_binding *, const unsigned char **, unsigned long long *, yvex_error *);
/* A non-null backend is consumed as an already-open CUDA owner before a large arena is registered. */
int yvex_runtime_residency_cuda_session_attach(yvex_runtime_residency *residency, yvex_backend **backend,
    unsigned long long maximum_device_bytes, int *uploaded, yvex_runtime_residency_summary *summary, yvex_error *err);
int yvex_runtime_residency_invalidate(yvex_runtime_residency *residency, yvex_error *err);
int yvex_runtime_state_residency_prepare(yvex_runtime_state_residency **out, yvex_backend *backend,
    const yvex_graph_attention_capacity_plan *capacity, const yvex_attention_state_provider *provider,
    unsigned long long prior_host_bytes, unsigned long long maximum_host_bytes, unsigned long long prior_device_bytes,
    unsigned long long maximum_device_bytes, yvex_error *err);
typedef enum { YVEX_RUNTIME_STATE_BEGIN = 0, YVEX_RUNTIME_STATE_STAGE } yvex_runtime_state_action;
int yvex_runtime_state_residency_transition(yvex_runtime_state_residency *,
    const yvex_attention_state_provider *, const yvex_attention_publication *,
    unsigned long long, unsigned long long, yvex_runtime_state_action, yvex_error *);
int yvex_runtime_state_residency_prepare_commit(
    yvex_runtime_state_residency *residency, yvex_error *err);
void yvex_runtime_state_residency_publish_commit(
    yvex_runtime_state_residency *residency);
void yvex_runtime_state_residency_abort(yvex_runtime_state_residency *residency);
int yvex_runtime_state_residency_reset(yvex_runtime_state_residency *residency, yvex_error *err);
int yvex_runtime_state_residency_invalidate(yvex_runtime_state_residency *residency, yvex_error *err);
int yvex_runtime_state_residency_close(yvex_runtime_state_residency **residency, yvex_error *err);
int yvex_runtime_state_residency_summary_copy(const yvex_runtime_state_residency *residency,
                                              yvex_runtime_state_residency_summary *out, yvex_error *err);
/* A cleanup failure may publish an unpublished model in out; close retries exact ownership. */
int yvex_model_engine_open(yvex_model_engine **out, const yvex_model_engine_open_request *request,
                            yvex_model_engine_failure *failure, yvex_error *err);
int yvex_model_engine_validate(yvex_model_engine *model, yvex_model_engine_failure *failure, yvex_error *err);
int yvex_model_engine_summary_copy(const yvex_model_engine *model, yvex_model_engine_summary *out, yvex_error *err);
void yvex_model_engine_close(yvex_model_engine **model);
const yvex_model_engine_view *yvex_model_engine_view_get(const yvex_model_engine *model);
typedef struct {
    yvex_backend_kind backend;
    unsigned long long maximum_host_bytes, maximum_device_bytes;
    const yvex_attention_state_provider_factory *attention_state_factory;
    const yvex_sequence_state_plan *sequence_state_plan;
} yvex_runtime_session_open_request;
typedef struct {
    int open, busy, cancelled, invalidated;
    yvex_backend_kind backend;
    unsigned long long engine_generation;
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
    unsigned long long sequence_state_binding_count, sequence_state_generation;
    unsigned long long sequence_committed_state_bytes;
    unsigned long long sequence_candidate_state_bytes;
    unsigned long long sequence_host_state_bytes;
    unsigned long long sequence_device_state_bytes;
    int host_workspace_owned, host_workspace_pinned;
    int device_index, compute_capability_major, compute_capability_minor;
    unsigned long long total_device_bytes, sustainable_read_bytes_per_second, sustainable_copy_bytes_per_second;
    unsigned long long sustainable_coherent_host_bytes_per_second;
    char device_name[128], bandwidth_evidence_identity[YVEX_SHA256_HEX_CAP];
    char engine_specialization_identity[YVEX_SHA256_HEX_CAP];
    char residency_identity[YVEX_SHA256_HEX_CAP], workspace_identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_session_summary;
typedef struct {
    const yvex_model_engine *engine;
    yvex_backend *backend;
    const yvex_attention_state_provider *attention_state_provider;
    const yvex_attention_state_provider *draft_attention_state_provider;
    yvex_attention_workspace *attention_workspace;
    yvex_runtime_state_residency *state_residency;
    yvex_runtime_state_residency *draft_state_residency;
    yvex_sequence_state *sequence_state;
} yvex_runtime_session_view;
/* A cleanup failure may retain an unpublished closing session in out; retry close discharges it. */
int yvex_runtime_session_open(yvex_runtime_execution_session **out, yvex_model_engine *model,
    const yvex_runtime_session_open_request *request, yvex_model_engine_failure *failure,
    yvex_error *err);
int yvex_runtime_session_prepare_attention_workspace(yvex_runtime_execution_session *session,
    yvex_runtime_execution_mode mode, yvex_runtime_execution_scope scope,
    yvex_attention_evidence_level evidence_level,
    const yvex_graph_attention_capacity_plan *capacity,
    unsigned long long physical_row_capacity, unsigned long long minimum_bytes,
    yvex_model_engine_failure *failure, yvex_error *err);
int yvex_runtime_session_summary_copy(const yvex_runtime_execution_session *session,
                                      yvex_runtime_session_summary *out, yvex_error *err);
int yvex_runtime_session_close(yvex_runtime_execution_session **session, yvex_error *err);
const yvex_runtime_session_view *yvex_runtime_session_view_get(const yvex_runtime_execution_session *session);
int yvex_runtime_device_view_bind(yvex_execution_device_view *out, yvex_execution_device_value_kind kind,
    yvex_model_engine *model, yvex_runtime_execution_session *session,
    const yvex_attention_state_provider *provider,
    const yvex_runtime_execution_profile *profile, const yvex_device_tensor *tensor,
    unsigned long long offset, unsigned long long rows, unsigned long long columns,
    yvex_error *err);
int yvex_runtime_cleanup_lease_acquire(
    yvex_runtime_cleanup_lease **out, const yvex_model_engine_open_request *model_request,
    const yvex_runtime_session_open_request *session_request,
    yvex_model_engine **borrowed_model, yvex_runtime_execution_session **borrowed_session,
    yvex_model_engine_failure *failure, yvex_error *err);
int yvex_runtime_cleanup_lease_session_open(
    yvex_runtime_cleanup_lease *lease, const yvex_runtime_session_open_request *request,
    yvex_runtime_execution_session **borrowed_session,
    yvex_model_engine_failure *failure, yvex_error *err);
int yvex_runtime_cleanup_lease_close(yvex_runtime_cleanup_lease **lease, yvex_error *err);
typedef int (*yvex_runtime_cleanup_release_fn)(void **context, yvex_error *err);
int yvex_runtime_cleanup_lease_adopt(yvex_runtime_cleanup_lease *lease, void *context,
    yvex_runtime_cleanup_release_fn release, yvex_error *err);
int yvex_runtime_workspace_identity_compute(
    const char *runtime_model_identity, yvex_backend_kind backend,
    unsigned long long maximum_host_bytes, unsigned long long maximum_device_bytes,
    unsigned long long workspace_bytes, unsigned long long host_workspace_bytes,
    const char *capacity_identity, char output[YVEX_SHA256_HEX_CAP], yvex_error *err);
#ifdef __cplusplus
}
#endif
#endif /* INCLUDE_YVEX_INTERNAL_RUNTIME_H_INCLUDED */
