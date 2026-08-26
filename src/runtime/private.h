/*
 * Runtime model and session objects cross the core and transactional-state owners but never leave
 * the runtime subsystem. The model is immutable after publication; each session exclusively owns
 * its mutable backend, attention-state, residency, and workspace resources.
 */
#ifndef SRC_RUNTIME_PRIVATE_H_INCLUDED
#define SRC_RUNTIME_PRIVATE_H_INCLUDED

#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <yvex/internal/backend.h>
#include <yvex/internal/decode.h>
#include <yvex/internal/generation.h>
#include <yvex/internal/graph.h>
#include <yvex/internal/logits.h>
#include <yvex/internal/runtime.h>
#include <yvex/internal/engine_scheduler.h>

static inline yvex_attention_evidence_level runtime_attention_evidence(
    yvex_execution_evidence_profile profile)
{
    static const yvex_attention_evidence_level levels[] = {
        YVEX_ATTENTION_EVIDENCE_NONE,
        YVEX_ATTENTION_EVIDENCE_STAGES,
        YVEX_ATTENTION_EVIDENCE_FULL};
    return (unsigned int)profile < sizeof(levels) / sizeof(levels[0])
               ? levels[profile]
               : YVEX_ATTENTION_EVIDENCE_NONE;
}

static inline int runtime_engine_scheduler_options_valid(
    int enabled, unsigned long long width)
{
    return (enabled == 0 || enabled == 1) &&
           (enabled ? width >= 2ull && width < 64ull : !width);
}

/* A workload profile is usable only inside the engine generation that admitted it. */
static inline int runtime_execution_profile_matches(
    const yvex_runtime_execution_profile *profile,
    const yvex_model_engine_summary *model,
    const yvex_runtime_session_summary *session)
{
    return profile && model && session && model->sealed && model->valid &&
           profile->schema_version == YVEX_RUNTIME_EXECUTION_PROFILE_SCHEMA_V1 &&
           yvex_sha256_hex_valid(profile->identity) &&
           yvex_sha256_hex_valid(profile->engine_specialization_identity) &&
           yvex_sha256_hex_valid(profile->kernel_bundle_identity) &&
           yvex_sha256_hex_valid(profile->workload_profile_identity) &&
           profile->engine_generation &&
           profile->engine_generation == model->engine_generation &&
           profile->engine_generation == session->engine_generation &&
           strcmp(profile->engine_specialization_identity,
                  model->engine_specialization_identity) == 0 &&
           strcmp(profile->engine_specialization_identity,
                  session->engine_specialization_identity) == 0;
}

#define YVEX_GENERATION_LIFECYCLE_ACTIVE 1u
#define YVEX_GENERATION_LIFECYCLE_CLOSING 2u
#define YVEX_GENERATION_LIFECYCLE_CLOSED 6u

struct yvex_runtime_binding {
    yvex_runtime_binding_summary summary;
    yvex_complete_artifact_admission admission;
    yvex_materialization_summary materialization;
    yvex_materialized_tensor_binding *materialized;
    yvex_runtime_descriptor_summary descriptor;
    yvex_runtime_tensor_binding *runtime;
    yvex_physical_execution_ir *physical_execution;
    yvex_transformer_family_policy transformer_policy;
    yvex_logits_family_policy logits_policy;
    yvex_speculation_family_policy speculation_policy;
    yvex_tokenizer_family_policy tokenizer_policy;
    yvex_attention_summary attention, draft_attention;
    yvex_attention_layer_plan *layers, *draft_layers;
    yvex_compiled_model_plan *plan;
};

/* Exact legacy wire records exist only to authenticate and narrow accepted v14 bindings. */
typedef struct {
    unsigned int schema_version;
    unsigned long long decision_count, encoded_bytes;
    unsigned long long consumer_counts[YVEX_EXECUTION_CONSUMER_COUNT];
    unsigned long long layout_counts[4], placement_counts[5];
    char physical_variant_identity[YVEX_SHA256_HEX_CAP];
    char identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_binding_physical_summary_v14;

typedef struct {
    unsigned int schema_version;
    unsigned long long terminal_tensor_id;
    unsigned int role, scope;
    unsigned long long layer_index, predictor_index, expert_count;
    unsigned int canonical_qtype;
    unsigned long long canonical_row_width, canonical_row_count;
    unsigned long long encoded_offset, encoded_bytes, alignment;
    unsigned int consumer, layout, placement, sharing, activation;
    unsigned long long supported_width_mask, maximum_context, worklist_width_mask;
    unsigned long long tensor_core_minimum;
    unsigned int required_backend, required_compute_major, required_compute_minor;
    unsigned int evidence, fallback;
    int derived_asset_required;
    char kernel_family[YVEX_EXECUTION_TEXT_CAP];
    char tensor_core_kernel_family[YVEX_EXECUTION_TEXT_CAP];
    char terminal_identity[YVEX_SHA256_HEX_CAP];
    char decision_identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_binding_physical_decision_v14;

int yvex_runtime_private_binding_physical_v14_import(
    yvex_physical_execution_ir **out,
    const yvex_runtime_binding_physical_summary_v14 *summary,
    const yvex_runtime_binding_physical_decision_v14 *decisions,
    unsigned long long count, yvex_error *err);
int yvex_runtime_private_binding_policies_valid(
    const yvex_runtime_binding *binding);
int yvex_runtime_private_binding_policies_match_model(
    const yvex_model_execution_descriptor *model,
    const yvex_transformer_family_policy *transformer,
    const yvex_logits_family_policy *logits,
    const yvex_speculation_family_policy *speculation);

typedef struct runtime_engine_scheduler runtime_engine_scheduler;
typedef struct runtime_engine_work runtime_engine_work;
typedef int (*runtime_engine_work_execute)(
    runtime_engine_work *const *tickets,
    unsigned long long ticket_count, yvex_error *err);

typedef enum {
    RUNTIME_ENGINE_WORK_PHYSICAL = 0,
    RUNTIME_ENGINE_WORK_RENDEZVOUS
} runtime_engine_work_kind;

struct runtime_engine_work {
    yvex_execution_compatibility_key key;
    unsigned long long row_count, actual_width, group_size, coalescing_limit_ns;
    runtime_engine_work_execute execute;
    void *context;
    int (*cancel_requested)(void *context);
    void *cancel_context;
    yvex_error failure;
    yvex_expert_worklist_observation worklists;
    runtime_engine_work_kind kind;
    int status, done;
};

typedef struct {
    yvex_model_engine *model;
    yvex_runtime_execution_session *session;
    yvex_backend *backend;
    const yvex_transformer_plan_summary *transformer;
    const yvex_runtime_execution_profile *execution_profile;
    yvex_tensor_scope tensor_scope;
    yvex_execution_phase phase;
    yvex_execution_class execution_class;
    unsigned long long maximum_width;
    int (*cancel_requested)(void *context);
    void *cancel_context;
} runtime_engine_step_request;

typedef struct {
    yvex_model_engine *model;
    yvex_runtime_execution_session *session;
    yvex_runtime_moe_context *moe;
    yvex_backend *backend;
    const yvex_transformer_plan_summary *transformer;
    const yvex_moe_layer_plan *layer;
    const yvex_attention_publication *attention;
    const yvex_runtime_execution_profile *execution_profile;
    const unsigned int *token_ids;
    const yvex_device_tensor *device_rows;
    yvex_device_tensor *device_outputs;
    yvex_device_tensor *batch_device_rows, *batch_device_outputs;
    float *expanded_rows, *combined_rows, *routed_rows, *shared_rows;
    float *post_rows, *combination_rows;
    unsigned int *batch_token_ids;
    yvex_execution_batch_source *batch_sources;
    yvex_execution_batch_row *batch_rows;
    unsigned long long layer_ordinal, row_count, row_capacity, admitted_width;
    yvex_execution_batch_provenance provenance;
    yvex_execution_phase phase;
    yvex_execution_class execution_class;
    yvex_moe_row_batch_result *result;
    yvex_runtime_transformer_block_result *transformer_result;
    int (*cancel_requested)(void *context);
    void *cancel_context;
} runtime_engine_moe_request;

int yvex_runtime_private_engine_scheduler_open(
    runtime_engine_scheduler **out, unsigned long long queue_capacity,
    yvex_error *err);
int yvex_runtime_private_engine_scheduler_start(
    runtime_engine_scheduler *scheduler, yvex_error *err);
int yvex_runtime_private_engine_scheduler_set_producers(
    runtime_engine_scheduler *scheduler, unsigned long long producers,
    yvex_error *err);
int yvex_runtime_private_engine_scheduler_submit(
    runtime_engine_scheduler *scheduler,
    runtime_engine_work *ticket, yvex_error *err);
int yvex_runtime_private_engine_scheduler_snapshot(
    const runtime_engine_scheduler *scheduler,
    yvex_engine_scheduler_summary *summary, yvex_error *err);
int yvex_runtime_private_engine_scheduler_close(
    runtime_engine_scheduler **scheduler, yvex_error *err);
int yvex_runtime_private_model_scheduler_acquire(
    yvex_model_engine *model, unsigned long long maximum_width,
    yvex_error *err);
int yvex_runtime_private_model_scheduler_release(
    yvex_model_engine *model, yvex_error *err);
int yvex_runtime_private_model_scheduler_finish(
    yvex_model_engine *model, int *acquired, yvex_error *err);
int yvex_runtime_private_model_scheduler_producer_enter(
    yvex_model_engine *model, yvex_error *err);
int yvex_runtime_private_model_scheduler_producer_leave(
    yvex_model_engine *model, yvex_error *err);
int yvex_runtime_private_engine_scheduler_producer_finish(
    yvex_model_engine *model, int *active, int status, yvex_error *err);
int yvex_runtime_private_engine_scheduler_step_enter(
    const runtime_engine_step_request *request, int *active,
    yvex_error *err);
int yvex_runtime_private_engine_scheduler_moe_execute(
    const runtime_engine_moe_request *request, yvex_error *err);
int yvex_runtime_private_generation_logits_project(
    yvex_runtime_generation_context *context,
    const yvex_runtime_logits_source *source,
    yvex_runtime_logits_row_result *result, yvex_error *err);
static inline int runtime_binding_maximum_tensor_bytes(
    const yvex_runtime_binding *binding, unsigned long long *maximum)
{
    unsigned long long index;
    if (!binding || !maximum || !binding->materialized ||
        !binding->summary.tensor_count) return 0;
    *maximum = 0ull;
    for (index = 0ull; index < binding->summary.tensor_count; ++index) {
        unsigned long long bytes = binding->materialized[index].encoded_bytes;
        if (!bytes) return 0;
        if (bytes > *maximum) *maximum = bytes;
    }
    return *maximum != 0ull;
}

int yvex_runtime_private_binding_refuse(
    yvex_runtime_binding_failure *failure, yvex_runtime_binding_failure_code code,
    const char *field, const char *path, unsigned long long record,
    unsigned long long expected, unsigned long long actual, yvex_status status,
    const char *reason, yvex_error *err);
int yvex_runtime_private_compiled_plan_valid(
    const yvex_runtime_binding *binding);
int yvex_runtime_private_residency_execution_view(
    const yvex_runtime_residency *residency,
    const yvex_materialized_tensor_binding *binding,
    const unsigned char **data, unsigned long long *bytes,
    yvex_execution_layout_class *layout, yvex_error *err);
int yvex_runtime_private_residency_backing_bytes(
    const yvex_runtime_binding *binding, yvex_backend *backend,
    yvex_runtime_weight_placement placement, unsigned long long *bytes,
    yvex_error *err);

#define YVEX_ENGINE_SPECIALIZATION_SCHEMA_V1 1u
#define YVEX_ENGINE_IMPLEMENTATION_CAP 8u
typedef struct {
    unsigned int schema_version;
    yvex_engine_implementation implementation, fallback_implementation;
    yvex_execution_activation_class activation, fallback_activation;
    unsigned long long supported_width_mask, worklist_width_mask, tensor_core_minimum;
    char identity[YVEX_SHA256_HEX_CAP];
} yvex_engine_implementation_record;
typedef struct {
    unsigned int schema_version;
    yvex_backend_kind backend;
    int device_index, compute_major, compute_minor;
    unsigned long long package_decision_count, implementation_count;
    char package_execution_identity[YVEX_SHA256_HEX_CAP];
    char identity[YVEX_SHA256_HEX_CAP];
} yvex_engine_specialization_summary;
typedef struct yvex_engine_specialization {
    yvex_engine_implementation_record implementations[YVEX_ENGINE_IMPLEMENTATION_CAP];
    unsigned int *decision_handles;
    yvex_engine_specialization_summary summary;
} yvex_engine_specialization;

static inline void runtime_specialization_release(yvex_engine_specialization **specialization)
{
    if (!specialization || !*specialization) return;
    free((*specialization)->decision_handles);
    free(*specialization);
    *specialization = NULL;
}

static inline const yvex_engine_implementation_record *runtime_specialization_decision(
    const yvex_engine_specialization *specialization, unsigned long long decision_index)
{
    unsigned int handle;
    if (!specialization || decision_index >= specialization->summary.package_decision_count)
        return NULL;
    handle = specialization->decision_handles[decision_index];
    return handle < specialization->summary.implementation_count
               ? &specialization->implementations[handle] : NULL;
}

static inline const yvex_engine_implementation_record *runtime_specialization_tensor(
    const yvex_engine_specialization *specialization,
    const yvex_physical_execution_ir *package_execution, unsigned long long tensor_id)
{
    unsigned long long index;
    if (!specialization || !package_execution) return NULL;
    for (index = 0ull; index < specialization->summary.package_decision_count; ++index) {
        const yvex_physical_execution_decision *package =
            yvex_physical_execution_ir_decision_at(package_execution, index);
        if (package && package->terminal_tensor_id == tensor_id)
            return runtime_specialization_decision(specialization, index);
    }
    return NULL;
}

struct yvex_model_engine {
    const yvex_graph_execution_api *graph;
    char target_id[128];
    yvex_runtime_binding *binding;
    yvex_runtime_binding_summary binding_summary;
    yvex_artifact *artifact;
    yvex_gguf *gguf;
    yvex_tensor_table *tensors;
    yvex_complete_artifact_admission admission;
    yvex_materialization_plan *materialization_plan;
    yvex_materialization_session *materialization;
    yvex_runtime_descriptor *descriptor;
    const yvex_physical_execution_ir *physical_execution;
    yvex_engine_specialization *specializations[2];
    yvex_attention_plan *attention;
    yvex_attention_plan *draft_attention;
    yvex_tokenizer *tokenizer;
    yvex_runtime_residency *residency;
    yvex_backend *opening_backend;
    yvex_model_engine_summary summary;
    yvex_model_engine_view view;
    pthread_mutex_t lifecycle_mutex;
    runtime_engine_scheduler *engine_scheduler;
    struct yvex_runtime_execution_session *sessions;
    unsigned long long active_sessions, engine_scheduler_references;
    unsigned long long engine_scheduler_producers;
    unsigned long long next_session_ordinal;
    unsigned long long scheduler_maximum_width;
    int lifecycle_mutex_ready, close_requested, dependent_invalidation_pending;
};

struct yvex_runtime_execution_session {
    yvex_model_engine *engine;
    const yvex_engine_specialization *specialization;
    yvex_backend *backend;
    yvex_attention_state_provider attention_state_provider;
    yvex_attention_state_provider_factory attention_state_factory;
    yvex_attention_state_provider draft_attention_state_provider;
    yvex_attention_state_provider_factory draft_attention_state_factory;
    yvex_attention_workspace *attention_workspace;
    yvex_runtime_state_residency *state_residency;
    yvex_runtime_state_residency *draft_state_residency;
    yvex_device_tensor *workspace;
    yvex_runtime_session_summary summary;
    yvex_runtime_session_view view;
    pthread_mutex_t lifecycle_mutex;
    pthread_cond_t idle_condition;
    pthread_t execution_owner;
    struct yvex_runtime_execution_session *engine_previous, *engine_next;
    unsigned long long maximum_host_bytes, maximum_device_bytes;
    unsigned long long batch_source_ordinal;
    char batch_source_identity[YVEX_SHA256_HEX_CAP];
    int lifecycle_mutex_ready, idle_condition_ready, execution_owner_ready;
    int closing, engine_registered, engine_reserved;
    int engine_release_pending, invalidation_pending, host_workspace_cleanup_pending;
    int attention_state_provider_ready, draft_attention_state_provider_ready;
    int state_resolver_attached;
};

typedef struct {
    const yvex_attention_state_provider *provider;
    yvex_runtime_state_residency *residency;
    yvex_attention_operation_scope operation_scope;
    yvex_sha256 output_hash;
    unsigned long long output_values, active_layer_ordinal;
    unsigned long long pending_layer_ordinal, pending_layer_count;
    int hash_output, layer_active;
    char last_delta_identity[YVEX_SHA256_HEX_CAP];
} runtime_attention_state_bridge;

yvex_attention_probe_state_provider yvex_runtime_private_attention_state_provider(
    runtime_attention_state_bridge *bridge);
int yvex_runtime_private_attention_state_pristine(
    const yvex_attention_state_provider *provider, int *pristine,
    yvex_error *err);
int yvex_runtime_private_attention_state_abort(
    void *context, yvex_attention_failure *failure, yvex_error *err);

struct yvex_runtime_cleanup_lease {
    yvex_model_engine *model;
    yvex_runtime_execution_session *session;
    void *dependent_context;
    yvex_runtime_cleanup_release_fn dependent_release;
};

struct yvex_runtime_generation_context {
    yvex_model_engine *model;
    yvex_runtime_execution_session *session;
    const yvex_model_engine_view *model_view;
    const yvex_tokenizer *tokenizer;
    yvex_runtime_transformer_context *transformer;
    yvex_runtime_decode_context *decode;
    yvex_runtime_logits_context *logits;
    yvex_runtime_sampling_context *sampling;
    yvex_runtime_speculation_context *speculation;
    yvex_tokenizer_decoder *decoder;
    yvex_token_sequence *sequence;
    yvex_runtime_generation_options options;
    yvex_runtime_generation_plan_summary plan;
    yvex_runtime_execution_profile execution_profile;
    yvex_execution_hardware_profile hardware_profile;
    yvex_backend_bandwidth_evidence bandwidth_evidence;
    yvex_execution_workload_profile workload_profile;
    yvex_execution_capacity_plan capacity_plan;
    unsigned long long system_capacity_bytes, system_reserve_bytes;
    unsigned long long sampling_workspace_bytes;
    yvex_execution_phase_measurement phase_measurements[YVEX_EXECUTION_ROOFLINE_PHASE_COUNT];
    unsigned long long phase_measurement_count;
    unsigned int *additional_stops;
    float *hidden, *logits_row;
    unsigned long long hidden_count, logits_count, workspace_bytes;
    atomic_uint lifecycle;
    atomic_ullong admission_failures;
    pthread_mutex_t drain_mutex;
    pthread_cond_t drain_condition;
    unsigned long long execution_count, failure_count, cancellation_count;
    int drain_mutex_ready, drain_condition_ready, continuation_allowed;
    int device_selection;
};

int yvex_runtime_private_generation_enter(
    yvex_runtime_generation_context *context, yvex_error *err);
void yvex_runtime_private_generation_leave(
    yvex_runtime_generation_context *context, int rc, int executed);

typedef enum {
    YVEX_RUNTIME_REFUSE_MODEL_LOCK_UNAVAILABLE = 0,
    YVEX_RUNTIME_REFUSE_MODEL_INVALID_OR_DRAINING,
    YVEX_RUNTIME_REFUSE_ARTIFACT_IDENTITY,
    YVEX_RUNTIME_REFUSE_MODEL_OPEN_REQUEST,
    YVEX_RUNTIME_REFUSE_FAMILY_ADAPTER,
    YVEX_RUNTIME_REFUSE_MODEL_ALLOCATION,
    YVEX_RUNTIME_REFUSE_MODEL_LOCK_INITIALIZATION,
    YVEX_RUNTIME_REFUSE_MODEL_REQUIRED,
    YVEX_RUNTIME_REFUSE_MODEL_UNSEALED,
    YVEX_RUNTIME_REFUSE_DRIFT_COUNTER,
    YVEX_RUNTIME_REFUSE_HOST_RESIDENCY,
    YVEX_RUNTIME_REFUSE_CUDA_EAGER,
    YVEX_RUNTIME_REFUSE_SESSION_REQUEST,
    YVEX_RUNTIME_REFUSE_SESSION_ALLOCATION,
    YVEX_RUNTIME_REFUSE_SESSION_LOCK_INITIALIZATION,
    YVEX_RUNTIME_REFUSE_SESSION_CONDITION_INITIALIZATION,
    YVEX_RUNTIME_REFUSE_WORKSPACE_IDENTITY,
    YVEX_RUNTIME_REFUSE_SESSION_RESOURCE_INJECTION,
    YVEX_RUNTIME_REFUSE_MODEL_DRAINING_PUBLICATION,
    YVEX_RUNTIME_REFUSE_WORKSPACE_REQUEST,
    YVEX_RUNTIME_REFUSE_WORKSPACE_LOCK,
    YVEX_RUNTIME_REFUSE_WORKSPACE_SESSION_STATE,
    YVEX_RUNTIME_REFUSE_WORKSPACE_STATE,
    YVEX_RUNTIME_REFUSE_WORKSPACE_ALREADY_SEALED,
    YVEX_RUNTIME_REFUSE_WORKSPACE_BUDGET,
    YVEX_RUNTIME_REFUSE_WORKSPACE_CAPABILITY_INJECTION,
    YVEX_RUNTIME_REFUSE_SESSION_REQUIRED,
    YVEX_RUNTIME_REFUSE_SESSION_INVALIDATED,
    YVEX_RUNTIME_REFUSE_SESSION_CLOSING,
    YVEX_RUNTIME_REFUSE_SESSION_BUSY,
    YVEX_RUNTIME_REFUSE_SESSION_CANCELLED,
    YVEX_RUNTIME_REFUSE_BINDING_ADMISSION,
    YVEX_RUNTIME_REFUSE_ADAPTER_CAPABILITY,
    YVEX_RUNTIME_REFUSE_ADAPTER_CAPABILITY_STALE,
    YVEX_RUNTIME_REFUSE_ARTIFACT_DRIFT,
    YVEX_RUNTIME_REFUSE_DEVICE_CAPABILITY,
    YVEX_RUNTIME_REFUSE_CUDA_CAPABILITY,
    YVEX_RUNTIME_REFUSE_SESSION_OPEN_CLEANUP,
    YVEX_RUNTIME_REFUSE_DEVICE_WORKSPACE_BUDGET,
    YVEX_RUNTIME_REFUSE_CLEANUP_LEASE,
    YVEX_RUNTIME_REFUSE_CLEANUP_LEASE_ALLOCATION,
    YVEX_RUNTIME_REFUSE_CLEANUP_LEASE_SESSION,
    YVEX_RUNTIME_REFUSE_OPEN_BINDING,
    YVEX_RUNTIME_REFUSE_OPEN_ADAPTER,
    YVEX_RUNTIME_REFUSE_OPEN_LOGICAL_TRANSFORM,
    YVEX_RUNTIME_REFUSE_OPEN_HOST_BUDGET,
    YVEX_RUNTIME_REFUSE_OPEN_PROCESS_MEMORY,
    YVEX_RUNTIME_REFUSE_OPEN_SYSTEM_MEMORY,
    YVEX_RUNTIME_REFUSE_OPEN_STARTUP_CAPACITY,
    YVEX_RUNTIME_REFUSE_OPEN_ARTIFACT,
    YVEX_RUNTIME_REFUSE_OPEN_MATERIALIZATION,
    YVEX_RUNTIME_REFUSE_OPEN_IMPORT,
    YVEX_RUNTIME_REFUSE_OPEN_PHYSICAL_EXECUTION,
    YVEX_RUNTIME_REFUSE_OPEN_IMPORTED_IDENTITY,
    YVEX_RUNTIME_REFUSE_OPEN_TOKENIZER,
    YVEX_RUNTIME_REFUSE_OPEN_SEAL,
    YVEX_RUNTIME_REFUSE_OPEN_CAPABILITIES,
    YVEX_RUNTIME_REFUSE_OPEN_RESIDENCY,
    YVEX_RUNTIME_REFUSE_OPEN_RESIDENCY_COMPLETE,
    YVEX_RUNTIME_REFUSE_OPEN_DRIFT,
    YVEX_RUNTIME_REFUSE_COUNT
} yvex_runtime_private_refusal_id;

void yvex_runtime_private_failure_record(
    yvex_model_engine_failure *failure, yvex_model_engine_failure_code code,
    const char *field, unsigned long long expected,
    unsigned long long actual, const char *reason);
int yvex_runtime_private_reject(
    yvex_model_engine_failure *failure, yvex_model_engine_failure_code code,
    const char *field, unsigned long long expected,
    unsigned long long actual, const char *reason, yvex_error *err,
    yvex_status status);
int yvex_runtime_private_refuse(
    yvex_model_engine_failure *failure, yvex_runtime_private_refusal_id id,
    unsigned long long expected, unsigned long long actual, yvex_error *err);
int yvex_runtime_private_success(yvex_error *err);
int yvex_runtime_private_memory_capacity(
    unsigned long long *total_bytes, unsigned long long *available_bytes,
    int *process_limited);
unsigned long long yvex_runtime_private_system_reserve(
    unsigned long long capacity_bytes);
int yvex_runtime_private_weight_placement_select(
    const yvex_runtime_binding *binding, yvex_backend_kind backend_kind,
    yvex_backend *backend,
    yvex_runtime_weight_placement *placement, yvex_error *err);
int yvex_runtime_private_generation_capacity_preflight(
    const yvex_runtime_binding *binding, yvex_backend *backend,
    const yvex_runtime_generation_options *options,
    unsigned long long *required_bytes, unsigned long long *available_bytes,
    yvex_error *err);
int yvex_runtime_private_attention_workspace_required(
    const yvex_attention_summary *summary,
    const yvex_attention_layer_plan *layers, unsigned long long layer_count,
    const yvex_graph_attention_capacity_plan *capacity,
    yvex_attention_execution_mode mode,
    yvex_attention_operation_scope scope,
    yvex_attention_evidence_level evidence_level,
    unsigned long long physical_row_capacity, int deferred,
    unsigned long long *required_bytes, yvex_error *err);
int yvex_runtime_private_session_invalidate(
    yvex_runtime_execution_session *session, int include_state, yvex_error *err);
int yvex_runtime_private_session_workspace_discard(
    yvex_runtime_execution_session *session, yvex_error *err);
int yvex_runtime_private_session_capabilities_bind(
    yvex_runtime_execution_session *session,
    yvex_model_engine_failure *failure, int require_workspace,
    yvex_error *err);
int yvex_runtime_private_model_specialization_prepare(
    yvex_model_engine *model, yvex_backend_kind backend_kind,
    yvex_backend *backend, const yvex_engine_specialization **out,
    yvex_error *err);
int yvex_runtime_private_session_prepare_persistent_scope_state_locked(
    yvex_runtime_execution_session *session, yvex_tensor_scope scope,
    const yvex_graph_attention_capacity_plan *capacity,
    yvex_model_engine_failure *failure, yvex_error *err);
int yvex_runtime_private_state_residency_resolve(
    const void *context, const void *host, unsigned long long bytes,
    unsigned long long *device_address);
yvex_runtime_profile_mode yvex_runtime_generation_profile_mode(
    yvex_runtime_trace_policy policy);
int yvex_runtime_generation_workload_identity(
    const yvex_runtime_generation_context *context,
    const yvex_runtime_generation_turn_request *turn,
    char output[YVEX_SHA256_HEX_CAP]);
int yvex_runtime_generation_profile_phase(
    yvex_runtime_profile_record *profile, yvex_runtime_profile_phase phase,
    unsigned long long elapsed, yvex_error *err);
int yvex_runtime_generation_profile_transformer(
    yvex_runtime_profile_record *profile,
    const yvex_runtime_transformer_result *value, yvex_error *err);
int yvex_runtime_generation_profile_decode(
    yvex_runtime_profile_record *profile,
    const yvex_runtime_decode_step_result *value, yvex_error *err);

#endif
