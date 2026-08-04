/*
 * Runtime model and session objects cross the core and transactional-state owners but never leave
 * the runtime subsystem. The model is immutable after publication; each session exclusively owns
 * its mutable backend, attention-state, residency, and workspace resources.
 */
#ifndef SRC_RUNTIME_PRIVATE_H_INCLUDED
#define SRC_RUNTIME_PRIVATE_H_INCLUDED

#include <pthread.h>
#include <stdatomic.h>
#include <yvex/internal/decode.h>
#include <yvex/internal/generation.h>
#include <yvex/internal/graph.h>
#include <yvex/internal/runtime.h>

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

#define YVEX_GENERATION_LIFECYCLE_ACTIVE 1u
#define YVEX_GENERATION_LIFECYCLE_CLOSING 2u
#define YVEX_GENERATION_LIFECYCLE_CLOSED 6u

struct yvex_runtime_model {
    const yvex_runtime_family_adapter *adapter;
    yvex_runtime_binding *binding;
    yvex_runtime_binding_summary binding_summary;
    yvex_artifact *artifact;
    yvex_gguf *gguf;
    yvex_tensor_table *tensors;
    yvex_complete_artifact_admission admission;
    yvex_materialization_plan *materialization_plan;
    yvex_materialization_session *materialization;
    yvex_runtime_descriptor *descriptor;
    yvex_physical_execution_ir *physical_execution;
    yvex_attention_plan *attention;
    yvex_attention_plan *draft_attention;
    yvex_tokenizer *tokenizer;
    yvex_runtime_residency *residency;
    yvex_runtime_model_summary summary;
    yvex_runtime_model_view view;
    pthread_mutex_t lifecycle_mutex;
    struct yvex_runtime_execution_session *sessions;
    unsigned long long active_sessions;
    int lifecycle_mutex_ready, close_requested, dependent_invalidation_pending;
};

struct yvex_runtime_execution_session {
    yvex_runtime_model *model;
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
    struct yvex_runtime_execution_session *model_previous, *model_next;
    unsigned long long maximum_host_bytes, maximum_device_bytes;
    int lifecycle_mutex_ready, idle_condition_ready, execution_owner_ready;
    int closing, model_registered, model_reserved;
    int model_release_pending, invalidation_pending, host_workspace_cleanup_pending;
    int attention_state_provider_ready, draft_attention_state_provider_ready;
    int state_resolver_attached;
};

struct yvex_runtime_cleanup_lease {
    yvex_runtime_model *model;
    yvex_runtime_execution_session *session;
    void *dependent_context;
    yvex_runtime_cleanup_release_fn dependent_release;
};

struct yvex_runtime_generation_context {
    yvex_runtime_model *model;
    yvex_runtime_execution_session *session;
    const yvex_runtime_model_view *model_view;
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
    yvex_compiled_execution_profile execution_profile;
    yvex_execution_hardware_profile hardware_profile;
    yvex_execution_workload_profile workload_profile;
    yvex_execution_capacity_plan capacity_plan;
    yvex_execution_shape_registry *execution_shapes;
    unsigned int *additional_stops;
    float *hidden, *logits_row, *anchor_probabilities;
    unsigned long long hidden_count, logits_count, workspace_bytes;
    atomic_uint lifecycle;
    atomic_ullong admission_failures;
    pthread_mutex_t drain_mutex;
    pthread_cond_t drain_condition;
    unsigned long long execution_count, failure_count, cancellation_count;
    int drain_mutex_ready, drain_condition_ready, continuation_allowed;
};

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
    YVEX_RUNTIME_REFUSE_OPEN_SYSTEM_MEMORY,
    YVEX_RUNTIME_REFUSE_OPEN_ARTIFACT,
    YVEX_RUNTIME_REFUSE_OPEN_MATERIALIZATION,
    YVEX_RUNTIME_REFUSE_OPEN_IMPORT,
    YVEX_RUNTIME_REFUSE_OPEN_PHYSICAL_EXECUTION,
    YVEX_RUNTIME_REFUSE_OPEN_IMPORTED_IDENTITY,
    YVEX_RUNTIME_REFUSE_OPEN_TOKENIZER,
    YVEX_RUNTIME_REFUSE_OPEN_SEAL,
    YVEX_RUNTIME_REFUSE_OPEN_BUILD,
    YVEX_RUNTIME_REFUSE_OPEN_CAPABILITIES,
    YVEX_RUNTIME_REFUSE_OPEN_RESIDENCY,
    YVEX_RUNTIME_REFUSE_OPEN_RESIDENCY_COMPLETE,
    YVEX_RUNTIME_REFUSE_OPEN_DRIFT,
    YVEX_RUNTIME_REFUSE_COUNT
} yvex_runtime_private_refusal_id;

void yvex_runtime_private_failure_record(
    yvex_runtime_model_failure *failure, yvex_runtime_model_failure_code code,
    const char *field, unsigned long long expected,
    unsigned long long actual, const char *reason);
int yvex_runtime_private_reject(
    yvex_runtime_model_failure *failure, yvex_runtime_model_failure_code code,
    const char *field, unsigned long long expected,
    unsigned long long actual, const char *reason, yvex_error *err,
    yvex_status status);
int yvex_runtime_private_refuse(
    yvex_runtime_model_failure *failure, yvex_runtime_private_refusal_id id,
    unsigned long long expected, unsigned long long actual, yvex_error *err);
int yvex_runtime_private_success(yvex_error *err);
int yvex_runtime_private_session_invalidate(
    yvex_runtime_execution_session *session, int include_state, yvex_error *err);
int yvex_runtime_private_session_workspace_discard(
    yvex_runtime_execution_session *session, yvex_error *err);
int yvex_runtime_private_session_capabilities_bind(
    yvex_runtime_execution_session *session,
    yvex_runtime_model_failure *failure, int require_workspace,
    yvex_error *err);
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
