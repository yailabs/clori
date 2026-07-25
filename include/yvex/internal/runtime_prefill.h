/* Owner: runtime activation prefill.
 * Owns: identity-bound activation chunks, bounded tensor-file admission, and chunk coordination.
 * Does not own: prompt text, tokenization, embeddings, attention math, MoE, or generation.
 * Invariants: one sealed input covers every selected layer and each committed chunk is all-or-none.
 * Boundary: typed activations enter the existing runtime/session attention path.
 * Purpose: expose the production activation-prefill contract to operator and future transformer owners.
 * Inputs: sealed runtime identities, ordered layer activations, and session-owned persistent state.
 * Effects: admits immutable input, executes bounded chunks, and advances committed session position.
 * Failure: malformed input or failed chunks preserve the last committed prefix and release input resources. */
#ifndef INCLUDE_YVEX_INTERNAL_RUNTIME_PREFILL_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_RUNTIME_PREFILL_H_INCLUDED

#include <yvex/internal/runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_RUNTIME_ACTIVATION_INPUT_SCHEMA_V1 1u
#define YVEX_RUNTIME_ACTIVATION_INPUT_SUFFIX ".yvex-activations"

typedef struct {
    unsigned int schema_version;
    yvex_attention_operation_scope operation_scope;
    unsigned long long token_start, token_count, layer_count;
    unsigned long long payload_bytes;
    char logical_model_identity[YVEX_SHA256_HEX_CAP];
    char runtime_numeric_identity[YVEX_SHA256_HEX_CAP];
    char runtime_descriptor_identity[YVEX_SHA256_HEX_CAP];
    char attention_plan_identity[YVEX_SHA256_HEX_CAP];
    char payload_digest[YVEX_SHA256_HEX_CAP];
    char input_identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_activation_input_summary;

typedef struct {
    unsigned long long ordinal, layer_index, width, stride;
    unsigned long long payload_offset, payload_bytes;
    char layer_identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_activation_layer_record;

typedef struct yvex_runtime_activation_input yvex_runtime_activation_input;

typedef struct {
    unsigned long long maximum_file_bytes;
} yvex_runtime_activation_input_limits;

typedef struct {
    const char *logical_model_identity;
    const char *runtime_numeric_identity;
    const char *runtime_descriptor_identity;
    const char *attention_plan_identity;
    const yvex_attention_plan *attention;
    yvex_attention_operation_scope operation_scope;
} yvex_runtime_activation_input_expectation;

/* Purpose: derive one ordered layer identity from semantic plan facts only. */
int yvex_runtime_activation_layer_identity_compute(
    const char *attention_plan_identity, unsigned long long ordinal,
    const yvex_attention_layer_plan *layer,
    yvex_attention_operation_scope operation_scope,
    char output[YVEX_SHA256_HEX_CAP], yvex_error *err);

/* Purpose: seal canonical records and payload without serializing native structure bytes. */
int yvex_runtime_activation_input_seal(
    yvex_runtime_activation_input_summary *summary,
    yvex_runtime_activation_layer_record *records,
    const float *payload, yvex_error *err);

/* Purpose: transactionally serialize one already sealed activation input. */
int yvex_runtime_activation_input_write(
    const char *path, const yvex_runtime_activation_input_summary *summary,
    const yvex_runtime_activation_layer_record *records,
    const float *payload, yvex_error *err);

/* Purpose: admit a bounded immutable tensor file through a stable open handle. */
int yvex_runtime_activation_input_open_file(
    yvex_runtime_activation_input **out, const char *path,
    const yvex_runtime_activation_input_limits *limits, yvex_error *err);

/* Purpose: admit caller-owned memory through the same canonical fact validator. */
int yvex_runtime_activation_input_open_memory(
    yvex_runtime_activation_input **out,
    const yvex_runtime_activation_input_summary *summary,
    const yvex_runtime_activation_layer_record *records,
    const float *payload, yvex_error *err);

/* Purpose: revalidate an open file snapshot before or after execution. */
int yvex_runtime_activation_input_validate(
    const yvex_runtime_activation_input *input, yvex_error *err);

const yvex_runtime_activation_input_summary *
yvex_runtime_activation_input_summary_get(
    const yvex_runtime_activation_input *input);
int yvex_runtime_activation_input_view(
    const yvex_runtime_activation_input *input, unsigned long long ordinal,
    unsigned long long token_offset, unsigned long long token_count,
    const float **values, unsigned long long *stride, yvex_error *err);
void yvex_runtime_activation_input_close(yvex_runtime_activation_input **input);

typedef struct {
    yvex_backend_kind backend;
    yvex_runtime_execution_mode mode;
    yvex_runtime_execution_scope operation_scope;
    unsigned long long chunk_tokens, context_capacity;
    unsigned long long maximum_host_bytes, maximum_device_bytes;
    int (*cancel_requested)(void *context);
    void *cancel_context;
} yvex_runtime_activation_prefill_request;

typedef struct {
    int completed;
    unsigned long long token_start, token_count, chunk_count;
    unsigned long long committed_prefix, position_before, position_after;
    unsigned long long generation_before, generation_after;
    unsigned long long layers_executed, bindings_executed;
    unsigned long long swa_layers_executed, csa_layers_executed, hca_layers_executed;
    char input_identity[YVEX_SHA256_HEX_CAP];
    char tensor_output_digest[YVEX_SHA256_HEX_CAP];
    char persistent_state_digest[YVEX_SHA256_HEX_CAP];
    char execution_identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_activation_prefill_result;

/* Purpose: execute all admitted attention layers over ordered activation chunks.
 * Inputs: one model/session, admitted input, explicit backend/mode, and bounded capacity.
 * Effects: commits persistent state once per complete all-layer chunk.
 * Failure: the failing chunk aborts while earlier committed chunks remain authoritative.
 * Boundary: publishes attention outputs and state only, never a complete transformer hidden state. */
int yvex_runtime_activation_prefill_execute(
    yvex_runtime_model *model, yvex_runtime_execution_session *session,
    const yvex_runtime_activation_input *input,
    const yvex_runtime_activation_prefill_request *request,
    yvex_runtime_activation_prefill_result *result,
    yvex_runtime_model_failure *failure, yvex_error *err);

/* Purpose: adapt the existing graph-attention operator request to activation prefill. */
int yvex_runtime_activation_prefill_operator_execute(
    const yvex_graph_attention_operator_request *request,
    yvex_graph_attention_operator_result *result,
    yvex_runtime_cleanup_lease **retained_cleanup, yvex_error *err);

#ifdef __cplusplus
}
#endif
#endif /* INCLUDE_YVEX_INTERNAL_RUNTIME_PREFILL_H_INCLUDED */
