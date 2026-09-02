/* Execute one heterogeneous autoregressive decoder over session-owned mixed sequence state. */
#ifndef INCLUDE_YVEX_INTERNAL_DECODER_EXECUTION_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_DECODER_EXECUTION_H_INCLUDED

#include <yvex/internal/device_view.h>
#include <yvex/internal/runtime.h>
#include <yvex/internal/transformer.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_RUNTIME_DECODER_EXECUTION_SCHEMA_V1 1u

typedef struct yvex_runtime_decoder_execution_context
    yvex_runtime_decoder_execution_context;

typedef struct {
    unsigned long long context_capacity, token_capacity;
    unsigned long long maximum_host_bytes, maximum_device_bytes;
    int (*cancel_requested)(void *context);
    void *cancel_context;
    const yvex_runtime_execution_profile *execution_profile;
} yvex_runtime_decoder_execution_options;

typedef struct {
    const unsigned int *token_ids;
    unsigned long long token_start, token_count;
    const char *input_identity;
} yvex_runtime_decoder_execution_request;

typedef struct {
    unsigned int schema_version;
    int completed;
    unsigned long long token_start, token_count, position_after;
    unsigned long long layers_executed, attention_layers, recurrent_layers;
    unsigned long long linear_operations, accelerated_matrix_operations;
    unsigned long long kernel_launches, h2d_bytes, d2h_bytes, d2d_bytes;
    unsigned long long recurrent_state_bytes, convolution_state_bytes;
    unsigned long long embedding_nanoseconds, layer_nanoseconds;
    unsigned long long final_nanoseconds;
    yvex_execution_device_view device_hidden;
    char decoder_plan_identity[YVEX_SHA256_HEX_CAP];
    char input_identity[YVEX_SHA256_HEX_CAP];
    char persistent_state_identity[YVEX_SHA256_HEX_CAP];
    char normalized_hidden_digest[YVEX_SHA256_HEX_CAP];
    char execution_identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_decoder_execution_result;

int yvex_runtime_decoder_execution_context_open(
    yvex_runtime_decoder_execution_context **out, yvex_model_engine *model,
    yvex_runtime_execution_session *session,
    const yvex_runtime_decoder_execution_options *options, yvex_error *err);
const yvex_decoder_plan *yvex_runtime_decoder_execution_plan(
    const yvex_runtime_decoder_execution_context *context);
int yvex_runtime_decoder_execution_execute(
    yvex_runtime_decoder_execution_context *context,
    const yvex_runtime_decoder_execution_request *request,
    yvex_runtime_decoder_execution_result *result, yvex_error *err);
int yvex_runtime_decoder_execution_context_close(
    yvex_runtime_decoder_execution_context **context, yvex_error *err);

#ifdef __cplusplus
}
#endif
#endif
