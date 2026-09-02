/*
 * Immutable execution topology for heterogeneous autoregressive decoders.
 *
 * The plan binds family-authored semantic topology to one operator graph and projects the
 * recurrent subset into session-owned sequence state. Physical tensor bindings and backend
 * executables are lower-owner concerns.
 */
#ifndef INCLUDE_YVEX_INTERNAL_DECODER_PLAN_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_DECODER_PLAN_H_INCLUDED

#include <yvex/internal/compiler.h>
#include <yvex/internal/operator_graph.h>
#include <yvex/internal/sequence_mixer.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_DECODER_PLAN_SCHEMA_V1 1u
#define YVEX_DECODER_LAYER_PLAN_SCHEMA_V1 1u
#define YVEX_DECODER_NO_ATTENTION (~0ull)

typedef struct {
    unsigned int schema_version;
    unsigned long long ordinal, layer_index, attention_ordinal;
    yvex_semantic_decoder_mixer mixer;
    yvex_semantic_decoder_ffn feed_forward;
    unsigned long long hidden_width, intermediate_width;
    double normalization_epsilon;
    int mixer_output_gate;
    yvex_gated_delta_plan gated_delta;
    char identity[YVEX_SHA256_HEX_BYTES];
} yvex_decoder_layer_plan;

typedef struct {
    unsigned int schema_version;
    unsigned long long family_adapter_id, family_adapter_version;
    unsigned long long layer_count, attention_layer_count, recurrent_layer_count;
    unsigned long long hidden_width, intermediate_width, vocabulary_size;
    unsigned long long maximum_context;
    unsigned long long convolution_state_bytes, recurrent_state_bytes;
    char logical_model_identity[YVEX_SHA256_HEX_BYTES];
    char semantic_model_identity[YVEX_SHA256_HEX_BYTES];
    char model_execution_identity[YVEX_SHA256_HEX_BYTES];
    char operator_graph_identity[YVEX_SHA256_HEX_BYTES];
    char decoder_plan_identity[YVEX_SHA256_HEX_BYTES];
} yvex_decoder_plan_summary;

typedef struct yvex_decoder_plan yvex_decoder_plan;

int yvex_decoder_plan_compile(
    yvex_decoder_plan **out, const yvex_semantic_model_ir *semantic_model,
    const yvex_operator_graph_ir *operator_graph, yvex_error *err);
int yvex_decoder_plan_import(
    yvex_decoder_plan **out, const yvex_decoder_plan_summary *summary,
    const yvex_decoder_layer_plan *layers, yvex_error *err);
const yvex_decoder_plan_summary *yvex_decoder_plan_summary_get(
    const yvex_decoder_plan *plan);
const yvex_decoder_layer_plan *yvex_decoder_plan_layer_at(
    const yvex_decoder_plan *plan, unsigned long long ordinal);
int yvex_decoder_plan_sequence_state(
    const yvex_decoder_plan *plan, yvex_sequence_state_plan *out,
    yvex_error *err);
void yvex_decoder_plan_close(yvex_decoder_plan **plan);

#ifdef __cplusplus
}
#endif
#endif
