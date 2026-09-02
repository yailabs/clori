/* Family-neutral decoder topology for heterogeneous token mixers and dense feed-forward blocks. */
#ifndef INCLUDE_YVEX_INTERNAL_SEMANTIC_DECODER_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_SEMANTIC_DECODER_H_INCLUDED

#include <yvex/core.h>
#include <yvex/model.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    YVEX_SEMANTIC_DECODER_MIXER_FULL_CAUSAL_ATTENTION = 1,
    YVEX_SEMANTIC_DECODER_MIXER_GATED_DELTA
} yvex_semantic_decoder_mixer;

typedef enum {
    YVEX_SEMANTIC_DECODER_FFN_DENSE_SILU_GATED = 1
} yvex_semantic_decoder_ffn;

typedef enum {
    YVEX_NORMALIZATION_WEIGHT_UNKNOWN = 0,
    YVEX_NORMALIZATION_WEIGHT_DIRECT,
    YVEX_NORMALIZATION_WEIGHT_ONE_PLUS
} yvex_normalization_weight_convention;

#define YVEX_SEQUENCE_MIXER_GATED_DELTA_SCHEMA_V2 2u
#define YVEX_SEMANTIC_DECODER_IDENTITY_CAP 65u
typedef enum {
    YVEX_SEQUENCE_MIXER_NUMERIC_UNKNOWN = 0,
    YVEX_SEQUENCE_MIXER_NUMERIC_F32_RECURRENCE
} yvex_sequence_mixer_numeric_contract;

typedef struct {
    unsigned int schema_version;
    unsigned long long query_heads, key_heads, value_heads;
    unsigned long long key_head_dimension, value_head_dimension;
    unsigned long long convolution_kernel;
    yvex_dtype projected_dtype, convolution_state_dtype;
    yvex_dtype recurrent_state_dtype, accumulation_dtype, output_dtype;
    yvex_sequence_mixer_numeric_contract numeric_contract;
    yvex_normalization_weight_convention output_normalization_weight_convention;
    double qk_normalization_epsilon, output_normalization_epsilon;
    double query_scale;
    int deterministic;
} yvex_gated_delta_requirement;

int yvex_semantic_gated_delta_requirement_identity(
    const yvex_gated_delta_requirement *requirement,
    char identity[YVEX_SEMANTIC_DECODER_IDENTITY_CAP]);

typedef struct {
    unsigned long long ordinal, layer_index;
    yvex_tensor_scope tensor_scope;
    yvex_semantic_decoder_mixer mixer;
    yvex_semantic_decoder_ffn feed_forward;
    unsigned long long hidden_width, intermediate_width;
    yvex_normalization_weight_convention normalization_weight_convention;
    double normalization_epsilon;
    int mixer_output_gate;
    yvex_gated_delta_requirement gated_delta;
} yvex_semantic_decoder_layer;

typedef int (*yvex_semantic_decoder_layer_fn)(
    const void *context, unsigned long long index,
    yvex_semantic_decoder_layer *output);

#ifdef __cplusplus
}
#endif
#endif
