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

/* Scalar-transition selective SSM semantics, independent of physical scan lowering.
 * Geometry describes F32 recurrent/causal-history values, never a KV cache. */
#define YVEX_SELECTIVE_SSD_SCHEMA_V1 1u
typedef struct {
    unsigned int schema_version;
    unsigned long long heads, head_dimension, state_dimension, groups;
    unsigned long long convolution_kernel, normalization_groups;
    double normalization_epsilon, time_step_minimum, time_step_maximum;
    int time_step_unbounded, norm_before_gate;
} yvex_selective_ssd_requirement;

typedef struct {
    unsigned int schema_version;
    yvex_selective_ssd_requirement requirement;
    unsigned long long width, convolution_width, projection_width;
    unsigned long long convolution_state_values, recurrent_state_values;
    char identity[YVEX_SEMANTIC_DECODER_IDENTITY_CAP];
} yvex_selective_ssd_geometry;

int yvex_selective_ssd_geometry_seal(
    yvex_selective_ssd_geometry *geometry, const yvex_selective_ssd_requirement *requirement,
    yvex_error *err);
int yvex_selective_ssd_geometry_validate(
    const yvex_selective_ssd_geometry *geometry, yvex_error *err);

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
