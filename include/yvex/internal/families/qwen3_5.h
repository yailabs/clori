/* Qwen3.5 semantic architecture; product releases remain separate source identities. */
#ifndef INCLUDE_YVEX_INTERNAL_FAMILIES_QWEN3_5_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_FAMILIES_QWEN3_5_H_INCLUDED

#include <yvex/core.h>
#include <yvex/internal/source.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_QWEN3_8_27B_TARGET_ID "qwen3.8-27b"
#define YVEX_QWEN3_5_FAMILY_KEY "qwen3_5"
#define YVEX_QWEN3_5_TEXT_MODEL_TYPE "qwen3_5_text"
#define YVEX_QWEN3_5_LAYER_CAP 64u
#define YVEX_QWEN3_5_GENERATION_STOP_CAP 4u
#define YVEX_QWEN3_5_ADAPTER_ID 0x5157454e335f35ull
#define YVEX_QWEN3_5_ADAPTER_VERSION 1ull

typedef enum {
    YVEX_QWEN3_5_LAYER_LINEAR_ATTENTION = 1,
    YVEX_QWEN3_5_LAYER_FULL_ATTENTION = 2
} yvex_qwen3_5_layer_kind;

typedef enum {
    YVEX_QWEN3_5_FAILURE_NONE = 0,
    YVEX_QWEN3_5_FAILURE_INVALID_ARGUMENT,
    YVEX_QWEN3_5_FAILURE_SOURCE_NOT_VERIFIED,
    YVEX_QWEN3_5_FAILURE_SOURCE_IDENTITY,
    YVEX_QWEN3_5_FAILURE_MISSING_CONFIG,
    YVEX_QWEN3_5_FAILURE_MALFORMED_CONFIG,
    YVEX_QWEN3_5_FAILURE_CONFIGURATION,
    YVEX_QWEN3_5_FAILURE_GENERATION_POLICY,
    YVEX_QWEN3_5_FAILURE_ALLOCATION
} yvex_qwen3_5_failure_code;

typedef struct {
    yvex_qwen3_5_failure_code code;
    char field[64];
    const char *reason;
} yvex_qwen3_5_failure;

typedef struct {
    unsigned long long hidden_size, layer_count, vocabulary_size;
    unsigned long long intermediate_size, maximum_positions;
    unsigned long long full_attention_interval, full_attention_layers;
    unsigned long long linear_attention_layers;
    unsigned long long attention_heads, kv_heads, attention_head_dimension;
    unsigned long long rotary_dimension, rope_theta;
    unsigned long long mrope_sections[3];
    unsigned long long linear_key_heads, linear_value_heads;
    unsigned long long linear_key_head_dimension, linear_value_head_dimension;
    unsigned long long linear_convolution_kernel;
    unsigned long long mtp_hidden_layers;
    unsigned long long bos_token_id, eos_token_id;
    yvex_qwen3_5_layer_kind layers[YVEX_QWEN3_5_LAYER_CAP];
    char model_type[32], source_dtype[16], recurrent_state_dtype[16];
    char hidden_activation[16], output_gate_type[16];
    double partial_rotary_factor, rope_partial_rotary_factor;
    double rms_norm_epsilon, attention_dropout;
    int attention_output_gate, attention_bias, use_cache;
    int tied_embeddings, mtp_dedicated_embeddings, recurrent_state_f32;
    int mrope_interleaved;
} yvex_qwen3_5_text_architecture;

typedef struct {
    unsigned long long depth, hidden_size, intermediate_size, heads;
    unsigned long long output_hidden_size, position_count;
    unsigned long long patch_size, spatial_merge_size, temporal_patch_size;
    unsigned long long image_token_id, video_token_id;
    unsigned long long vision_start_token_id, vision_end_token_id;
    unsigned long long deepstack_index_count;
    char model_type[32], hidden_activation[32];
} yvex_qwen3_5_vision_architecture;

typedef struct {
    unsigned long long bos_token_id, pad_token_id;
    unsigned long long stop_token_ids[YVEX_QWEN3_5_GENERATION_STOP_CAP];
    unsigned long long stop_token_count, top_k;
    double temperature, top_p;
    int do_sample;
} yvex_qwen3_5_generation_policy;

typedef struct {
    yvex_qwen3_5_text_architecture text;
    yvex_qwen3_5_vision_architecture vision;
    yvex_qwen3_5_generation_policy generation;
    char product_id[64], semantic_family[32], source_revision[65];
    char transformers_version[32], architecture_identity[65];
    int source_multimodal, text_specialization;
    int vision_execution_deferred, mtp_acceleration_deferred;
} yvex_qwen3_5_architecture;

typedef struct yvex_qwen3_5_model yvex_qwen3_5_model;

typedef struct {
    unsigned int schema_version;
    int (*open)(yvex_qwen3_5_model **out,
                const yvex_source_verification *verification,
                yvex_qwen3_5_failure *failure, yvex_error *err);
    void (*close)(yvex_qwen3_5_model **model);
    const yvex_qwen3_5_architecture *(*architecture)(
        const yvex_qwen3_5_model *model);
    yvex_qwen3_5_layer_kind (*layer_kind)(
        const yvex_qwen3_5_model *model, unsigned long long layer);
    const char *(*failure_name)(yvex_qwen3_5_failure_code code);
} yvex_qwen3_5_api;

const yvex_qwen3_5_api *yvex_model_register_qwen3_5(void);

#ifdef __cplusplus
}
#endif
#endif
