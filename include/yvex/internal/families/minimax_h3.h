/*
 * Bind the exact MiniMax-H3 FL2VA source to one component-aware logical target.
 *
 * The target owns source interpretation and Transformation IR composition only. It exposes no
 * artifact, numerical graph, runtime, backend, solver, VAE execution, or media capability.
 */
#ifndef INCLUDE_YVEX_INTERNAL_FAMILIES_MINIMAX_H3_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_FAMILIES_MINIMAX_H3_H_INCLUDED

#include <stddef.h>
#include <yvex/core.h>
#include <yvex/source.h>

typedef struct yvex_transform_ir yvex_transform_ir;

#define YVEX_MINIMAX_H3_TARGET_ID "minimax-h3-fl2va"
#define YVEX_MINIMAX_H3_REPOSITORY "MiniMaxAI/MiniMax-H3"
#define YVEX_MINIMAX_H3_REVISION "b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08"
#define YVEX_MINIMAX_H3_SUBTREE "FL2VA"
#define YVEX_MINIMAX_H3_SOURCE_TREE_IDENTITY \
    "91972f8e4e6562562456c339b43eed1fba5f7b9d7fb13987f495b416a5109b5e"
#define YVEX_MINIMAX_H3_SOURCE_INVENTORY_IDENTITY \
    "c37f859ce8cccf2465adcd0e31f0d21d603ec41cccd15301c8cf467d651625e3"
#define YVEX_MINIMAX_H3_MODEL_INDEX_IDENTITY \
    "d1113e0f123c69f79cd0de35ca1771606ebc3ec924270d257b771f96f584aa6b"
#define YVEX_MINIMAX_H3_LOGICAL_COMPONENTS 8ull
#define YVEX_MINIMAX_H3_WEIGHTED_COMPONENTS 4ull
#define YVEX_MINIMAX_H3_SOURCE_FILES 83ull
#define YVEX_MINIMAX_H3_SHARDS 29ull
#define YVEX_MINIMAX_H3_TENSORS 3240ull
#define YVEX_MINIMAX_H3_ELEMENTS 69235580593ull
#define YVEX_MINIMAX_H3_TENSOR_BYTES 144016000740ull
#define YVEX_MINIMAX_H3_SOURCE_BYTES 144051204180ull
#define YVEX_MINIMAX_H3_NO_COORDINATE (~0ull)

typedef enum {
    YVEX_MINIMAX_H3_COMPONENT_PIPELINE = 0,
    YVEX_MINIMAX_H3_COMPONENT_PROCESSOR,
    YVEX_MINIMAX_H3_COMPONENT_TOKENIZER,
    YVEX_MINIMAX_H3_COMPONENT_TEXT_ENCODER,
    YVEX_MINIMAX_H3_COMPONENT_TRANSFORMER,
    YVEX_MINIMAX_H3_COMPONENT_VIDEO_VAE,
    YVEX_MINIMAX_H3_COMPONENT_AUDIO_VAE,
    YVEX_MINIMAX_H3_COMPONENT_LATENT_CONTROLLER,
    YVEX_MINIMAX_H3_COMPONENT_COUNT
} yvex_minimax_h3_component_id;

typedef enum {
    YVEX_MINIMAX_H3_PHASE_PREPARE = 1,
    YVEX_MINIMAX_H3_PHASE_CONDITION,
    YVEX_MINIMAX_H3_PHASE_LATENT_INITIALIZE,
    YVEX_MINIMAX_H3_PHASE_LATENT_ITERATE,
    YVEX_MINIMAX_H3_PHASE_VIDEO_DECODE,
    YVEX_MINIMAX_H3_PHASE_AUDIO_DECODE,
    YVEX_MINIMAX_H3_PHASE_MEDIA_PUBLISH
} yvex_minimax_h3_phase;

typedef enum {
    YVEX_MINIMAX_H3_LIFETIME_METADATA = 1,
    YVEX_MINIMAX_H3_LIFETIME_PHASE,
    YVEX_MINIMAX_H3_LIFETIME_REQUEST_IMMUTABLE,
    YVEX_MINIMAX_H3_LIFETIME_REQUEST_MUTABLE,
    YVEX_MINIMAX_H3_LIFETIME_OUTPUT_TRANSACTION
} yvex_minimax_h3_lifetime;

typedef enum {
    YVEX_MINIMAX_H3_SHARING_INDEPENDENT = 0,
    YVEX_MINIMAX_H3_SHARING_LOGICALLY_SHARED,
    YVEX_MINIMAX_H3_SHARING_DUPLICATED_SOURCE
} yvex_minimax_h3_sharing;

typedef enum {
    YVEX_MINIMAX_H3_TRANSFORM_IDENTITY = 0
} yvex_minimax_h3_transform;

typedef enum {
    YVEX_MINIMAX_H3_DATA_TEXT = 1u << 0,
    YVEX_MINIMAX_H3_DATA_MEDIA_GRID = 1u << 1,
    YVEX_MINIMAX_H3_DATA_TOKEN_IDS = 1u << 2,
    YVEX_MINIMAX_H3_DATA_CONDITIONING = 1u << 3,
    YVEX_MINIMAX_H3_DATA_VIDEO_LATENT = 1u << 4,
    YVEX_MINIMAX_H3_DATA_AUDIO_LATENT = 1u << 5,
    YVEX_MINIMAX_H3_DATA_RGB_FRAMES = 1u << 6,
    YVEX_MINIMAX_H3_DATA_STEREO_SAMPLES = 1u << 7,
    YVEX_MINIMAX_H3_DATA_SYNCHRONIZED_MEDIA = 1u << 8
} yvex_minimax_h3_data_class;

typedef enum {
    YVEX_MINIMAX_H3_ROLE_TEXT_EMBEDDING = 1,
    YVEX_MINIMAX_H3_ROLE_TEXT_OUTPUT_HEAD,
    YVEX_MINIMAX_H3_ROLE_TEXT_ATTENTION_Q,
    YVEX_MINIMAX_H3_ROLE_TEXT_ATTENTION_K,
    YVEX_MINIMAX_H3_ROLE_TEXT_ATTENTION_V,
    YVEX_MINIMAX_H3_ROLE_TEXT_ATTENTION_OUT,
    YVEX_MINIMAX_H3_ROLE_TEXT_QK_NORM,
    YVEX_MINIMAX_H3_ROLE_TEXT_RMS_NORM,
    YVEX_MINIMAX_H3_ROLE_TEXT_MLP,
    YVEX_MINIMAX_H3_ROLE_VISION_PATCH,
    YVEX_MINIMAX_H3_ROLE_VISION_POSITION,
    YVEX_MINIMAX_H3_ROLE_VISION_ATTENTION,
    YVEX_MINIMAX_H3_ROLE_VISION_NORM,
    YVEX_MINIMAX_H3_ROLE_VISION_MLP,
    YVEX_MINIMAX_H3_ROLE_VISION_MERGER,
    YVEX_MINIMAX_H3_ROLE_VISION_DEEPSTACK,
    YVEX_MINIMAX_H3_ROLE_CONDITION_PROJECTION,
    YVEX_MINIMAX_H3_ROLE_VIDEO_PATCH,
    YVEX_MINIMAX_H3_ROLE_AUDIO_PATCH,
    YVEX_MINIMAX_H3_ROLE_TIMESTEP_PROJECTION,
    YVEX_MINIMAX_H3_ROLE_TOKEN_REFINER_ATTENTION,
    YVEX_MINIMAX_H3_ROLE_TOKEN_REFINER_NORM,
    YVEX_MINIMAX_H3_ROLE_TOKEN_REFINER_MLP,
    YVEX_MINIMAX_H3_ROLE_OMNI_ATTENTION,
    YVEX_MINIMAX_H3_ROLE_OMNI_QK_NORM,
    YVEX_MINIMAX_H3_ROLE_OMNI_NORM,
    YVEX_MINIMAX_H3_ROLE_OMNI_MLP,
    YVEX_MINIMAX_H3_ROLE_OMNI_ADALN,
    YVEX_MINIMAX_H3_ROLE_OMNI_POSITION,
    YVEX_MINIMAX_H3_ROLE_FINAL_NORM,
    YVEX_MINIMAX_H3_ROLE_FINAL_ADALN,
    YVEX_MINIMAX_H3_ROLE_FINAL_VIDEO,
    YVEX_MINIMAX_H3_ROLE_FINAL_AUDIO,
    YVEX_MINIMAX_H3_ROLE_VIDEO_ENCODER_CONV,
    YVEX_MINIMAX_H3_ROLE_VIDEO_ENCODER_NORM,
    YVEX_MINIMAX_H3_ROLE_VIDEO_RESAMPLE,
    YVEX_MINIMAX_H3_ROLE_VIDEO_LATENT_PROJECTION,
    YVEX_MINIMAX_H3_ROLE_VIDEO_DECODER_PROJECTION,
    YVEX_MINIMAX_H3_ROLE_VIDEO_DECODER_TOKEN,
    YVEX_MINIMAX_H3_ROLE_VIDEO_DECODER_ATTENTION,
    YVEX_MINIMAX_H3_ROLE_VIDEO_DECODER_NORM,
    YVEX_MINIMAX_H3_ROLE_VIDEO_DECODER_MLP,
    YVEX_MINIMAX_H3_ROLE_VIDEO_DECODER_SCALE,
    YVEX_MINIMAX_H3_ROLE_VIDEO_OUTPUT,
    YVEX_MINIMAX_H3_ROLE_AUDIO_ENCODER_CONV,
    YVEX_MINIMAX_H3_ROLE_AUDIO_RESAMPLE,
    YVEX_MINIMAX_H3_ROLE_AUDIO_LATENT_PROJECTION,
    YVEX_MINIMAX_H3_ROLE_AUDIO_PRE_ATTENTION,
    YVEX_MINIMAX_H3_ROLE_AUDIO_PRE_NORM,
    YVEX_MINIMAX_H3_ROLE_AUDIO_PRE_MLP,
    YVEX_MINIMAX_H3_ROLE_AUDIO_DECODER_CONV,
    YVEX_MINIMAX_H3_ROLE_AUDIO_FILTER,
    YVEX_MINIMAX_H3_ROLE_AUDIO_ACTIVATION,
    YVEX_MINIMAX_H3_ROLE_AUDIO_OUTPUT,
    YVEX_MINIMAX_H3_ROLE_COUNT
} yvex_minimax_h3_role;

typedef enum {
    YVEX_MINIMAX_H3_FAILURE_NONE = 0,
    YVEX_MINIMAX_H3_FAILURE_INVALID_ARGUMENT,
    YVEX_MINIMAX_H3_FAILURE_SOURCE_ACQUISITION,
    YVEX_MINIMAX_H3_FAILURE_SOURCE_INVENTORY,
    YVEX_MINIMAX_H3_FAILURE_SOURCE_IDENTITY,
    YVEX_MINIMAX_H3_FAILURE_COMPONENT_COVERAGE,
    YVEX_MINIMAX_H3_FAILURE_COMPONENT_CYCLE,
    YVEX_MINIMAX_H3_FAILURE_PHASE_ORDER,
    YVEX_MINIMAX_H3_FAILURE_ARCHITECTURE,
    YVEX_MINIMAX_H3_FAILURE_TENSOR_ROLE,
    YVEX_MINIMAX_H3_FAILURE_SHAPE,
    YVEX_MINIMAX_H3_FAILURE_DTYPE,
    YVEX_MINIMAX_H3_FAILURE_SOURCE_RANGE,
    YVEX_MINIMAX_H3_FAILURE_TRANSFORMATION,
    YVEX_MINIMAX_H3_FAILURE_RESOURCE_BUDGET,
    YVEX_MINIMAX_H3_FAILURE_ALLOCATION
} yvex_minimax_h3_failure_code;

typedef struct {
    yvex_minimax_h3_failure_code code;
    yvex_minimax_h3_component_id component;
    yvex_minimax_h3_role role;
    unsigned long long tensor_index;
    char source_name[256];
} yvex_minimax_h3_failure;

typedef struct {
    yvex_minimax_h3_component_id id;
    const char *canonical_id;
    const char *implementation_class;
    const char *configuration_path;
    char identity[65];
    unsigned int dependency_mask;
    unsigned int input_classes;
    unsigned int output_classes;
    yvex_minimax_h3_phase phase;
    yvex_minimax_h3_lifetime lifetime;
    unsigned long long file_count;
    unsigned long long shard_count;
    unsigned long long tensor_count;
    unsigned long long element_count;
    unsigned long long payload_bytes;
    int weighted;
    int release_after_phase;
    int request_local;
} yvex_minimax_h3_component;

typedef struct {
    unsigned long long text_layers;
    unsigned long long text_width;
    unsigned long long text_ffn_width;
    unsigned long long text_query_heads;
    unsigned long long text_kv_heads;
    unsigned long long text_head_dimension;
    unsigned long long vocabulary_size;
    unsigned long long rope_theta;
    unsigned long long mrope_sections[3];
    unsigned long long vision_layers;
    unsigned long long vision_width;
    unsigned long long vision_ffn_width;
    unsigned long long vision_heads;
    unsigned long long vision_patch[3];
    unsigned long long vision_merge;
    unsigned long long deepstack_layers[3];
    unsigned long long conditioning_width;
    int tied_embeddings;
} yvex_minimax_h3_encoder_signature;

typedef struct {
    unsigned long long blocks;
    unsigned long long width;
    unsigned long long ffn_width;
    unsigned long long heads;
    unsigned long long head_dimension;
    unsigned long long token_refiner_blocks;
    unsigned long long conditioning_width;
    unsigned long long video_channels;
    unsigned long long audio_channels;
    unsigned long long video_patch[3];
    unsigned long long timestep_input;
    unsigned long long timestep_hidden;
    unsigned long long timestep_output;
    unsigned long long adaln_width;
    unsigned long long final_adaln_width;
    unsigned long long video_head_width;
    unsigned long long audio_head_width;
    int qk_normalization;
} yvex_minimax_h3_omni_signature;

typedef struct {
    unsigned long long latent_channels;
    unsigned long long media_channels;
    unsigned long long spatial_ratio;
    unsigned long long temporal_ratio;
    unsigned long long decoder_blocks;
    unsigned long long decoder_heads;
    unsigned long long tile_size;
    unsigned long long tile_overlap;
    unsigned long long clip_length;
    unsigned long long token_drop;
    int causal_encoder;
    int causal_decoder;
} yvex_minimax_h3_video_vae_signature;

typedef struct {
    unsigned long long latent_channels;
    unsigned long long output_channels;
    unsigned long long sample_rate;
    unsigned long long encoder_rate_product;
    unsigned long long decoder_rate_product;
    unsigned long long latent_steps_per_second;
} yvex_minimax_h3_audio_vae_signature;

typedef struct {
    yvex_minimax_h3_encoder_signature encoder;
    yvex_minimax_h3_omni_signature omni;
    yvex_minimax_h3_video_vae_signature video_vae;
    yvex_minimax_h3_audio_vae_signature audio_vae;
    unsigned long long bf16_tensors;
    unsigned long long f32_tensors;
    char identity[65];
} yvex_minimax_h3_architecture;

typedef struct {
    yvex_minimax_h3_component_id component;
    yvex_minimax_h3_role role;
    const char *source_name;
    const char *shard_name;
    yvex_native_dtype source_dtype;
    unsigned int rank;
    unsigned long long source_shape[YVEX_NATIVE_WEIGHT_MAX_DIMS];
    unsigned long long logical_shape[YVEX_NATIVE_WEIGHT_MAX_DIMS];
    unsigned long long relative_begin;
    unsigned long long relative_end;
    unsigned long long elements;
    unsigned long long layer_index;
    unsigned long long subcomponent_index;
    unsigned long long repeated_index;
    yvex_minimax_h3_phase phase;
    yvex_minimax_h3_lifetime lifetime;
    unsigned long long unresolved_requirement_identity;
    yvex_minimax_h3_transform transform;
    yvex_minimax_h3_sharing sharing;
    char destination_identity[65];
} yvex_minimax_h3_tensor_role;

typedef struct {
    char source_acquisition_identity[65];
    char source_snapshot_identity[65];
    char component_manifest_identity[65];
    char architecture_identity[65];
    char role_map_identity[65];
    char target_identity[65];
    char unresolved_requirements_identity[65];
    unsigned long long source_snapshot_key;
    unsigned long long component_count;
    unsigned long long weighted_component_count;
    unsigned long long source_file_count;
    unsigned long long shard_count;
    unsigned long long tensor_count;
    unsigned long long element_count;
    unsigned long long payload_bytes;
    unsigned long long unmapped_tensors;
    unsigned long long duplicate_mappings;
    unsigned int output_classes;
    int source_verified;
    int architecture_admitted;
    int roles_complete;
} yvex_minimax_h3_summary;

typedef struct yvex_minimax_h3_target yvex_minimax_h3_target;
typedef struct {
    const char *source_root;
} yvex_minimax_h3_open_options;
typedef struct {
    int (*open)(yvex_minimax_h3_target **out,
                const yvex_minimax_h3_open_options *options,
                yvex_minimax_h3_failure *failure,
                yvex_error *err);
    void (*close)(yvex_minimax_h3_target **target);
    const yvex_minimax_h3_summary *(*summary)(const yvex_minimax_h3_target *target);
    const yvex_minimax_h3_architecture *(*architecture)(
        const yvex_minimax_h3_target *target);
    const yvex_minimax_h3_component *(*component_at)(
        const yvex_minimax_h3_target *target, unsigned long long index);
    const yvex_minimax_h3_tensor_role *(*role_at)(
        const yvex_minimax_h3_target *target, unsigned long long index);
    const char *(*failure_name)(yvex_minimax_h3_failure_code code);
    const char *(*role_name)(yvex_minimax_h3_role role);
    const char *(*component_name)(yvex_minimax_h3_component_id component);
    int (*components_canonical)(
        yvex_minimax_h3_component components[YVEX_MINIMAX_H3_COMPONENT_COUNT],
        char manifest_identity[65], yvex_minimax_h3_failure *failure,
        yvex_error *err);
    int (*component_graph_validate)(
        const yvex_minimax_h3_component *components, size_t component_count,
        yvex_minimax_h3_failure *failure, yvex_error *err);
    int (*architecture_canonical)(
        yvex_minimax_h3_architecture *architecture,
        yvex_minimax_h3_failure *failure, yvex_error *err);
    int (*tensor_classify)(
        const yvex_native_weight_info *tensor, yvex_minimax_h3_tensor_role *role,
        yvex_minimax_h3_failure *failure, yvex_error *err);
} yvex_minimax_h3_api;
typedef struct {
    int (*build)(yvex_transform_ir **out,
                 char derivation_identity[65],
                 const yvex_minimax_h3_target *target,
                 yvex_error *err);
} yvex_minimax_h3_transform_api;

const yvex_minimax_h3_api *yvex_model_register_minimax_h3(void);
const yvex_minimax_h3_transform_api *yvex_model_minimax_h3_transform_api(void);

#endif /* INCLUDE_YVEX_INTERNAL_FAMILIES_MINIMAX_H3_H_INCLUDED */
