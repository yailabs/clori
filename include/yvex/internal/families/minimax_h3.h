/* Bind exact FL2VA facts and bounded execution without promoting a family runtime. */
#ifndef INCLUDE_YVEX_INTERNAL_FAMILIES_MINIMAX_H3_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_FAMILIES_MINIMAX_H3_H_INCLUDED
#include <stddef.h>
#include <yvex/core.h>
#include <yvex/source.h>
#include <yvex/internal/source_payload.h>
typedef struct yvex_transform_ir yvex_transform_ir;
typedef struct yvex_transform_binding yvex_transform_binding;
typedef struct yvex_artifact yvex_artifact;
typedef struct yvex_gguf yvex_gguf;
typedef struct yvex_tensor_table yvex_tensor_table;
typedef struct yvex_complete_artifact_admission yvex_complete_artifact_admission;
typedef struct yvex_artifact_admission_failure yvex_artifact_admission_failure;
typedef struct yvex_materialization_session yvex_materialization_session;
typedef struct yvex_backend yvex_backend;
typedef struct yvex_runtime_component_session yvex_runtime_component_session;
typedef struct yvex_runtime_latent_result yvex_runtime_latent_result;
typedef struct yvex_runtime_latent_evaluator_result yvex_runtime_latent_evaluator_result;
typedef struct yvex_runtime_av_layout_output yvex_runtime_av_layout_output;
typedef struct yvex_runtime_av_layout_result yvex_runtime_av_layout_result;
typedef struct yvex_component_encoded_weight yvex_minimax_h3_encoded_weight;
#define YVEX_MINIMAX_H3_TARGET_ID "minimax-h3-fl2va"
#define YVEX_MINIMAX_H3_REPOSITORY "MiniMaxAI/MiniMax-H3"
#define YVEX_MINIMAX_H3_REVISION "b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08"
#define YVEX_MINIMAX_H3_SUBTREE "FL2VA"
#define YVEX_MINIMAX_H3_SOURCE_TREE_IDENTITY "91972f8e4e6562562456c339b43eed1fba5f7b9d7fb13987f495b416a5109b5e"
#define YVEX_MINIMAX_H3_SOURCE_INVENTORY_IDENTITY "c37f859ce8cccf2465adcd0e31f0d21d603ec41cccd15301c8cf467d651625e3"
#define YVEX_MINIMAX_H3_MODEL_INDEX_IDENTITY "d1113e0f123c69f79cd0de35ca1771606ebc3ec924270d257b771f96f584aa6b"
#define YVEX_MINIMAX_H3_LOGICAL_COMPONENTS 8ull
#define YVEX_MINIMAX_H3_WEIGHTED_COMPONENTS 4ull
#define YVEX_MINIMAX_H3_PHASE_EDGES 7ull
#define YVEX_MINIMAX_H3_SOURCE_FILES 83ull
#define YVEX_MINIMAX_H3_SHARDS 29ull
#define YVEX_MINIMAX_H3_TENSORS 3240ull
#define YVEX_MINIMAX_H3_ELEMENTS 69235580593ull
#define YVEX_MINIMAX_H3_TENSOR_BYTES 144016000740ull
#define YVEX_MINIMAX_H3_SOURCE_BYTES 144051204180ull
#define YVEX_MINIMAX_H3_NO_COORDINATE (~0ull)
/* Exact source-faithful Audio VAE physical boundary emitted by YVEX. */
#define YVEX_MINIMAX_H3_AUDIO_COMPONENT_IDENTITY "be921beb8581b44624aaad452f30f77f1e204159ae8fe11da455d5208dc4e62b"
#define YVEX_MINIMAX_H3_AUDIO_SNAPSHOT_IDENTITY "897ceaff08708f431132c6643bc8f1041ace8c0444a3ea248bbf727fc7da9943"
#define YVEX_MINIMAX_H3_AUDIO_MANIFEST_IDENTITY "715f2359aaff048ccca8207976421af5f9f76b08b6f24986b3cc186d2822bc0e"
#define YVEX_MINIMAX_H3_AUDIO_ARCHITECTURE_IDENTITY "47a03bbac2b5346771f70ae39155920f9b1c6e6cec17f2639dd0cbedfa90b517"
#define YVEX_MINIMAX_H3_AUDIO_ROLE_MAP_IDENTITY "61e7a2cfc29e6dd3da966878f5388f1472a406d7e33ba34ef65f44b61f08f013"
#define YVEX_MINIMAX_H3_AUDIO_UNRESOLVED_IDENTITY "935ae0a2371b15131b8920a879462484ebd3f5526ff5a97ef95c4e0af7b7cc1d"
#define YVEX_MINIMAX_H3_AUDIO_TRANSFORM_IDENTITY "e6f8f3ac2ae01157a57049f0db2439271585966174c0bfe202a5546471361ab3"
#define YVEX_MINIMAX_H3_AUDIO_PROFILE_NAME "minimax-h3-source-faithful-v1"
#define YVEX_MINIMAX_H3_AUDIO_PROFILE_IDENTITY "b8b5aa330a617b0fa33fdd1428e5fea9e8edcdd7f6a2ba6f530d378fbaddaa65"
#define YVEX_MINIMAX_H3_AUDIO_QUANT_IDENTITY "551609d790bd9af9a51297bacbc7d476bbe436239ee0ce86fb1daa896fccd2ec"
#define YVEX_MINIMAX_H3_AUDIO_PAYLOAD_PLAN_IDENTITY "c42dee2e548b9452707cb3327e55f09e3e9262bfc3d3d3665170bc9cfce1ffe4"
#define YVEX_MINIMAX_H3_AUDIO_PAYLOAD_BYTE_IDENTITY "59850eaaecfc00f777bbeb2506a231e57313940a4c0e00b4501472cbc1a5cbf2"
#define YVEX_MINIMAX_H3_AUDIO_PAYLOAD_IDENTITY "7eec5b07bbb6427611553b16670f9dc31969ae8ba602a79c0c7a2693a5fa168a"
#define YVEX_MINIMAX_H3_AUDIO_WRITER_PLAN_IDENTITY "40c89b292935ae03708df9a131d92fbd2fc2de6428550ade6f8c436294217271"
#define YVEX_MINIMAX_H3_AUDIO_ARTIFACT_IDENTITY "52a10c9f6f6e3b9b81569a95329f503fcb3cbddb224d12bf7851b4929d02e1c1"
#define YVEX_MINIMAX_H3_AUDIO_SOURCE_SNAPSHOT_KEY 9907051661387403075ull
#define YVEX_MINIMAX_H3_AUDIO_MAPPING_IDENTITY 15010405512422704850ull
#define YVEX_MINIMAX_H3_AUDIO_TENSORS 1087ull
#define YVEX_MINIMAX_H3_AUDIO_ELEMENTS 151326585ull
#define YVEX_MINIMAX_H3_AUDIO_PAYLOAD_BYTES 605306340ull
#define YVEX_MINIMAX_H3_AUDIO_FILE_BYTES 605401984ull
#define YVEX_MINIMAX_H3_TEXT_COMPONENT_IDENTITY "a4b9c13360aeaa03bbd4d9681b821575e6556bead71c226d0aa72fca5aca7382"
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
    const char *canonical_id, *implementation_class, *configuration_path;
    char identity[65];
    unsigned int dependency_mask, input_classes, output_classes;
    yvex_minimax_h3_phase phase;
    yvex_minimax_h3_lifetime lifetime;
    unsigned long long file_count, shard_count, tensor_count, element_count, payload_bytes;
    int weighted, release_after_phase, request_local;
} yvex_minimax_h3_component;
typedef struct {
    const char *json_path, *config_path, *pre_tokenizer, *prompt_policy;
    unsigned long long token_count;
} yvex_minimax_h3_tokenizer_spec;
typedef struct {
    yvex_minimax_h3_phase source_phase, destination_phase;
    unsigned int data_classes;
    yvex_minimax_h3_lifetime lifetime;
} yvex_minimax_h3_phase_edge;
typedef struct {
    unsigned long long text_layers, text_width, text_ffn_width;
    unsigned long long text_query_heads, text_kv_heads, text_head_dimension;
    unsigned long long vocabulary_size, rope_theta;
    unsigned long long mrope_sections[3];
    unsigned long long vision_layers, vision_width, vision_ffn_width, vision_heads;
    unsigned long long vision_patch[3];
    unsigned long long vision_merge;
    unsigned long long deepstack_layers[3];
    unsigned long long conditioning_width;
    int tied_embeddings;
} yvex_minimax_h3_encoder_signature;
typedef struct {
    unsigned long long blocks, width, ffn_width, heads, head_dimension;
    unsigned long long token_refiner_blocks, conditioning_width;
    unsigned long long video_channels, audio_channels;
    unsigned long long video_patch[3];
    unsigned long long audio_patch_steps, audio_patch_channels;
    unsigned long long timestep_input, timestep_hidden, timestep_output;
    unsigned long long adaln_width, final_adaln_width;
    unsigned long long video_head_width, audio_head_width;
    int qk_normalization;
} yvex_minimax_h3_omni_signature;
typedef struct {
    unsigned long long latent_channels, media_channels, base_channels, stage_count;
    unsigned long long channel_multipliers[6];
    unsigned long long spatial_down[6];
    unsigned long long spatial_up[6];
    unsigned long long temporal_down[6];
    unsigned long long spatial_ratio, temporal_ratio, residual_blocks;
    unsigned long long decoder_blocks, decoder_heads, decoder_head_dimension;
    unsigned long long decoder_rope_ratio_numerator, decoder_rope_ratio_denominator;
    unsigned long long decoder_rope_theta, tile_size, tile_overlap, clip_length, token_drop;
    int conv3d, isolated_temporal_group_norm, encoder_tiling, decoder_tiling;
    int parallel_tiling, causal_encoder, causal_decoder;
} yvex_minimax_h3_video_vae_signature;
typedef struct {
    unsigned long long latent_channels, output_channels, sample_rate;
    unsigned long long encoder_width, decoder_width, latent_projection_width;
    unsigned long long encoder_stage_count;
    unsigned long long encoder_rates[5];
    unsigned long long decoder_stage_count;
    unsigned long long decoder_rates[7];
    unsigned long long encoder_rate_product, decoder_rate_product, latent_steps_per_second;
    int attention_projection;
} yvex_minimax_h3_audio_vae_signature;
typedef struct {
    yvex_minimax_h3_encoder_signature encoder;
    yvex_minimax_h3_omni_signature omni;
    yvex_minimax_h3_video_vae_signature video_vae;
    yvex_minimax_h3_audio_vae_signature audio_vae;
    unsigned long long bf16_tensors, f32_tensors;
    char identity[65];
} yvex_minimax_h3_architecture;
typedef struct {
    yvex_minimax_h3_component_id component;
    yvex_minimax_h3_role role;
    const char *source_name, *shard_name;
    yvex_native_dtype source_dtype;
    unsigned int rank;
    unsigned long long source_shape[YVEX_NATIVE_WEIGHT_MAX_DIMS];
    unsigned long long logical_shape[YVEX_NATIVE_WEIGHT_MAX_DIMS];
    unsigned long long relative_begin, relative_end, elements;
    unsigned long long layer_index, subcomponent_index, repeated_index;
    yvex_minimax_h3_phase phase;
    yvex_minimax_h3_lifetime lifetime;
    unsigned long long unresolved_requirement_identity;
    yvex_minimax_h3_transform transform;
    yvex_minimax_h3_sharing sharing;
    char destination_identity[65];
} yvex_minimax_h3_tensor_role;
typedef struct {
    char source_acquisition_identity[65], source_snapshot_identity[65];
    char component_manifest_identity[65], phase_dag_identity[65];
    char architecture_identity[65], role_map_identity[65];
    char target_identity[65], unresolved_requirements_identity[65];
    unsigned long long source_snapshot_key, component_count, weighted_component_count;
    unsigned long long phase_edge_count, source_file_count, shard_count, tensor_count;
    unsigned long long element_count, payload_bytes, unmapped_tensors, duplicate_mappings;
    unsigned int output_classes;
    int source_verified, architecture_admitted, roles_complete;
} yvex_minimax_h3_summary;
typedef struct {
    const float *video_mean, *video_std, *audio_mean, *audio_std, *pixel_mean, *pixel_std;
    unsigned long long video_channels, audio_channels, pixel_channels;
} yvex_minimax_h3_latent_normalization;
typedef struct yvex_minimax_h3_target yvex_minimax_h3_target;
typedef struct { const char *source_root; } yvex_minimax_h3_open_options;
typedef struct {
    int (*open)(yvex_minimax_h3_target **out,
                const yvex_minimax_h3_open_options *options,
                yvex_minimax_h3_failure *failure,
                yvex_error *err);
    void (*close)(yvex_minimax_h3_target **target);
    const yvex_minimax_h3_summary *(*summary)(const yvex_minimax_h3_target *target);
    const yvex_source_acquisition *(*acquisition)(const yvex_minimax_h3_target *target);
    const yvex_minimax_h3_architecture *(*architecture)(
        const yvex_minimax_h3_target *target);
    const yvex_minimax_h3_component *(*component_at)(
        const yvex_minimax_h3_target *target, unsigned long long index);
    const yvex_minimax_h3_tokenizer_spec *(*tokenizer_spec)(void);
    const yvex_minimax_h3_phase_edge *(*phase_edge_at)(unsigned long long index);
    const yvex_minimax_h3_tensor_role *(*role_at)(
        const yvex_minimax_h3_target *target, unsigned long long index);
    const char *(*failure_name)(yvex_minimax_h3_failure_code code);
    const char *(*role_name)(yvex_minimax_h3_role role);
    const char *(*component_name)(yvex_minimax_h3_component_id component);
    int (*components_canonical)(yvex_minimax_h3_component components[YVEX_MINIMAX_H3_COMPONENT_COUNT],
        char manifest_identity[65], yvex_minimax_h3_failure *failure, yvex_error *err);
    int (*component_graph_validate)(
        const yvex_minimax_h3_component *components, size_t component_count,
        yvex_minimax_h3_failure *failure, yvex_error *err);
    int (*phase_graph_validate)(
        const yvex_minimax_h3_phase_edge *edges, size_t edge_count,
        yvex_minimax_h3_failure *failure, yvex_error *err);
    int (*architecture_canonical)(yvex_minimax_h3_architecture *architecture,
        yvex_minimax_h3_failure *failure, yvex_error *err);
    const yvex_minimax_h3_latent_normalization *(*latent_normalization)(void);
    int (*tensor_classify)(const yvex_native_weight_info *tensor, yvex_minimax_h3_tensor_role *role,
        yvex_minimax_h3_failure *failure, yvex_error *err);
} yvex_minimax_h3_api;
typedef struct {
    int (*build)(yvex_transform_ir **out,
                 char derivation_identity[65],
                 const yvex_minimax_h3_target *target,
                 yvex_error *err);
    int (*build_component)(yvex_transform_ir **out,
                           char derivation_identity[65],
                           const yvex_minimax_h3_target *target,
                           yvex_minimax_h3_component_id component,
                           yvex_error *err);
} yvex_minimax_h3_transform_api;
typedef enum {
    YVEX_MINIMAX_H3_HANDOFF_NONE = 0,
    YVEX_MINIMAX_H3_HANDOFF_INVALID_ARGUMENT,
    YVEX_MINIMAX_H3_HANDOFF_TARGET,
    YVEX_MINIMAX_H3_HANDOFF_SNAPSHOT,
    YVEX_MINIMAX_H3_HANDOFF_PAYLOAD_SESSION,
    YVEX_MINIMAX_H3_HANDOFF_TRANSFORMATION,
    YVEX_MINIMAX_H3_HANDOFF_BINDING,
    YVEX_MINIMAX_H3_HANDOFF_PAYLOAD_PLAN,
    YVEX_MINIMAX_H3_HANDOFF_ALLOCATION
} yvex_minimax_h3_handoff_code;
typedef struct {
    yvex_minimax_h3_handoff_code code;
    yvex_minimax_h3_failure target_failure;
    yvex_source_payload_failure payload_failure;
} yvex_minimax_h3_handoff_failure;
typedef struct {
    const char *source_root;
    yvex_minimax_h3_component_id component;
    yvex_source_payload_budget budget;
    size_t chunk_bytes;
    size_t page_bytes;
} yvex_minimax_h3_handoff_options;
typedef struct {
    yvex_minimax_h3_component_id component;
    char component_identity[65];
    char source_snapshot_identity[65];
    char payload_identity[65];
    char transform_identity[65];
    char derivation_identity[65];
    unsigned long long shards, tensors, elements, payload_bytes;
    unsigned long long planned_ranges, planned_chunks, payload_execution_bytes_read;
    int complete;
} yvex_minimax_h3_handoff_summary;
typedef struct yvex_minimax_h3_handoff yvex_minimax_h3_handoff;
typedef struct {
    int (*open)(yvex_minimax_h3_handoff **out,
                const yvex_minimax_h3_handoff_options *options,
                yvex_minimax_h3_handoff_failure *failure, yvex_error *err);
    void (*close)(yvex_minimax_h3_handoff **handoff);
    const yvex_minimax_h3_handoff_summary *(*summary)(
        const yvex_minimax_h3_handoff *handoff);
    const yvex_minimax_h3_target *(*target)(const yvex_minimax_h3_handoff *handoff);
    const yvex_transform_ir *(*transform_ir)(const yvex_minimax_h3_handoff *handoff);
    const yvex_transform_binding *(*binding)(const yvex_minimax_h3_handoff *handoff);
    yvex_source_payload_session *(*session)(yvex_minimax_h3_handoff *handoff);
    const yvex_source_payload_plan *(*plan)(const yvex_minimax_h3_handoff *handoff);
} yvex_minimax_h3_handoff_api;
#define YVEX_MINIMAX_H3_COMPONENT_EXECUTION_NONE 0u
#define YVEX_MINIMAX_H3_COMPONENT_EXECUTION_INVALID_ARGUMENT 1u
#define YVEX_MINIMAX_H3_COMPONENT_EXECUTION_LIFECYCLE 2u
#define YVEX_MINIMAX_H3_COMPONENT_EXECUTION_MISSING_TENSOR 3u
#define YVEX_MINIMAX_H3_COMPONENT_EXECUTION_TENSOR_CONTRACT 4u
#define YVEX_MINIMAX_H3_COMPONENT_EXECUTION_BUDGET 5u
#define YVEX_MINIMAX_H3_COMPONENT_EXECUTION_MATERIALIZATION 6u
#define YVEX_MINIMAX_H3_COMPONENT_EXECUTION_NUMERIC 7u
#define YVEX_MINIMAX_H3_COMPONENT_EXECUTION_CANCELLED 8u
typedef unsigned int yvex_minimax_h3_component_execution_code;
typedef struct yvex_component_execution_failure yvex_minimax_h3_component_execution_failure;
typedef int (*yvex_minimax_h3_cancelled_fn)(void *context);
typedef struct yvex_runtime_av_audio_decode_options yvex_minimax_h3_audio_decode_options;
typedef struct yvex_runtime_av_audio_decode_result yvex_minimax_h3_audio_decode_result;
typedef struct yvex_runtime_av_video_decode_options yvex_minimax_h3_video_decode_options;
typedef struct yvex_runtime_av_video_decode_result yvex_minimax_h3_video_decode_result;
typedef struct yvex_runtime_av_conditioning_result yvex_minimax_h3_conditioning_result;
enum {
    YVEX_MINIMAX_H3_TEXT_MAX_TOKENS = 256u,
    YVEX_MINIMAX_H3_OMNI_MAX_PACKED_ROWS = 2048u
};
typedef enum {
    YVEX_MINIMAX_H3_TEXT_EMBEDDING = 0, YVEX_MINIMAX_H3_TEXT_INPUT_NORM,
    YVEX_MINIMAX_H3_TEXT_Q_PROJECTION, YVEX_MINIMAX_H3_TEXT_K_PROJECTION,
    YVEX_MINIMAX_H3_TEXT_V_PROJECTION, YVEX_MINIMAX_H3_TEXT_O_PROJECTION,
    YVEX_MINIMAX_H3_TEXT_Q_NORM, YVEX_MINIMAX_H3_TEXT_K_NORM,
    YVEX_MINIMAX_H3_TEXT_POST_NORM, YVEX_MINIMAX_H3_TEXT_GATE_PROJECTION,
    YVEX_MINIMAX_H3_TEXT_UP_PROJECTION, YVEX_MINIMAX_H3_TEXT_DOWN_PROJECTION,
    YVEX_MINIMAX_H3_TEXT_WEIGHT_COUNT,
    YVEX_MINIMAX_H3_TEXT_LAYER_WEIGHT_COUNT = YVEX_MINIMAX_H3_TEXT_WEIGHT_COUNT - 1,
    YVEX_MINIMAX_H3_TEXT_CONDITIONING_LAYERS = 50
} yvex_minimax_h3_text_weight_slot;
typedef enum {
    YVEX_MINIMAX_H3_OMNI_NORM1 = 0, YVEX_MINIMAX_H3_OMNI_QKV,
    YVEX_MINIMAX_H3_OMNI_Q_NORM, YVEX_MINIMAX_H3_OMNI_K_NORM,
    YVEX_MINIMAX_H3_OMNI_ATTENTION_OUT, YVEX_MINIMAX_H3_OMNI_NORM2,
    YVEX_MINIMAX_H3_OMNI_FC1, YVEX_MINIMAX_H3_OMNI_FC2, YVEX_MINIMAX_H3_OMNI_ADALN_WEIGHT,
    YVEX_MINIMAX_H3_OMNI_ADALN_BIAS, YVEX_MINIMAX_H3_OMNI_BLOCK_WEIGHT_COUNT
} yvex_minimax_h3_omni_weight_slot;
typedef enum {
    YVEX_MINIMAX_H3_OMNI_AUDIO_WEIGHT = 0, YVEX_MINIMAX_H3_OMNI_AUDIO_BIAS,
    YVEX_MINIMAX_H3_OMNI_VIDEO_WEIGHT, YVEX_MINIMAX_H3_OMNI_VIDEO_BIAS,
    YVEX_MINIMAX_H3_OMNI_CONDITION_WEIGHT, YVEX_MINIMAX_H3_OMNI_CONDITION_BIAS,
    YVEX_MINIMAX_H3_OMNI_TIME_IN_WEIGHT, YVEX_MINIMAX_H3_OMNI_TIME_IN_BIAS,
    YVEX_MINIMAX_H3_OMNI_TIME_OUT_WEIGHT, YVEX_MINIMAX_H3_OMNI_TIME_OUT_BIAS,
    YVEX_MINIMAX_H3_OMNI_REFINER_WEIGHTS,
    YVEX_MINIMAX_H3_OMNI_REFINER_FINAL = YVEX_MINIMAX_H3_OMNI_REFINER_WEIGHTS + 16,
    YVEX_MINIMAX_H3_OMNI_ROPE_INV_FREQ,
    YVEX_MINIMAX_H3_OMNI_FINAL_NORM, YVEX_MINIMAX_H3_OMNI_FINAL_ADALN_WEIGHT,
    YVEX_MINIMAX_H3_OMNI_FINAL_ADALN_BIAS, YVEX_MINIMAX_H3_OMNI_VIDEO_OUT_WEIGHT,
    YVEX_MINIMAX_H3_OMNI_VIDEO_OUT_BIAS, YVEX_MINIMAX_H3_OMNI_AUDIO_OUT_WEIGHT,
    YVEX_MINIMAX_H3_OMNI_AUDIO_OUT_BIAS, YVEX_MINIMAX_H3_OMNI_EXTERNAL_WEIGHT_COUNT
} yvex_minimax_h3_omni_external_slot;
typedef struct {
    unsigned long long packed_rows, block_count, resident_bytes, kernel_launches;
    unsigned long long h2d_bytes, d2h_bytes, device_bytes;
    char residency_identity[65], execution_identity[65];
    int complete;
} yvex_minimax_h3_omni_result;
typedef struct {
    const float *video, *audio, *conditioning, *timesteps, *position_ids;
    const unsigned int *video_indices, *audio_indices, *text_indices;
    const unsigned int *timestep_indices, *token_tags;
    unsigned long long video_rows, audio_rows, text_rows, timestep_count, packed_rows;
    unsigned long long block_count;
    float *video_output, *audio_output;
    unsigned long long video_output_capacity, audio_output_capacity;
} yvex_minimax_h3_omni_transformer_request;
typedef struct {
    unsigned long long video_rows, audio_rows, text_rows, packed_rows, block_count;
    unsigned long long resident_bytes, kernel_launches, h2d_bytes, d2h_bytes, device_bytes;
    char residency_identity[65], execution_identity[65];
    int complete;
} yvex_minimax_h3_omni_transformer_result;
typedef struct {
    int (*text_embed_cuda)(yvex_backend *backend, const unsigned char *encoded,
        unsigned long long encoded_bytes, unsigned int qtype, unsigned long long row_count,
        unsigned long long row_width, unsigned long long row_bytes,
        const char *residency_identity, unsigned long long resident_bytes,
        const unsigned int *token_ids, unsigned long long token_count, float *output,
        unsigned long long output_capacity, yvex_minimax_h3_conditioning_result *result,
        yvex_error *err);
    int (*text_layer_cuda)(yvex_backend *backend,
        const yvex_minimax_h3_encoded_weight *weights, unsigned long long layer_count,
        const char *residency_identity, unsigned long long resident_bytes,
        const unsigned int *token_ids, unsigned long long token_count, float *output,
        unsigned long long output_capacity, yvex_minimax_h3_conditioning_result *result,
        yvex_error *err);
    int (*omni_blocks_cuda)(yvex_backend *backend,
        const yvex_minimax_h3_encoded_weight *weights, unsigned long long block_count,
        const char *residency_identity, unsigned long long resident_bytes,
        const float *hidden, const float *temb, unsigned long long timestep_count,
        const float *position_ids, const unsigned int *adaln_indices, unsigned long long packed_rows,
        float *output, unsigned long long output_capacity,
        yvex_minimax_h3_omni_result *result, yvex_error *err);
    int (*omni_transformer_cuda)(yvex_backend *backend,
        const yvex_minimax_h3_encoded_weight *external_weights,
        const yvex_minimax_h3_encoded_weight *block_weights, const char *residency_identity,
        unsigned long long resident_bytes, const yvex_minimax_h3_omni_transformer_request *request,
        yvex_minimax_h3_omni_transformer_result *result, yvex_error *err);
} yvex_minimax_h3_backend_api;
typedef struct yvex_runtime_av_plan yvex_minimax_h3_t2va_plan;
typedef struct yvex_runtime_av_latent_context yvex_minimax_h3_t2va_omni_context;
typedef yvex_runtime_latent_evaluator_result yvex_minimax_h3_t2va_omni_result;
typedef struct {
    int (*t2va_plan_build)(yvex_minimax_h3_t2va_plan *, unsigned long long,
        unsigned long long, unsigned long long, unsigned long long, unsigned int, yvex_error *);
    int (*scheduler_step)(float *output, const float *sample, const float *velocity,
                          unsigned long long values, float timestep, float sigma,
                          float sigma_next, yvex_error *err);
    int (*t2va_latent_execute)(const yvex_minimax_h3_t2va_plan *,
        const yvex_minimax_h3_t2va_omni_context *, unsigned long long, unsigned long long,
        float *, unsigned long long, float *, unsigned long long, yvex_runtime_latent_result *,
        yvex_minimax_h3_t2va_omni_result *, yvex_error *);
    int (*t2va_layout_build)(const yvex_minimax_h3_t2va_plan *, const yvex_runtime_av_layout_output *,
        yvex_runtime_av_layout_result *, yvex_error *);
    int (*component_admit)(const char *component,
        const yvex_artifact *artifact, const yvex_gguf *gguf,
        const yvex_tensor_table *tensors, yvex_complete_artifact_admission *out,
        yvex_artifact_admission_failure *failure, yvex_error *err);
    int (*text_encoder_artifact_cuda)(
        const yvex_artifact *artifact, const yvex_gguf *gguf, const yvex_tensor_table *tensors,
        const unsigned int *token_ids, unsigned long long token_count, unsigned long long layer_count,
        float *output, unsigned long long output_capacity,
        unsigned long long maximum_host_bytes, unsigned long long maximum_device_bytes,
        yvex_minimax_h3_conditioning_result *result, yvex_error *err);
    int (*transformer_component_cuda)(yvex_runtime_component_session *,
        const yvex_minimax_h3_omni_transformer_request *,
        yvex_minimax_h3_omni_transformer_result *, yvex_error *);
    int (*audio_vae_decode_cpu)(yvex_materialization_session *session,
        const yvex_minimax_h3_audio_decode_options *options,
        yvex_minimax_h3_audio_decode_result *result, yvex_minimax_h3_component_execution_failure *failure,
        yvex_error *err);
    int (*audio_vae_execute_artifact_cpu)(
        const yvex_artifact *artifact, const yvex_gguf *gguf, const yvex_tensor_table *tensors,
        const yvex_minimax_h3_audio_decode_options *options,
        yvex_minimax_h3_audio_decode_result *result, yvex_minimax_h3_component_execution_failure *failure,
        yvex_error *err);
    int (*audio_vae_execute_artifact_cuda)(
        const yvex_artifact *, const yvex_gguf *, const yvex_tensor_table *,
        const yvex_minimax_h3_audio_decode_options *, unsigned long long,
        yvex_minimax_h3_audio_decode_result *, yvex_minimax_h3_component_execution_failure *,
        yvex_error *);
    int (*video_vae_decode_cpu)(yvex_materialization_session *session,
        const yvex_minimax_h3_video_decode_options *options,
        yvex_minimax_h3_video_decode_result *result, yvex_minimax_h3_component_execution_failure *failure,
        yvex_error *err);
    int (*video_vae_decode_cuda)(yvex_runtime_component_session *,
        const yvex_minimax_h3_video_decode_options *, yvex_minimax_h3_video_decode_result *,
        yvex_minimax_h3_component_execution_failure *, yvex_error *);
    int (*video_vae_execute_artifact_cpu)(
        const yvex_artifact *artifact, const yvex_gguf *gguf, const yvex_tensor_table *tensors,
        const yvex_minimax_h3_video_decode_options *options,
        yvex_minimax_h3_video_decode_result *result, yvex_minimax_h3_component_execution_failure *failure,
        yvex_error *err);
    int (*video_vae_execute_artifact_cuda)(
        const yvex_artifact *, const yvex_gguf *, const yvex_tensor_table *,
        const yvex_minimax_h3_video_decode_options *, unsigned long long,
        yvex_minimax_h3_video_decode_result *, yvex_minimax_h3_component_execution_failure *,
        yvex_error *);
} yvex_minimax_h3_graph_api;
const yvex_minimax_h3_api *yvex_model_register_minimax_h3(void);
const yvex_minimax_h3_transform_api *yvex_model_minimax_h3_transform_api(void);
const yvex_minimax_h3_handoff_api *yvex_model_minimax_h3_handoff_api(void);
const yvex_minimax_h3_graph_api *yvex_graph_register_minimax_h3(void);
const yvex_minimax_h3_backend_api *yvex_backend_register_minimax_h3(void);
#endif /* INCLUDE_YVEX_INTERNAL_FAMILIES_MINIMAX_H3_H_INCLUDED */
