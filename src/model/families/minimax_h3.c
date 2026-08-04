/*
 * Admit the exact MiniMax-H3 FL2VA source as one component-aware logical target.
 *
 * Source bytes are authenticated before headers become a snapshot. Family logic then assigns
 * architecture, component, phase, lifetime, and terminal roles without executing payloads or
 * choosing artifacts, kernels, solver steps, runtime placement, or media formats.
 */
#include <yvex/internal/families/minimax_h3.h>

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <yvex/internal/core.h>
#include <yvex/internal/source.h>

#define MINIMAX_UNKNOWN_NONE 0ull
#define MINIMAX_UNKNOWN_MROPE 1ull
#define MINIMAX_UNKNOWN_MASK 2ull
#define MINIMAX_UNKNOWN_SCHEDULER 3ull
#define MINIMAX_UNKNOWN_VIDEO_EXECUTION 4ull
#define MINIMAX_UNKNOWN_AUDIO_EXECUTION 5ull

static const char text_config_identity[] =
    "d2dd0c60d01b9e195d9447c52da61c7302d28828524914c044d9c6e1b81d0427";
static const char transformer_config_identity[] =
    "f619093a231fcfbcc3d035bec26c50ad864e7331a500d5c519f5045dc1e50458";
static const char video_config_identity[] =
    "3edd2cdd1ebc823c868be55ef917e1b3b8a398fde4d3150dae44a3bf05d9f627";
static const char video_source_config_identity[] =
    "66c68f541e6578ce613ce7a0fc985eb59097038829e49f7535e6d08e6d95ab12";
static const char audio_config_identity[] =
    "d8f3bcc62e23c7e9806970fa63cca6139c06faa3797cf9c94034f60db8512771";
static const char audio_metadata_identity[] =
    "755d0529d43b2b5c83590f6f44ca659bc68e6a21b01d5669c93e8b2965749bff";
static const char tokenizer_config_identity[] =
    "a07e942ac874baa13758de8d1fbdb186683cc03416b5589e1b6671c6b3057c68";

struct yvex_minimax_h3_target {
    char source_root[YVEX_PATH_CAP];
    yvex_source_acquisition *acquisition;
    yvex_native_weight_table *weights;
    yvex_minimax_h3_component components[YVEX_MINIMAX_H3_COMPONENT_COUNT];
    yvex_minimax_h3_architecture architecture;
    yvex_minimax_h3_tensor_role *roles;
    yvex_minimax_h3_summary summary;
};

static const char *const component_names[] = {
    "pipeline", "processor", "tokenizer", "text_encoder",
    "transformer", "video_vae", "audio_vae", "latent_controller"
};

static const char *const role_names[] = {
    "invalid", "text-embedding", "text-output-head", "text-attention-q",
    "text-attention-k", "text-attention-v", "text-attention-out", "text-qk-norm",
    "text-rms-norm", "text-mlp", "vision-patch", "vision-position",
    "vision-attention", "vision-norm", "vision-mlp", "vision-merger",
    "vision-deepstack", "condition-projection", "video-patch", "audio-patch",
    "timestep-projection", "token-refiner-attention", "token-refiner-norm",
    "token-refiner-mlp", "omni-attention", "omni-qk-norm", "omni-norm",
    "omni-mlp", "omni-adaln", "omni-position", "final-norm", "final-adaln",
    "final-video", "final-audio", "video-encoder-conv", "video-encoder-norm",
    "video-resample", "video-latent-projection", "video-decoder-projection",
    "video-decoder-token", "video-decoder-attention", "video-decoder-norm",
    "video-decoder-mlp", "video-decoder-scale", "video-output",
    "audio-encoder-conv", "audio-resample", "audio-latent-projection",
    "audio-pre-attention", "audio-pre-norm", "audio-pre-mlp", "audio-decoder-conv",
    "audio-filter", "audio-activation", "audio-output"
};

static const char *const failure_names[] = {
    "none", "invalid-argument", "source-acquisition", "source-inventory",
    "source-identity", "component-coverage", "component-cycle", "phase-order",
    "architecture", "tensor-role", "shape", "dtype", "source-range",
    "transformation", "resource-budget", "allocation"
};

static const char *family_component_name(yvex_minimax_h3_component_id component)
{
    return (unsigned int)component < YVEX_MINIMAX_H3_COMPONENT_COUNT
               ? component_names[component] : "unknown";
}

static const char *family_role_name(yvex_minimax_h3_role role)
{
    return (unsigned int)role < YVEX_MINIMAX_H3_ROLE_COUNT
               ? role_names[role] : "unknown";
}

static const char *family_failure_name(yvex_minimax_h3_failure_code code)
{
    size_t count = sizeof(failure_names) / sizeof(failure_names[0]);

    return (unsigned int)code < count ? failure_names[code] : "unknown";
}

static int family_refuse(yvex_minimax_h3_failure *failure,
                         yvex_minimax_h3_failure_code code,
                         yvex_minimax_h3_component_id component,
                         yvex_minimax_h3_role role,
                         unsigned long long tensor_index,
                         const char *source_name,
                         yvex_status status,
                         yvex_error *err,
                         const char *message)
{
    if (failure) {
        memset(failure, 0, sizeof(*failure));
        failure->code = code;
        failure->component = component;
        failure->role = role;
        failure->tensor_index = tensor_index;
        yvex_core_text_copy(failure->source_name, sizeof(failure->source_name),
                            source_name ? source_name : "");
    }
    yvex_error_setf(err, status, "model.minimax_h3", "%s: %s",
                    family_failure_name(code), message);
    return status;
}

static int identity_finish(yvex_sha256 *hash, char output[65])
{
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];

    if (!yvex_sha256_final(hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

static int identity_u32(yvex_sha256 *hash, unsigned int value)
{
    return yvex_sha256_update_u64(hash, value);
}

static unsigned long long identity_key(const char identity[65])
{
    unsigned long long value = 0u;
    unsigned int index;

    if (!yvex_sha256_hex_valid(identity)) return 0u;
    for (index = 0u; index < 16u; ++index) {
        unsigned char ch = (unsigned char)identity[index];
        unsigned long long digit = isdigit(ch) ? (unsigned long long)(ch - '0')
                                                 : (unsigned long long)(tolower(ch) - 'a' + 10);
        value = (value << 4u) | digit;
    }
    return value ? value : 1u;
}

static const yvex_source_acquisition_file *acquisition_file_find(
    const yvex_source_acquisition *acquisition,
    const char *path)
{
    const yvex_source_acquisition_facts *facts =
        yvex_source_acquisition_facts_get(acquisition);
    unsigned long long index = 0u;

    for (index = 0u; facts && index < facts->file_count; ++index) {
        const yvex_source_acquisition_file *file =
            yvex_source_acquisition_file_at(acquisition, index);
        if (file && strcmp(file->path, path) == 0) return file;
    }
    return NULL;
}

static int source_inventory_identity(const yvex_source_acquisition *acquisition,
                                     char output[65])
{
    const yvex_source_acquisition_facts *facts =
        yvex_source_acquisition_facts_get(acquisition);
    yvex_sha256 hash;
    unsigned long long index;

    if (!facts) return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.minimax-h3.source-inventory.v1") ||
        !yvex_sha256_update_text(&hash, facts->repository) ||
        !yvex_sha256_update_text(&hash, facts->revision) ||
        !yvex_sha256_update_text(&hash, facts->subtree) ||
        !yvex_sha256_update_u64(&hash, facts->file_count)) return 0;
    for (index = 0u; index < facts->file_count; ++index) {
        const yvex_source_acquisition_file *file =
            yvex_source_acquisition_file_at(acquisition, index);
        if (!file || !yvex_sha256_update_text(&hash, file->path) ||
            !yvex_sha256_update_u64(&hash, file->expected_size) ||
            !yvex_sha256_update_text(&hash, file->git_oid) ||
            !yvex_sha256_update_text(&hash, file->lfs_oid) ||
            !yvex_sha256_update_text(&hash, file->xet_hash)) return 0;
    }
    return identity_finish(&hash, output);
}

static int source_snapshot_identity(const yvex_source_acquisition_facts *acquisition,
                                    const yvex_native_weight_table *weights,
                                    char output[65])
{
    yvex_sha256 hash;
    unsigned long long index;

    if (!acquisition || !weights) return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.minimax-h3.source-snapshot.v1") ||
        !yvex_sha256_update_text(&hash, acquisition->acquisition_identity) ||
        !yvex_sha256_update_u64(&hash, yvex_native_weight_table_count(weights))) return 0;
    for (index = 0u; index < yvex_native_weight_table_count(weights); ++index) {
        const yvex_native_weight_info *tensor =
            yvex_native_weight_table_at(weights, index);
        unsigned int dimension;
        if (!tensor || !yvex_sha256_update_text(&hash, tensor->name) ||
            !yvex_sha256_update_text(&hash, tensor->shard_path) ||
            !identity_u32(&hash, (unsigned int)tensor->dtype) ||
            !identity_u32(&hash, tensor->rank)) return 0;
        for (dimension = 0u; dimension < tensor->rank; ++dimension)
            if (!yvex_sha256_update_u64(&hash, tensor->dims[dimension])) return 0;
        if (!yvex_sha256_update_u64(&hash, tensor->data_start) ||
            !yvex_sha256_update_u64(&hash, tensor->data_end)) return 0;
    }
    return identity_finish(&hash, output);
}

static int tensor_elements(const yvex_native_weight_info *tensor,
                           unsigned long long *elements)
{
    unsigned long long product = 1u;
    unsigned int dimension;

    if (!tensor || !elements || !tensor->rank ||
        tensor->rank > YVEX_NATIVE_WEIGHT_MAX_DIMS) return 0;
    for (dimension = 0u; dimension < tensor->rank; ++dimension) {
        if (!tensor->dims[dimension] ||
            !yvex_core_u64_mul(product, tensor->dims[dimension], &product)) return 0;
    }
    *elements = product;
    return 1;
}

static int source_population_validate(yvex_minimax_h3_target *target,
                                      yvex_minimax_h3_failure *failure,
                                      yvex_error *err)
{
    yvex_native_weight_summary native;
    unsigned long long bf16 = 0u;
    unsigned long long f32 = 0u;
    unsigned long long elements = 0u;
    unsigned long long index;

    if (yvex_native_weight_table_summary(target->weights, &native, err) != YVEX_OK)
        return family_refuse(failure, YVEX_MINIMAX_H3_FAILURE_SOURCE_INVENTORY,
                             YVEX_MINIMAX_H3_COMPONENT_COUNT, 0, 0u, NULL,
                             YVEX_ERR_FORMAT, err, "native source summary is unavailable");
    for (index = 0u; index < native.tensor_count; ++index) {
        const yvex_native_weight_info *tensor =
            yvex_native_weight_table_at(target->weights, index);
        unsigned long long count;
        if (!tensor_elements(tensor, &count) ||
            !yvex_core_u64_add(elements, count, &elements)) {
            return family_refuse(failure, YVEX_MINIMAX_H3_FAILURE_SHAPE,
                                 YVEX_MINIMAX_H3_COMPONENT_COUNT, 0, index,
                                 tensor ? tensor->name : NULL, YVEX_ERR_BOUNDS, err,
                                 "source tensor shape overflows element accounting");
        }
        if (tensor->dtype == YVEX_NATIVE_DTYPE_BF16) bf16++;
        else if (tensor->dtype == YVEX_NATIVE_DTYPE_F32) f32++;
        else return family_refuse(failure, YVEX_MINIMAX_H3_FAILURE_DTYPE,
                                  YVEX_MINIMAX_H3_COMPONENT_COUNT, 0, index,
                                  tensor->name, YVEX_ERR_FORMAT, err,
                                  "source tensor dtype is outside the admitted BF16/F32 set");
    }
    if (native.shard_count != YVEX_MINIMAX_H3_SHARDS ||
        native.tensor_count != YVEX_MINIMAX_H3_TENSORS ||
        native.total_tensor_bytes != YVEX_MINIMAX_H3_TENSOR_BYTES ||
        elements != YVEX_MINIMAX_H3_ELEMENTS || bf16 != 1580u || f32 != 1660u) {
        return family_refuse(failure, YVEX_MINIMAX_H3_FAILURE_SOURCE_INVENTORY,
                             YVEX_MINIMAX_H3_COMPONENT_COUNT, 0, 0u, NULL,
                             YVEX_ERR_FORMAT, err,
                             "weighted source population differs from the pinned target");
    }
    target->architecture.bf16_tensors = bf16;
    target->architecture.f32_tensors = f32;
    target->summary.shard_count = native.shard_count;
    target->summary.tensor_count = native.tensor_count;
    target->summary.element_count = elements;
    target->summary.payload_bytes = native.total_tensor_bytes;
    return YVEX_OK;
}

static int source_components_validate(yvex_minimax_h3_target *target,
                                      yvex_minimax_h3_failure *failure,
                                      yvex_error *err)
{
    static const yvex_source_component_inventory_spec specs[] = {
        {"text_encoder", "FL2VA/text_encoder/model.safetensors.index.json",
         "FL2VA/text_encoder/", 14u, 1058u},
        {"transformer", "FL2VA/transformer/model.safetensors.index.json",
         "FL2VA/transformer/", 13u, 535u},
        {"video_vae", NULL, "FL2VA/video_vae/", 1u, 560u},
        {"audio_vae", NULL, "FL2VA/audio_vae/", 1u, 1087u}
    };
    yvex_source_component_inventory_facts facts;

    if (yvex_source_component_inventory_verify(
            target->source_root, target->weights,
            specs, sizeof(specs) / sizeof(specs[0]), &facts, err) == YVEX_OK) {
        return YVEX_OK;
    }
    (void)facts;
    return family_refuse(failure, YVEX_MINIMAX_H3_FAILURE_SOURCE_INVENTORY,
                         YVEX_MINIMAX_H3_COMPONENT_COUNT, 0, 0u, NULL,
                         YVEX_ERR_FORMAT, err,
                         "component shard indexes do not reconcile with headers");
}

static int source_file_identity_validate(const yvex_minimax_h3_target *target,
                                         const char *path,
                                         const char *identity,
                                         yvex_minimax_h3_failure *failure,
                                         yvex_error *err)
{
    const yvex_source_acquisition_file *file =
        acquisition_file_find(target->acquisition, path);

    if (file && strcmp(file->actual_sha256, identity) == 0) return YVEX_OK;
    return family_refuse(failure, YVEX_MINIMAX_H3_FAILURE_SOURCE_IDENTITY,
                         YVEX_MINIMAX_H3_COMPONENT_COUNT, 0, 0u, path,
                         YVEX_ERR_FORMAT, err,
                         "critical source configuration identity differs from the pinned source");
}

static void component_set(yvex_minimax_h3_component *component,
                          yvex_minimax_h3_component_id id,
                          const char *implementation_class,
                          const char *configuration_path,
                          unsigned int dependencies,
                          unsigned int inputs,
                          unsigned int outputs,
                          yvex_minimax_h3_phase phase,
                          yvex_minimax_h3_lifetime lifetime,
                          unsigned long long files,
                          unsigned long long shards,
                          unsigned long long tensors,
                          unsigned long long elements,
                          unsigned long long bytes)
{
    memset(component, 0, sizeof(*component));
    component->id = id;
    component->canonical_id = family_component_name(id);
    component->implementation_class = implementation_class;
    component->configuration_path = configuration_path;
    component->dependency_mask = dependencies;
    component->input_classes = inputs;
    component->output_classes = outputs;
    component->phase = phase;
    component->lifetime = lifetime;
    component->file_count = files;
    component->shard_count = shards;
    component->tensor_count = tensors;
    component->element_count = elements;
    component->payload_bytes = bytes;
    component->weighted = tensors != 0u;
    component->release_after_phase = id != YVEX_MINIMAX_H3_COMPONENT_PIPELINE;
    component->request_local = id == YVEX_MINIMAX_H3_COMPONENT_LATENT_CONTROLLER;
}

static int component_identity_compute(yvex_minimax_h3_component *component,
                                      const char *configuration_identity)
{
    yvex_sha256 hash;

    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.minimax-h3.component.v1") ||
        !identity_u32(&hash, component->id) ||
        !yvex_sha256_update_text(&hash, component->canonical_id) ||
        !yvex_sha256_update_text(&hash, component->implementation_class) ||
        !yvex_sha256_update_text(&hash, component->configuration_path) ||
        !yvex_sha256_update_text(&hash, configuration_identity) ||
        !identity_u32(&hash, component->dependency_mask) ||
        !identity_u32(&hash, component->input_classes) ||
        !identity_u32(&hash, component->output_classes) ||
        !identity_u32(&hash, component->phase) ||
        !identity_u32(&hash, component->lifetime) ||
        !yvex_sha256_update_u64(&hash, component->file_count) ||
        !yvex_sha256_update_u64(&hash, component->shard_count) ||
        !yvex_sha256_update_u64(&hash, component->tensor_count) ||
        !yvex_sha256_update_u64(&hash, component->element_count) ||
        !yvex_sha256_update_u64(&hash, component->payload_bytes) ||
        !identity_u32(&hash, component->weighted) ||
        !identity_u32(&hash, component->release_after_phase) ||
        !identity_u32(&hash, component->request_local)) return 0;
    return identity_finish(&hash, component->identity);
}

static int component_visit(const yvex_minimax_h3_component *components,
                           unsigned int index,
                           unsigned char state[YVEX_MINIMAX_H3_COMPONENT_COUNT])
{
    unsigned int dependency;

    if (state[index] == 1u) return 0;
    if (state[index] == 2u) return 1;
    state[index] = 1u;
    for (dependency = 0u; dependency < YVEX_MINIMAX_H3_COMPONENT_COUNT; ++dependency) {
        if ((components[index].dependency_mask & (1u << dependency)) &&
            !component_visit(components, dependency, state)) return 0;
    }
    state[index] = 2u;
    return 1;
}

static int component_graph_validate(
    const yvex_minimax_h3_component *components, size_t component_count,
    yvex_minimax_h3_failure *failure, yvex_error *err)
{
    yvex_minimax_h3_component ordered[YVEX_MINIMAX_H3_COMPONENT_COUNT];
    unsigned char present[YVEX_MINIMAX_H3_COMPONENT_COUNT] = {0};
    unsigned char state[YVEX_MINIMAX_H3_COMPONENT_COUNT] = {0};
    size_t index;

    if (!components || component_count != YVEX_MINIMAX_H3_COMPONENT_COUNT) {
        return family_refuse(failure, YVEX_MINIMAX_H3_FAILURE_COMPONENT_COVERAGE,
                             YVEX_MINIMAX_H3_COMPONENT_COUNT, 0, component_count, NULL,
                             YVEX_ERR_FORMAT, err, "exactly eight logical components are required");
    }
    memset(ordered, 0, sizeof(ordered));
    for (index = 0u; index < component_count; ++index) {
        const yvex_minimax_h3_component *component = &components[index];
        if ((unsigned int)component->id >= YVEX_MINIMAX_H3_COMPONENT_COUNT ||
            present[component->id]) {
            return family_refuse(failure, YVEX_MINIMAX_H3_FAILURE_COMPONENT_COVERAGE,
                                 component->id, 0, index, NULL, YVEX_ERR_FORMAT, err,
                                 "component ID is absent, unknown, or duplicated");
        }
        present[component->id] = 1u;
        ordered[component->id] = *component;
    }
    for (index = 0u; index < YVEX_MINIMAX_H3_COMPONENT_COUNT; ++index) {
        const yvex_minimax_h3_component *component = &ordered[index];
        unsigned int dependency;
        if (!present[index] || !component->canonical_id ||
            strcmp(component->canonical_id, family_component_name(component->id)) != 0 ||
            component->dependency_mask & ~((1u << YVEX_MINIMAX_H3_COMPONENT_COUNT) - 1u) ||
            component->dependency_mask & (1u << component->id) ||
            !component->output_classes ||
            (component->weighted && (!component->shard_count || !component->tensor_count))) {
            return family_refuse(failure, YVEX_MINIMAX_H3_FAILURE_COMPONENT_COVERAGE,
                                 component->id, 0, index, NULL, YVEX_ERR_FORMAT, err,
                                 "required component facts are absent or inconsistent");
        }
        for (dependency = 0u; dependency < YVEX_MINIMAX_H3_COMPONENT_COUNT; ++dependency) {
            if ((component->dependency_mask & (1u << dependency)) &&
                ordered[dependency].phase > component->phase) {
                return family_refuse(failure, YVEX_MINIMAX_H3_FAILURE_PHASE_ORDER,
                                     component->id, 0, index, NULL, YVEX_ERR_FORMAT, err,
                                     "component phase precedes one of its dependencies");
            }
        }
        if (!component_visit(ordered, (unsigned int)index, state)) {
            return family_refuse(failure, YVEX_MINIMAX_H3_FAILURE_COMPONENT_CYCLE,
                                 component->id, 0, index, NULL, YVEX_ERR_FORMAT, err,
                                 "component dependency graph contains a cycle");
        }
    }
    return YVEX_OK;
}

static int components_validate(yvex_minimax_h3_target *target,
                               yvex_minimax_h3_failure *failure,
                               yvex_error *err)
{
    static const char *const configuration_identities[] = {
        YVEX_MINIMAX_H3_MODEL_INDEX_IDENTITY,
        tokenizer_config_identity, tokenizer_config_identity, text_config_identity,
        transformer_config_identity, video_config_identity, audio_config_identity,
        transformer_config_identity
    };
    yvex_sha256 hash;
    unsigned int index;

    component_set(&target->components[YVEX_MINIMAX_H3_COMPONENT_PIPELINE],
                  YVEX_MINIMAX_H3_COMPONENT_PIPELINE, "MiniMaxH3Pipeline",
                  "FL2VA/model_index.json", 0u, YVEX_MINIMAX_H3_DATA_TEXT,
                  YVEX_MINIMAX_H3_DATA_TEXT, YVEX_MINIMAX_H3_PHASE_PREPARE,
                  YVEX_MINIMAX_H3_LIFETIME_METADATA, 3u, 0u, 0u, 0u, 0u);
    component_set(&target->components[YVEX_MINIMAX_H3_COMPONENT_PROCESSOR],
                  YVEX_MINIMAX_H3_COMPONENT_PROCESSOR, "Qwen3VLProcessor",
                  "FL2VA/processor/preprocessor_config.json", 1u << 0,
                  YVEX_MINIMAX_H3_DATA_TEXT | YVEX_MINIMAX_H3_DATA_MEDIA_GRID,
                  YVEX_MINIMAX_H3_DATA_TOKEN_IDS | YVEX_MINIMAX_H3_DATA_MEDIA_GRID,
                  YVEX_MINIMAX_H3_PHASE_PREPARE, YVEX_MINIMAX_H3_LIFETIME_PHASE,
                  7u, 0u, 0u, 0u, 0u);
    component_set(&target->components[YVEX_MINIMAX_H3_COMPONENT_TOKENIZER],
                  YVEX_MINIMAX_H3_COMPONENT_TOKENIZER, "Qwen2TokenizerFast",
                  "FL2VA/tokenizer/tokenizer_config.json", 1u << 0,
                  YVEX_MINIMAX_H3_DATA_TEXT, YVEX_MINIMAX_H3_DATA_TOKEN_IDS,
                  YVEX_MINIMAX_H3_PHASE_PREPARE, YVEX_MINIMAX_H3_LIFETIME_PHASE,
                  4u, 0u, 0u, 0u, 0u);
    component_set(&target->components[YVEX_MINIMAX_H3_COMPONENT_TEXT_ENCODER],
                  YVEX_MINIMAX_H3_COMPONENT_TEXT_ENCODER,
                  "MiniMaxH3Qwen3VLHFEncoder", "FL2VA/text_encoder/config.json",
                  (1u << 1) | (1u << 2),
                  YVEX_MINIMAX_H3_DATA_TOKEN_IDS | YVEX_MINIMAX_H3_DATA_MEDIA_GRID,
                  YVEX_MINIMAX_H3_DATA_CONDITIONING, YVEX_MINIMAX_H3_PHASE_CONDITION,
                  YVEX_MINIMAX_H3_LIFETIME_PHASE, 23u, 14u, 1058u,
                  33357390064ull, 66714780128ull);
    component_set(&target->components[YVEX_MINIMAX_H3_COMPONENT_TRANSFORMER],
                  YVEX_MINIMAX_H3_COMPONENT_TRANSFORMER, "MiniMaxH3DiTModel",
                  "FL2VA/transformer/config.json", (1u << 3) | (1u << 7),
                  YVEX_MINIMAX_H3_DATA_CONDITIONING |
                      YVEX_MINIMAX_H3_DATA_VIDEO_LATENT |
                      YVEX_MINIMAX_H3_DATA_AUDIO_LATENT,
                  YVEX_MINIMAX_H3_DATA_VIDEO_LATENT |
                      YVEX_MINIMAX_H3_DATA_AUDIO_LATENT,
                  YVEX_MINIMAX_H3_PHASE_LATENT_ITERATE,
                  YVEX_MINIMAX_H3_LIFETIME_PHASE, 15u, 13u, 535u,
                  33122992912ull, 66280430144ull);
    component_set(&target->components[YVEX_MINIMAX_H3_COMPONENT_VIDEO_VAE],
                  YVEX_MINIMAX_H3_COMPONENT_VIDEO_VAE, "MiniMaxH3VideoVAE",
                  "FL2VA/video_vae/config.json", 1u << 4,
                  YVEX_MINIMAX_H3_DATA_VIDEO_LATENT, YVEX_MINIMAX_H3_DATA_RGB_FRAMES,
                  YVEX_MINIMAX_H3_PHASE_VIDEO_DECODE, YVEX_MINIMAX_H3_LIFETIME_PHASE,
                  18u, 1u, 560u, 2603871032ull, 10415484128ull);
    component_set(&target->components[YVEX_MINIMAX_H3_COMPONENT_AUDIO_VAE],
                  YVEX_MINIMAX_H3_COMPONENT_AUDIO_VAE, "MiniMaxH3AudioVAE",
                  "FL2VA/audio_vae/config.json", 1u << 4,
                  YVEX_MINIMAX_H3_DATA_AUDIO_LATENT,
                  YVEX_MINIMAX_H3_DATA_STEREO_SAMPLES,
                  YVEX_MINIMAX_H3_PHASE_AUDIO_DECODE, YVEX_MINIMAX_H3_LIFETIME_PHASE,
                  13u, 1u, 1087u, 151326585ull, 605306340ull);
    component_set(&target->components[YVEX_MINIMAX_H3_COMPONENT_LATENT_CONTROLLER],
                  YVEX_MINIMAX_H3_COMPONENT_LATENT_CONTROLLER, "source-declared-null-scheduler",
                  "FL2VA/model_index.json", 1u << 3, YVEX_MINIMAX_H3_DATA_CONDITIONING,
                  YVEX_MINIMAX_H3_DATA_VIDEO_LATENT | YVEX_MINIMAX_H3_DATA_AUDIO_LATENT,
                  YVEX_MINIMAX_H3_PHASE_LATENT_INITIALIZE,
                  YVEX_MINIMAX_H3_LIFETIME_REQUEST_MUTABLE, 0u, 0u, 0u, 0u, 0u);

    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.minimax-h3.component-manifest.v1"))
        goto identity_failure;
    for (index = 0u; index < YVEX_MINIMAX_H3_COMPONENT_COUNT; ++index) {
        yvex_minimax_h3_component *component = &target->components[index];
        if (component->id != (yvex_minimax_h3_component_id)index ||
            !component->canonical_id || !component->canonical_id[0] ||
            (component->weighted && (!component->shard_count || !component->tensor_count)) ||
            !component->output_classes ||
            !component_identity_compute(component, configuration_identities[index])) {
            return family_refuse(failure, YVEX_MINIMAX_H3_FAILURE_COMPONENT_COVERAGE,
                                 component->id, 0, 0u, NULL, YVEX_ERR_FORMAT, err,
                                 "component declaration is incomplete");
        }
        if (!yvex_sha256_update_text(&hash, component->identity)) goto identity_failure;
    }
    if (component_graph_validate(
            target->components, YVEX_MINIMAX_H3_COMPONENT_COUNT,
            failure, err) != YVEX_OK) return yvex_error_code(err);
    if (!identity_finish(&hash, target->summary.component_manifest_identity))
        goto identity_failure;
    target->summary.component_count = YVEX_MINIMAX_H3_LOGICAL_COMPONENTS;
    target->summary.weighted_component_count = YVEX_MINIMAX_H3_WEIGHTED_COMPONENTS;
    target->summary.output_classes = YVEX_MINIMAX_H3_DATA_SYNCHRONIZED_MEDIA;
    return YVEX_OK;

identity_failure:
    return family_refuse(failure, YVEX_MINIMAX_H3_FAILURE_COMPONENT_COVERAGE,
                         YVEX_MINIMAX_H3_COMPONENT_COUNT, 0, 0u, NULL,
                         YVEX_ERR_STATE, err, "component identity construction failed");
}

static int components_canonical(
    yvex_minimax_h3_component components[YVEX_MINIMAX_H3_COMPONENT_COUNT],
    char manifest_identity[65], yvex_minimax_h3_failure *failure,
    yvex_error *err)
{
    yvex_minimax_h3_target target;
    int rc;

    if (!components || !manifest_identity) {
        return family_refuse(failure, YVEX_MINIMAX_H3_FAILURE_INVALID_ARGUMENT,
                             YVEX_MINIMAX_H3_COMPONENT_COUNT, 0, 0u, NULL,
                             YVEX_ERR_INVALID_ARG, err, "component outputs are required");
    }
    memset(&target, 0, sizeof(target));
    rc = components_validate(&target, failure, err);
    if (rc != YVEX_OK) return rc;
    memcpy(components, target.components, sizeof(target.components));
    yvex_core_text_copy(manifest_identity, 65u,
                        target.summary.component_manifest_identity);
    return YVEX_OK;
}

static int architecture_build(yvex_minimax_h3_target *target,
                              yvex_minimax_h3_failure *failure,
                              yvex_error *err)
{
    yvex_minimax_h3_architecture *architecture = &target->architecture;
    yvex_sha256 hash;
    const unsigned long long facts[] = {
        64u, 5120u, 25600u, 64u, 8u, 128u, 151936u, 5000000u,
        24u, 20u, 20u, 27u, 1152u, 4304u, 16u, 2u, 16u, 16u, 2u,
        8u, 16u, 24u, 5120u, 50u, 5376u, 14336u, 56u, 128u, 2u,
        5120u, 24u, 32u, 1u, 2u, 2u, 256u, 5376u, 2688u, 96768u,
        10752u, 96u, 32u, 24u, 3u, 16u, 4u, 36u, 32u, 256u, 64u,
        17u, 3u, 32u, 2u, 32000u, 800u, 800u, 40u, 1580u, 1660u
    };
    size_t index;

    memset(architecture, 0, sizeof(*architecture));
    architecture->encoder = (yvex_minimax_h3_encoder_signature){
        64u, 5120u, 25600u, 64u, 8u, 128u, 151936u, 5000000u,
        {24u, 20u, 20u}, 27u, 1152u, 4304u, 16u, {2u, 16u, 16u},
        2u, {8u, 16u, 24u}, 5120u, 0
    };
    architecture->omni = (yvex_minimax_h3_omni_signature){
        50u, 5376u, 14336u, 56u, 128u, 2u, 5120u, 24u, 32u,
        {1u, 2u, 2u}, 256u, 5376u, 2688u, 96768u, 10752u, 96u, 32u, 1
    };
    architecture->video_vae = (yvex_minimax_h3_video_vae_signature){
        24u, 3u, 16u, 4u, 36u, 32u, 256u, 64u, 17u, 3u, 1, 0
    };
    architecture->audio_vae = (yvex_minimax_h3_audio_vae_signature){
        32u, 2u, 32000u, 800u, 800u, 40u
    };
    architecture->bf16_tensors = 1580u;
    architecture->f32_tensors = 1660u;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.minimax-h3.architecture.v1") ||
        !yvex_sha256_update_text(&hash, text_config_identity) ||
        !yvex_sha256_update_text(&hash, transformer_config_identity) ||
        !yvex_sha256_update_text(&hash, video_config_identity) ||
        !yvex_sha256_update_text(&hash, video_source_config_identity) ||
        !yvex_sha256_update_text(&hash, audio_config_identity) ||
        !yvex_sha256_update_text(&hash, audio_metadata_identity)) goto fail;
    for (index = 0u; index < sizeof(facts) / sizeof(facts[0]); ++index)
        if (!yvex_sha256_update_u64(&hash, facts[index])) goto fail;
    if (!identity_finish(&hash, architecture->identity)) goto fail;
    yvex_core_text_copy(target->summary.architecture_identity,
                        sizeof(target->summary.architecture_identity),
                        architecture->identity);
    target->summary.architecture_admitted = 1;
    return YVEX_OK;

fail:
    return family_refuse(failure, YVEX_MINIMAX_H3_FAILURE_ARCHITECTURE,
                         YVEX_MINIMAX_H3_COMPONENT_COUNT, 0, 0u, NULL,
                         YVEX_ERR_STATE, err, "architecture identity construction failed");
}

static int architecture_canonical(
    yvex_minimax_h3_architecture *architecture,
    yvex_minimax_h3_failure *failure, yvex_error *err)
{
    yvex_minimax_h3_target target;
    int rc;

    if (!architecture) {
        return family_refuse(failure, YVEX_MINIMAX_H3_FAILURE_INVALID_ARGUMENT,
                             YVEX_MINIMAX_H3_COMPONENT_COUNT, 0, 0u, NULL,
                             YVEX_ERR_INVALID_ARG, err, "architecture output is required");
    }
    memset(&target, 0, sizeof(target));
    rc = architecture_build(&target, failure, err);
    if (rc == YVEX_OK) *architecture = target.architecture;
    return rc;
}

static int unresolved_requirements_build(yvex_minimax_h3_target *target,
                                         yvex_minimax_h3_failure *failure,
                                         yvex_error *err)
{
    static const char *const requirements[] = {
        "omni-mm-rope-numerical-contract", "attention-masks",
        "conditioning-placement", "latent-rng", "solver", "timestep-schedule",
        "iteration-count", "latent-update-equation", "guidance",
        "output-geometry-defaults", "media-codec-container",
        "synchronization-timestamps", "adaln-precomputation-validity",
        "quantization-or-f32-narrowing-validity"
    };
    yvex_sha256 hash;
    size_t index;

    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.minimax-h3.unresolved.v1") ||
        !yvex_sha256_update_u64(&hash, sizeof(requirements) / sizeof(requirements[0])))
        goto fail;
    for (index = 0u; index < sizeof(requirements) / sizeof(requirements[0]); ++index)
        if (!yvex_sha256_update_text(&hash, requirements[index])) goto fail;
    if (identity_finish(&hash, target->summary.unresolved_requirements_identity))
        return YVEX_OK;
fail:
    return family_refuse(failure, YVEX_MINIMAX_H3_FAILURE_ARCHITECTURE,
                         YVEX_MINIMAX_H3_COMPONENT_COUNT, 0, 0u, NULL,
                         YVEX_ERR_STATE, err, "unresolved requirement identity construction failed");
}

static int text_starts_with(const char *text, const char *prefix)
{
    return text && prefix && strncmp(text, prefix, strlen(prefix)) == 0;
}

static unsigned long long coordinate_after(const char *name, const char *marker)
{
    const char *position = strstr(name, marker);
    char *end = NULL;
    unsigned long long value;

    if (!position) return YVEX_MINIMAX_H3_NO_COORDINATE;
    position += strlen(marker);
    if (!isdigit((unsigned char)*position)) return YVEX_MINIMAX_H3_NO_COORDINATE;
    value = strtoull(position, &end, 10);
    return end != position && (*end == '.' || *end == '\0')
               ? value : YVEX_MINIMAX_H3_NO_COORDINATE;
}

static yvex_minimax_h3_component_id tensor_component(const char *shard)
{
    if (text_starts_with(shard, "FL2VA/text_encoder/"))
        return YVEX_MINIMAX_H3_COMPONENT_TEXT_ENCODER;
    if (text_starts_with(shard, "FL2VA/transformer/"))
        return YVEX_MINIMAX_H3_COMPONENT_TRANSFORMER;
    if (text_starts_with(shard, "FL2VA/video_vae/"))
        return YVEX_MINIMAX_H3_COMPONENT_VIDEO_VAE;
    if (text_starts_with(shard, "FL2VA/audio_vae/"))
        return YVEX_MINIMAX_H3_COMPONENT_AUDIO_VAE;
    return YVEX_MINIMAX_H3_COMPONENT_COUNT;
}

static yvex_minimax_h3_role text_encoder_role(const char *name)
{
    if (strcmp(name, "model.language_model.embed_tokens.weight") == 0)
        return YVEX_MINIMAX_H3_ROLE_TEXT_EMBEDDING;
    if (strcmp(name, "lm_head.weight") == 0)
        return YVEX_MINIMAX_H3_ROLE_TEXT_OUTPUT_HEAD;
    if (strstr(name, "model.language_model.layers.")) {
        if (strstr(name, ".self_attn.q_proj.")) return YVEX_MINIMAX_H3_ROLE_TEXT_ATTENTION_Q;
        if (strstr(name, ".self_attn.k_proj.")) return YVEX_MINIMAX_H3_ROLE_TEXT_ATTENTION_K;
        if (strstr(name, ".self_attn.v_proj.")) return YVEX_MINIMAX_H3_ROLE_TEXT_ATTENTION_V;
        if (strstr(name, ".self_attn.o_proj.")) return YVEX_MINIMAX_H3_ROLE_TEXT_ATTENTION_OUT;
        if (strstr(name, ".self_attn.q_norm.") || strstr(name, ".self_attn.k_norm."))
            return YVEX_MINIMAX_H3_ROLE_TEXT_QK_NORM;
        if (strstr(name, ".mlp.")) return YVEX_MINIMAX_H3_ROLE_TEXT_MLP;
        if (strstr(name, "layernorm")) return YVEX_MINIMAX_H3_ROLE_TEXT_RMS_NORM;
    }
    if (strcmp(name, "model.language_model.norm.weight") == 0)
        return YVEX_MINIMAX_H3_ROLE_TEXT_RMS_NORM;
    if (strstr(name, "model.visual.patch_embed.")) return YVEX_MINIMAX_H3_ROLE_VISION_PATCH;
    if (strstr(name, "model.visual.pos_embed.")) return YVEX_MINIMAX_H3_ROLE_VISION_POSITION;
    if (strstr(name, "model.visual.blocks.")) {
        if (strstr(name, ".attn.")) return YVEX_MINIMAX_H3_ROLE_VISION_ATTENTION;
        if (strstr(name, ".norm")) return YVEX_MINIMAX_H3_ROLE_VISION_NORM;
        if (strstr(name, ".mlp.")) return YVEX_MINIMAX_H3_ROLE_VISION_MLP;
    }
    if (strstr(name, "model.visual.deepstack_merger_list."))
        return YVEX_MINIMAX_H3_ROLE_VISION_DEEPSTACK;
    if (strstr(name, "model.visual.merger.")) return YVEX_MINIMAX_H3_ROLE_VISION_MERGER;
    return 0;
}

static yvex_minimax_h3_role transformer_role(const char *name)
{
    if (text_starts_with(name, "video_patch_proj.")) return YVEX_MINIMAX_H3_ROLE_VIDEO_PATCH;
    if (text_starts_with(name, "audio_patch_proj.")) return YVEX_MINIMAX_H3_ROLE_AUDIO_PATCH;
    if (text_starts_with(name, "condition_proj.")) return YVEX_MINIMAX_H3_ROLE_CONDITION_PROJECTION;
    if (text_starts_with(name, "time_embedder.")) return YVEX_MINIMAX_H3_ROLE_TIMESTEP_PROJECTION;
    if (text_starts_with(name, "rope.")) return YVEX_MINIMAX_H3_ROLE_OMNI_POSITION;
    if (text_starts_with(name, "token_refiner.")) {
        if (strstr(name, ".attn.q_norm.") || strstr(name, ".attn.k_norm.") ||
            strstr(name, ".norm") || strstr(name, "final_norm."))
            return YVEX_MINIMAX_H3_ROLE_TOKEN_REFINER_NORM;
        if (strstr(name, ".attn.")) return YVEX_MINIMAX_H3_ROLE_TOKEN_REFINER_ATTENTION;
        if (strstr(name, ".mlp.")) return YVEX_MINIMAX_H3_ROLE_TOKEN_REFINER_MLP;
    }
    if (text_starts_with(name, "blocks.")) {
        if (strstr(name, ".adaln_proj.")) return YVEX_MINIMAX_H3_ROLE_OMNI_ADALN;
        if (strstr(name, ".attn.q_norm.") || strstr(name, ".attn.k_norm."))
            return YVEX_MINIMAX_H3_ROLE_OMNI_QK_NORM;
        if (strstr(name, ".attn.")) return YVEX_MINIMAX_H3_ROLE_OMNI_ATTENTION;
        if (strstr(name, ".norm")) return YVEX_MINIMAX_H3_ROLE_OMNI_NORM;
        if (strstr(name, ".mlp.")) return YVEX_MINIMAX_H3_ROLE_OMNI_MLP;
    }
    if (text_starts_with(name, "final_layer.adaln_proj.")) return YVEX_MINIMAX_H3_ROLE_FINAL_ADALN;
    if (text_starts_with(name, "final_layer.norm.")) return YVEX_MINIMAX_H3_ROLE_FINAL_NORM;
    if (text_starts_with(name, "final_layer.video_out.")) return YVEX_MINIMAX_H3_ROLE_FINAL_VIDEO;
    if (text_starts_with(name, "final_layer.audio_out.")) return YVEX_MINIMAX_H3_ROLE_FINAL_AUDIO;
    return 0;
}

static yvex_minimax_h3_role video_vae_role(const char *name)
{
    if (text_starts_with(name, "encoder.")) {
        if (strstr(name, ".norm")) return YVEX_MINIMAX_H3_ROLE_VIDEO_ENCODER_NORM;
        if (strstr(name, ".downsample.")) return YVEX_MINIMAX_H3_ROLE_VIDEO_RESAMPLE;
        return YVEX_MINIMAX_H3_ROLE_VIDEO_ENCODER_CONV;
    }
    if (text_starts_with(name, "quant_conv.")) return YVEX_MINIMAX_H3_ROLE_VIDEO_LATENT_PROJECTION;
    if (text_starts_with(name, "post_quant_conv."))
        return YVEX_MINIMAX_H3_ROLE_VIDEO_DECODER_PROJECTION;
    if (text_starts_with(name, "decoder.transformer_blocks.")) {
        if (strstr(name, ".attn.")) return YVEX_MINIMAX_H3_ROLE_VIDEO_DECODER_ATTENTION;
        if (strstr(name, ".norm")) return YVEX_MINIMAX_H3_ROLE_VIDEO_DECODER_NORM;
        if (strstr(name, ".ff.")) return YVEX_MINIMAX_H3_ROLE_VIDEO_DECODER_MLP;
        if (strstr(name, ".scale")) return YVEX_MINIMAX_H3_ROLE_VIDEO_DECODER_SCALE;
    }
    if (strcmp(name, "decoder.mask_token") == 0 || strcmp(name, "decoder.register_tokens") == 0)
        return YVEX_MINIMAX_H3_ROLE_VIDEO_DECODER_TOKEN;
    if (text_starts_with(name, "decoder.x_embedder."))
        return YVEX_MINIMAX_H3_ROLE_VIDEO_DECODER_PROJECTION;
    if (text_starts_with(name, "decoder.norm_out.")) return YVEX_MINIMAX_H3_ROLE_VIDEO_DECODER_NORM;
    if (text_starts_with(name, "decoder.proj_out.")) return YVEX_MINIMAX_H3_ROLE_VIDEO_OUTPUT;
    return 0;
}

static yvex_minimax_h3_role audio_vae_role(const char *name)
{
    if (text_starts_with(name, "encoder.")) return YVEX_MINIMAX_H3_ROLE_AUDIO_ENCODER_CONV;
    if (text_starts_with(name, "mean_proj.") || text_starts_with(name, "logs_proj."))
        return YVEX_MINIMAX_H3_ROLE_AUDIO_LATENT_PROJECTION;
    if (text_starts_with(name, "dec_in_proj.")) return YVEX_MINIMAX_H3_ROLE_AUDIO_LATENT_PROJECTION;
    if (text_starts_with(name, "pre_block.attn.")) return YVEX_MINIMAX_H3_ROLE_AUDIO_PRE_ATTENTION;
    if (text_starts_with(name, "pre_block.norm") || strstr(name, "pre_block.mlp.norm."))
        return YVEX_MINIMAX_H3_ROLE_AUDIO_PRE_NORM;
    if (text_starts_with(name, "pre_block.mlp.")) return YVEX_MINIMAX_H3_ROLE_AUDIO_PRE_MLP;
    if (text_starts_with(name, "pre_block.proj.")) return YVEX_MINIMAX_H3_ROLE_AUDIO_PRE_ATTENTION;
    if (strstr(name, ".filter")) return YVEX_MINIMAX_H3_ROLE_AUDIO_FILTER;
    if (strstr(name, ".activation") || strstr(name, ".activations."))
        return YVEX_MINIMAX_H3_ROLE_AUDIO_ACTIVATION;
    if (text_starts_with(name, "decoder.ups.")) return YVEX_MINIMAX_H3_ROLE_AUDIO_RESAMPLE;
    if (text_starts_with(name, "decoder.conv_post.")) return YVEX_MINIMAX_H3_ROLE_AUDIO_OUTPUT;
    if (text_starts_with(name, "decoder.")) return YVEX_MINIMAX_H3_ROLE_AUDIO_DECODER_CONV;
    return 0;
}

static unsigned long long role_unknown(yvex_minimax_h3_component_id component,
                                       yvex_minimax_h3_role role)
{
    if (component == YVEX_MINIMAX_H3_COMPONENT_TRANSFORMER) {
        if (role == YVEX_MINIMAX_H3_ROLE_OMNI_POSITION) return MINIMAX_UNKNOWN_MROPE;
        if (role == YVEX_MINIMAX_H3_ROLE_TIMESTEP_PROJECTION) return MINIMAX_UNKNOWN_SCHEDULER;
        if (role == YVEX_MINIMAX_H3_ROLE_OMNI_ATTENTION ||
            role == YVEX_MINIMAX_H3_ROLE_TOKEN_REFINER_ATTENTION) return MINIMAX_UNKNOWN_MASK;
    }
    if (component == YVEX_MINIMAX_H3_COMPONENT_VIDEO_VAE)
        return MINIMAX_UNKNOWN_VIDEO_EXECUTION;
    if (component == YVEX_MINIMAX_H3_COMPONENT_AUDIO_VAE)
        return MINIMAX_UNKNOWN_AUDIO_EXECUTION;
    return MINIMAX_UNKNOWN_NONE;
}

static int tensor_shape_is(const yvex_native_weight_info *tensor,
                           unsigned long long first,
                           unsigned long long second)
{
    return tensor->rank == 2u && tensor->dims[0] == first && tensor->dims[1] == second;
}

static int tensor_name_ambiguous(yvex_minimax_h3_component_id component,
                                 const char *name)
{
    if (component == YVEX_MINIMAX_H3_COMPONENT_TEXT_ENCODER)
        return (strstr(name, ".self_attn.") && strstr(name, ".mlp.")) ||
               (strstr(name, "model.visual.blocks.") && strstr(name, ".attn.") &&
                strstr(name, ".mlp."));
    if (component == YVEX_MINIMAX_H3_COMPONENT_TRANSFORMER)
        return strstr(name, ".attn.") && strstr(name, ".mlp.");
    if (component == YVEX_MINIMAX_H3_COMPONENT_VIDEO_VAE)
        return strstr(name, ".attn.") && strstr(name, ".ff.");
    return 0;
}

static int tensor_role_classify(const yvex_native_weight_info *tensor,
                                yvex_minimax_h3_tensor_role *role,
                                yvex_minimax_h3_failure *failure,
                                yvex_error *err)
{
    yvex_minimax_h3_component_id component;
    yvex_minimax_h3_role classified = 0;
    yvex_sha256 hash;
    unsigned int dimension;

    if (!tensor || !role) {
        return family_refuse(failure, YVEX_MINIMAX_H3_FAILURE_INVALID_ARGUMENT,
                             YVEX_MINIMAX_H3_COMPONENT_COUNT, 0, 0u, NULL,
                             YVEX_ERR_INVALID_ARG, err, "tensor role inputs are required");
    }
    component = tensor_component(tensor->shard_path);
    if (component == YVEX_MINIMAX_H3_COMPONENT_COUNT || !tensor->name ||
        !tensor->name[0] || tensor->rank == 0u ||
        tensor->rank > YVEX_NATIVE_WEIGHT_MAX_DIMS ||
        tensor->data_end <= tensor->data_start ||
        tensor->data_bytes != tensor->data_end - tensor->data_start) {
        return family_refuse(failure, YVEX_MINIMAX_H3_FAILURE_SOURCE_RANGE,
                             component, 0, 0u, tensor->name,
                             YVEX_ERR_FORMAT, err, "tensor source range or owner is malformed");
    }
    if ((component == YVEX_MINIMAX_H3_COMPONENT_TEXT_ENCODER &&
         tensor->dtype != YVEX_NATIVE_DTYPE_BF16) ||
        ((component == YVEX_MINIMAX_H3_COMPONENT_VIDEO_VAE ||
          component == YVEX_MINIMAX_H3_COMPONENT_AUDIO_VAE) &&
         tensor->dtype != YVEX_NATIVE_DTYPE_F32) ||
        (component == YVEX_MINIMAX_H3_COMPONENT_TRANSFORMER &&
         tensor->dtype != YVEX_NATIVE_DTYPE_BF16 &&
         tensor->dtype != YVEX_NATIVE_DTYPE_F32)) {
        return family_refuse(failure, YVEX_MINIMAX_H3_FAILURE_DTYPE,
                             component, 0, 0u, tensor->name,
                             YVEX_ERR_FORMAT, err, "tensor dtype contradicts its component source");
    }
    if (tensor_name_ambiguous(component, tensor->name)) {
        return family_refuse(failure, YVEX_MINIMAX_H3_FAILURE_TENSOR_ROLE,
                             component, 0, 0u, tensor->name,
                             YVEX_ERR_FORMAT, err, "tensor name admits multiple canonical roles");
    }
    if (component == YVEX_MINIMAX_H3_COMPONENT_TEXT_ENCODER)
        classified = text_encoder_role(tensor->name);
    else if (component == YVEX_MINIMAX_H3_COMPONENT_TRANSFORMER)
        classified = transformer_role(tensor->name);
    else if (component == YVEX_MINIMAX_H3_COMPONENT_VIDEO_VAE)
        classified = video_vae_role(tensor->name);
    else if (component == YVEX_MINIMAX_H3_COMPONENT_AUDIO_VAE)
        classified = audio_vae_role(tensor->name);
    if (!classified) {
        return family_refuse(failure, YVEX_MINIMAX_H3_FAILURE_TENSOR_ROLE,
                             component, 0, 0u, tensor->name, YVEX_ERR_FORMAT, err,
                             "source tensor has no exact canonical family role");
    }
    if (((classified == YVEX_MINIMAX_H3_ROLE_TEXT_EMBEDDING ||
          classified == YVEX_MINIMAX_H3_ROLE_TEXT_OUTPUT_HEAD) &&
         !tensor_shape_is(tensor, 151936u, 5120u)) ||
        (classified == YVEX_MINIMAX_H3_ROLE_OMNI_NORM &&
         strstr(tensor->name, "blocks.") &&
         !(tensor->rank == 1u && tensor->dims[0] == 5376u))) {
        return family_refuse(failure, YVEX_MINIMAX_H3_FAILURE_SHAPE,
                             component, classified, 0u, tensor->name,
                             YVEX_ERR_FORMAT, err, "tensor shape contradicts its canonical role");
    }
    memset(role, 0, sizeof(*role));
    role->component = component;
    role->role = classified;
    role->source_name = tensor->name;
    role->shard_name = tensor->shard_path;
    role->source_dtype = tensor->dtype;
    role->rank = tensor->rank;
    role->relative_begin = tensor->data_start;
    role->relative_end = tensor->data_end;
    role->sharing = YVEX_MINIMAX_H3_SHARING_INDEPENDENT;
    role->transform = YVEX_MINIMAX_H3_TRANSFORM_IDENTITY;
    if (!tensor_elements(tensor, &role->elements)) {
        return family_refuse(failure, YVEX_MINIMAX_H3_FAILURE_SHAPE,
                             component, classified, 0u, tensor->name,
                             YVEX_ERR_BOUNDS, err, "tensor shape is not an admitted logical shape");
    }
    for (dimension = 0u; dimension < tensor->rank; ++dimension) {
        role->source_shape[dimension] = tensor->dims[dimension];
        role->logical_shape[dimension] = tensor->dims[dimension];
    }
    if (component == YVEX_MINIMAX_H3_COMPONENT_TEXT_ENCODER) {
        role->phase = YVEX_MINIMAX_H3_PHASE_CONDITION;
        role->layer_index = coordinate_after(tensor->name, ".layers.");
        if (role->layer_index == YVEX_MINIMAX_H3_NO_COORDINATE)
            role->layer_index = coordinate_after(tensor->name, ".blocks.");
        role->subcomponent_index = coordinate_after(tensor->name, "deepstack_merger_list.");
    } else if (component == YVEX_MINIMAX_H3_COMPONENT_TRANSFORMER) {
        role->phase = YVEX_MINIMAX_H3_PHASE_LATENT_ITERATE;
        role->layer_index = coordinate_after(tensor->name, "blocks.");
        role->subcomponent_index = text_starts_with(tensor->name, "token_refiner.") ? 1u : 0u;
    } else if (component == YVEX_MINIMAX_H3_COMPONENT_VIDEO_VAE) {
        role->phase = YVEX_MINIMAX_H3_PHASE_VIDEO_DECODE;
        role->layer_index = coordinate_after(tensor->name, "transformer_blocks.");
        role->subcomponent_index = coordinate_after(tensor->name, "encoder.down.");
    } else {
        role->phase = YVEX_MINIMAX_H3_PHASE_AUDIO_DECODE;
        role->layer_index = coordinate_after(tensor->name, "decoder.resblocks.");
        role->subcomponent_index = coordinate_after(tensor->name, "encoder.block.");
    }
    role->repeated_index = role->layer_index;
    role->lifetime = YVEX_MINIMAX_H3_LIFETIME_PHASE;
    role->unresolved_requirement_identity = role_unknown(component, classified);
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.minimax-h3.logical-tensor.v1") ||
        !identity_u32(&hash, component) || !identity_u32(&hash, classified) ||
        !yvex_sha256_update_text(&hash, tensor->name) ||
        !yvex_sha256_update_text(&hash, tensor->shard_path) ||
        !identity_finish(&hash, role->destination_identity)) {
        return family_refuse(failure, YVEX_MINIMAX_H3_FAILURE_TENSOR_ROLE,
                             component, classified, 0u, tensor->name,
                             YVEX_ERR_STATE, err, "logical tensor identity construction failed");
    }
    return YVEX_OK;
}

static int tensor_classify(
    const yvex_native_weight_info *tensor, yvex_minimax_h3_tensor_role *role,
    yvex_minimax_h3_failure *failure, yvex_error *err)
{
    return tensor_role_classify(tensor, role, failure, err);
}

static int role_map_build(yvex_minimax_h3_target *target,
                          yvex_minimax_h3_failure *failure,
                          yvex_error *err)
{
    unsigned long long count = yvex_native_weight_table_count(target->weights);
    yvex_sha256 hash;
    unsigned long long index = 0u;

    if (count > SIZE_MAX / sizeof(*target->roles)) {
        return family_refuse(failure, YVEX_MINIMAX_H3_FAILURE_RESOURCE_BUDGET,
                             YVEX_MINIMAX_H3_COMPONENT_COUNT, 0, count, NULL,
                             YVEX_ERR_BOUNDS, err, "tensor role allocation exceeds addressable memory");
    }
    target->roles = (yvex_minimax_h3_tensor_role *)calloc((size_t)count,
                                                           sizeof(*target->roles));
    if (!target->roles) {
        return family_refuse(failure, YVEX_MINIMAX_H3_FAILURE_ALLOCATION,
                             YVEX_MINIMAX_H3_COMPONENT_COUNT, 0, count, NULL,
                             YVEX_ERR_NOMEM, err, "tensor role allocation failed");
    }
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.minimax-h3.tensor-role-map.v1") ||
        !yvex_sha256_update_text(&hash, target->summary.source_snapshot_identity) ||
        !yvex_sha256_update_u64(&hash, count)) goto fail;
    for (index = 0u; index < count; ++index) {
        const yvex_native_weight_info *tensor =
            yvex_native_weight_table_at(target->weights, index);
        yvex_minimax_h3_tensor_role *role = &target->roles[index];
        unsigned int dimension;
        if (tensor_classify(tensor, role, failure, err) != YVEX_OK) {
            if (failure) failure->tensor_index = index;
            target->summary.unmapped_tensors++;
            return yvex_error_code(err);
        }
        if (!identity_u32(&hash, role->component) || !identity_u32(&hash, role->role) ||
            !yvex_sha256_update_text(&hash, role->source_name) ||
            !yvex_sha256_update_text(&hash, role->shard_name) ||
            !identity_u32(&hash, role->source_dtype) || !identity_u32(&hash, role->rank))
            goto fail;
        for (dimension = 0u; dimension < role->rank; ++dimension)
            if (!yvex_sha256_update_u64(&hash, role->source_shape[dimension]) ||
                !yvex_sha256_update_u64(&hash, role->logical_shape[dimension])) goto fail;
        if (!yvex_sha256_update_u64(&hash, role->relative_begin) ||
            !yvex_sha256_update_u64(&hash, role->relative_end) ||
            !yvex_sha256_update_u64(&hash, role->elements) ||
            !yvex_sha256_update_u64(&hash, role->layer_index) ||
            !yvex_sha256_update_u64(&hash, role->subcomponent_index) ||
            !yvex_sha256_update_u64(&hash, role->repeated_index) ||
            !identity_u32(&hash, role->phase) || !identity_u32(&hash, role->lifetime) ||
            !yvex_sha256_update_u64(&hash, role->unresolved_requirement_identity) ||
            !identity_u32(&hash, role->transform) || !identity_u32(&hash, role->sharing) ||
            !yvex_sha256_update_text(&hash, role->destination_identity)) goto fail;
    }
    if (!identity_finish(&hash, target->summary.role_map_identity)) goto fail;
    target->summary.roles_complete = count == YVEX_MINIMAX_H3_TENSORS;
    return YVEX_OK;
fail:
    return family_refuse(failure, YVEX_MINIMAX_H3_FAILURE_TENSOR_ROLE,
                         YVEX_MINIMAX_H3_COMPONENT_COUNT, 0, index, NULL,
                         YVEX_ERR_STATE, err, "tensor role identity construction failed");
}

static int target_identity_build(yvex_minimax_h3_target *target,
                                 yvex_minimax_h3_failure *failure,
                                 yvex_error *err)
{
    yvex_sha256 hash;

    yvex_sha256_init(&hash);
    if (yvex_sha256_update_text(&hash, "yvex.minimax-h3.logical-target.v1") &&
        yvex_sha256_update_text(&hash, YVEX_MINIMAX_H3_TARGET_ID) &&
        yvex_sha256_update_text(&hash, target->summary.source_snapshot_identity) &&
        yvex_sha256_update_text(&hash, target->summary.component_manifest_identity) &&
        yvex_sha256_update_text(&hash, target->summary.architecture_identity) &&
        yvex_sha256_update_text(&hash, target->summary.role_map_identity) &&
        yvex_sha256_update_text(&hash, target->summary.unresolved_requirements_identity) &&
        identity_finish(&hash, target->summary.target_identity)) return YVEX_OK;
    return family_refuse(failure, YVEX_MINIMAX_H3_FAILURE_TRANSFORMATION,
                         YVEX_MINIMAX_H3_COMPONENT_COUNT, 0, 0u, NULL,
                         YVEX_ERR_STATE, err, "logical target identity construction failed");
}

static void target_close(yvex_minimax_h3_target **target_pointer)
{
    yvex_minimax_h3_target *target;

    if (!target_pointer || !*target_pointer) return;
    target = *target_pointer;
    free(target->roles);
    target->roles = NULL;
    yvex_native_weight_table_close(target->weights);
    target->weights = NULL;
    yvex_source_acquisition_release(&target->acquisition);
    memset(target, 0, sizeof(*target));
    free(target);
    *target_pointer = NULL;
}

static int source_identity_validate(yvex_minimax_h3_target *target,
                                    yvex_minimax_h3_failure *failure,
                                    yvex_error *err)
{
    const yvex_source_acquisition_facts *facts =
        yvex_source_acquisition_facts_get(target->acquisition);
    char inventory_identity[65];

    if (!facts || !facts->complete ||
        strcmp(facts->repository, YVEX_MINIMAX_H3_REPOSITORY) != 0 ||
        strcmp(facts->revision, YVEX_MINIMAX_H3_REVISION) != 0 ||
        strcmp(facts->subtree, YVEX_MINIMAX_H3_SUBTREE) != 0 ||
        facts->file_count != YVEX_MINIMAX_H3_SOURCE_FILES ||
        facts->shard_count != YVEX_MINIMAX_H3_SHARDS ||
        facts->source_bytes != YVEX_MINIMAX_H3_SOURCE_BYTES ||
        !source_inventory_identity(target->acquisition, inventory_identity) ||
        strcmp(inventory_identity, YVEX_MINIMAX_H3_SOURCE_INVENTORY_IDENTITY) != 0) {
        return family_refuse(failure, YVEX_MINIMAX_H3_FAILURE_SOURCE_IDENTITY,
                             YVEX_MINIMAX_H3_COMPONENT_COUNT, 0, 0u, NULL,
                             YVEX_ERR_FORMAT, err,
                             "acquired source inventory differs from the pinned FL2VA allowlist");
    }
    yvex_core_text_copy(target->summary.source_acquisition_identity,
                        sizeof(target->summary.source_acquisition_identity),
                        facts->acquisition_identity);
    target->summary.source_file_count = facts->file_count;
    if (source_file_identity_validate(target, "FL2VA/model_index.json",
                                      YVEX_MINIMAX_H3_MODEL_INDEX_IDENTITY,
                                      failure, err) != YVEX_OK ||
        source_file_identity_validate(target, "FL2VA/text_encoder/config.json",
                                      text_config_identity, failure, err) != YVEX_OK ||
        source_file_identity_validate(target, "FL2VA/transformer/config.json",
                                      transformer_config_identity, failure, err) != YVEX_OK ||
        source_file_identity_validate(target, "FL2VA/video_vae/config.json",
                                      video_config_identity, failure, err) != YVEX_OK ||
        source_file_identity_validate(target, "FL2VA/video_vae/source/config.json",
                                      video_source_config_identity, failure, err) != YVEX_OK ||
        source_file_identity_validate(target, "FL2VA/audio_vae/config.json",
                                      audio_config_identity, failure, err) != YVEX_OK ||
        source_file_identity_validate(target, "FL2VA/audio_vae/metadata.json",
                                      audio_metadata_identity, failure, err) != YVEX_OK ||
        source_file_identity_validate(target, "FL2VA/tokenizer/tokenizer_config.json",
                                      tokenizer_config_identity, failure, err) != YVEX_OK ||
        source_file_identity_validate(target, "FL2VA/processor/tokenizer_config.json",
                                      tokenizer_config_identity, failure, err) != YVEX_OK)
        return yvex_error_code(err);
    target->summary.source_verified = 1;
    return YVEX_OK;
}

static int target_open(yvex_minimax_h3_target **out,
                       const yvex_minimax_h3_open_options *options,
                       yvex_minimax_h3_failure *failure,
                       yvex_error *err)
{
    yvex_minimax_h3_target *target = NULL;
    yvex_source_acquisition_options acquisition_options;
    yvex_source_acquisition_failure acquisition_failure;
    const yvex_source_acquisition_facts *acquisition_facts;
    yvex_native_weight_options native_options;
    int rc;

    if (out) *out = NULL;
    if (failure) memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    if (!out || !options || !options->source_root || !options->source_root[0] ||
        strlen(options->source_root) >= YVEX_PATH_CAP) {
        return family_refuse(failure, YVEX_MINIMAX_H3_FAILURE_INVALID_ARGUMENT,
                             YVEX_MINIMAX_H3_COMPONENT_COUNT, 0, 0u, NULL,
                             YVEX_ERR_INVALID_ARG, err, "exact source root is required");
    }
    target = (yvex_minimax_h3_target *)calloc(1u, sizeof(*target));
    if (!target) {
        return family_refuse(failure, YVEX_MINIMAX_H3_FAILURE_ALLOCATION,
                             YVEX_MINIMAX_H3_COMPONENT_COUNT, 0, 0u, NULL,
                             YVEX_ERR_NOMEM, err, "logical target allocation failed");
    }
    yvex_core_text_copy(target->source_root, sizeof(target->source_root), options->source_root);
    yvex_source_acquisition_options_default(&acquisition_options);
    acquisition_options.source_root = target->source_root;
    acquisition_options.expected_repository = YVEX_MINIMAX_H3_REPOSITORY;
    acquisition_options.expected_revision = YVEX_MINIMAX_H3_REVISION;
    acquisition_options.expected_subtree = YVEX_MINIMAX_H3_SUBTREE;
    acquisition_options.maximum_files = YVEX_MINIMAX_H3_SOURCE_FILES;
    acquisition_options.maximum_source_bytes = YVEX_MINIMAX_H3_SOURCE_BYTES;
    acquisition_options.verify_digests = 1;
    rc = yvex_source_acquisition_open(&target->acquisition, &acquisition_options,
                                      &acquisition_failure, err);
    if (rc != YVEX_OK) {
        rc = family_refuse(failure, YVEX_MINIMAX_H3_FAILURE_SOURCE_ACQUISITION,
                           YVEX_MINIMAX_H3_COMPONENT_COUNT, 0,
                           acquisition_failure.file_index, acquisition_failure.path,
                           rc, err, "immutable source acquisition admission failed");
        goto fail;
    }
    rc = source_identity_validate(target, failure, err);
    if (rc != YVEX_OK) goto fail;
    native_options.source_dir = target->source_root;
    native_options.recursive = 1;
    native_options.include_metadata = 0;
    rc = yvex_native_weight_table_open(&target->weights, &native_options, err);
    if (rc != YVEX_OK) {
        rc = family_refuse(failure, YVEX_MINIMAX_H3_FAILURE_SOURCE_INVENTORY,
                           YVEX_MINIMAX_H3_COMPONENT_COUNT, 0, 0u, NULL, rc, err,
                           "bounded safetensors header admission failed");
        goto fail;
    }
    rc = source_components_validate(target, failure, err);
    if (rc != YVEX_OK) goto fail;
    rc = source_population_validate(target, failure, err);
    if (rc != YVEX_OK) goto fail;
    acquisition_facts = yvex_source_acquisition_facts_get(target->acquisition);
    if (!source_snapshot_identity(acquisition_facts, target->weights,
                                  target->summary.source_snapshot_identity)) {
        rc = family_refuse(failure, YVEX_MINIMAX_H3_FAILURE_SOURCE_IDENTITY,
                           YVEX_MINIMAX_H3_COMPONENT_COUNT, 0, 0u, NULL,
                           YVEX_ERR_STATE, err, "source snapshot identity construction failed");
        goto fail;
    }
    target->summary.source_snapshot_key =
        identity_key(target->summary.source_snapshot_identity);
    rc = unresolved_requirements_build(target, failure, err);
    if (rc != YVEX_OK) goto fail;
    rc = components_canonical(
        target->components, target->summary.component_manifest_identity,
        failure, err);
    if (rc != YVEX_OK) goto fail;
    target->summary.component_count = YVEX_MINIMAX_H3_LOGICAL_COMPONENTS;
    target->summary.weighted_component_count = YVEX_MINIMAX_H3_WEIGHTED_COMPONENTS;
    target->summary.output_classes = YVEX_MINIMAX_H3_DATA_SYNCHRONIZED_MEDIA;
    rc = architecture_canonical(&target->architecture, failure, err);
    if (rc != YVEX_OK) goto fail;
    yvex_core_text_copy(target->summary.architecture_identity,
                        sizeof(target->summary.architecture_identity),
                        target->architecture.identity);
    target->summary.architecture_admitted = 1;
    rc = role_map_build(target, failure, err);
    if (rc != YVEX_OK) goto fail;
    rc = target_identity_build(target, failure, err);
    if (rc != YVEX_OK) goto fail;
    *out = target;
    return YVEX_OK;

fail:
    target_close(&target);
    return rc;
}

static const yvex_minimax_h3_summary *target_summary(
    const yvex_minimax_h3_target *target)
{
    return target ? &target->summary : NULL;
}

static const yvex_minimax_h3_architecture *target_architecture(
    const yvex_minimax_h3_target *target)
{
    return target ? &target->architecture : NULL;
}

static const yvex_minimax_h3_component *target_component_at(
    const yvex_minimax_h3_target *target, unsigned long long index)
{
    return target && index < YVEX_MINIMAX_H3_COMPONENT_COUNT
               ? &target->components[index] : NULL;
}

static const yvex_minimax_h3_tensor_role *target_role_at(
    const yvex_minimax_h3_target *target, unsigned long long index)
{
    return target && index < target->summary.tensor_count ? &target->roles[index] : NULL;
}

const yvex_minimax_h3_api *yvex_model_register_minimax_h3(void)
{
    static const yvex_minimax_h3_api api = {
        target_open, target_close, target_summary, target_architecture,
        target_component_at, target_role_at,
        family_failure_name, family_role_name, family_component_name,
        components_canonical, component_graph_validate,
        architecture_canonical, tensor_classify
    };

    return &api;
}
