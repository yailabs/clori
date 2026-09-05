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
#include <yvex/internal/compilation.h>
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
static const yvex_minimax_h3_phase_edge phase_edges[] = {
    {YVEX_MINIMAX_H3_PHASE_PREPARE, YVEX_MINIMAX_H3_PHASE_CONDITION,
     YVEX_MINIMAX_H3_DATA_TOKEN_IDS | YVEX_MINIMAX_H3_DATA_MEDIA_GRID,
     YVEX_MINIMAX_H3_LIFETIME_PHASE},
    {YVEX_MINIMAX_H3_PHASE_CONDITION, YVEX_MINIMAX_H3_PHASE_LATENT_INITIALIZE,
     YVEX_MINIMAX_H3_DATA_CONDITIONING, YVEX_MINIMAX_H3_LIFETIME_REQUEST_IMMUTABLE},
    {YVEX_MINIMAX_H3_PHASE_LATENT_INITIALIZE, YVEX_MINIMAX_H3_PHASE_LATENT_ITERATE,
     YVEX_MINIMAX_H3_DATA_VIDEO_LATENT | YVEX_MINIMAX_H3_DATA_AUDIO_LATENT,
     YVEX_MINIMAX_H3_LIFETIME_REQUEST_MUTABLE},
    {YVEX_MINIMAX_H3_PHASE_LATENT_ITERATE, YVEX_MINIMAX_H3_PHASE_VIDEO_DECODE,
     YVEX_MINIMAX_H3_DATA_VIDEO_LATENT, YVEX_MINIMAX_H3_LIFETIME_REQUEST_IMMUTABLE},
    {YVEX_MINIMAX_H3_PHASE_LATENT_ITERATE, YVEX_MINIMAX_H3_PHASE_AUDIO_DECODE,
     YVEX_MINIMAX_H3_DATA_AUDIO_LATENT, YVEX_MINIMAX_H3_LIFETIME_REQUEST_IMMUTABLE},
    {YVEX_MINIMAX_H3_PHASE_VIDEO_DECODE, YVEX_MINIMAX_H3_PHASE_MEDIA_PUBLISH,
     YVEX_MINIMAX_H3_DATA_RGB_FRAMES, YVEX_MINIMAX_H3_LIFETIME_OUTPUT_TRANSACTION},
    {YVEX_MINIMAX_H3_PHASE_AUDIO_DECODE, YVEX_MINIMAX_H3_PHASE_MEDIA_PUBLISH,
     YVEX_MINIMAX_H3_DATA_STEREO_SAMPLES, YVEX_MINIMAX_H3_LIFETIME_OUTPUT_TRANSACTION}
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
static const yvex_minimax_h3_tokenizer_spec tokenizer_spec = {
    "FL2VA/tokenizer/tokenizer.json", "FL2VA/tokenizer/tokenizer_config.json",
    "qwen2", "verbatim-no-special-v1", 151669ull};
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

static int phase_graph_validate(
    const yvex_minimax_h3_phase_edge *edges, size_t edge_count,
    yvex_minimax_h3_failure *failure, yvex_error *err)
{
    unsigned int publication_inputs = 0u;
    size_t index;

    if (!edges || edge_count != YVEX_MINIMAX_H3_PHASE_EDGES) {
        return family_refuse(failure, YVEX_MINIMAX_H3_FAILURE_COMPONENT_COVERAGE,
                             YVEX_MINIMAX_H3_COMPONENT_COUNT, 0, edge_count, NULL,
                             YVEX_ERR_FORMAT, err, "exactly seven phase edges are required");
    }
    for (index = 0u; index < edge_count; ++index) {
        const yvex_minimax_h3_phase_edge *edge = &edges[index];
        size_t prior;

        if (edge->source_phase < YVEX_MINIMAX_H3_PHASE_PREPARE ||
            edge->source_phase > YVEX_MINIMAX_H3_PHASE_MEDIA_PUBLISH ||
            edge->destination_phase < YVEX_MINIMAX_H3_PHASE_PREPARE ||
            edge->destination_phase > YVEX_MINIMAX_H3_PHASE_MEDIA_PUBLISH ||
            edge->source_phase >= edge->destination_phase || !edge->data_classes ||
            edge->lifetime < YVEX_MINIMAX_H3_LIFETIME_METADATA ||
            edge->lifetime > YVEX_MINIMAX_H3_LIFETIME_OUTPUT_TRANSACTION) {
            return family_refuse(failure, YVEX_MINIMAX_H3_FAILURE_PHASE_ORDER,
                                 YVEX_MINIMAX_H3_COMPONENT_COUNT, 0, index, NULL,
                                 YVEX_ERR_FORMAT, err,
                                 "phase edge endpoint, order, data, or lifetime is invalid");
        }
        for (prior = 0u; prior < index; ++prior) {
            if (edges[prior].source_phase == edge->source_phase &&
                edges[prior].destination_phase == edge->destination_phase &&
                edges[prior].data_classes == edge->data_classes) {
                return family_refuse(failure, YVEX_MINIMAX_H3_FAILURE_COMPONENT_COVERAGE,
                                     YVEX_MINIMAX_H3_COMPONENT_COUNT, 0, index, NULL,
                                     YVEX_ERR_FORMAT, err, "phase edge is duplicated");
            }
        }
        if (edge->destination_phase == YVEX_MINIMAX_H3_PHASE_MEDIA_PUBLISH) {
            if (edge->lifetime != YVEX_MINIMAX_H3_LIFETIME_OUTPUT_TRANSACTION) {
                return family_refuse(failure, YVEX_MINIMAX_H3_FAILURE_PHASE_ORDER,
                                     YVEX_MINIMAX_H3_COMPONENT_COUNT, 0, index, NULL,
                                     YVEX_ERR_FORMAT, err,
                                     "publication inputs must belong to the output transaction");
            }
            publication_inputs |= edge->data_classes;
        }
    }
    if ((publication_inputs & (YVEX_MINIMAX_H3_DATA_RGB_FRAMES |
                               YVEX_MINIMAX_H3_DATA_STEREO_SAMPLES)) !=
        (YVEX_MINIMAX_H3_DATA_RGB_FRAMES | YVEX_MINIMAX_H3_DATA_STEREO_SAMPLES)) {
        return family_refuse(failure, YVEX_MINIMAX_H3_FAILURE_COMPONENT_COVERAGE,
                             YVEX_MINIMAX_H3_COMPONENT_COUNT, 0, edge_count, NULL,
                             YVEX_ERR_FORMAT, err,
                             "media publication requires both video frames and stereo samples");
    }
    return YVEX_OK;
}

static int phase_graph_identity(char output[65])
{
    yvex_sha256 hash;
    size_t index;

    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.minimax-h3.phase-dag.v1") ||
        !yvex_sha256_update_u64(&hash, YVEX_MINIMAX_H3_PHASE_EDGES)) return 0;
    for (index = 0u; index < YVEX_MINIMAX_H3_PHASE_EDGES; ++index) {
        if (!identity_u32(&hash, phase_edges[index].source_phase) ||
            !identity_u32(&hash, phase_edges[index].destination_phase) ||
            !identity_u32(&hash, phase_edges[index].data_classes) ||
            !identity_u32(&hash, phase_edges[index].lifetime)) return 0;
    }
    return identity_finish(&hash, output);
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

    target->summary.output_classes = YVEX_MINIMAX_H3_DATA_SYNCHRONIZED_MEDIA;
    if (phase_graph_validate(phase_edges, YVEX_MINIMAX_H3_PHASE_EDGES,
                             failure, err) != YVEX_OK ||
        !phase_graph_identity(target->summary.phase_dag_identity))
        goto identity_failure;
    target->summary.phase_edge_count = YVEX_MINIMAX_H3_PHASE_EDGES;
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
    if (!yvex_sha256_update_text(&hash, target->summary.phase_dag_identity) ||
        !identity_u32(&hash, target->summary.output_classes) ||
        !identity_finish(&hash, target->summary.component_manifest_identity))
        goto identity_failure;
    target->summary.component_count = YVEX_MINIMAX_H3_LOGICAL_COMPONENTS;
    target->summary.weighted_component_count = YVEX_MINIMAX_H3_WEIGHTED_COMPONENTS;
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
        5120u, 24u, 32u, 1u, 2u, 2u, 1u, 32u, 256u, 5376u, 2688u, 96768u,
        10752u, 96u, 32u, 24u, 3u, 128u, 6u, 1u, 2u, 2u, 4u, 4u, 8u,
        2u, 2u, 2u, 2u, 1u, 1u, 1u, 2u, 2u, 2u, 2u, 1u,
        1u, 2u, 2u, 1u, 1u, 1u, 16u, 4u, 2u, 36u, 32u, 64u,
        3u, 4u, 100u, 256u, 64u, 17u, 3u, 1u, 1u, 1u, 1u, 1u, 1u, 0u,
        32u, 2u, 32000u, 64u, 1024u, 2048u, 5u, 2u, 4u, 4u, 5u, 5u,
        7u, 5u, 5u, 2u, 2u, 2u, 2u, 2u, 800u, 800u, 40u, 1u, 1580u, 1660u
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
        {1u, 2u, 2u}, 1u, 32u, 256u, 5376u, 2688u, 96768u, 10752u, 96u, 32u, 1
    };
    architecture->video_vae = (yvex_minimax_h3_video_vae_signature){
        .latent_channels = 24u, .media_channels = 3u,
        .base_channels = 128u, .stage_count = 6u,
        .channel_multipliers = {1u, 2u, 2u, 4u, 4u, 8u},
        .spatial_down = {2u, 2u, 2u, 2u, 1u, 1u},
        .spatial_up = {1u, 2u, 2u, 2u, 2u, 1u},
        .temporal_down = {1u, 2u, 2u, 1u, 1u, 1u},
        .spatial_ratio = 16u, .temporal_ratio = 4u, .residual_blocks = 2u,
        .decoder_blocks = 36u, .decoder_heads = 32u, .decoder_head_dimension = 64u,
        .decoder_rope_ratio_numerator = 3u, .decoder_rope_ratio_denominator = 4u,
        .decoder_rope_theta = 100u, .tile_size = 256u, .tile_overlap = 64u,
        .clip_length = 17u, .token_drop = 3u, .conv3d = 1,
        .isolated_temporal_group_norm = 1, .encoder_tiling = 1, .decoder_tiling = 1,
        .parallel_tiling = 1, .causal_encoder = 1, .causal_decoder = 0
    };
    architecture->audio_vae = (yvex_minimax_h3_audio_vae_signature){
        .latent_channels = 32u, .output_channels = 2u, .sample_rate = 32000u,
        .encoder_width = 64u, .decoder_width = 1024u, .latent_projection_width = 2048u,
        .encoder_stage_count = 5u,
        .encoder_rates = {2u, 4u, 4u, 5u, 5u},
        .decoder_stage_count = 7u,
        .decoder_rates = {5u, 5u, 2u, 2u, 2u, 2u, 2u},
        .encoder_rate_product = 800u, .decoder_rate_product = 800u,
        .latent_steps_per_second = 40u, .attention_projection = 1
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
static const yvex_minimax_h3_latent_normalization *latent_normalization(void)
{
    static const float pixel_mean[3] = {0.485f, 0.456f, 0.406f};
    static const float pixel_std[3] = {0.229f, 0.224f, 0.225f};
    static const float video_mean[24] = {
        0.858090341f, -0.960659146f, 1.06616402f, -0.509032547f,
        -0.272758186f, -1.36754143f, -0.255325496f, -0.269075543f,
        -0.537684083f, -0.0464097299f, 0.665737033f, 0.196901277f,
        -0.546060801f, -0.403534204f, -0.236830249f, 0.259284526f,
        -0.301339447f, 0.211341992f, -1.12068486f, 0.358193338f,
        -0.0422514379f, 0.260482997f, 0.228640929f, 0.705603182f,
    };
    static const float video_std[24] = {
        1.22237742f, 1.27672637f, 1.68317747f, 1.75494552f,
        1.56362164f, 2.19414353f, 0.965313792f, 1.0569886f,
        0.841948926f, 0.772995293f, 1.89559376f, 0.946841836f,
        0.799680948f, 0.449889004f, 0.719739974f, 0.693629324f,
        2.96109509f, 2.76941991f, 3.04961848f, 2.10880542f,
        3.27622628f, 3.1627357f, 2.2816813f, 2.61278439f,
    };
    static const float audio_mean[32] = {
        -0.0202116873f, 0.387646645f, -0.0439827964f, -0.285915136f,
        0.0817968622f, -0.357826412f, 0.0406238101f, -0.0155253448f,
        -0.223362476f, 0.182100683f, 0.29417789f, -0.0790116787f,
        -0.0568150729f, -0.369902819f, -0.316163152f, 0.590595126f,
        -0.0521395691f, 0.0136731602f, -0.0369164795f, 0.0973266065f,
        -0.339466244f, -0.306856781f, -0.24504599f, -0.0346985236f,
        0.0286803227f, -0.212177798f, -0.16782631f, 0.322128803f,
        -0.122305587f, 0.435660481f, -0.0502599217f, 0.397925824f,
    };
    static const float audio_std[32] = {
        1.68955243f, 2.76263738f, 1.79453444f, 1.68016815f,
        1.63902271f, 2.77882981f, 1.76590896f, 1.61997581f,
        2.63365245f, 1.85393572f, 2.50564981f, 1.81101918f,
        1.95796573f, 1.66854978f, 1.49224699f, 3.29867029f,
        1.94918048f, 1.87200034f, 1.833408f, 1.64880705f,
        1.61769581f, 1.91314495f, 1.56952453f, 1.69436598f,
        1.83184206f, 1.5540638f, 1.93449306f, 1.59919822f,
        1.71804595f, 1.63072193f, 1.8661226f, 1.56137681f,
    };
    static const yvex_minimax_h3_latent_normalization facts = {
        video_mean, video_std, audio_mean, audio_std, pixel_mean, pixel_std, 24ull, 32ull, 3ull};
    return &facts;
}
static int architecture_canonical(yvex_minimax_h3_architecture *architecture,
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
int yvex_model_minimax_h3_media_target_profile(yvex_media_target_profile *out, yvex_error *err)
{
    yvex_minimax_h3_architecture architecture = {0};
    yvex_minimax_h3_failure failure = {0};
    const yvex_minimax_h3_latent_normalization *normalization;
    int rc;
    if (!out) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "minimax-h3.media-profile",
                       "media target profile output is required");
        return YVEX_ERR_INVALID_ARG;
    }
    rc = architecture_canonical(&architecture, &failure, err);
    normalization = rc == YVEX_OK ? latent_normalization() : NULL;
    if (rc != YVEX_OK || !normalization) return rc != YVEX_OK ? rc : YVEX_ERR_STATE;
    *out = (yvex_media_target_profile){
        .schema_version = YVEX_MEDIA_TARGET_PROFILE_SCHEMA_V2,
        .target = YVEX_MINIMAX_H3_TARGET_ID, .family = "minimax-h3",
        .source_identity = YVEX_MINIMAX_H3_SOURCE_TREE_IDENTITY,
        .text_artifact = "physical-v3/text_encoder.gguf",
        .transformer_artifact = "physical-v4/transformer.gguf",
        .video_artifact = "physical/video_vae.gguf",
        .audio_artifact = "physical/audio_vae.gguf",
        .tiers = {{"preview", 192ull, 192ull, 124ull, 1},
                  {"preview-256", 256ull, 256ull, 124ull, 0},
                  {"preview-384", 384ull, 384ull, 124ull, 0},
                  {"source-768", 768ull, 768ull, 345ull, 0},
                  {"smoke", 32ull, 32ull, 345ull, 0}}, .tier_count = 5ull,
        .fps_numerator = 24ull, .fps_denominator = 1ull,
        .audio_sample_rate = architecture.audio_vae.sample_rate, .seed = 42ull,
        .keyframe_encode_seed = 42ull,
        .maximum_host_bytes = 96ull << 30u, .maximum_device_bytes = 64ull << 30u,
        .maximum_workspace_bytes = 16ull << 30u, .maximum_file_bytes = 2ull << 30u,
        .video_temporal_ratio = architecture.video_vae.temporal_ratio,
        .video_clip_length = architecture.video_vae.clip_length,
        .video_token_drop = architecture.video_vae.token_drop,
        .video_spatial_ratio = architecture.video_vae.spatial_ratio,
        .video_tile_size = architecture.video_vae.tile_size,
        .video_minimum_tile_overlap = architecture.video_vae.tile_overlap,
        .video_mean = normalization->video_mean, .video_std = normalization->video_std,
        .audio_mean = normalization->audio_mean, .audio_std = normalization->audio_std,
        .pixel_mean = normalization->pixel_mean, .pixel_std = normalization->pixel_std,
        .video_channels = normalization->video_channels,
        .audio_channels = normalization->audio_channels,
        .pixel_channels = normalization->pixel_channels,
        .audio_output_channels = architecture.audio_vae.output_channels,
        .audio_samples_per_step = architecture.audio_vae.decoder_rate_product,
        .frames_per_chunk = 17ull, .frame_remainder = 5ull,
        .minimum_frames = 124ull, .maximum_frames = 345ull,
        .minimum_inference_steps = 2ull, .maximum_inference_steps = 64ull,
        .released_sigma_grid_points = 50ull,
        .canvas_multiple = 32ull, .canvas_short_edge = 768ull,
        .minimum_canvas_pixels = 768ull * 768ull,
        .maximum_canvas_pixels = 768ull * 1344ull,
        .released_width = 1344ull, .released_height = 768ull,
        .minimum_duration_milliseconds = 5000ull,
        .maximum_duration_milliseconds = 15000ull,
        .minimum_aspect_numerator = 1ull, .minimum_aspect_denominator = 4ull,
        .maximum_aspect_numerator = 4ull, .maximum_aspect_denominator = 1ull};
    yvex_error_clear(err);
    return YVEX_OK;
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
        strcmp(facts->repository, YVEX_SOURCE_MINIMAX_H3_REPOSITORY) != 0 ||
        strcmp(facts->revision, YVEX_SOURCE_MINIMAX_H3_REVISION) != 0 ||
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
    acquisition_options.expected_repository = YVEX_SOURCE_MINIMAX_H3_REPOSITORY;
    acquisition_options.expected_revision = YVEX_SOURCE_MINIMAX_H3_REVISION;
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
    rc = components_validate(target, failure, err);
    if (rc != YVEX_OK) goto fail;
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

static const yvex_source_acquisition *target_acquisition(
    const yvex_minimax_h3_target *target)
{
    return target ? target->acquisition : NULL;
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

static const yvex_minimax_h3_tokenizer_spec *target_tokenizer_spec(void)
{
    return &tokenizer_spec;
}

static const yvex_minimax_h3_phase_edge *target_phase_edge_at(unsigned long long index)
{
    return index < YVEX_MINIMAX_H3_PHASE_EDGES ? &phase_edges[index] : NULL;
}

static const yvex_minimax_h3_tensor_role *target_role_at(
    const yvex_minimax_h3_target *target, unsigned long long index)
{
    return target && index < target->summary.tensor_count ? &target->roles[index] : NULL;
}

const yvex_minimax_h3_api *yvex_model_register_minimax_h3(void)
{
    static const yvex_minimax_h3_api api = {
        target_open, target_close, target_summary, target_acquisition, target_architecture,
        target_component_at, target_tokenizer_spec, target_phase_edge_at, target_role_at,
        family_failure_name, family_role_name, family_component_name,
        components_canonical, component_graph_validate, phase_graph_validate,
        architecture_canonical, latent_normalization, tensor_classify
    };

    return &api;
}
static yvex_transform_dtype minimax_transform_dtype(yvex_native_dtype dtype)
{
    if (dtype == YVEX_NATIVE_DTYPE_BF16) return YVEX_TRANSFORM_DTYPE_BF16;
    if (dtype == YVEX_NATIVE_DTYPE_F32) return YVEX_TRANSFORM_DTYPE_F32;
    return YVEX_TRANSFORM_DTYPE_UNKNOWN;
}

static yvex_transform_subsystem minimax_transform_subsystem(yvex_minimax_h3_role role)
{
    if ((role >= YVEX_MINIMAX_H3_ROLE_TEXT_ATTENTION_Q &&
         role <= YVEX_MINIMAX_H3_ROLE_TEXT_QK_NORM) ||
        role == YVEX_MINIMAX_H3_ROLE_VISION_ATTENTION ||
        role == YVEX_MINIMAX_H3_ROLE_TOKEN_REFINER_ATTENTION ||
        role == YVEX_MINIMAX_H3_ROLE_OMNI_ATTENTION ||
        role == YVEX_MINIMAX_H3_ROLE_OMNI_QK_NORM ||
        role == YVEX_MINIMAX_H3_ROLE_VIDEO_DECODER_ATTENTION ||
        role == YVEX_MINIMAX_H3_ROLE_AUDIO_PRE_ATTENTION)
        return YVEX_TRANSFORM_SUBSYSTEM_ATTENTION;
    if (role == YVEX_MINIMAX_H3_ROLE_TEXT_RMS_NORM ||
        role == YVEX_MINIMAX_H3_ROLE_VISION_NORM ||
        role == YVEX_MINIMAX_H3_ROLE_TOKEN_REFINER_NORM ||
        role == YVEX_MINIMAX_H3_ROLE_OMNI_NORM ||
        role == YVEX_MINIMAX_H3_ROLE_FINAL_NORM ||
        role == YVEX_MINIMAX_H3_ROLE_VIDEO_ENCODER_NORM ||
        role == YVEX_MINIMAX_H3_ROLE_VIDEO_DECODER_NORM ||
        role == YVEX_MINIMAX_H3_ROLE_AUDIO_PRE_NORM)
        return YVEX_TRANSFORM_SUBSYSTEM_NORMALIZATION;
    if (role == YVEX_MINIMAX_H3_ROLE_TEXT_OUTPUT_HEAD ||
        role == YVEX_MINIMAX_H3_ROLE_FINAL_VIDEO ||
        role == YVEX_MINIMAX_H3_ROLE_FINAL_AUDIO ||
        role == YVEX_MINIMAX_H3_ROLE_VIDEO_OUTPUT ||
        role == YVEX_MINIMAX_H3_ROLE_AUDIO_OUTPUT)
        return YVEX_TRANSFORM_SUBSYSTEM_OUTPUT;
    return YVEX_TRANSFORM_SUBSYSTEM_AUXILIARY;
}

typedef struct {
    const yvex_minimax_h3_target *target;
    yvex_minimax_h3_component_id component;
    int component_only;
    int transformer_q8;
} minimax_transform_projection;
static int minimax_transform_project(void *context, yvex_transform_recipe_sink *sink,
                                     yvex_transform_failure *failure,
                                     yvex_error *err)
{
    minimax_transform_projection *projection = context;
    const yvex_minimax_h3_api *family = yvex_model_register_minimax_h3();
    const yvex_minimax_h3_summary *summary = family->summary(projection->target);
    unsigned long long source_index, terminal_ordinal = 0ull;

    for (source_index = 0ull; source_index < summary->tensor_count; ++source_index) {
        const yvex_minimax_h3_tensor_role *role =
            family->role_at(projection->target, source_index);
        const yvex_minimax_h3_component *component;
        yvex_transform_source_spec source = {0};
        yvex_transform_recipe recipe = {0};
        unsigned int dimension;
        int rc;

        if (!role)
            return yvex_transform_fail(
                failure, YVEX_TRANSFORM_FAILURE_MISSING_SOURCE, source_index,
                YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
                YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID, 1ull,
                0ull, 0u, err, "minimax_h3_transform");
        if (projection->component_only && role->component != projection->component)
            continue;
        if (role->transform != YVEX_MINIMAX_H3_TRANSFORM_IDENTITY)
            return yvex_transform_fail(
                failure, YVEX_TRANSFORM_FAILURE_UNSUPPORTED_OPERATION,
                source_index, YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
                YVEX_TRANSFORM_IR_NO_ID, YVEX_TRANSFORM_IR_NO_ID,
                YVEX_MINIMAX_H3_TRANSFORM_IDENTITY, role->transform, 0u, err,
                "minimax_h3_transform");
        component = family->component_at(projection->target, role->component);
        if (!component) {
            yvex_error_set(err, YVEX_ERR_STATE, "minimax_h3_transform",
                           "tensor role references an absent component");
            return YVEX_ERR_STATE;
        }
        source.source_name = role->source_name;
        source.shard_name = role->shard_name;
        source.source_tensor_index = source_index;
        source.requirement_index = terminal_ordinal;
        source.source_snapshot_identity = summary->source_snapshot_key;
        source.source_dtype = role->source_dtype;
        source.value_dtype = minimax_transform_dtype(role->source_dtype);
        source.shape.rank = role->rank;
        for (dimension = 0u; dimension < role->rank; ++dimension)
            source.shape.dims[dimension] = role->source_shape[dimension];
        source.relative_begin = role->relative_begin;
        source.relative_end = role->relative_end;
        source.requirement_identity = yvex_transform_hash_string(role->destination_identity);
        source.scope = role->layer_index == YVEX_MINIMAX_H3_NO_COORDINATE
                           ? YVEX_TRANSFORM_SCOPE_GLOBAL
                           : YVEX_TRANSFORM_SCOPE_MAIN_LAYER;
        source.subsystem = minimax_transform_subsystem(role->role);
        source.role_hint = YVEX_TENSOR_ROLE_UNKNOWN;
        source.component_identity = yvex_transform_hash_string(component->identity);
        source.semantic_role = role->role;
        source.phase_identity = role->phase;
        source.lifetime_identity = role->lifetime;
        source.unresolved_requirement_identity = role->unresolved_requirement_identity;
        source.layer_index = role->layer_index;
        source.auxiliary_index = YVEX_TRANSFORM_IR_NO_ID;
        source.expert_index = YVEX_TRANSFORM_IR_NO_ID;
        source.required_uses = 1ull;
        recipe.sources = &source;
        recipe.source_count = 1ull;
        recipe.terminal.semantic_id = source.requirement_identity;
        recipe.terminal.shape = source.shape;
        recipe.terminal.dtype = source.value_dtype;
        recipe.terminal.precision.flags = YVEX_TRANSFORM_PRECISION_EXACT;
        recipe.terminal.precision.allowed_physical_classes =
            source.value_dtype == YVEX_TRANSFORM_DTYPE_BF16
                ? YVEX_TRANSFORM_PHYSICAL_BF16 : YVEX_TRANSFORM_PHYSICAL_F32;
        if (projection->transformer_q8 &&
            role->component == YVEX_MINIMAX_H3_COMPONENT_TRANSFORMER &&
            role->layer_index != YVEX_MINIMAX_H3_NO_COORDINATE &&
            (YVEX_MINIMAX_H3_TRANSFORMER_Q8_ROLE_MASK & (1ull << role->role))) {
            recipe.terminal.precision.flags |= YVEX_TRANSFORM_PRECISION_QUANTIZABLE_WEIGHT |
                                               YVEX_TRANSFORM_PRECISION_REFERENCE_COMPUTE;
            recipe.terminal.precision.allowed_physical_classes |=
                YVEX_TRANSFORM_PHYSICAL_QUANTIZED;
            recipe.terminal.precision.approximation_allowed = 1;
            recipe.terminal.precision.reference_compute_required = 1;
        }
        recipe.terminal.logical_key.scope = source.scope;
        recipe.terminal.logical_key.subsystem = source.subsystem;
        recipe.terminal.logical_key.role = YVEX_TENSOR_ROLE_UNKNOWN;
        recipe.terminal.logical_key.component_identity = source.component_identity;
        recipe.terminal.logical_key.semantic_role = role->role;
        recipe.terminal.logical_key.phase_identity = role->phase;
        recipe.terminal.logical_key.lifetime_identity = role->lifetime;
        recipe.terminal.logical_key.layer_index = role->layer_index;
        recipe.terminal.logical_key.auxiliary_index = YVEX_TRANSFORM_IR_NO_ID;
        recipe.terminal.logical_key.group_index = source_index;
        recipe.operation.kind = YVEX_TRANSFORM_OP_IDENTITY;
        recipe.operation.numeric = YVEX_TRANSFORM_NUMERIC_EXACT;
        recipe.operation.ordering = YVEX_TRANSFORM_ORDER_INPUT;
        recipe.operation.payload_execution_required = 1;
        rc = yvex_transform_recipe_sink_add(sink, &recipe, failure, err);
        if (rc != YVEX_OK) return rc;
        ++terminal_ordinal;
    }
    return YVEX_OK;
}

static void minimax_transform_header(yvex_transform_header *header,
                                     const yvex_minimax_h3_summary *facts,
                                     const yvex_minimax_h3_component *component)
{
    memset(header, 0, sizeof(*header));
    header->schema_version = YVEX_TRANSFORM_IR_COMPONENT_SCHEMA_VERSION;
    header->logical_model_identity = component ? component->identity : facts->target_identity;
    header->source_snapshot_identity = facts->source_snapshot_key;
    header->coverage_identity = yvex_transform_hash_string(
        component ? component->identity : facts->role_map_identity);
    header->required_payload_identity = facts->source_acquisition_identity;
    header->payload_trust_class = "complete-sha256-verified-source";
    header->component_manifest_identity = facts->component_manifest_identity;
    header->architecture_identity = facts->architecture_identity;
    header->role_map_identity = facts->role_map_identity;
    header->unresolved_requirements_identity = facts->unresolved_requirements_identity;
    header->expected_source_count = component ? component->tensor_count : facts->tensor_count;
    header->expected_terminal_count = header->expected_source_count;
    header->header_scan_count = component ? component->shard_count : facts->shard_count;
}

static int minimax_derivation_identity(
    const yvex_transform_ir *ir, const yvex_minimax_h3_summary *facts,
    const yvex_minimax_h3_component *component, char output[65], yvex_error *err)
{
    const yvex_transform_ir_summary *summary = yvex_transform_ir_summary_get(ir);
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];

    yvex_sha256_init(&hash);
    if (!summary || !summary->complete ||
        !yvex_sha256_update_text(
            &hash, component ? "yvex.minimax-h3.component-derivation.v1"
                             : "yvex.minimax-h3.target-derivation.v1") ||
        !yvex_sha256_update_text(&hash, facts->target_identity) ||
        (component && !yvex_sha256_update_text(&hash, component->identity)) ||
        !yvex_sha256_update_text(&hash, summary->transform_identity) ||
        !yvex_sha256_final(&hash, digest)) {
        yvex_error_set(err, YVEX_ERR_STATE, "minimax_h3_transform",
                       "target derivation identity construction failed");
        return YVEX_ERR_STATE;
    }
    yvex_sha256_hex(digest, output);
    return YVEX_OK;
}
static int minimax_transform_build_selected(
    yvex_transform_ir **out, char derivation_identity[65], const yvex_minimax_h3_target *target,
    const yvex_minimax_h3_component *component, int transformer_q8, yvex_error *err)
{
    const yvex_minimax_h3_api *family = yvex_model_register_minimax_h3();
    const yvex_minimax_h3_summary *facts = family->summary(target);
    yvex_transform_header header;
    yvex_transform_builder_options options = {0};
    yvex_transform_failure failure = {0};
    minimax_transform_projection projection;
    int rc;

    if (out) *out = NULL;
    if (derivation_identity) derivation_identity[0] = '\0';
    if (!out || !derivation_identity || !facts ||
        (component && !component->weighted) || !facts->source_verified ||
        !facts->architecture_admitted || !facts->roles_complete) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "minimax_h3_transform",
                       "verified source, architecture, and complete roles are required");
        return YVEX_ERR_INVALID_ARG;
    }
    minimax_transform_header(&header, facts, component);
    projection.target = target;
    projection.component = component ? component->id : YVEX_MINIMAX_H3_COMPONENT_COUNT;
    projection.component_only = component != NULL;
    projection.transformer_q8 = transformer_q8;
    yvex_transform_budget_default(&options.budget);
    rc = yvex_transform_recipe_compile(out, &header, minimax_transform_project, &projection,
                                       &options, &failure, err);
    if (rc == YVEX_OK)
        rc = minimax_derivation_identity(*out, facts, component, derivation_identity, err);
    if (rc != YVEX_OK) yvex_transform_ir_release(out);
    return rc;
}
static int minimax_transform_build(yvex_transform_ir **out,
                                   char derivation_identity[65],
                                   const yvex_minimax_h3_target *target,
                                   yvex_error *err)
{
    return minimax_transform_build_selected(out, derivation_identity, target, NULL, 0, err);
}

static int minimax_transform_build_component(yvex_transform_ir **out,
                                             char derivation_identity[65],
                                             const yvex_minimax_h3_target *target,
                                             yvex_minimax_h3_component_id component_id,
                                             yvex_error *err)
{
    const yvex_minimax_h3_api *family = yvex_model_register_minimax_h3();
    const yvex_minimax_h3_component *component = family->component_at(target, component_id);
    return minimax_transform_build_selected(out, derivation_identity, target, component, 0, err);
}

const yvex_minimax_h3_transform_api *yvex_model_minimax_h3_transform_api(void)
{
    static const yvex_minimax_h3_transform_api api = {
        minimax_transform_build, minimax_transform_build_component
    };

    return &api;
}

struct yvex_minimax_h3_handoff {
    yvex_minimax_h3_target *target;
    yvex_source_tensor_snapshot *snapshot;
    yvex_source_payload_session *session;
    yvex_transform_ir *transform_ir;
    yvex_transform_binding *binding;
    yvex_source_payload_plan *plan;
    yvex_minimax_h3_handoff_summary summary;
};

static int minimax_handoff_refuse(yvex_minimax_h3_handoff_failure *failure,
                                  yvex_minimax_h3_handoff_code code,
                                  yvex_status status, yvex_error *err,
                                  const char *message)
{
    if (failure) failure->code = code;
    if (err && yvex_error_code(err) == YVEX_OK)
        yvex_error_set(err, status, "minimax_h3_handoff", message);
    return status;
}

static void minimax_handoff_close(yvex_minimax_h3_handoff **address)
{
    yvex_minimax_h3_handoff *handoff;

    if (!address || !*address) return;
    handoff = *address;
    *address = NULL;
    yvex_source_payload_plan_close(handoff->plan);
    yvex_transform_binding_release(&handoff->binding);
    yvex_transform_ir_release(&handoff->transform_ir);
    (void)yvex_source_payload_session_release(&handoff->session, NULL, NULL);
    yvex_source_tensor_snapshot_release(handoff->snapshot);
    yvex_model_register_minimax_h3()->close(&handoff->target);
    memset(handoff, 0, sizeof(*handoff));
    free(handoff);
}

static int minimax_handoff_plan_build(
    yvex_minimax_h3_handoff *handoff, const yvex_minimax_h3_component *component,
    const yvex_minimax_h3_handoff_options *options,
    const yvex_source_payload_budget *budget,
    yvex_minimax_h3_handoff_failure *failure, yvex_error *err)
{
    const yvex_transform_binding_summary *binding =
        yvex_transform_binding_summary_get(handoff->binding);
    int rc;

    if (!binding || !binding->complete || binding->source_count != component->tensor_count)
        return minimax_handoff_refuse(
            failure, YVEX_MINIMAX_H3_HANDOFF_TRANSFORMATION, YVEX_ERR_STATE, err,
            "component transform binding does not cover its exact source set");
    rc = yvex_transform_binding_payload_plan_build(
        &handoff->plan, handoff->binding,
        options->chunk_bytes ? options->chunk_bytes : budget->chunk_bytes,
        options->page_bytes ? options->page_bytes : budget->page_bytes,
        failure ? &failure->payload_failure : NULL, err);
    if (rc != YVEX_OK && failure)
        failure->code = YVEX_MINIMAX_H3_HANDOFF_PAYLOAD_PLAN;
    return rc;
}

static int minimax_handoff_summary_seal(
    yvex_minimax_h3_handoff *handoff, const yvex_minimax_h3_summary *target,
    const yvex_minimax_h3_component *component,
    yvex_minimax_h3_handoff_failure *failure, yvex_error *err)
{
    const yvex_source_payload_plan_summary *plan =
        yvex_source_payload_plan_summary_get(handoff->plan);
    const yvex_transform_ir_summary *transform =
        yvex_transform_ir_summary_get(handoff->transform_ir);
    yvex_source_payload_session_facts session;
    int rc = yvex_source_payload_session_facts_get(handoff->session, &session, err);

    if (rc != YVEX_OK || !plan || !transform || !transform->complete ||
        session.state != YVEX_SOURCE_PAYLOAD_STATE_READY ||
        plan->range_count != component->tensor_count)
        return minimax_handoff_refuse(failure, YVEX_MINIMAX_H3_HANDOFF_PAYLOAD_PLAN,
                                      YVEX_ERR_STATE, err,
                                      "component handoff did not seal");
    handoff->summary.component = component->id;
    yvex_core_text_copy(handoff->summary.component_identity,
                        sizeof(handoff->summary.component_identity), component->identity);
    yvex_core_text_copy(handoff->summary.source_snapshot_identity,
                        sizeof(handoff->summary.source_snapshot_identity),
                        target->source_snapshot_identity);
    yvex_core_text_copy(handoff->summary.payload_identity,
                        sizeof(handoff->summary.payload_identity), session.payload_identity);
    yvex_core_text_copy(handoff->summary.transform_identity,
                        sizeof(handoff->summary.transform_identity),
                        transform->transform_identity);
    handoff->summary.shards = component->shard_count;
    handoff->summary.tensors = component->tensor_count;
    handoff->summary.elements = component->element_count;
    handoff->summary.payload_bytes = component->payload_bytes;
    handoff->summary.planned_ranges = plan->range_count;
    handoff->summary.planned_chunks = plan->chunk_count;
    handoff->summary.payload_execution_bytes_read = transform->payload_bytes_read;
    handoff->summary.complete = 1;
    return YVEX_OK;
}

static int minimax_handoff_open(yvex_minimax_h3_handoff **out,
                                const yvex_minimax_h3_handoff_options *options,
                                yvex_minimax_h3_handoff_failure *failure,
                                yvex_error *err)
{
    const yvex_minimax_h3_api *family = yvex_model_register_minimax_h3();
    yvex_minimax_h3_handoff *handoff = NULL;
    yvex_minimax_h3_open_options target_options;
    const yvex_minimax_h3_summary *target;
    const yvex_minimax_h3_component *component;
    const yvex_source_acquisition *acquisition;
    yvex_source_payload_open_options payload = {0};
    yvex_transform_failure transform_failure;
    int rc;

    if (out) *out = NULL;
    if (failure) memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    if (!out || !options || !options->source_root || !options->source_root[0] ||
        options->component >= YVEX_MINIMAX_H3_COMPONENT_COUNT)
        return minimax_handoff_refuse(failure, YVEX_MINIMAX_H3_HANDOFF_INVALID_ARGUMENT,
                                      YVEX_ERR_INVALID_ARG, err,
                                      "one exact source root and component are required");
    handoff = (yvex_minimax_h3_handoff *)calloc(1u, sizeof(*handoff));
    if (!handoff)
        return minimax_handoff_refuse(failure, YVEX_MINIMAX_H3_HANDOFF_ALLOCATION,
                                      YVEX_ERR_NOMEM, err, "handoff allocation failed");
    target_options.source_root = options->source_root;
    rc = family->open(&handoff->target, &target_options,
                      failure ? &failure->target_failure : NULL, err);
    if (rc != YVEX_OK) {
        if (failure) failure->code = YVEX_MINIMAX_H3_HANDOFF_TARGET;
        goto fail;
    }
    target = family->summary(handoff->target);
    component = family->component_at(handoff->target, options->component);
    acquisition = family->acquisition(handoff->target);
    if (!target || !component || !component->weighted || !acquisition) {
        rc = minimax_handoff_refuse(
            failure, YVEX_MINIMAX_H3_HANDOFF_INVALID_ARGUMENT, YVEX_ERR_INVALID_ARG, err,
            "only one admitted weighted component can own a payload handoff");
        goto fail;
    }
    rc = yvex_source_acquisition_snapshot_create(
        &handoff->snapshot, acquisition, options->source_root,
        target->source_snapshot_key, err);
    if (rc != YVEX_OK) {
        if (failure) failure->code = YVEX_MINIMAX_H3_HANDOFF_SNAPSHOT;
        goto fail;
    }
    payload.acquisition = acquisition;
    payload.acquired_source_root = options->source_root;
    payload.acquired_target_id = YVEX_MINIMAX_H3_TARGET_ID;
    payload.acquired_family_key = "minimax-h3";
    payload.acquired_payload_identity = target->source_acquisition_identity;
    payload.acquired_source_snapshot_identity = target->source_snapshot_key;
    payload.snapshot = handoff->snapshot;
    payload.budget = options->budget;
    if (!payload.budget.maximum_shards)
        yvex_source_payload_budget_default(&payload.budget);
    rc = yvex_source_payload_session_open(
        &handoff->session, &payload, failure ? &failure->payload_failure : NULL, err);
    if (rc != YVEX_OK) {
        if (failure) failure->code = YVEX_MINIMAX_H3_HANDOFF_PAYLOAD_SESSION;
        goto fail;
    }
    rc = minimax_transform_build_selected(
        &handoff->transform_ir, handoff->summary.derivation_identity,
        handoff->target, component, options->transformer_q8, err);
    if (rc != YVEX_OK) {
        if (failure) failure->code = YVEX_MINIMAX_H3_HANDOFF_TRANSFORMATION;
        goto fail;
    }
    rc = yvex_transform_binding_create(
        &handoff->binding, handoff->transform_ir, handoff->session, NULL,
        &transform_failure, err);
    if (rc != YVEX_OK) {
        if (failure) failure->code = YVEX_MINIMAX_H3_HANDOFF_BINDING;
        goto fail;
    }
    rc = minimax_handoff_plan_build(handoff, component, options, &payload.budget,
                                    failure, err);
    if (rc == YVEX_OK)
        rc = minimax_handoff_summary_seal(handoff, target, component, failure, err);
    if (rc != YVEX_OK) goto fail;
    *out = handoff;
    return YVEX_OK;
fail:
    minimax_handoff_close(&handoff);
    return rc;
}
static const yvex_minimax_h3_handoff_summary *minimax_handoff_summary(
    const yvex_minimax_h3_handoff *handoff)
{
    return handoff ? &handoff->summary : NULL;
}
static const yvex_minimax_h3_target *minimax_handoff_target(
    const yvex_minimax_h3_handoff *handoff)
{
    return handoff ? handoff->target : NULL;
}
static const yvex_transform_ir *minimax_handoff_transform_ir(
    const yvex_minimax_h3_handoff *handoff)
{
    return handoff ? handoff->transform_ir : NULL;
}
static const yvex_transform_binding *minimax_handoff_binding(
    const yvex_minimax_h3_handoff *handoff)
{
    return handoff ? handoff->binding : NULL;
}
static yvex_source_payload_session *minimax_handoff_session(
    yvex_minimax_h3_handoff *handoff)
{
    return handoff ? handoff->session : NULL;
}
static const yvex_source_payload_plan *minimax_handoff_plan(
    const yvex_minimax_h3_handoff *handoff)
{
    return handoff ? handoff->plan : NULL;
}

const yvex_minimax_h3_handoff_api *yvex_model_minimax_h3_handoff_api(void)
{
    static const yvex_minimax_h3_handoff_api api = {
        minimax_handoff_open, minimax_handoff_close, minimax_handoff_summary,
        minimax_handoff_target, minimax_handoff_transform_ir, minimax_handoff_binding,
        minimax_handoff_session, minimax_handoff_plan
    };

    return &api;
}
