/* Compare the exact MiniMax-H3 Transformer envelope with an independent CUDA oracle. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include <yvex/artifact.h>
#include <yvex/gguf.h>
#include <yvex/internal/artifact.h>
#include <yvex/internal/backend.h>
#include <yvex/internal/component.h>
#include <yvex/internal/core.h>
#include <yvex/internal/families/minimax_h3.h>
#include <yvex/internal/joint_transformer.h>
#include <yvex/internal/latent.h>
#include <yvex/internal/media.h>
#include <yvex/internal/runtime.h>
#include "src/backend/cuda/component_ops.h"

enum { VIDEO_VALUES = 96u, AUDIO_VALUES = 32u, CONDITION_VALUES = 5120u };

static const char *const external_names[YVEX_TRANSFORMER_JOINT_EXTERNAL_WEIGHT_COUNT] = {
    "audio_patch_proj.weight", "audio_patch_proj.bias",
    "video_patch_proj.weight", "video_patch_proj.bias",
    "condition_proj.weight", "condition_proj.bias",
    "time_embedder.proj_in.weight", "time_embedder.proj_in.bias",
    "time_embedder.proj_out.weight", "time_embedder.proj_out.bias",
    "token_refiner.blocks.0.norm1.weight", "token_refiner.blocks.0.attn.qkv_proj.weight",
    "token_refiner.blocks.0.attn.q_norm.weight", "token_refiner.blocks.0.attn.k_norm.weight",
    "token_refiner.blocks.0.attn.out_proj.weight", "token_refiner.blocks.0.norm2.weight",
    "token_refiner.blocks.0.mlp.fc1.weight", "token_refiner.blocks.0.mlp.fc2.weight",
    "token_refiner.blocks.1.norm1.weight", "token_refiner.blocks.1.attn.qkv_proj.weight",
    "token_refiner.blocks.1.attn.q_norm.weight", "token_refiner.blocks.1.attn.k_norm.weight",
    "token_refiner.blocks.1.attn.out_proj.weight", "token_refiner.blocks.1.norm2.weight",
    "token_refiner.blocks.1.mlp.fc1.weight", "token_refiner.blocks.1.mlp.fc2.weight",
    "token_refiner.final_norm.weight", "rope.inv_freq", "final_layer.norm.weight",
    "final_layer.adaln_proj.linear.weight", "final_layer.adaln_proj.linear.bias",
    "final_layer.video_out.weight", "final_layer.video_out.bias",
    "final_layer.audio_out.weight", "final_layer.audio_out.bias",
};

static const char *const block_suffixes[YVEX_TRANSFORMER_JOINT_BLOCK_WEIGHT_COUNT] = {
    "norm1.weight", "attn.qkv_proj.weight", "attn.q_norm.weight", "attn.k_norm.weight",
    "attn.out_proj.weight", "norm2.weight", "mlp.fc1.weight", "mlp.fc2.weight",
    "adaln_proj.linear.weight", "adaln_proj.linear.bias",
};

static const yvex_materialized_tensor_binding *binding_find(
    const yvex_materialization_session *session, const char *name)
{
    unsigned long long index;
    for (index = 0ull;; ++index) {
        const yvex_materialized_tensor_binding *binding =
            yvex_materialization_session_tensor_at(session, index);
        if (!binding || strcmp(binding->name, name) == 0) return binding;
    }
}

static int weight_bind(yvex_minimax_h3_encoded_weight *weight,
                       const yvex_materialized_tensor_binding *binding,
                       unsigned char *encoded)
{
    if (!weight || !binding || !encoded || !binding->row_count) return 0;
    weight->encoded = encoded;
    weight->encoded_bytes = binding->encoded_bytes;
    weight->row_count = binding->row_count;
    weight->row_width = binding->row_width;
    weight->row_bytes = binding->encoded_bytes / binding->row_count;
    weight->qtype = binding->qtype;
    return 1;
}

static int selected_weights_load(
    yvex_materialization_session *session, unsigned char **arena_out,
    unsigned long long *arena_bytes_out,
    yvex_minimax_h3_encoded_weight external[YVEX_TRANSFORMER_JOINT_EXTERNAL_WEIGHT_COUNT],
    yvex_minimax_h3_encoded_weight blocks[YVEX_TRANSFORMER_JOINT_BLOCK_WEIGHT_COUNT],
    unsigned long long selected_block, char identity[65], yvex_error *err)
{
    const yvex_materialized_tensor_binding *bindings[
        YVEX_TRANSFORMER_JOINT_EXTERNAL_WEIGHT_COUNT + YVEX_TRANSFORMER_JOINT_BLOCK_WEIGHT_COUNT];
    char block_names[YVEX_TRANSFORMER_JOINT_BLOCK_WEIGHT_COUNT][96];
    yvex_materialization_failure failure;
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES], *arena;
    unsigned long long index, total = 0ull, cursor = 0ull;
    for (index = 0ull; index < YVEX_TRANSFORMER_JOINT_EXTERNAL_WEIGHT_COUNT; ++index)
        bindings[index] = binding_find(session, external_names[index]);
    for (index = 0ull; index < YVEX_TRANSFORMER_JOINT_BLOCK_WEIGHT_COUNT; ++index) {
        int length = snprintf(block_names[index], sizeof(block_names[index]),
                              "blocks.%llu.%s", selected_block, block_suffixes[index]);
        if (length < 0 || (size_t)length >= sizeof(block_names[index])) return YVEX_ERR_BOUNDS;
        bindings[YVEX_TRANSFORMER_JOINT_EXTERNAL_WEIGHT_COUNT + index] =
            binding_find(session, block_names[index]);
    }
    for (index = 0ull; index < sizeof(bindings) / sizeof(bindings[0]); ++index)
        if (!bindings[index] || !yvex_core_u64_add(total, bindings[index]->encoded_bytes, &total)) {
            yvex_error_set(err, YVEX_ERR_FORMAT, "minimax-h3.transformer-proof.binding",
                           "the selected Transformer weight set is unavailable");
            return YVEX_ERR_FORMAT;
        }
    arena = mmap(NULL, (size_t)total, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (arena == MAP_FAILED) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "minimax-h3.transformer-proof.arena",
                       "selected Transformer proof residency allocation failed");
        return YVEX_ERR_NOMEM;
    }
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.minimax-h3.transformer-envelope.proof.v1"))
        goto failed;
    for (index = 0ull; index < sizeof(bindings) / sizeof(bindings[0]); ++index) {
        const yvex_materialized_tensor_binding *binding = bindings[index];
        if (yvex_materialization_session_read(session, binding, 0ull, arena + cursor,
                                               (size_t)binding->encoded_bytes, &failure, err) != YVEX_OK ||
            !yvex_sha256_update_text(&hash, binding->name) ||
            !yvex_sha256_update_u64(&hash, binding->encoded_bytes) ||
            !yvex_sha256_update(&hash, arena + cursor, (size_t)binding->encoded_bytes))
            goto failed;
        if (index < YVEX_TRANSFORMER_JOINT_EXTERNAL_WEIGHT_COUNT) {
            if (!weight_bind(external + index, binding, arena + cursor)) goto failed;
        } else if (!weight_bind(blocks + index - YVEX_TRANSFORMER_JOINT_EXTERNAL_WEIGHT_COUNT,
                                binding, arena + cursor)) goto failed;
        cursor += binding->encoded_bytes;
    }
    if (!yvex_sha256_final(&hash, digest)) goto failed;
    yvex_sha256_hex(digest, identity);
    *arena_out = arena;
    *arena_bytes_out = total;
    return YVEX_OK;
failed:
    munmap(arena, (size_t)total);
    if (yvex_error_code(err) == YVEX_OK)
        yvex_error_set(err, YVEX_ERR_STATE, "minimax-h3.transformer-proof.identity",
                       "selected Transformer proof identity could not be sealed");
    return yvex_error_code(err);
}

static int file_read(const char *path, float *values, unsigned long long count)
{
    FILE *file = fopen(path, "rb");
    int valid;
    if (!file) return 0;
    valid = fread(values, sizeof(*values), (size_t)count, file) == count && fgetc(file) == EOF;
    if (fclose(file) != 0) valid = 0;
    return valid;
}

static int file_read_u32(const char *path, unsigned int *values, unsigned long long count)
{
    FILE *file = fopen(path, "rb");
    int valid;
    if (!file) return 0;
    valid = fread(values, sizeof(*values), (size_t)count, file) == count && fgetc(file) == EOF;
    if (fclose(file) != 0) valid = 0;
    return valid;
}

static int fixture_path(char output[1024], const char *root, const char *name)
{
    int length;
    if (!root || !root[0] || !name || !name[0]) return 0;
    length = snprintf(output, 1024u, "%s/%s", root, name);
    return length > 0 && length < 1024;
}

static int file_write(const char *path, const float *values, unsigned long long count)
{
    FILE *file = fopen(path, "wb");
    int valid;
    if (!file) return 0;
    valid = fwrite(values, sizeof(*values), (size_t)count, file) == count;
    if (fclose(file) != 0) valid = 0;
    return valid;
}

static int file_write_atomic(const char *path, const float *values, unsigned long long count)
{
    char temporary[1024];
    FILE *file;
    int length = snprintf(temporary, sizeof(temporary), "%s.part", path), valid;
    if (length < 0 || (size_t)length >= sizeof(temporary)) return 0;
    file = fopen(temporary, "wb");
    if (!file) return 0;
    valid = fwrite(values, sizeof(*values), (size_t)count, file) == count &&
            fflush(file) == 0 && fsync(fileno(file)) == 0;
    if (fclose(file) != 0) valid = 0;
    if (valid && rename(temporary, path) != 0) valid = 0;
    if (!valid) remove(temporary);
    return valid;
}

typedef struct {
    const char *root;
    unsigned long long step_count;
} latent_checkpoint_context;

static int latent_checkpoint_write(
    const latent_checkpoint_context *context, const char *domain, const char *kind,
    unsigned long long step, const float *values, unsigned long long count)
{
    char path[1024];
    int length = snprintf(path, sizeof(path), "%s/%s.%s.step-%03llu.f32",
                          context->root, domain, kind, step);
    return length > 0 && (size_t)length < sizeof(path) && file_write_atomic(path, values, count);
}

static int latent_checkpoint_observe(
    void *opaque, const yvex_runtime_latent_observation *observation, yvex_error *err)
{
    const latent_checkpoint_context *context = opaque;
    unsigned long long middle = context ? context->step_count / 2ull : 0ull;
    int selected = 0, valid;
    const char *kind;
    if (!context || !context->root || !observation) return YVEX_OK;
    if (observation->stage == YVEX_RUNTIME_LATENT_OBSERVATION_INITIAL) {
        selected = 1; kind = "initial";
    } else if (observation->stage == YVEX_RUNTIME_LATENT_OBSERVATION_EVALUATED &&
               (observation->completed_steps == 0ull ||
                observation->completed_steps == middle ||
                observation->completed_steps + 1ull == context->step_count)) {
        selected = 1; kind = "velocity";
    } else if (observation->stage == YVEX_RUNTIME_LATENT_OBSERVATION_ADVANCED &&
               (observation->completed_steps == 1ull ||
                observation->completed_steps == middle + 1ull ||
                observation->completed_steps == context->step_count)) {
        selected = 1; kind = "state";
    } else if (observation->stage == YVEX_RUNTIME_LATENT_OBSERVATION_FINAL) {
        selected = 1; kind = "final";
    } else {
        return YVEX_OK;
    }
    valid = latent_checkpoint_write(context, "video", kind, observation->completed_steps,
                                    kind[0] == 'v' ? observation->video_velocity
                                                   : observation->video_state,
                                    observation->video_values) &&
            latent_checkpoint_write(context, "audio", kind, observation->completed_steps,
                                    kind[0] == 'v' ? observation->audio_velocity
                                                   : observation->audio_state,
                                    observation->audio_values);
    if (!selected || valid) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    yvex_error_set(err, YVEX_ERR_IO, "minimax-h3.latent-checkpoint",
                   "a selected latent checkpoint could not be published atomically");
    return YVEX_ERR_IO;
}

typedef struct {
    const char *root;
} block_checkpoint_context;

static int block_checkpoint_observe(
    void *opaque, const yvex_transformer_joint_block_observation *observation,
    yvex_error *err)
{
    const block_checkpoint_context *context = opaque;
    const char *every_block = getenv("YVEX_MINIMAX_H3_BLOCK_CHECKPOINT_EVERY");
    char path[1024];
    unsigned long long completed;
    int length;
    if (!context || !context->root || !observation) return YVEX_OK;
    completed = observation->completed_blocks;
    if ((!every_block || strcmp(every_block, "1") != 0) &&
        completed != 1ull && completed != 2ull && completed != 5ull &&
        completed != 10ull && completed != 20ull && completed != 30ull &&
        completed != 40ull && completed != 50ull)
        return YVEX_OK;
    length = snprintf(path, sizeof(path), "%s/hidden.block-%03llu.f32",
                      context->root, completed);
    if (length > 0 && (size_t)length < sizeof(path) &&
        file_write_atomic(path, observation->values, observation->value_count)) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    yvex_error_set(err, YVEX_ERR_IO, "minimax-h3.block-checkpoint",
                   "a selected Omni block checkpoint could not be published atomically");
    return YVEX_ERR_IO;
}

typedef struct {
    const char *root;
    yvex_transformer_joint_scope scope;
    unsigned long long block;
} stage_checkpoint_context;

static int stage_checkpoint_observe(
    void *opaque, const yvex_transformer_joint_stage_observation *observation,
    yvex_error *err)
{
    const stage_checkpoint_context *context = opaque;
    char path[1024];
    int length;
    if (!context || !context->root || !observation) return YVEX_OK;
    length = snprintf(path, sizeof(path), "%s/scope-%u-block-%03llu-stage-%02u.f32",
                      context->root, (unsigned int)observation->scope,
                      observation->block, (unsigned int)observation->stage);
    if (length > 0 && (size_t)length < sizeof(path) &&
        file_write_atomic(path, observation->values, observation->value_count)) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    yvex_error_set(err, YVEX_ERR_IO, "minimax-h3.stage-checkpoint",
                   "a selected Omni stage checkpoint could not be published atomically");
    return YVEX_ERR_IO;
}

static int compare(const char *name, const float *reference, const float *output,
                   unsigned long long count, unsigned long long block_count)
{
    double error = 0.0, reference_norm = 0.0, output_norm = 0.0, dot = 0.0;
    double relative, cosine, scaled;
    float maximum = 0.0f, scale = 0.0f;
    /* The full stack accumulates legal CUDA reduction-order drift; cosine and scale bounds
       prevent that aggregate allowance from hiding a semantic block error. */
    double maximum_relative = block_count == 1ull
                                  ? (strcmp(name, "video") == 0 ? 0.02 : 0.01)
                                  : 0.04;
    double minimum_cosine = block_count == 1ull
                                ? (strcmp(name, "video") == 0 ? 0.9998 : 0.9999)
                                : 0.9995;
    double maximum_scaled = block_count == 1ull
                                ? (strcmp(name, "video") == 0 ? 0.02 : 0.01)
                                : 0.04;
    unsigned long long index;
    for (index = 0ull; index < count; ++index) {
        float difference = fabsf(reference[index] - output[index]);
        if (difference > maximum) maximum = difference;
        if (fabsf(reference[index]) > scale) scale = fabsf(reference[index]);
        error += (double)difference * difference;
        reference_norm += (double)reference[index] * reference[index];
        output_norm += (double)output[index] * output[index];
        dot += (double)reference[index] * output[index];
    }
    relative = sqrt(error / reference_norm);
    cosine = dot / sqrt(reference_norm * output_norm);
    scaled = maximum / scale;
    printf("%s_max_absolute=%.9g %s_relative_l2=%.9g %s_cosine=%.12g %s_scaled_absolute=%.9g\n",
           name, maximum, name, relative, name, cosine, name, scaled);
    return relative <= maximum_relative && cosine >= minimum_cosine &&
           scaled <= maximum_scaled;
}

static int refusal_checks(
    yvex_backend *backend,
    const yvex_minimax_h3_encoded_weight *external,
    const yvex_minimax_h3_encoded_weight *blocks, const char *identity,
    unsigned long long arena_bytes, const yvex_minimax_h3_omni_transformer_request *valid,
    yvex_error *err)
{
    yvex_minimax_h3_omni_transformer_request request = *valid;
    yvex_minimax_h3_omni_transformer_result result;
    unsigned int duplicate_text[1] = {0u};
    float invalid_timestep[1] = {1.25f};
    int rc;
    request.text_indices = duplicate_text;
    rc = yvex_cuda_transformer_joint_execute(
        backend, external, blocks, identity, arena_bytes, &request, &result, err);
    if (rc != YVEX_ERR_FORMAT || result.complete) return YVEX_ERR_STATE;
    request = *valid;
    request.timesteps = invalid_timestep;
    rc = yvex_cuda_transformer_joint_execute(
        backend, external, blocks, identity, arena_bytes, &request, &result, err);
    if (rc != YVEX_ERR_FORMAT || result.complete) return YVEX_ERR_STATE;
    request = *valid;
    request.video_output_capacity--;
    rc = yvex_cuda_transformer_joint_execute(
        backend, external, blocks, identity, arena_bytes, &request, &result, err);
    if (rc != YVEX_ERR_INVALID_ARG || result.complete) return YVEX_ERR_STATE;
    request = *valid;
    request.video_output_physical = valid->audio_output_physical;
    rc = yvex_cuda_transformer_joint_execute(
        backend, external, blocks, identity, arena_bytes, &request, &result, err);
    if (rc != YVEX_ERR_INVALID_ARG || result.complete) return YVEX_ERR_STATE;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int specialize_outputs(
    const yvex_transformer_joint_recipe *recipe,
    yvex_transformer_linear_physical_plan *video,
    yvex_transformer_linear_physical_plan *audio, yvex_error *err)
{
    yvex_runtime_av_generation_request request = {
        .component_backend = YVEX_BACKEND_KIND_CUDA};
    int rc = yvex_runtime_media_request_specialize(
        &request, recipe ? recipe->identity_domain : NULL,
        recipe ? &recipe->video_output : NULL,
        recipe ? &recipe->audio_output : NULL, err);
    if (rc == YVEX_OK) {
        *video = request.video_output_specialization;
        *audio = request.audio_output_specialization;
    }
    return rc;
}

static int execute(const yvex_artifact *artifact, const yvex_gguf *gguf,
                   const yvex_tensor_table *tensors,
                   yvex_minimax_h3_omni_transformer_request *request,
                   yvex_minimax_h3_omni_transformer_result *result, yvex_error *err)
{
    const yvex_minimax_h3_graph_api *graph = yvex_graph_register_minimax_h3();
    yvex_complete_artifact_admission admission;
    yvex_artifact_admission_failure admission_failure;
    yvex_materialization_options options;
    yvex_materialization_failure failure;
    yvex_materialization_plan *plan = NULL;
    yvex_materialization_session *session = NULL;
    yvex_minimax_h3_encoded_weight external[YVEX_TRANSFORMER_JOINT_EXTERNAL_WEIGHT_COUNT] = {{0}};
    yvex_minimax_h3_encoded_weight blocks[YVEX_TRANSFORMER_JOINT_BLOCK_WEIGHT_COUNT] = {{0}};
    yvex_backend_options backend_options = {0};
    yvex_backend_tensor_desc descriptor = {0};
    yvex_backend *backend = NULL;
    yvex_device_tensor *resident = NULL;
    unsigned char *arena = NULL, *registered = NULL;
    unsigned long long arena_bytes = 0ull;
    char identity[65] = {0};
    int attached = 0, rc, cleanup_rc;
    yvex_error cleanup;
    request->recipe = graph ? graph->omni_recipe : NULL;
    if (!graph || !request->recipe) {
        yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "minimax-h3.transformer-proof",
                       "the admitted joint Transformer recipe is unavailable");
        rc = YVEX_ERR_UNSUPPORTED;
    } else {
        rc = specialize_outputs(
            request->recipe,
            &request->video_output_physical, &request->audio_output_physical, err);
    }
    if (rc == YVEX_OK) {
        rc = graph->component_admit(
            "transformer", artifact, gguf, tensors, NULL, &admission, NULL,
            &admission_failure, err);
    }
    yvex_materialization_options_default(&options);
    options.max_chunk_bytes = 64ull * 1024ull * 1024ull;
    if (rc == YVEX_OK) rc = yvex_materialization_plan_build(
        &plan, &admission, artifact, gguf, tensors, NULL, &options, &failure, err);
    if (rc == YVEX_OK) rc = yvex_materialization_session_open(
        &session, plan, artifact, &options, &failure, err);
    if (rc == YVEX_OK) rc = yvex_materialization_session_commit(session, &failure, err);
    if (rc == YVEX_OK) rc = selected_weights_load(
        session, &arena, &arena_bytes, external, blocks, 0ull, identity, err);
    backend_options.kind = YVEX_BACKEND_KIND_CUDA;
    backend_options.memory_limit_bytes = 16ull * 1024ull * 1024ull * 1024ull;
    if (rc == YVEX_OK) rc = yvex_backend_open(&backend, &backend_options, err);
    descriptor.name = "minimax-h3-transformer-envelope-proof-residency";
    descriptor.dtype = YVEX_DTYPE_I8;
    descriptor.rank = 1u;
    descriptor.dims[0] = descriptor.bytes = arena_bytes;
    registered = arena;
    if (rc == YVEX_OK)
        rc = yvex_backend_resident_alloc(backend, &descriptor, &resident, &registered, err);
    if (rc == YVEX_OK && registered != arena) rc = YVEX_ERR_STATE;
    if (rc == YVEX_OK) {
        rc = yvex_backend_resident_attach(backend, arena, arena_bytes, resident, 1ull, err);
        attached = rc == YVEX_OK;
    }
    if (rc == YVEX_OK)
        rc = refusal_checks(backend, external, blocks, identity, arena_bytes,
                            request, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_joint_execute(
            backend, external, blocks, identity, arena_bytes, request, result, err);
    if (attached) {
        yvex_error_clear(&cleanup);
        cleanup_rc = yvex_backend_resident_detach(backend, &cleanup);
        if (cleanup_rc != YVEX_OK) { rc = cleanup_rc; if (err) *err = cleanup; }
    }
    if (resident) {
        yvex_error_clear(&cleanup);
        cleanup_rc = yvex_backend_tensor_release(backend, &resident, &cleanup);
        if (cleanup_rc != YVEX_OK) { rc = cleanup_rc; if (err) *err = cleanup; }
    }
    yvex_error_clear(&cleanup);
    cleanup_rc = yvex_backend_close_checked(&backend, &cleanup);
    if (cleanup_rc != YVEX_OK) { rc = cleanup_rc; if (err) *err = cleanup; }
    if (arena) munmap(arena, (size_t)arena_bytes);
    yvex_materialization_session_close(session);
    yvex_materialization_plan_close(plan);
    return rc;
}

static int execute_selected_block(
    const char *artifact_path, const char *fixture_root, const char *hidden_path,
    const char *time_path, const char *output_path, unsigned long long video_rows,
    unsigned long long audio_rows, unsigned long long text_rows,
    unsigned long long timestep_count, unsigned long long selected_block)
{
    const yvex_minimax_h3_graph_api *graph = yvex_graph_register_minimax_h3();
    yvex_artifact_options artifact_options = {.path = artifact_path, .readonly = 1};
    yvex_artifact *artifact = NULL;
    yvex_gguf *gguf = NULL;
    yvex_tensor_table *tensors = NULL;
    yvex_complete_artifact_admission admission;
    yvex_artifact_admission_failure admission_failure;
    yvex_materialization_options materialization_options;
    yvex_materialization_failure materialization_failure;
    yvex_materialization_plan *plan = NULL;
    yvex_materialization_session *session = NULL;
    yvex_minimax_h3_encoded_weight external[YVEX_TRANSFORMER_JOINT_EXTERNAL_WEIGHT_COUNT] = {{0}};
    yvex_minimax_h3_encoded_weight block[YVEX_TRANSFORMER_JOINT_BLOCK_WEIGHT_COUNT] = {{0}};
    yvex_backend_options backend_options = {0};
    yvex_backend_tensor_desc descriptor = {0};
    yvex_backend *backend = NULL;
    yvex_device_tensor *resident = NULL;
    yvex_transformer_joint_block_result result = {0};
    stage_checkpoint_context checkpoint = {
        .root = getenv("YVEX_MINIMAX_H3_STAGE_CHECKPOINT_ROOT"),
        .scope = YVEX_TRANSFORMER_JOINT_SCOPE_OMNI,
        .block = 1ull};
    yvex_transformer_joint_block_options options = {
        .stage_observer = checkpoint.root ? stage_checkpoint_observe : NULL,
        .stage_observer_context = checkpoint.root ? &checkpoint : NULL,
        .observed_stage_block = 1ull,
        .observed_stage_scope = YVEX_TRANSFORMER_JOINT_SCOPE_OMNI,
        .observed_stage = YVEX_TRANSFORMER_JOINT_STAGE_COUNT};
    unsigned char *arena = NULL, *registered = NULL;
    float *hidden = NULL, *time = NULL, *positions = NULL, *output = NULL, *reference = NULL;
    unsigned int *tags = NULL, *timestep_indices = NULL, *adaln_indices = NULL;
    unsigned long long arena_bytes = 0ull, rows, values, row, bytes;
    char identity[65] = {0}, positions_path[1024], tags_path[1024];
    char timestep_indices_path[1024], reference_name[64], reference_path[1024];
    yvex_error err, cleanup;
    int attached = 0, rc = YVEX_OK, cleanup_rc, matches = 0;
    if (!graph || !graph->omni_recipe || selected_block >= 50ull || !video_rows ||
        !audio_rows || !text_rows || !timestep_count || timestep_count > 64ull ||
        !yvex_core_u64_add(video_rows, audio_rows, &rows) ||
        !yvex_core_u64_add(rows, text_rows, &rows) ||
        !yvex_core_u64_mul(rows, 5376ull, &values) ||
        !yvex_core_u64_mul(values, sizeof(float), &bytes) || bytes > SIZE_MAX ||
        !fixture_path(positions_path, fixture_root, "positions.f32") ||
        !fixture_path(tags_path, fixture_root, "tags.u32") ||
        !fixture_path(timestep_indices_path, fixture_root, "timestep_indices.u32") ||
        snprintf(reference_name, sizeof(reference_name), "hidden.block-%03llu.f32",
                 selected_block + 1ull) <= 0 ||
        !fixture_path(reference_path, fixture_root, reference_name) ||
        !(hidden = malloc((size_t)bytes)) || !(output = malloc((size_t)bytes)) ||
        !(reference = malloc((size_t)bytes)) ||
        !(time = malloc((size_t)(timestep_count * 2688ull * sizeof(float)))) ||
        !(positions = malloc((size_t)(rows * 3ull * sizeof(float)))) ||
        !(tags = malloc((size_t)(rows * sizeof(*tags)))) ||
        !(timestep_indices = malloc((size_t)(rows * sizeof(*timestep_indices)))) ||
        !(adaln_indices = malloc((size_t)(rows * sizeof(*adaln_indices))))) {
        rc = YVEX_ERR_INVALID_ARG;
        goto cleanup;
    }
    if (!file_read(hidden_path, hidden, values) ||
        !file_read(time_path, time, timestep_count * 2688ull) ||
        !file_read(positions_path, positions, rows * 3ull) ||
        !file_read_u32(tags_path, tags, rows) ||
        !file_read_u32(timestep_indices_path, timestep_indices, rows) ||
        !file_read(reference_path, reference, values)) {
        rc = YVEX_ERR_IO;
        goto cleanup;
    }
    for (row = 0ull; row < rows; ++row) {
        if (tags[row] >= 3u || timestep_indices[row] >= timestep_count) {
            rc = YVEX_ERR_FORMAT;
            goto cleanup;
        }
        adaln_indices[row] = timestep_indices[row] * 3u + tags[row];
    }
    yvex_error_clear(&err);
    rc = yvex_artifact_open(&artifact, &artifact_options, &err);
    if (rc == YVEX_OK) rc = yvex_gguf_open(&gguf, artifact, &err);
    if (rc == YVEX_OK) rc = yvex_tensor_table_from_gguf(&tensors, gguf, &err);
    if (rc == YVEX_OK)
        rc = graph->component_admit(
            "transformer", artifact, gguf, tensors, NULL, &admission, NULL,
            &admission_failure, &err);
    yvex_materialization_options_default(&materialization_options);
    materialization_options.max_chunk_bytes = 64ull * 1024ull * 1024ull;
    if (rc == YVEX_OK)
        rc = yvex_materialization_plan_build(
            &plan, &admission, artifact, gguf, tensors, NULL, &materialization_options,
            &materialization_failure, &err);
    if (rc == YVEX_OK)
        rc = yvex_materialization_session_open(
            &session, plan, artifact, &materialization_options, &materialization_failure, &err);
    if (rc == YVEX_OK)
        rc = yvex_materialization_session_commit(session, &materialization_failure, &err);
    if (rc == YVEX_OK)
        rc = selected_weights_load(session, &arena, &arena_bytes, external, block,
                                   selected_block, identity, &err);
    backend_options.kind = YVEX_BACKEND_KIND_CUDA;
    backend_options.memory_limit_bytes = 16ull * 1024ull * 1024ull * 1024ull;
    if (rc == YVEX_OK) rc = yvex_backend_open(&backend, &backend_options, &err);
    descriptor.name = "minimax-h3-selected-block-proof-residency";
    descriptor.dtype = YVEX_DTYPE_I8;
    descriptor.rank = 1u;
    descriptor.dims[0] = descriptor.bytes = arena_bytes;
    registered = arena;
    if (rc == YVEX_OK)
        rc = yvex_backend_resident_alloc(backend, &descriptor, &resident, &registered, &err);
    if (rc == YVEX_OK && registered != arena) rc = YVEX_ERR_STATE;
    if (rc == YVEX_OK) {
        rc = yvex_backend_resident_attach(backend, arena, arena_bytes, resident, 1ull, &err);
        attached = rc == YVEX_OK;
    }
    options.inv_freq = (const float *)external[YVEX_TRANSFORMER_JOINT_ROPE_INV_FREQ].encoded;
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_joint_blocks_execute(
            backend, graph->omni_recipe, block, 1ull, identity, arena_bytes, hidden,
            time, timestep_count, positions, adaln_indices, rows, output, values,
            &result, &options, &err);
    if (rc == YVEX_OK && !file_write(output_path, output, values)) rc = YVEX_ERR_IO;
    if (rc == YVEX_OK) matches = compare("block", reference, output, values, 1ull);
    if (rc == YVEX_OK)
        printf("selected_block=%llu rows=%llu oracle_match=%s execution_identity=%s\n",
               selected_block + 1ull, rows, matches ? "yes" : "no", result.execution_identity);
    else
        fprintf(stderr, "selected_block=refused where=%s message=%s\n",
                yvex_error_where(&err), yvex_error_message(&err));
cleanup:
    if (attached) {
        yvex_error_clear(&cleanup);
        cleanup_rc = yvex_backend_resident_detach(backend, &cleanup);
        if (cleanup_rc != YVEX_OK && rc == YVEX_OK) rc = cleanup_rc;
    }
    if (resident) {
        yvex_error_clear(&cleanup);
        cleanup_rc = yvex_backend_tensor_release(backend, &resident, &cleanup);
        if (cleanup_rc != YVEX_OK && rc == YVEX_OK) rc = cleanup_rc;
    }
    yvex_error_clear(&cleanup);
    cleanup_rc = yvex_backend_close_checked(&backend, &cleanup);
    if (cleanup_rc != YVEX_OK && rc == YVEX_OK) rc = cleanup_rc;
    if (arena) munmap(arena, (size_t)arena_bytes);
    yvex_materialization_session_close(session);
    yvex_materialization_plan_close(plan);
    yvex_tensor_table_close(tensors);
    yvex_gguf_close(gguf);
    yvex_artifact_close(artifact);
    free(adaln_indices); free(timestep_indices); free(tags); free(positions);
    free(time); free(reference); free(output); free(hidden);
    return rc == YVEX_OK ? 0 : 1;
}

static int execute_artifact(const yvex_artifact *artifact, const yvex_gguf *gguf,
                            const yvex_tensor_table *tensors,
                            yvex_minimax_h3_omni_transformer_request *request,
                            yvex_minimax_h3_omni_transformer_result *result, yvex_error *err)
{
    const yvex_minimax_h3_graph_api *graph = yvex_graph_register_minimax_h3();
    yvex_complete_artifact_admission admission;
    yvex_artifact_admission_failure failure;
    yvex_runtime_component_session *session = NULL;
    yvex_component_execution component = {0};
    yvex_error cleanup;
    int rc;
    request->recipe = graph ? graph->omni_recipe : NULL;
    if (!graph || !request->recipe) {
        yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "minimax-h3.transformer-proof",
                       "the admitted joint Transformer recipe is unavailable");
        rc = YVEX_ERR_UNSUPPORTED;
    } else {
        rc = specialize_outputs(
            request->recipe,
            &request->video_output_physical, &request->audio_output_physical, err);
    }
    if (rc == YVEX_OK) {
        rc = graph->component_admit(
            "transformer", artifact, gguf, tensors, NULL, &admission, NULL, &failure, err);
    }
    if (rc == YVEX_OK)
        rc = yvex_runtime_component_session_open(
            &session, &admission, artifact, gguf, tensors, YVEX_BACKEND_KIND_CUDA,
            80ull * 1024ull * 1024ull * 1024ull, 16ull * 1024ull * 1024ull * 1024ull, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_component_session_borrow(session, &component, err);
    if (rc == YVEX_OK)
        rc = graph->transformer_component_execute(&component, request, result, err);
    yvex_error_clear(&cleanup);
    if (yvex_runtime_component_session_close(&session, &cleanup) != YVEX_OK && rc == YVEX_OK) {
        rc = yvex_error_code(&cleanup);
        if (err) *err = cleanup;
    }
    return rc;
}

static int float_identity(const char *domain, const float *values,
                          unsigned long long count, char output[65])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, domain) || !yvex_sha256_update_u64(&hash, count)) return 0;
    for (index = 0ull; index < count; ++index) {
        uint32_t bits;
        memcpy(&bits, values + index, sizeof(bits));
        if (!yvex_sha256_update_u64(&hash, bits)) return 0;
    }
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

static int execute_latent(const char *path, const char *conditioning_path,
                          unsigned long long block_count, unsigned int steps)
{
    const yvex_minimax_h3_graph_api *graph = yvex_graph_register_minimax_h3();
    const yvex_minimax_h3_latent_normalization *normalization =
        yvex_model_register_minimax_h3()->latent_normalization();
    yvex_artifact_options options = {.path = path, .readonly = 1};
    yvex_artifact *artifact = NULL;
    yvex_gguf *gguf = NULL;
    yvex_tensor_table *tensors = NULL;
    yvex_complete_artifact_admission admission;
    yvex_artifact_admission_failure failure;
    yvex_runtime_component_session *session = NULL;
    yvex_component_execution component = {0};
    yvex_component_resource_summary resources = {0};
    const int optional_fallback =
        getenv("YVEX_TEST_JOINT_PREPARED_OPTIONAL_FAILURE") != NULL;
    yvex_minimax_h3_t2va_plan plan;
    yvex_runtime_av_layout_result layout_result;
    yvex_runtime_latent_result latent_result;
    yvex_minimax_h3_t2va_omni_result omni_result;
    yvex_runtime_av_unpack_result unpack_result;
    float conditioning[CONDITION_VALUES], positions[57], video[192], audio[512];
    float video_input[192], audio_input[512];
    unsigned int tags[19], video_indices[2], audio_indices[16], text_indices[1];
    unsigned int timestep_indices[19];
    yvex_runtime_av_layout_output layout = {
        positions, 57ull, tags, video_indices, audio_indices, text_indices,
        19ull, 2ull, 16ull, 1ull,
    };
    yvex_runtime_av_unpack_output component_inputs = {
        video_input, audio_input, 192ull, 512ull,
    };
    yvex_runtime_av_unpack_request unpack = {
        .schema_version = YVEX_RUNTIME_AV_UNPACK_SCHEMA_V1,
        .video_row_capacity = 192ull, .audio_row_capacity = 512ull,
        .maximum_workspace_bytes = (192ull + 512ull) * sizeof(float),
    };
    yvex_media_plan_request plan_request = {
        .schema_version = YVEX_RUNTIME_AV_PLAN_SCHEMA_V1,
        .text_tokens = 1ull, .width = 32ull, .height = 32ull, .frames = 5ull,
        .inference_steps = steps,
    };
    yvex_media_layout_request layout_request = {0};
    yvex_minimax_h3_t2va_omni_context context = {0};
    yvex_transformer_linear_physical_plan video_specialization = {0};
    yvex_transformer_linear_physical_plan audio_specialization = {0};
    char conditioning_identity[65];
    yvex_error err, cleanup;
    int rc, cleanup_rc;
    if (!path || !conditioning_path || !normalization || !block_count || block_count > 50ull ||
        !steps || steps > 64u || !file_read(conditioning_path, conditioning, CONDITION_VALUES) ||
        !float_identity("yvex.minimax-h3.conditioning.fixture.v1", conditioning,
                        CONDITION_VALUES, conditioning_identity))
        return 2;
    yvex_error_clear(&err);
    rc = yvex_artifact_open(&artifact, &options, &err);
    if (rc == YVEX_OK) rc = yvex_gguf_open(&gguf, artifact, &err);
    if (rc == YVEX_OK) rc = yvex_tensor_table_from_gguf(&tensors, gguf, &err);
    if (rc == YVEX_OK)
        rc = graph->component_admit("transformer", artifact, gguf, tensors,
                                    NULL, &admission, NULL, &failure, &err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_component_session_open(
            &session, &admission, artifact, gguf, tensors, YVEX_BACKEND_KIND_CUDA,
            80ull * 1024ull * 1024ull * 1024ull, 16ull * 1024ull * 1024ull * 1024ull, &err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_component_session_borrow(session, &component, &err);
    if (rc == YVEX_OK) rc = graph->t2va_plan_build(&plan, &plan_request, &err);
    layout_request.plan = &plan;
    if (rc == YVEX_OK)
        rc = graph->t2va_layout_build(&layout_request, &layout, &layout_result, &err);
    if (rc == YVEX_OK)
        rc = specialize_outputs(
            graph->omni_recipe,
            &video_specialization, &audio_specialization, &err);
    context.transformer_component = &component;
    context.conditioning = conditioning;
    context.conditioning_capacity = CONDITION_VALUES;
    context.layout = &layout;
    context.layout_result = &layout_result;
    context.video_output_specialization = &video_specialization;
    context.audio_output_specialization = &audio_specialization;
    context.timestep_indices = timestep_indices;
    context.timestep_capacity = 19ull;
    context.block_count = block_count;
    context.conditioning_identity = conditioning_identity;
    if (rc == YVEX_OK)
        rc = graph->t2va_latent_execute(
            &plan, &context, 42ull, (192ull + 512ull) * sizeof(float) * 4ull,
            video, 192ull, audio, 512ull, &latent_result, &omni_result, &err);
    if (rc == YVEX_OK &&
        (!latent_result.completed || !omni_result.complete ||
         omni_result.model_evaluations != steps ||
         latent_result.transaction.state != YVEX_EXECUTION_TRANSACTION_COMMITTED ||
         latent_result.transaction.started_quanta != steps ||
         latent_result.transaction.completed_quanta != steps ||
         latent_result.transaction.safe_points != steps ||
         latent_result.transaction.retained_resources != 1ull)) {
        yvex_error_set(&err, YVEX_ERR_STATE, "minimax-h3.latent-proof",
                       "the exact resident latent iteration did not complete");
        rc = YVEX_ERR_STATE;
    }
    if (rc == YVEX_OK)
        rc = yvex_component_execution_resource_summary(
            &component, &resources, &err);
    if (rc == YVEX_OK &&
        (!resources.ready || resources.retained_by_transaction ||
         resources.preparation_count != 1ull || resources.use_count != steps ||
         resources.reuse_count + 1ull != resources.use_count ||
         resources.rebuild_count || !resources.host_arena_bytes ||
         !resources.device_arena_bytes ||
         (optional_fallback
              ? (resources.request_ready || resources.condition_ready ||
                 resources.resource_count != 1ull ||
                 resources.request_prepared_bytes ||
                 resources.condition_prepared_bytes)
              : (!resources.request_ready || !resources.condition_ready ||
                 resources.resource_count != 3ull ||
                 !resources.request_prepared_bytes ||
                 !resources.condition_prepared_bytes)))) {
        yvex_error_set(&err, YVEX_ERR_STATE, "minimax-h3.resource-proof",
                       "the iterative request did not retain and reuse its prepared resources");
        rc = YVEX_ERR_STATE;
    }
    if (rc == YVEX_OK) {
        unpack.plan = &plan; unpack.video_rows = video; unpack.audio_rows = audio;
        unpack.video_channel_mean = normalization->video_mean;
        unpack.video_channel_std = normalization->video_std;
        unpack.audio_channel_mean = normalization->audio_mean;
        unpack.audio_channel_std = normalization->audio_std;
        unpack.video_channel_count = normalization->video_channels;
        unpack.audio_channel_count = normalization->audio_channels;
        unpack.latent_execution_identity = latent_result.execution_identity;
        rc = yvex_runtime_av_unpack(&unpack, &component_inputs, &unpack_result, &err);
    }
    if (rc == YVEX_OK && (!unpack_result.complete || unpack_result.video_values != 192ull ||
                          unpack_result.audio_values != 512ull)) {
        yvex_error_set(&err, YVEX_ERR_STATE, "minimax-h3.vae-input-proof",
                       "the final latents did not become exact VAE component inputs");
        rc = YVEX_ERR_STATE;
    }
    if (rc == YVEX_OK)
        printf("t2va_latent=accepted steps=%u blocks=%llu packed_rows=%llu\n"
               "kernel_launches=%llu peak_device_bytes=%llu\n"
               "execution_quanta=%llu safe_points=%llu transaction_setup_ns=%llu "
               "safe_point_ns=%llu\nresource_prepare_ns=%llu arena_host_bytes=%llu "
               "arena_device_bytes=%llu resource_uses=%llu resource_reuses=%llu "
               "last_execution_allocations=%llu\nrequest_prepared_bytes=%llu "
               "condition_prepared_bytes=%llu resource_count=%llu resource_rebuilds=%llu\n"
               "plan_identity=%s\n"
               "layout_identity=%s\nevaluator_identity=%s\nlatent_identity=%s\n"
               "transformer_chain_identity=%s\nresidency_identity=%s\n"
               "vae_input_identity=%s\n",
               steps, block_count, plan.packed_rows, omni_result.kernel_launches,
               omni_result.peak_device_bytes, latent_result.transaction.completed_quanta,
               latent_result.transaction.safe_points,
               latent_result.transaction.setup_nanoseconds,
               latent_result.transaction.safe_point_nanoseconds,
               resources.preparation_nanoseconds, resources.host_arena_bytes,
               resources.device_arena_bytes, resources.use_count,
               resources.reuse_count, resources.last_execution_allocation_events,
               resources.request_prepared_bytes,
               resources.condition_prepared_bytes, resources.resource_count,
               resources.rebuild_count,
               plan.identity, layout_result.layout_identity,
               omni_result.evaluator_identity, latent_result.execution_identity,
               omni_result.execution_chain_identity, omni_result.residency_identity,
               unpack_result.input_identity);
    else
        fprintf(stderr, "t2va_latent=refused where=%s message=%s\n",
                yvex_error_where(&err), yvex_error_message(&err));
    yvex_error_clear(&cleanup);
    cleanup_rc = yvex_runtime_component_session_close(&session, &cleanup);
    if (cleanup_rc != YVEX_OK && rc == YVEX_OK) rc = cleanup_rc;
    yvex_tensor_table_close(tensors);
    yvex_gguf_close(gguf);
    yvex_artifact_close(artifact);
    return rc == YVEX_OK ? 0 : 1;
}

typedef struct {
    float *video, *audio, *conditioning, *timesteps, *positions;
    float *video_output, *audio_output, *video_reference, *audio_reference;
    unsigned int *video_indices, *audio_indices, *text_indices;
    unsigned int *timestep_indices, *tags;
} request_fixture;

static void request_fixture_close(request_fixture *fixture)
{
    if (!fixture) return;
    free(fixture->tags); free(fixture->timestep_indices); free(fixture->text_indices);
    free(fixture->audio_indices); free(fixture->video_indices);
    free(fixture->audio_reference); free(fixture->video_reference);
    free(fixture->audio_output); free(fixture->video_output); free(fixture->positions);
    free(fixture->timesteps); free(fixture->conditioning); free(fixture->audio);
    free(fixture->video);
    memset(fixture, 0, sizeof(*fixture));
}

static int request_fixture_allocate(request_fixture *fixture,
                                    unsigned long long video_rows,
                                    unsigned long long audio_rows,
                                    unsigned long long text_rows,
                                    unsigned long long packed_rows,
                                    unsigned long long timestep_count)
{
#define ALLOCATE(target, count, type) \
    ((count) <= SIZE_MAX / sizeof(type) && \
     ((target) = calloc((size_t)(count), sizeof(type))) != NULL)
    if (!fixture || !ALLOCATE(fixture->video, video_rows * 96ull, float) ||
        !ALLOCATE(fixture->audio, audio_rows * 32ull, float) ||
        !ALLOCATE(fixture->conditioning, text_rows * 5120ull, float) ||
        !ALLOCATE(fixture->timesteps, timestep_count, float) ||
        !ALLOCATE(fixture->positions, packed_rows * 3ull, float) ||
        !ALLOCATE(fixture->video_output, video_rows * 96ull, float) ||
        !ALLOCATE(fixture->audio_output, audio_rows * 32ull, float) ||
        !ALLOCATE(fixture->video_reference, video_rows * 96ull, float) ||
        !ALLOCATE(fixture->audio_reference, audio_rows * 32ull, float) ||
        !ALLOCATE(fixture->video_indices, video_rows, unsigned int) ||
        !ALLOCATE(fixture->audio_indices, audio_rows, unsigned int) ||
        !ALLOCATE(fixture->text_indices, text_rows, unsigned int) ||
        !ALLOCATE(fixture->timestep_indices, packed_rows, unsigned int) ||
        !ALLOCATE(fixture->tags, packed_rows, unsigned int)) {
        request_fixture_close(fixture);
        return 0;
    }
#undef ALLOCATE
    return 1;
}

static int request_fixture_read(request_fixture *fixture, const char *root,
                                unsigned long long video_rows,
                                unsigned long long audio_rows,
                                unsigned long long text_rows,
                                unsigned long long packed_rows,
                                unsigned long long timestep_count)
{
    static const char *const names[] = {
        "video.f32", "audio.f32", "conditioning.f32", "timesteps.f32", "positions.f32",
        "video.oracle.f32", "audio.oracle.f32", "video_indices.u32", "audio_indices.u32",
        "text_indices.u32", "timestep_indices.u32", "tags.u32",
    };
    char paths[sizeof(names) / sizeof(names[0])][1024];
    unsigned long long index;
    for (index = 0ull; index < sizeof(names) / sizeof(names[0]); ++index)
        if (!fixture_path(paths[index], root, names[index])) return 0;
    return file_read(paths[0], fixture->video, video_rows * 96ull) &&
           file_read(paths[1], fixture->audio, audio_rows * 32ull) &&
           file_read(paths[2], fixture->conditioning, text_rows * 5120ull) &&
           file_read(paths[3], fixture->timesteps, timestep_count) &&
           file_read(paths[4], fixture->positions, packed_rows * 3ull) &&
           file_read(paths[5], fixture->video_reference, video_rows * 96ull) &&
           file_read(paths[6], fixture->audio_reference, audio_rows * 32ull) &&
           file_read_u32(paths[7], fixture->video_indices, video_rows) &&
           file_read_u32(paths[8], fixture->audio_indices, audio_rows) &&
           file_read_u32(paths[9], fixture->text_indices, text_rows) &&
           file_read_u32(paths[10], fixture->timestep_indices, packed_rows) &&
           file_read_u32(paths[11], fixture->tags, packed_rows);
}

static int execute_request_fixture(
    const char *artifact_path, const char *fixture_root, const char *video_output_path,
    const char *audio_output_path, unsigned long long video_rows,
    unsigned long long audio_rows, unsigned long long text_rows,
    unsigned long long block_count, unsigned long long timestep_count)
{
    yvex_artifact_options options = {.path = artifact_path, .readonly = 1};
    yvex_artifact *artifact = NULL;
    yvex_tensor_table *tensors = NULL;
    yvex_gguf *gguf = NULL;
    yvex_minimax_h3_omni_transformer_request request = {0};
    yvex_minimax_h3_omni_transformer_result result = {0};
    request_fixture fixture = {0};
    block_checkpoint_context checkpoint = {
        .root = getenv("YVEX_MINIMAX_H3_BLOCK_CHECKPOINT_ROOT")};
    stage_checkpoint_context stage_checkpoint = {
        .root = getenv("YVEX_MINIMAX_H3_STAGE_CHECKPOINT_ROOT")};
    const char *stage_block = getenv("YVEX_MINIMAX_H3_STAGE_CHECKPOINT_BLOCK");
    const char *stage_value = getenv("YVEX_MINIMAX_H3_STAGE_CHECKPOINT_STAGE");
    const char *stage_scope = getenv("YVEX_MINIMAX_H3_STAGE_CHECKPOINT_SCOPE");
    char *stage_block_end = NULL;
    char *stage_value_end = NULL;
    yvex_error err;
    unsigned long long packed_rows, video_values, audio_values;
    int video_matches = 0, audio_matches = 0;
    int rc = YVEX_OK;
    if (stage_checkpoint.root) {
        unsigned long long selected_stage = stage_value
                                                ? strtoull(stage_value, &stage_value_end, 10)
                                                : YVEX_TRANSFORMER_JOINT_STAGE_COUNT;
        stage_checkpoint.block = stage_block ? strtoull(stage_block, &stage_block_end, 10) : 0ull;
        stage_checkpoint.scope = stage_scope && strcmp(stage_scope, "refiner") == 0
                                     ? YVEX_TRANSFORMER_JOINT_SCOPE_REFINER
                                     : YVEX_TRANSFORMER_JOINT_SCOPE_OMNI;
        if (!stage_block || !stage_block_end || *stage_block_end ||
            (!stage_checkpoint.block &&
             stage_checkpoint.scope != YVEX_TRANSFORMER_JOINT_SCOPE_REFINER) ||
            (stage_scope && strcmp(stage_scope, "omni") != 0 &&
             strcmp(stage_scope, "refiner") != 0) ||
            (stage_value && (!stage_value_end || *stage_value_end)) ||
            selected_stage > YVEX_TRANSFORMER_JOINT_STAGE_COUNT)
            return 2;
        request.observed_stage_scope = stage_checkpoint.scope;
        request.observed_stage = (yvex_transformer_joint_stage)selected_stage;
    }
    if (!video_rows || !audio_rows || !text_rows || !block_count || block_count > 50ull ||
        !timestep_count || timestep_count > 64ull ||
        !yvex_core_u64_add(video_rows, audio_rows, &packed_rows) ||
        !yvex_core_u64_add(packed_rows, text_rows, &packed_rows) ||
        packed_rows > YVEX_MINIMAX_H3_OMNI_MAX_PACKED_ROWS ||
        !yvex_core_u64_mul(video_rows, 96ull, &video_values) ||
        !yvex_core_u64_mul(audio_rows, 32ull, &audio_values) ||
        !request_fixture_allocate(&fixture, video_rows, audio_rows, text_rows,
                                  packed_rows, timestep_count) ||
        !request_fixture_read(&fixture, fixture_root, video_rows, audio_rows, text_rows,
                              packed_rows, timestep_count)) {
        request_fixture_close(&fixture);
        return 2;
    }
    request.video = fixture.video; request.audio = fixture.audio;
    request.conditioning = fixture.conditioning; request.timesteps = fixture.timesteps;
    request.position_ids = fixture.positions; request.video_indices = fixture.video_indices;
    request.audio_indices = fixture.audio_indices; request.text_indices = fixture.text_indices;
    request.timestep_indices = fixture.timestep_indices; request.token_tags = fixture.tags;
    request.video_rows = video_rows; request.audio_rows = audio_rows; request.text_rows = text_rows;
    request.packed_rows = packed_rows; request.timestep_count = timestep_count;
    request.block_count = block_count; request.video_output = fixture.video_output;
    request.audio_output = fixture.audio_output; request.video_output_capacity = video_values;
    request.audio_output_capacity = audio_values;
    request.block_observer = checkpoint.root ? block_checkpoint_observe : NULL;
    request.block_observer_context = checkpoint.root ? &checkpoint : NULL;
    request.stage_observer = stage_checkpoint.root ? stage_checkpoint_observe : NULL;
    request.stage_observer_context = stage_checkpoint.root ? &stage_checkpoint : NULL;
    request.observed_stage_block = stage_checkpoint.block;
    yvex_error_clear(&err);
    rc = yvex_artifact_open(&artifact, &options, &err);
    if (rc == YVEX_OK) rc = yvex_gguf_open(&gguf, artifact, &err);
    if (rc == YVEX_OK) rc = yvex_tensor_table_from_gguf(&tensors, gguf, &err);
    if (rc == YVEX_OK)
        rc = execute_artifact(artifact, gguf, tensors, &request, &result, &err);
    if (rc == YVEX_OK &&
        (!file_write(video_output_path, fixture.video_output, video_values) ||
         !file_write(audio_output_path, fixture.audio_output, audio_values)))
        rc = YVEX_ERR_IO;
    if (rc == YVEX_OK) {
        video_matches = compare("video", fixture.video_reference, fixture.video_output,
                                video_values, block_count);
        audio_matches = compare("audio", fixture.audio_reference, fixture.audio_output,
                                audio_values, block_count);
        if (!video_matches || !audio_matches) {
            yvex_error_set(
                &err, YVEX_ERR_FORMAT, "minimax-h3.transformer-request-proof.oracle",
                "the request-shaped Transformer envelope differs from its BF16 oracle");
            rc = YVEX_ERR_FORMAT;
        }
    }
    if (rc == YVEX_OK)
        printf("omni_transformer_request=accepted rows=%llu blocks=%llu resident_bytes=%llu\n"
               "kernel_launches=%llu device_bytes=%llu\nresidency_identity=%s\n"
               "execution_identity=%s\n", packed_rows, block_count, result.resident_bytes,
               result.kernel_launches, result.device_bytes, result.residency_identity,
               result.execution_identity);
    else
        fprintf(stderr, "omni_transformer_request=refused where=%s message=%s\n",
                yvex_error_where(&err), yvex_error_message(&err));
    request_fixture_close(&fixture);
    yvex_tensor_table_close(tensors); yvex_gguf_close(gguf); yvex_artifact_close(artifact);
    return rc == YVEX_OK ? 0 : 1;
}

static int latent_fixture_read(request_fixture *fixture, const char *root,
                               unsigned long long video_values,
                               unsigned long long audio_values,
                               unsigned long long conditioning_values)
{
    char conditioning_path[1024], video_path[1024], audio_path[1024];
    return fixture_path(conditioning_path, root, "conditioning.f32") &&
           fixture_path(video_path, root, "video.oracle.f32") &&
           fixture_path(audio_path, root, "audio.oracle.f32") &&
           file_read(conditioning_path, fixture->conditioning, conditioning_values) &&
           file_read(video_path, fixture->video_reference, video_values) &&
           file_read(audio_path, fixture->audio_reference, audio_values);
}

static int execute_latent_fixture(
    const char *artifact_path, const char *fixture_root, const char *video_output_path,
    const char *audio_output_path, unsigned long long width, unsigned long long height,
    unsigned long long frames, unsigned long long text_rows, unsigned long long block_count,
    unsigned int steps, unsigned long long seed)
{
    const yvex_minimax_h3_graph_api *graph = yvex_graph_register_minimax_h3();
    yvex_artifact_options options = {.path = artifact_path, .readonly = 1};
    yvex_artifact *artifact = NULL;
    yvex_gguf *gguf = NULL;
    yvex_tensor_table *tensors = NULL;
    yvex_complete_artifact_admission admission;
    yvex_artifact_admission_failure failure;
    yvex_runtime_component_session *session = NULL;
    yvex_component_execution component = {0};
    yvex_component_resource_summary resources = {0};
    const int optional_fallback =
        getenv("YVEX_TEST_JOINT_PREPARED_OPTIONAL_FAILURE") != NULL;
    yvex_minimax_h3_t2va_plan plan = {0};
    yvex_runtime_av_layout_result layout_result = {0};
    yvex_runtime_latent_result latent_result = {0};
    yvex_minimax_h3_t2va_omni_result omni_result = {0};
    yvex_minimax_h3_t2va_omni_context context = {0};
    yvex_transformer_linear_physical_plan video_specialization = {0};
    yvex_transformer_linear_physical_plan audio_specialization = {0};
    yvex_runtime_av_layout_output layout = {0};
    request_fixture fixture = {0};
    yvex_media_plan_request plan_request = {
        .schema_version = YVEX_RUNTIME_AV_PLAN_SCHEMA_V1,
        .text_tokens = text_rows, .width = width, .height = height, .frames = frames,
        .inference_steps = steps,
    };
    yvex_media_layout_request layout_request = {0};
    unsigned long long video_values, audio_values, conditioning_values, workspace_bytes;
    char conditioning_identity[65];
    const char *checkpoint_root = getenv("YVEX_MINIMAX_H3_LATENT_CHECKPOINT_ROOT");
    latent_checkpoint_context checkpoint = {checkpoint_root, steps};
    yvex_error err, cleanup;
    int rc = YVEX_OK, cleanup_rc;
    if (!artifact_path || !fixture_root || !video_output_path || !audio_output_path ||
        !width || !height || !frames || !text_rows || !block_count || block_count > 50ull ||
        !steps || steps > 64u || !graph ||
        graph->t2va_plan_build(&plan, &plan_request, &err) != YVEX_OK ||
        plan.packed_rows > YVEX_MINIMAX_H3_OMNI_MAX_PACKED_ROWS ||
        !yvex_core_u64_mul(plan.video_rows, plan.video_value_width, &video_values) ||
        !yvex_core_u64_mul(plan.audio_rows, plan.audio_value_width, &audio_values) ||
        !yvex_core_u64_mul(text_rows, 5120ull, &conditioning_values) ||
        !yvex_core_u64_add(video_values, audio_values, &workspace_bytes) ||
        !yvex_core_u64_mul(workspace_bytes, 4ull * sizeof(float), &workspace_bytes) ||
        !request_fixture_allocate(&fixture, plan.video_rows, plan.audio_rows, text_rows,
                                  plan.packed_rows, 2ull) ||
        !latent_fixture_read(&fixture, fixture_root, video_values, audio_values,
                             conditioning_values) ||
        !float_identity("yvex.minimax-h3.conditioning.fixture.v1", fixture.conditioning,
                        conditioning_values, conditioning_identity)) {
        request_fixture_close(&fixture);
        return 2;
    }
    layout = (yvex_runtime_av_layout_output){
        fixture.positions, plan.packed_rows * 3ull, fixture.tags,
        fixture.video_indices, fixture.audio_indices, fixture.text_indices,
        plan.packed_rows, plan.video_rows, plan.audio_rows, text_rows,
    };
    yvex_error_clear(&err);
    rc = yvex_artifact_open(&artifact, &options, &err);
    if (rc == YVEX_OK) rc = yvex_gguf_open(&gguf, artifact, &err);
    if (rc == YVEX_OK) rc = yvex_tensor_table_from_gguf(&tensors, gguf, &err);
    if (rc == YVEX_OK)
        rc = graph->component_admit("transformer", artifact, gguf, tensors,
                                    NULL, &admission, NULL, &failure, &err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_component_session_open(
            &session, &admission, artifact, gguf, tensors, YVEX_BACKEND_KIND_CUDA,
            80ull * 1024ull * 1024ull * 1024ull, 16ull * 1024ull * 1024ull * 1024ull, &err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_component_session_borrow(session, &component, &err);
    layout_request.plan = &plan;
    if (rc == YVEX_OK)
        rc = graph->t2va_layout_build(&layout_request, &layout, &layout_result, &err);
    if (rc == YVEX_OK)
        rc = specialize_outputs(
            graph->omni_recipe,
            &video_specialization, &audio_specialization, &err);
    context.transformer_component = &component;
    context.conditioning = fixture.conditioning;
    context.conditioning_capacity = conditioning_values;
    context.layout = &layout;
    context.layout_result = &layout_result;
    context.video_output_specialization = &video_specialization;
    context.audio_output_specialization = &audio_specialization;
    context.timestep_indices = fixture.timestep_indices;
    context.timestep_capacity = plan.packed_rows;
    context.block_count = block_count;
    context.conditioning_identity = conditioning_identity;
    context.observe = checkpoint_root && checkpoint_root[0] ? latent_checkpoint_observe : NULL;
    context.observer_context = &checkpoint;
    if (rc == YVEX_OK)
        rc = graph->t2va_latent_execute(
            &plan, &context, seed, workspace_bytes, fixture.video_output, video_values,
            fixture.audio_output, audio_values, &latent_result, &omni_result, &err);
    if (rc == YVEX_OK &&
        (latent_result.transaction.state != YVEX_EXECUTION_TRANSACTION_COMMITTED ||
         latent_result.transaction.completed_quanta != steps ||
         latent_result.transaction.safe_points != steps ||
         latent_result.transaction.retained_resources != 1ull)) {
        yvex_error_set(&err, YVEX_ERR_STATE, "minimax-h3.latent-request.transaction",
                       "the iterative request did not retain one admitted execution resource");
        rc = YVEX_ERR_STATE;
    }
    if (rc == YVEX_OK)
        rc = yvex_component_execution_resource_summary(
            &component, &resources, &err);
    if (rc == YVEX_OK &&
        (!resources.ready || resources.retained_by_transaction ||
         resources.preparation_count != 1ull || resources.use_count != steps ||
         resources.reuse_count + 1ull != resources.use_count ||
         resources.rebuild_count || !resources.host_arena_bytes ||
         !resources.device_arena_bytes ||
         (optional_fallback
              ? (resources.request_ready || resources.condition_ready ||
                 resources.resource_count != 1ull ||
                 resources.request_prepared_bytes ||
                 resources.condition_prepared_bytes)
              : (!resources.request_ready || !resources.condition_ready ||
                 resources.resource_count != 3ull ||
                 !resources.request_prepared_bytes ||
                 !resources.condition_prepared_bytes)))) {
        yvex_error_set(&err, YVEX_ERR_STATE, "minimax-h3.resource-proof",
                       "prepared request and condition state did not remain reusable");
        rc = YVEX_ERR_STATE;
    }
    if (rc == YVEX_OK &&
        (!file_write(video_output_path, fixture.video_output, video_values) ||
         !file_write(audio_output_path, fixture.audio_output, audio_values)))
        rc = YVEX_ERR_IO;
    if (rc == YVEX_OK) {
        int video_matches = compare("video", fixture.video_reference, fixture.video_output,
                                    video_values, block_count);
        int audio_matches = compare("audio", fixture.audio_reference, fixture.audio_output,
                                    audio_values, block_count);
        if (!video_matches || !audio_matches) {
            yvex_error_set(&err, YVEX_ERR_FORMAT, "minimax-h3.latent-iteration-proof.oracle",
                           "the iterative latent result differs from its independent oracle");
            rc = YVEX_ERR_FORMAT;
        }
    }
    if (rc == YVEX_OK)
        printf("t2va_latent_request=accepted rows=%llu blocks=%llu steps=%u seed=%llu\n"
               "kernel_launches=%llu peak_device_bytes=%llu\n"
               "execution_quanta=%llu safe_points=%llu transaction_setup_ns=%llu "
               "safe_point_ns=%llu\nresource_prepare_ns=%llu arena_host_bytes=%llu "
               "arena_device_bytes=%llu resource_uses=%llu resource_reuses=%llu "
               "last_execution_allocations=%llu\nrequest_prepared_bytes=%llu "
               "condition_prepared_bytes=%llu resource_count=%llu resource_rebuilds=%llu\n"
               "plan_identity=%s\n"
               "layout_identity=%s\nlatent_identity=%s\ntransformer_chain_identity=%s\n",
               plan.packed_rows, block_count, steps, seed, omni_result.kernel_launches,
               omni_result.peak_device_bytes, latent_result.transaction.completed_quanta,
               latent_result.transaction.safe_points,
               latent_result.transaction.setup_nanoseconds,
               latent_result.transaction.safe_point_nanoseconds,
               resources.preparation_nanoseconds, resources.host_arena_bytes,
               resources.device_arena_bytes, resources.use_count,
               resources.reuse_count, resources.last_execution_allocation_events,
               resources.request_prepared_bytes,
               resources.condition_prepared_bytes, resources.resource_count,
               resources.rebuild_count,
               plan.identity, layout_result.layout_identity,
               latent_result.execution_identity, omni_result.execution_chain_identity);
    else
        fprintf(stderr, "t2va_latent_request=refused where=%s message=%s\n",
                yvex_error_where(&err), yvex_error_message(&err));
    yvex_error_clear(&cleanup);
    cleanup_rc = yvex_runtime_component_session_close(&session, &cleanup);
    if (cleanup_rc != YVEX_OK && rc == YVEX_OK) rc = cleanup_rc;
    request_fixture_close(&fixture);
    yvex_tensor_table_close(tensors);
    yvex_gguf_close(gguf);
    yvex_artifact_close(artifact);
    return rc == YVEX_OK ? 0 : 1;
}

int main(int argc, char **argv)
{
    yvex_artifact_options options = {0};
    yvex_artifact *artifact = NULL;
    yvex_tensor_table *tensors = NULL;
    yvex_gguf *gguf = NULL;
    yvex_minimax_h3_omni_transformer_request request = {0};
    yvex_minimax_h3_omni_transformer_result result = {0};
    yvex_error err;
    float video[VIDEO_VALUES], audio[AUDIO_VALUES], conditioning[CONDITION_VALUES];
    float video_output[VIDEO_VALUES] = {0}, audio_output[AUDIO_VALUES] = {0};
    float video_reference[VIDEO_VALUES], audio_reference[AUDIO_VALUES];
    float timesteps[1] = {0.625f};
    float positions[9] = {0, 1, 2, 0, 0, 0, 3, 0, 0};
    unsigned int video_indices[1] = {0}, text_indices[1] = {1}, audio_indices[1] = {2};
    unsigned int timestep_indices[3] = {0, 0, 0}, tags[3] = {0, 1, 2};
    const char *artifact_mode = getenv("YVEX_MINIMAX_H3_GRAPH_ARTIFACT");
    const char *blocks_text = getenv("YVEX_MINIMAX_H3_BLOCKS");
    char *blocks_end = NULL;
    unsigned long long block_count = blocks_text ? strtoull(blocks_text, &blocks_end, 10) : 1ull;
    int rc = YVEX_OK;
    if (argc == 12 && strcmp(argv[2], "block-request") == 0) {
        char *ends[5] = {0};
        unsigned long long values[5];
        int index;
        for (index = 0; index < 5; ++index)
            values[index] = strtoull(argv[index + 7], &ends[index], 10);
        for (index = 0; index < 5; ++index)
            if (!ends[index] || *ends[index]) return 2;
        return execute_selected_block(
            argv[1], argv[3], argv[4], argv[5], argv[6], values[0], values[1],
            values[2], values[3], values[4]);
    }
    if (argc == 13 && strcmp(argv[2], "latent-request") == 0) {
        char *ends[7] = {0};
        unsigned long long values[7];
        int index;
        for (index = 0; index < 7; ++index)
            values[index] = strtoull(argv[index + 6], &ends[index], 10);
        for (index = 0; index < 7; ++index)
            if (!ends[index] || *ends[index]) return 2;
        if (values[5] > UINT_MAX) return 2;
        return execute_latent_fixture(
            argv[1], argv[3], argv[4], argv[5], values[0], values[1], values[2],
            values[3], values[4], (unsigned int)values[5], values[6]);
    }
    if (argc == 11 && strcmp(argv[2], "request") == 0) {
        char *ends[5] = {0};
        unsigned long long values[5];
        int index;
        for (index = 0; index < 5; ++index) values[index] = strtoull(argv[index + 6], &ends[index], 10);
        for (index = 0; index < 5; ++index)
            if (!ends[index] || *ends[index]) return 2;
        return execute_request_fixture(argv[1], argv[3], argv[4], argv[5],
                                       values[0], values[1], values[2], values[3], values[4]);
    }
    if (argc == 6 && strcmp(argv[2], "latent") == 0) {
        char *latent_blocks_end = NULL, *steps_end = NULL;
        unsigned long long latent_blocks = strtoull(argv[4], &latent_blocks_end, 10);
        unsigned long long latent_steps = strtoull(argv[5], &steps_end, 10);
        if (!latent_blocks_end || *latent_blocks_end || !steps_end || *steps_end ||
            latent_steps > UINT_MAX) return 2;
        return execute_latent(argv[1], argv[3], latent_blocks, (unsigned int)latent_steps);
    }
    if (argc != 9) {
        fprintf(stderr, "usage: minimax_h3_transformer GGUF VIDEO AUDIO CONDITIONING "
                        "VIDEO_OUT AUDIO_OUT VIDEO_REFERENCE AUDIO_REFERENCE\n");
        return 2;
    }
    if ((blocks_text && (!blocks_end || *blocks_end)) || block_count < 1ull ||
        block_count > 50ull || (block_count != 1ull &&
        (!artifact_mode || strcmp(artifact_mode, "1") != 0))) return 2;
    if (!file_read(argv[2], video, VIDEO_VALUES) || !file_read(argv[3], audio, AUDIO_VALUES) ||
        !file_read(argv[4], conditioning, CONDITION_VALUES) ||
        !file_read(argv[7], video_reference, VIDEO_VALUES) ||
        !file_read(argv[8], audio_reference, AUDIO_VALUES)) return 2;
    request.video = video; request.audio = audio; request.conditioning = conditioning;
    request.timesteps = timesteps; request.position_ids = positions;
    request.video_indices = video_indices; request.audio_indices = audio_indices;
    request.text_indices = text_indices; request.timestep_indices = timestep_indices;
    request.token_tags = tags; request.video_rows = request.audio_rows = request.text_rows = 1ull;
    request.timestep_count = 1ull; request.packed_rows = 3ull; request.block_count = block_count;
    request.video_output = video_output; request.audio_output = audio_output;
    request.video_output_capacity = VIDEO_VALUES; request.audio_output_capacity = AUDIO_VALUES;
    options.path = argv[1]; options.readonly = 1;
    yvex_error_clear(&err);
    rc = yvex_artifact_open(&artifact, &options, &err);
    if (rc == YVEX_OK) rc = yvex_gguf_open(&gguf, artifact, &err);
    if (rc == YVEX_OK) rc = yvex_tensor_table_from_gguf(&tensors, gguf, &err);
    if (rc == YVEX_OK)
        rc = artifact_mode && strcmp(artifact_mode, "1") == 0
            ? execute_artifact(artifact, gguf, tensors, &request, &result, &err)
            : execute(artifact, gguf, tensors, &request, &result, &err);
    if (rc == YVEX_OK && (!file_write(argv[5], video_output, VIDEO_VALUES) ||
                          !file_write(argv[6], audio_output, AUDIO_VALUES))) rc = YVEX_ERR_IO;
    if (rc == YVEX_OK) {
        int video_valid = compare("video", video_reference, video_output,
                                  VIDEO_VALUES, block_count);
        int audio_valid = compare("audio", audio_reference, audio_output,
                                  AUDIO_VALUES, block_count);
        if (!video_valid || !audio_valid) {
            yvex_error_set(&err, YVEX_ERR_FORMAT, "minimax-h3.transformer-proof.oracle",
                           "YVEX Transformer envelope differs from the independent BF16 oracle");
            rc = YVEX_ERR_FORMAT;
        }
    }
    if (rc == YVEX_OK)
        printf("omni_transformer=accepted rows=%llu blocks=%llu resident_bytes=%llu\n"
               "kernel_launches=%llu device_bytes=%llu\nresidency_identity=%s\nexecution_identity=%s\n",
               result.packed_rows, result.block_count, result.resident_bytes,
               result.kernel_launches, result.device_bytes, result.residency_identity,
               result.execution_identity);
    else fprintf(stderr, "omni_transformer=refused where=%s message=%s\n",
                 yvex_error_where(&err), yvex_error_message(&err));
    yvex_tensor_table_close(tensors); yvex_gguf_close(gguf); yvex_artifact_close(artifact);
    return rc == YVEX_OK ? 0 : 1;
}
