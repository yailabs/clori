/* Compare the exact MiniMax-H3 Transformer envelope with an independent CUDA oracle. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include <yvex/artifact.h>
#include <yvex/gguf.h>
#include <yvex/internal/artifact.h>
#include <yvex/internal/backend.h>
#include <yvex/internal/core.h>
#include <yvex/internal/families/minimax_h3.h>
#include <yvex/internal/latent.h>
#include <yvex/internal/runtime.h>

enum { VIDEO_VALUES = 96u, AUDIO_VALUES = 32u, CONDITION_VALUES = 5120u };

static const char *const external_names[YVEX_MINIMAX_H3_OMNI_EXTERNAL_WEIGHT_COUNT] = {
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

static const char *const block_suffixes[YVEX_MINIMAX_H3_OMNI_BLOCK_WEIGHT_COUNT] = {
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
    yvex_minimax_h3_encoded_weight external[YVEX_MINIMAX_H3_OMNI_EXTERNAL_WEIGHT_COUNT],
    yvex_minimax_h3_encoded_weight blocks[YVEX_MINIMAX_H3_OMNI_BLOCK_WEIGHT_COUNT],
    char identity[65], yvex_error *err)
{
    const yvex_materialized_tensor_binding *bindings[
        YVEX_MINIMAX_H3_OMNI_EXTERNAL_WEIGHT_COUNT + YVEX_MINIMAX_H3_OMNI_BLOCK_WEIGHT_COUNT];
    char block_names[YVEX_MINIMAX_H3_OMNI_BLOCK_WEIGHT_COUNT][96];
    yvex_materialization_failure failure;
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES], *arena;
    unsigned long long index, total = 0ull, cursor = 0ull;
    for (index = 0ull; index < YVEX_MINIMAX_H3_OMNI_EXTERNAL_WEIGHT_COUNT; ++index)
        bindings[index] = binding_find(session, external_names[index]);
    for (index = 0ull; index < YVEX_MINIMAX_H3_OMNI_BLOCK_WEIGHT_COUNT; ++index) {
        int length = snprintf(block_names[index], sizeof(block_names[index]),
                              "blocks.0.%s", block_suffixes[index]);
        if (length < 0 || (size_t)length >= sizeof(block_names[index])) return YVEX_ERR_BOUNDS;
        bindings[YVEX_MINIMAX_H3_OMNI_EXTERNAL_WEIGHT_COUNT + index] =
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
        if (index < YVEX_MINIMAX_H3_OMNI_EXTERNAL_WEIGHT_COUNT) {
            if (!weight_bind(external + index, binding, arena + cursor)) goto failed;
        } else if (!weight_bind(blocks + index - YVEX_MINIMAX_H3_OMNI_EXTERNAL_WEIGHT_COUNT,
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

static int file_write(const char *path, const float *values, unsigned long long count)
{
    FILE *file = fopen(path, "wb");
    int valid;
    if (!file) return 0;
    valid = fwrite(values, sizeof(*values), (size_t)count, file) == count;
    if (fclose(file) != 0) valid = 0;
    return valid;
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
    const yvex_minimax_h3_backend_api *family, yvex_backend *backend,
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
    rc = family->omni_transformer_cuda(backend, external, blocks, identity, arena_bytes,
                                       &request, &result, err);
    if (rc != YVEX_ERR_FORMAT || result.complete) return YVEX_ERR_STATE;
    request = *valid;
    request.timesteps = invalid_timestep;
    rc = family->omni_transformer_cuda(backend, external, blocks, identity, arena_bytes,
                                       &request, &result, err);
    if (rc != YVEX_ERR_FORMAT || result.complete) return YVEX_ERR_STATE;
    request = *valid;
    request.video_output_capacity--;
    rc = family->omni_transformer_cuda(backend, external, blocks, identity, arena_bytes,
                                       &request, &result, err);
    if (rc != YVEX_ERR_INVALID_ARG || result.complete) return YVEX_ERR_STATE;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int execute(const yvex_artifact *artifact, const yvex_gguf *gguf,
                   const yvex_tensor_table *tensors,
                   yvex_minimax_h3_omni_transformer_request *request,
                   yvex_minimax_h3_omni_transformer_result *result, yvex_error *err)
{
    const yvex_minimax_h3_graph_api *graph = yvex_graph_register_minimax_h3();
    const yvex_minimax_h3_backend_api *family = yvex_backend_register_minimax_h3();
    yvex_complete_artifact_admission admission;
    yvex_artifact_admission_failure admission_failure;
    yvex_materialization_options options;
    yvex_materialization_failure failure;
    yvex_materialization_plan *plan = NULL;
    yvex_materialization_session *session = NULL;
    yvex_minimax_h3_encoded_weight external[YVEX_MINIMAX_H3_OMNI_EXTERNAL_WEIGHT_COUNT] = {{0}};
    yvex_minimax_h3_encoded_weight blocks[YVEX_MINIMAX_H3_OMNI_BLOCK_WEIGHT_COUNT] = {{0}};
    yvex_backend_options backend_options = {0};
    yvex_backend_tensor_desc descriptor = {0};
    yvex_backend *backend = NULL;
    yvex_device_tensor *resident = NULL;
    unsigned char *arena = NULL, *registered = NULL;
    unsigned long long arena_bytes = 0ull;
    char identity[65] = {0};
    int attached = 0, rc, cleanup_rc;
    yvex_error cleanup;
    rc = graph->component_admit("transformer", artifact, gguf, tensors,
                                &admission, &admission_failure, err);
    yvex_materialization_options_default(&options);
    options.max_chunk_bytes = 64ull * 1024ull * 1024ull;
    if (rc == YVEX_OK) rc = yvex_materialization_plan_build(
        &plan, &admission, artifact, gguf, tensors, NULL, &options, &failure, err);
    if (rc == YVEX_OK) rc = yvex_materialization_session_open(
        &session, plan, artifact, &options, &failure, err);
    if (rc == YVEX_OK) rc = yvex_materialization_session_commit(session, &failure, err);
    if (rc == YVEX_OK) rc = selected_weights_load(
        session, &arena, &arena_bytes, external, blocks, identity, err);
    backend_options.kind = YVEX_BACKEND_KIND_CUDA;
    backend_options.memory_limit_bytes = 4ull * 1024ull * 1024ull * 1024ull;
    if (rc == YVEX_OK) rc = yvex_backend_open(&backend, &backend_options, err);
    descriptor.name = "minimax-h3-transformer-envelope-proof-residency";
    descriptor.dtype = YVEX_DTYPE_I8;
    descriptor.rank = 1u;
    descriptor.dims[0] = descriptor.bytes = arena_bytes;
    registered = arena;
    if (rc == YVEX_OK)
        rc = backend->vtable->resident_alloc(backend, &descriptor, &resident, &registered, err);
    if (rc == YVEX_OK && registered != arena) rc = YVEX_ERR_STATE;
    if (rc == YVEX_OK) {
        rc = yvex_backend_resident_attach(backend, arena, arena_bytes, resident, 1ull, err);
        attached = rc == YVEX_OK;
    }
    if (rc == YVEX_OK)
        rc = refusal_checks(family, backend, external, blocks, identity, arena_bytes,
                            request, err);
    if (rc == YVEX_OK)
        rc = family->omni_transformer_cuda(backend, external, blocks, identity, arena_bytes,
                                           request, result, err);
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

static int execute_artifact(const yvex_artifact *artifact, const yvex_gguf *gguf,
                            const yvex_tensor_table *tensors,
                            yvex_minimax_h3_omni_transformer_request *request,
                            yvex_minimax_h3_omni_transformer_result *result, yvex_error *err)
{
    const yvex_minimax_h3_graph_api *graph = yvex_graph_register_minimax_h3();
    yvex_complete_artifact_admission admission;
    yvex_artifact_admission_failure failure;
    yvex_runtime_component_session *session = NULL;
    yvex_error cleanup;
    int rc = graph->component_admit(
        "transformer", artifact, gguf, tensors, &admission, &failure, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_component_session_open(
            &session, &admission, artifact, gguf, tensors, YVEX_BACKEND_KIND_CUDA,
            80ull * 1024ull * 1024ull * 1024ull, 4ull * 1024ull * 1024ull * 1024ull, err);
    if (rc == YVEX_OK) rc = graph->transformer_component_cuda(session, request, result, err);
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
    yvex_minimax_h3_t2va_omni_context context = {0};
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
                                    &admission, &failure, &err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_component_session_open(
            &session, &admission, artifact, gguf, tensors, YVEX_BACKEND_KIND_CUDA,
            80ull * 1024ull * 1024ull * 1024ull, 4ull * 1024ull * 1024ull * 1024ull, &err);
    if (rc == YVEX_OK)
        rc = graph->t2va_plan_build(&plan, 1ull, 32ull, 32ull, 5ull, steps, &err);
    if (rc == YVEX_OK) rc = graph->t2va_layout_build(&plan, &layout, &layout_result, &err);
    context.transformer_session = session;
    context.conditioning = conditioning;
    context.conditioning_capacity = CONDITION_VALUES;
    context.layout = &layout;
    context.layout_result = &layout_result;
    context.timestep_indices = timestep_indices;
    context.timestep_capacity = 19ull;
    context.block_count = block_count;
    context.conditioning_identity = conditioning_identity;
    if (rc == YVEX_OK)
        rc = graph->t2va_latent_execute(
            &plan, &context, 42ull, (192ull + 512ull) * sizeof(float) * 4ull,
            video, 192ull, audio, 512ull, &latent_result, &omni_result, &err);
    if (rc == YVEX_OK && (!latent_result.completed || !omni_result.complete ||
                          omni_result.model_evaluations != steps)) {
        yvex_error_set(&err, YVEX_ERR_STATE, "minimax-h3.latent-proof",
                       "the exact resident latent iteration did not complete");
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
               "kernel_launches=%llu peak_device_bytes=%llu\nplan_identity=%s\n"
               "layout_identity=%s\nevaluator_identity=%s\nlatent_identity=%s\n"
               "transformer_chain_identity=%s\nresidency_identity=%s\n"
               "vae_input_identity=%s\n",
               steps, block_count, plan.packed_rows, omni_result.kernel_launches,
               omni_result.peak_device_bytes, plan.identity, layout_result.layout_identity,
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
