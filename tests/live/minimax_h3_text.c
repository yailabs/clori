/* Exercise the exact Text Encoder embedding through staged GB10 residency. */
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include <yvex/artifact.h>
#include <yvex/gguf.h>
#include <yvex/model.h>
#include <yvex/internal/artifact.h>
#include <yvex/internal/backend.h>
#include <yvex/internal/component.h>
#include <yvex/internal/core.h>
#include <yvex/internal/family_catalog.h>
#include <yvex/internal/families/minimax_h3.h>
#include <yvex/internal/image.h>
#include <yvex/internal/multimodal.h>
#include "src/backend/cuda/component_ops.h"

enum { TEXT_HIDDEN = 5120u };
/* The independent PyTorch CPU/CUDA BF16 oracle pair differs by these measured bounds. */
static const float layer_oracle_max_absolute = 0.046875f;
static const double layer_oracle_max_rmse = 0.002315;
static const float encoder_oracle_max_absolute = 0.375f;
static const double encoder_oracle_max_rmse = 0.026718;

static const char *const layer_weight_names[YVEX_BACKEND_TEXT_WEIGHT_COUNT] = {
    "model.language_model.embed_tokens.weight",
    "model.language_model.layers.0.input_layernorm.weight",
    "model.language_model.layers.0.self_attn.q_proj.weight",
    "model.language_model.layers.0.self_attn.k_proj.weight",
    "model.language_model.layers.0.self_attn.v_proj.weight",
    "model.language_model.layers.0.self_attn.o_proj.weight",
    "model.language_model.layers.0.self_attn.q_norm.weight",
    "model.language_model.layers.0.self_attn.k_norm.weight",
    "model.language_model.layers.0.post_attention_layernorm.weight",
    "model.language_model.layers.0.mlp.gate_proj.weight",
    "model.language_model.layers.0.mlp.up_proj.weight",
    "model.language_model.layers.0.mlp.down_proj.weight",
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

static int proof_weights_load(
    yvex_materialization_session *session, unsigned char **arena_out,
    unsigned long long *arena_bytes_out,
    yvex_backend_text_weight weights[YVEX_BACKEND_TEXT_WEIGHT_COUNT],
    char identity[65], yvex_error *err)
{
    const yvex_materialized_tensor_binding *bindings[YVEX_BACKEND_TEXT_WEIGHT_COUNT];
    yvex_materialization_failure failure;
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned char *arena;
    unsigned long long index, total = 0ull, cursor = 0ull;
    for (index = 0ull; index < YVEX_BACKEND_TEXT_WEIGHT_COUNT; ++index) {
        bindings[index] = binding_find(session, layer_weight_names[index]);
        if (!bindings[index] || !bindings[index]->row_count ||
            !yvex_core_u64_add(total, bindings[index]->encoded_bytes, &total)) {
            yvex_error_set(err, YVEX_ERR_FORMAT, "minimax-h3.text-proof.binding",
                           "the exact layer-zero weight set is unavailable");
            return YVEX_ERR_FORMAT;
        }
    }
    if (!total || total > SIZE_MAX) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "minimax-h3.text-proof.arena",
                       "the layer-zero proof residency exceeds the host range");
        return YVEX_ERR_BOUNDS;
    }
    arena = mmap(NULL, (size_t)total, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (arena == MAP_FAILED) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "minimax-h3.text-proof.arena",
                       "the layer-zero proof residency allocation failed");
        return YVEX_ERR_NOMEM;
    }
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.minimax-h3.text-layer-zero.proof.v1")) goto failed;
    for (index = 0ull; index < YVEX_BACKEND_TEXT_WEIGHT_COUNT; ++index) {
        const yvex_materialized_tensor_binding *binding = bindings[index];
        if (binding->encoded_bytes > SIZE_MAX ||
            yvex_materialization_session_read(
                session, binding, 0ull, arena + cursor, (size_t)binding->encoded_bytes,
                &failure, err) != YVEX_OK ||
            !yvex_sha256_update_text(&hash, binding->name) ||
            !yvex_sha256_update_u64(&hash, binding->encoded_bytes) ||
            !yvex_sha256_update(&hash, arena + cursor, (size_t)binding->encoded_bytes))
            goto failed;
        weights[index].encoded = arena + cursor;
        weights[index].encoded_bytes = binding->encoded_bytes;
        weights[index].row_count = binding->row_count;
        weights[index].row_width = binding->row_width;
        weights[index].row_bytes = binding->encoded_bytes / binding->row_count;
        weights[index].qtype = binding->qtype;
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
        yvex_error_set(err, YVEX_ERR_STATE, "minimax-h3.text-proof.identity",
                       "the layer-zero proof identity could not be sealed");
    return yvex_error_code(err);
}

static int layer_proof_execute(
    const yvex_artifact *artifact, const yvex_gguf *gguf,
    const yvex_tensor_table *tensors, const unsigned int *token,
    float output[TEXT_HIDDEN], yvex_minimax_h3_conditioning_result *result,
    yvex_error *err)
{
    const yvex_minimax_h3_graph_api *graph = yvex_graph_register_minimax_h3();
    const yvex_minimax_h3_api *model = yvex_model_register_minimax_h3();
    yvex_minimax_h3_architecture architecture;
    yvex_component_text_recipe geometry;
    yvex_backend_text_execution_result backend_result = {0};
    yvex_minimax_h3_failure architecture_failure;
    yvex_complete_artifact_admission admission;
    yvex_artifact_admission_failure admission_failure;
    yvex_materialization_options materialization_options;
    yvex_materialization_failure materialization_failure;
    yvex_materialization_plan *plan = NULL;
    yvex_materialization_session *session = NULL;
    yvex_backend_text_weight weights[YVEX_BACKEND_TEXT_WEIGHT_COUNT] = {{0}};
    yvex_backend_options backend_options = {0};
    yvex_backend_tensor_desc descriptor = {0};
    yvex_backend *backend = NULL;
    yvex_device_tensor *resident = NULL;
    unsigned char *arena = NULL, *registered = NULL;
    unsigned long long arena_bytes = 0ull;
    char identity[65] = {0};
    int attached = 0, rc, release_rc;
    yvex_error cleanup;
    if (!artifact || !gguf || !tensors || !token || !output || !result ||
        !graph || !model ||
        !model->architecture_canonical ||
        model->architecture_canonical(
            &architecture, &architecture_failure, err) != YVEX_OK) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "minimax-h3.text-proof",
                       "exact artifact views, token, output, and production backend are required");
        return YVEX_ERR_INVALID_ARG;
    }
    geometry = (yvex_component_text_recipe){
        .schema_version = YVEX_COMPONENT_TEXT_RECIPE_SCHEMA_V1,
        .semantic_identity = YVEX_MINIMAX_H3_TEXT_COMPONENT_IDENTITY,
        .layer_capacity = architecture.encoder.text_layers,
        .hidden_width = architecture.encoder.text_width,
        .ffn_width = architecture.encoder.text_ffn_width,
        .query_heads = architecture.encoder.text_query_heads,
        .kv_heads = architecture.encoder.text_kv_heads,
        .head_dimension = architecture.encoder.text_head_dimension,
        .vocabulary_size = architecture.encoder.vocabulary_size,
        .rope_theta = architecture.encoder.rope_theta,
        .normalization_epsilon = 1.0e-6f};
    rc = graph->component_admit(
        "text_encoder", artifact, gguf, tensors, NULL, &admission, NULL,
        &admission_failure, err);
    yvex_materialization_options_default(&materialization_options);
    materialization_options.max_chunk_bytes = 64ull * 1024ull * 1024ull;
    if (rc == YVEX_OK)
        rc = yvex_materialization_plan_build(
            &plan, &admission, artifact, gguf, tensors, NULL, &materialization_options,
            &materialization_failure, err);
    if (rc == YVEX_OK)
        rc = yvex_materialization_session_open(
            &session, plan, artifact, &materialization_options,
            &materialization_failure, err);
    if (rc == YVEX_OK)
        rc = yvex_materialization_session_commit(session, &materialization_failure, err);
    if (rc == YVEX_OK)
        rc = proof_weights_load(session, &arena, &arena_bytes, weights, identity, err);
    backend_options.kind = YVEX_BACKEND_KIND_CUDA;
    backend_options.memory_limit_bytes = 512ull * 1024ull * 1024ull;
    if (rc == YVEX_OK) rc = yvex_backend_open(&backend, &backend_options, err);
    descriptor.name = "minimax-h3-layer-zero-proof-residency";
    descriptor.dtype = YVEX_DTYPE_I8;
    descriptor.rank = 1u;
    descriptor.dims[0] = descriptor.bytes = arena_bytes;
    registered = arena;
    if (rc == YVEX_OK)
        rc = yvex_backend_resident_alloc(
            backend, &descriptor, &resident, &registered, err);
    if (rc == YVEX_OK && registered != arena) {
        yvex_error_set(err, YVEX_ERR_STATE, "minimax-h3.text-proof.residency",
                       "CUDA registration changed the selected proof address");
        rc = YVEX_ERR_STATE;
    }
    if (rc == YVEX_OK) {
        rc = yvex_backend_resident_attach(backend, arena, arena_bytes, resident, 1ull, err);
        attached = rc == YVEX_OK;
    }
    if (rc == YVEX_OK)
        rc = yvex_cuda_text_encoder_execute(
            backend, &geometry, weights, 1ull, identity, arena_bytes, token, 1ull,
            output, TEXT_HIDDEN, &backend_result, err);
    if (rc == YVEX_OK) {
        *result = (yvex_minimax_h3_conditioning_result){
            .token_count = backend_result.token_count,
            .hidden_width = backend_result.hidden_width,
            .layer_count = backend_result.layer_count,
            .resident_bytes = backend_result.resident_bytes,
            .kernel_launches = backend_result.kernel_launches,
            .h2d_bytes = backend_result.h2d_bytes,
            .d2h_bytes = backend_result.d2h_bytes,
            .device_bytes = backend_result.device_bytes,
            .complete = backend_result.complete};
        memcpy(result->residency_identity, backend_result.residency_identity,
               sizeof(result->residency_identity));
        memcpy(result->execution_identity, backend_result.execution_identity,
               sizeof(result->execution_identity));
    }
    if (attached) {
        yvex_error_clear(&cleanup);
        release_rc = yvex_backend_resident_detach(backend, &cleanup);
        if (release_rc != YVEX_OK) {
            rc = release_rc;
            if (err) *err = cleanup;
        }
    }
    if (resident) {
        yvex_error_clear(&cleanup);
        release_rc = yvex_backend_tensor_release(backend, &resident, &cleanup);
        if (release_rc != YVEX_OK) {
            rc = release_rc;
            if (err) *err = cleanup;
        }
    }
    yvex_error_clear(&cleanup);
    release_rc = yvex_backend_close_checked(&backend, &cleanup);
    if (release_rc != YVEX_OK) {
        rc = release_rc;
        if (err) *err = cleanup;
    }
    if (arena) munmap(arena, (size_t)arena_bytes);
    yvex_materialization_session_close(session);
    yvex_materialization_plan_close(plan);
    return rc;
}

static int reference_compare(const char *path, const float output[TEXT_HIDDEN],
                             int execution_mode)
{
    float reference[TEXT_HIDDEN];
    float maximum = 0.0f;
    double squared = 0.0;
    FILE *file = fopen(path, "rb");
    unsigned long long index;

    if (!file || fread(reference, sizeof(reference), 1u, file) != 1u || fgetc(file) != EOF) {
        if (file) fclose(file);
        fprintf(stderr, "text_reference_read=refused\n");
        return 0;
    }
    fclose(file);
    for (index = 0ull; index < TEXT_HIDDEN; ++index) {
        float absolute = fabsf(reference[index] - output[index]);
        if (absolute > maximum) maximum = absolute;
        squared += (double)absolute * (double)absolute;
    }
    squared = sqrt(squared / TEXT_HIDDEN);
    printf("oracle_max_absolute_error=%.9g oracle_rmse=%.9g\n", maximum, squared);
    if (!execution_mode) return maximum == 0.0f;
    if (execution_mode == 3)
        return maximum <= encoder_oracle_max_absolute && squared <= encoder_oracle_max_rmse;
    return maximum <= layer_oracle_max_absolute && squared <= layer_oracle_max_rmse;
}

static int output_write(const char *path, const float output[TEXT_HIDDEN])
{
    FILE *file = fopen(path, "wb");
    size_t written;
    int close_rc;

    if (!file) return 0;
    written = fwrite(output, sizeof(float), TEXT_HIDDEN, file);
    close_rc = fclose(file);
    return written == TEXT_HIDDEN && close_rc == 0;
}

typedef struct {
    const char *output_root;
    const char *reference_root;
} multimodal_observer;

static int evidence_path(char path[1024], const char *root, const char *name)
{
    int length = snprintf(path, 1024u, "%s/%s", root, name);
    return length > 0 && length < 1024;
}

static int evidence_write(const char *root, const char *name,
                          const void *data, size_t bytes)
{
    char path[1024];
    FILE *file;
    int valid;
    if (!evidence_path(path, root, name)) return 0;
    file = fopen(path, "wb");
    if (!file) return 0;
    valid = fwrite(data, 1u, bytes, file) == bytes;
    if (fclose(file) != 0) valid = 0;
    return valid;
}

static int evidence_exact_compare(const char *root, const char *name,
                                  const void *actual, size_t bytes)
{
    char path[1024];
    unsigned char *reference;
    FILE *file;
    int matches;
    if (!evidence_path(path, root, name) || !(reference = malloc(bytes))) return 0;
    file = fopen(path, "rb");
    matches = file && fread(reference, 1u, bytes, file) == bytes && fgetc(file) == EOF &&
              memcmp(reference, actual, bytes) == 0;
    if (file) fclose(file);
    free(reference);
    printf("multimodal_oracle=%s exact=%s\n", name, matches ? "yes" : "no");
    return matches;
}

static int evidence_float_compare(const char *root, const char *name,
                                  const float *actual, unsigned long long count,
                                  double maximum_relative_l2, double minimum_cosine,
                                  float maximum_absolute)
{
    char path[1024];
    float *reference;
    FILE *file;
    long double squared_error = 0.0L, squared_reference = 0.0L;
    long double squared_actual = 0.0L, dot = 0.0L;
    double relative_l2, cosine;
    float maximum = 0.0f;
    unsigned long long index;
    int accepted;
    if (!count || count > SIZE_MAX / sizeof(float) || !evidence_path(path, root, name) ||
        !(reference = malloc((size_t)count * sizeof(float)))) return 0;
    file = fopen(path, "rb");
    if (!file || fread(reference, sizeof(float), (size_t)count, file) != count ||
        fgetc(file) != EOF) {
        if (file) fclose(file);
        free(reference);
        return 0;
    }
    fclose(file);
    for (index = 0ull; index < count; ++index) {
        long double difference = (long double)actual[index] - reference[index];
        float absolute = fabsf(actual[index] - reference[index]);
        if (!isfinite(actual[index]) || !isfinite(reference[index])) {
            free(reference);
            return 0;
        }
        if (absolute > maximum) maximum = absolute;
        squared_error += difference * difference;
        squared_reference += (long double)reference[index] * reference[index];
        squared_actual += (long double)actual[index] * actual[index];
        dot += (long double)actual[index] * reference[index];
    }
    relative_l2 = squared_reference > 0.0L
                      ? (double)sqrtl(squared_error / squared_reference) : INFINITY;
    cosine = squared_reference > 0.0L && squared_actual > 0.0L
                 ? (double)(dot / sqrtl(squared_reference * squared_actual)) : -1.0;
    accepted = maximum <= maximum_absolute && relative_l2 <= maximum_relative_l2 &&
               cosine >= minimum_cosine;
    printf("multimodal_oracle=%s max_absolute=%.9g relative_l2=%.12g cosine=%.12g accepted=%s\n",
           name, maximum, relative_l2, cosine, accepted ? "yes" : "no");
    free(reference);
    return accepted;
}

static int multimodal_observe(
    void *opaque, const yvex_media_conditioning_observation *observation,
    yvex_error *err)
{
    multimodal_observer *observer = opaque;
    unsigned long long grid[3];
    size_t token_bytes, position_bytes, patch_bytes, merged_bytes, deepstack_bytes;
    if (!observer || !observer->output_root || !observation ||
        observation->token_count > SIZE_MAX / sizeof(unsigned int) ||
        observation->token_count > SIZE_MAX / (3u * sizeof(unsigned long long)) ||
        observation->patch_values > SIZE_MAX / sizeof(float) ||
        observation->merged_values > SIZE_MAX / sizeof(float) ||
        observation->deepstack_values > SIZE_MAX / sizeof(float))
        goto failed;
    token_bytes = (size_t)observation->token_count * sizeof(unsigned int);
    position_bytes = (size_t)observation->token_count * 3u * sizeof(unsigned long long);
    patch_bytes = (size_t)observation->patch_values * sizeof(float);
    merged_bytes = (size_t)observation->merged_values * sizeof(float);
    deepstack_bytes = (size_t)observation->deepstack_values * sizeof(float);
    grid[0] = observation->image_count;
    grid[1] = observation->grid_height;
    grid[2] = observation->grid_width;
    if (!evidence_write(observer->output_root, "tokens.yvex.u32",
                        observation->token_ids, token_bytes) ||
        !evidence_write(observer->output_root, "types.yvex.u32",
                        observation->token_types, token_bytes) ||
        !evidence_write(observer->output_root, "tags.yvex.u32",
                        observation->text_tags, token_bytes) ||
        !evidence_write(observer->output_root, "positions.yvex.u64",
                        observation->position_ids, position_bytes) ||
        !evidence_write(observer->output_root, "grid.yvex.u64", grid, sizeof(grid)) ||
        !evidence_write(observer->output_root, "vision-patches.yvex.f32",
                        observation->vision_patches, patch_bytes) ||
        !evidence_write(observer->output_root, "vision-merged.yvex.f32",
                        observation->vision_merged, merged_bytes) ||
        !evidence_write(observer->output_root, "vision-deepstack.yvex.f32",
                        observation->vision_deepstack, deepstack_bytes))
        goto failed;
    if (observer->reference_root &&
        (!evidence_exact_compare(observer->reference_root, "tokens.reference.u32",
                                 observation->token_ids, token_bytes) ||
         !evidence_exact_compare(observer->reference_root, "types.reference.u32",
                                 observation->token_types, token_bytes) ||
         !evidence_exact_compare(observer->reference_root, "tags.reference.u32",
                                 observation->text_tags, token_bytes) ||
         !evidence_exact_compare(observer->reference_root, "positions.reference.u64",
                                 observation->position_ids, position_bytes) ||
         !evidence_exact_compare(observer->reference_root, "grid.reference.u64",
                                 grid, sizeof(grid)) ||
         !evidence_float_compare(observer->reference_root, "vision-patches.reference.f32",
                                 observation->vision_patches, observation->patch_values,
                                 1.0e-6, 0.999999, 1.0e-6f) ||
         !evidence_float_compare(observer->reference_root, "vision-merged.reference.f32",
                                 observation->vision_merged, observation->merged_values,
                                 0.05, 0.999, INFINITY) ||
         !evidence_float_compare(observer->reference_root, "vision-deepstack.reference.f32",
                                 observation->vision_deepstack, observation->deepstack_values,
                                 0.03, 0.999, INFINITY))) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "minimax-h3.multimodal.oracle",
                       "multimodal processor or vision output differs from the released oracle");
        return YVEX_ERR_FORMAT;
    }
    return YVEX_OK;
failed:
    yvex_error_set(err, YVEX_ERR_IO, "minimax-h3.multimodal.evidence",
                   "multimodal evidence could not be written completely");
    return YVEX_ERR_IO;
}

static int vision_observe(
    void *opaque, unsigned int stage, unsigned long long index,
    const float *values, unsigned long long rows, unsigned long long width,
    yvex_error *err)
{
    multimodal_observer *observer = opaque;
    char name[64];
    unsigned long long count;
    int length;
    if (!observer || !observer->output_root || !values || !rows || !width ||
        !yvex_core_u64_mul(rows, width, &count) || count > SIZE_MAX / sizeof(float))
        goto failed;
    if (stage == YVEX_VISION_OBSERVE_PATCH)
        length = snprintf(name, sizeof(name), "vision-patch-projection.yvex.f32");
    else if (stage == YVEX_VISION_OBSERVE_POSITION)
        length = snprintf(name, sizeof(name), "vision-position.yvex.f32");
    else if (stage == YVEX_VISION_OBSERVE_BLOCK &&
             (index < 2ull || index == 8ull || index == 16ull ||
              index == 24ull || index == 26ull))
        length = snprintf(name, sizeof(name), "vision-block-%llu.yvex.f32", index);
    else if (index == 0ull && stage == YVEX_VISION_OBSERVE_NORM1)
        length = snprintf(name, sizeof(name), "vision-block-0-norm1.yvex.f32");
    else if (index == 0ull && stage == YVEX_VISION_OBSERVE_QKV)
        length = snprintf(name, sizeof(name), "vision-block-0-qkv.yvex.f32");
    else if (index == 0ull && stage == YVEX_VISION_OBSERVE_QUERY)
        length = snprintf(name, sizeof(name), "vision-block-0-query.yvex.f32");
    else if (index == 0ull && stage == YVEX_VISION_OBSERVE_KEY)
        length = snprintf(name, sizeof(name), "vision-block-0-key.yvex.f32");
    else if (index == 0ull && stage == YVEX_VISION_OBSERVE_ATTENTION)
        length = snprintf(name, sizeof(name), "vision-block-0-attention.yvex.f32");
    else if (index == 0ull && stage == YVEX_VISION_OBSERVE_ATTENTION_PROJECTION)
        length = snprintf(name, sizeof(name), "vision-block-0-attention-projection.yvex.f32");
    else if (index == 0ull && stage == YVEX_VISION_OBSERVE_NORM2)
        length = snprintf(name, sizeof(name), "vision-block-0-norm2.yvex.f32");
    else if (index == 0ull && stage == YVEX_VISION_OBSERVE_FF1)
        length = snprintf(name, sizeof(name), "vision-block-0-ff1.yvex.f32");
    else if (index == 0ull && stage == YVEX_VISION_OBSERVE_GELU)
        length = snprintf(name, sizeof(name), "vision-block-0-gelu.yvex.f32");
    else if (index == 0ull && stage == YVEX_VISION_OBSERVE_FF2)
        length = snprintf(name, sizeof(name), "vision-block-0-ff2.yvex.f32");
    else
        return YVEX_OK;
    if (length <= 0 || (size_t)length >= sizeof(name) ||
        !evidence_write(observer->output_root, name, values,
                        (size_t)count * sizeof(float)))
        goto failed;
    return YVEX_OK;
failed:
    yvex_error_set(err, YVEX_ERR_IO, "minimax-h3.vision.evidence",
                   "vision stage evidence could not be written completely");
    return YVEX_ERR_IO;
}

static int multimodal_execute(
    const char *artifact_path, const char *image_path, const char *prompt,
    const char *evidence_root, const char *reference_root)
{
    yvex_model_context model = {0};
    yvex_complete_artifact_admission admission;
    yvex_artifact_admission_failure failure = {0};
    yvex_runtime_component_session *session = NULL;
    yvex_runtime_av_conditioning_result result;
    yvex_media_condition condition = {
        .schema_version = YVEX_MEDIA_CONDITION_SCHEMA_V1,
        .kind = YVEX_MEDIA_CONDITION_IMAGE,
        .role = YVEX_MEDIA_CONDITION_FIRST,
        .source_path = image_path,
    };
    yvex_media_conditioning_request request = {0};
    multimodal_observer observer = {evidence_root, reference_root};
    yvex_image image = {0};
    float *output = NULL;
    unsigned int *tags = NULL;
    unsigned long long maximum_tokens = 256ull, output_values;
    char output_path[1024];
    yvex_error err, cleanup;
    int rc, cleanup_rc;

    output_values = maximum_tokens * TEXT_HIDDEN;
    output = calloc((size_t)output_values, sizeof(*output));
    tags = calloc((size_t)maximum_tokens, sizeof(*tags));
    if (!output || !tags || !evidence_path(output_path, evidence_root,
                                            "conditioning.yvex.f32")) {
        free(tags); free(output);
        return 2;
    }
    yvex_error_clear(&err);
    rc = yvex_model_context_open(artifact_path, &model, &err);
    if (rc == YVEX_OK)
        rc = yvex_family_tokenizer_open(&model.tokenizer, model.gguf, &err);
    if (rc == YVEX_OK)
        rc = yvex_graph_register_minimax_h3()->component_admit(
            "text_encoder", model.artifact, model.gguf, model.table, NULL,
            &admission, NULL, &failure, &err);
    if (rc == YVEX_OK)
        rc = yvex_image_decode_file(&image, image_path,
                                    256ull * 1024ull * 1024ull, &err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_component_session_open(
            &session, &admission, model.artifact, model.gguf, model.table,
            YVEX_BACKEND_KIND_CUDA, admission.payload_bytes,
            80ull * 1024ull * 1024ull * 1024ull, &err);
    request = (yvex_media_conditioning_request){
        .schema_version = YVEX_MEDIA_CONDITIONING_SCHEMA_V2,
        .prompt = prompt,
        .tokenizer = model.tokenizer,
        .conditions = &condition,
        .condition_images = &image,
        .condition_count = 1ull,
        .width = 192ull,
        .height = 192ull,
        .layer_count = 50ull,
        .maximum_prompt_tokens = maximum_tokens,
        .text_session = session,
        .conditioning = output,
        .text_tags = tags,
        .conditioning_capacity = output_values,
        .text_tag_capacity = maximum_tokens,
        .observe = multimodal_observe,
        .observer_context = &observer,
        .vision_observe = vision_observe,
        .vision_observer_context = &observer,
    };
    if (rc == YVEX_OK)
        rc = yvex_backend_minimax_h3_fl2va_condition(&request, &result, &err);
    yvex_error_clear(&cleanup);
    cleanup_rc = yvex_runtime_component_session_close(&session, &cleanup);
    if (cleanup_rc != YVEX_OK) {
        rc = cleanup_rc;
        err = cleanup;
    }
    if (rc == YVEX_OK &&
        !evidence_write(evidence_root, "conditioning.yvex.f32", output,
                        (size_t)(result.token_count * TEXT_HIDDEN) * sizeof(float)))
        rc = YVEX_ERR_IO;
    if (rc == YVEX_OK && reference_root &&
        !evidence_float_compare(reference_root, "conditioning.reference.f32", output,
                                result.token_count * TEXT_HIDDEN,
                                0.012, 0.9999, INFINITY)) {
        yvex_error_set(&err, YVEX_ERR_FORMAT, "minimax-h3.multimodal-conditioning.oracle",
                       "multimodal Qwen conditioning differs from the released oracle");
        rc = YVEX_ERR_FORMAT;
    }
    if (rc == YVEX_OK)
        printf("multimodal_conditioning=accepted tokens=%llu hidden=%llu layers=%llu "
               "kernels=%llu resident_bytes=%llu device_bytes=%llu\n"
               "prompt_identity=%s\nprocessor_identity=%s\nvision_identity=%s\n"
               "residency_identity=%s\nexecution_identity=%s\nmedia_identity=%s\n",
               result.token_count, result.hidden_width, result.layer_count,
               result.kernel_launches, result.resident_bytes, result.device_bytes,
               result.prompt_identity, result.processor_identity,
               result.vision_identity, result.residency_identity,
               result.execution_identity, result.media_identities[0]);
    else
        fprintf(stderr, "multimodal_conditioning=refused field=%s where=%s message=%s\n",
                failure.field, yvex_error_where(&err), yvex_error_message(&err));
    yvex_image_close(&image);
    yvex_model_context_close(&model);
    free(tags); free(output);
    return rc == YVEX_OK ? 0 : 1;
}

int main(int argc, char **argv)
{
    yvex_artifact_options options = {0};
    yvex_artifact *artifact = NULL;
    yvex_tensor_table *tensors = NULL;
    yvex_gguf *gguf = NULL;
    yvex_minimax_h3_conditioning_result result;
    float output[TEXT_HIDDEN];
    char *end = NULL;
    unsigned long token_value;
    unsigned int token;
    yvex_error err;
    int execution_mode = 0, rc;

    if ((argc == 6 || argc == 7) && strcmp(argv[1], "--multimodal") == 0)
        return multimodal_execute(argv[2], argv[3], argv[4], argv[5],
                                  argc == 7 ? argv[6] : NULL);
    if (argc != 5 && argc != 6) {
        fprintf(stderr,
                "usage: minimax_h3_text TEXT_GGUF TOKEN OUTPUT_F32 REFERENCE_F32 "
                "[layer0|layer0-proof|encoder50]\n"
                "       minimax_h3_text --multimodal TEXT_GGUF IMAGE_PNG PROMPT "
                "EVIDENCE_ROOT\n");
        return 2;
    }
    if (argc == 6) {
        if (strcmp(argv[5], "layer0") == 0) execution_mode = 1;
        else if (strcmp(argv[5], "layer0-proof") == 0) execution_mode = 2;
        else if (strcmp(argv[5], "encoder50") == 0) execution_mode = 3;
        else {
            fprintf(stderr, "text_mode=refused\n");
            return 2;
        }
    }
    errno = 0;
    token_value = strtoul(argv[2], &end, 10);
    if (errno || !end || *end || token_value > 151935ul) {
        fprintf(stderr, "text_token=refused\n");
        return 2;
    }
    token = (unsigned int)token_value;
    options.path = argv[1];
    options.readonly = 1;
    rc = yvex_artifact_open(&artifact, &options, &err);
    if (rc == YVEX_OK) rc = yvex_gguf_open(&gguf, artifact, &err);
    if (rc == YVEX_OK) rc = yvex_tensor_table_from_gguf(&tensors, gguf, &err);
    if (rc == YVEX_OK) {
        const yvex_minimax_h3_graph_api *api = yvex_graph_register_minimax_h3();
        if (execution_mode == 2)
            rc = layer_proof_execute(
                artifact, gguf, tensors, &token, output, &result, &err);
        else
            rc = api->text_encoder_artifact_execute(
                artifact, gguf, tensors, YVEX_BACKEND_KIND_CUDA, &token, 1ull,
                execution_mode == 3 ? 50ull : execution_mode == 1 ? 1ull : 0ull,
                output, TEXT_HIDDEN, 70ull * 1024ull * 1024ull * 1024ull,
                execution_mode ? 512ull * 1024ull * 1024ull : 256ull * 1024ull * 1024ull,
                &result, &err);
    }
    if (rc == YVEX_OK && !output_write(argv[3], output)) {
        yvex_error_set(&err, YVEX_ERR_IO, "minimax-h3.text.output",
                       "conditioning output could not be written completely");
        rc = YVEX_ERR_IO;
    }
    if (rc == YVEX_OK && !reference_compare(argv[4], output, execution_mode)) {
        yvex_error_set(&err, YVEX_ERR_FORMAT, "minimax-h3.text.oracle",
                       "YVEX conditioning differs from the independent BF16 oracle");
        rc = YVEX_ERR_FORMAT;
    }
    if (rc == YVEX_OK) {
        const char *mode = execution_mode == 2 ? "layer0-proof"
                           : execution_mode == 3 ? "encoder50"
                           : execution_mode == 1 ? "layer0" : "embedding";
        printf("text_conditioning=accepted mode=%s\n", mode);
        printf("tokens=%llu hidden=%llu layers=%llu resident_bytes=%llu\n",
               result.token_count, result.hidden_width, result.layer_count,
               result.resident_bytes);
        printf("kernel_launches=%llu h2d_bytes=%llu d2h_bytes=%llu device_bytes=%llu\n",
               result.kernel_launches, result.h2d_bytes, result.d2h_bytes, result.device_bytes);
        printf("residency_identity=%s\nexecution_identity=%s\n",
               result.residency_identity, result.execution_identity);
    } else {
        fprintf(stderr, "text_conditioning=refused where=%s message=%s\n",
                yvex_error_where(&err), yvex_error_message(&err));
    }
    yvex_tensor_table_close(tensors);
    yvex_gguf_close(gguf);
    yvex_artifact_close(artifact);
    return rc == YVEX_OK ? 0 : 1;
}
