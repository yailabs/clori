/* Exercise the exact Text Encoder embedding through staged GB10 residency. */
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include <yvex/artifact.h>
#include <yvex/gguf.h>
#include <yvex/internal/artifact.h>
#include <yvex/internal/backend.h>
#include <yvex/internal/component.h>
#include <yvex/internal/core.h>
#include <yvex/internal/families/minimax_h3.h>
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
        .embedding_identity_domain = "yvex.minimax-h3.text-conditioning.cuda.v1",
        .encoder_identity_domain = "yvex.minimax-h3.qwen-text-stack.cuda.v1",
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

    if (argc != 5 && argc != 6) {
        fprintf(stderr,
                "usage: minimax_h3_text TEXT_GGUF TOKEN OUTPUT_F32 REFERENCE_F32 "
                "[layer0|layer0-proof|encoder50]\n");
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
