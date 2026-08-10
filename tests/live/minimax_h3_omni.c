/* Compare one exact MiniMax-H3 Omni block with an independent BF16 CUDA oracle. */
#include <math.h>
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

enum { OMNI_ROWS = 3u, OMNI_HIDDEN = 5376u, OMNI_TIME = 2688u };
/* Independent PyTorch CUDA BF16 execution bounds the composed block across
 * different legal reduction orders. */
static const double oracle_max_relative_l2 = 0.01;
static const double oracle_min_cosine = 0.9999;
static const double oracle_max_scaled_absolute = 0.002;

static const char *const block_weight_names[YVEX_MINIMAX_H3_OMNI_BLOCK_WEIGHT_COUNT] = {
    "blocks.0.norm1.weight",
    "blocks.0.attn.qkv_proj.weight",
    "blocks.0.attn.q_norm.weight",
    "blocks.0.attn.k_norm.weight",
    "blocks.0.attn.out_proj.weight",
    "blocks.0.norm2.weight",
    "blocks.0.mlp.fc1.weight",
    "blocks.0.mlp.fc2.weight",
    "blocks.0.adaln_proj.linear.weight",
    "blocks.0.adaln_proj.linear.bias",
};

static const float block_positions[OMNI_ROWS * 3u] = {
    0.0f, 0.0f, 0.0f, 3.0f, -2.0f, 5.0f, 4.0f, 0.0f, 7.0f,
};
static const unsigned int block_adaln_indices[OMNI_ROWS] = {1u, 0u, 2u};

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

static int weights_load(
    yvex_materialization_session *session, unsigned char **arena_out,
    unsigned long long *arena_bytes_out,
    yvex_minimax_h3_encoded_weight weights[YVEX_MINIMAX_H3_OMNI_BLOCK_WEIGHT_COUNT],
    char identity[65], yvex_error *err)
{
    const yvex_materialized_tensor_binding *bindings[YVEX_MINIMAX_H3_OMNI_BLOCK_WEIGHT_COUNT];
    yvex_materialization_failure failure;
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned char *arena;
    unsigned long long index, total = 0ull, cursor = 0ull;
    for (index = 0ull; index < YVEX_MINIMAX_H3_OMNI_BLOCK_WEIGHT_COUNT; ++index) {
        bindings[index] = binding_find(session, block_weight_names[index]);
        if (!bindings[index] || !bindings[index]->row_count ||
            !yvex_core_u64_add(total, bindings[index]->encoded_bytes, &total)) {
            yvex_error_set(err, YVEX_ERR_FORMAT, "minimax-h3.omni-proof.binding",
                           "the exact block-zero weight set is unavailable");
            return YVEX_ERR_FORMAT;
        }
    }
    if (!total || total > SIZE_MAX ||
        (arena = mmap(NULL, (size_t)total, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)) == MAP_FAILED) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "minimax-h3.omni-proof.arena",
                       "the block-zero proof residency allocation failed");
        return YVEX_ERR_NOMEM;
    }
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.minimax-h3.omni-block-zero.proof.v1"))
        goto failed;
    for (index = 0ull; index < YVEX_MINIMAX_H3_OMNI_BLOCK_WEIGHT_COUNT; ++index) {
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
        yvex_error_set(err, YVEX_ERR_STATE, "minimax-h3.omni-proof.identity",
                       "the block-zero proof identity could not be sealed");
    return yvex_error_code(err);
}

static int file_read_exact(const char *path, float *values, unsigned long long count)
{
    FILE *file = fopen(path, "rb");
    int valid;
    if (!file) return 0;
    valid = fread(values, sizeof(*values), (size_t)count, file) == count && fgetc(file) == EOF;
    if (fclose(file) != 0) valid = 0;
    return valid;
}

static int output_write(const char *path, const float *values, unsigned long long count)
{
    FILE *file = fopen(path, "wb");
    int valid;
    if (!file) return 0;
    valid = fwrite(values, sizeof(*values), (size_t)count, file) == count;
    if (fclose(file) != 0) valid = 0;
    return valid;
}

static int reference_compare(const float *reference, const float *output,
                             unsigned long long count)
{
    double squared = 0.0, reference_squared = 0.0, output_squared = 0.0, dot = 0.0;
    double rmse, relative_l2, cosine, scaled_absolute;
    float maximum = 0.0f, reference_maximum = 0.0f;
    unsigned long long index;
    for (index = 0ull; index < count; ++index) {
        float difference = fabsf(reference[index] - output[index]);
        if (difference > maximum) maximum = difference;
        if (fabsf(reference[index]) > reference_maximum)
            reference_maximum = fabsf(reference[index]);
        squared += (double)difference * difference;
        reference_squared += (double)reference[index] * reference[index];
        output_squared += (double)output[index] * output[index];
        dot += (double)reference[index] * output[index];
    }
    rmse = sqrt(squared / count);
    relative_l2 = reference_squared > 0.0 ? sqrt(squared / reference_squared) : INFINITY;
    cosine = reference_squared > 0.0 && output_squared > 0.0
                 ? dot / sqrt(reference_squared * output_squared) : -1.0;
    scaled_absolute = reference_maximum > 0.0f ? maximum / reference_maximum : INFINITY;
    printf("oracle_max_absolute_error=%.9g oracle_rmse=%.9g\n", maximum, rmse);
    printf("oracle_relative_l2=%.9g oracle_cosine=%.12g "
           "oracle_scaled_absolute=%.9g\n", relative_l2, cosine, scaled_absolute);
    return relative_l2 <= oracle_max_relative_l2 && cosine >= oracle_min_cosine &&
           scaled_absolute <= oracle_max_scaled_absolute;
}

static int execute_block(
    const yvex_artifact *artifact, const yvex_gguf *gguf,
    const yvex_tensor_table *tensors, const float *hidden, const float *temb,
    float *output, yvex_minimax_h3_omni_result *result, yvex_error *err)
{
    const yvex_minimax_h3_graph_api *graph = yvex_graph_register_minimax_h3();
    const yvex_minimax_h3_backend_api *family = yvex_backend_register_minimax_h3();
    yvex_complete_artifact_admission admission;
    yvex_artifact_admission_failure admission_failure;
    yvex_materialization_options materialization_options;
    yvex_materialization_failure materialization_failure;
    yvex_materialization_plan *plan = NULL;
    yvex_materialization_session *session = NULL;
    yvex_minimax_h3_encoded_weight weights[YVEX_MINIMAX_H3_OMNI_BLOCK_WEIGHT_COUNT] = {{0}};
    yvex_backend_options backend_options = {0};
    yvex_backend_tensor_desc descriptor = {0};
    yvex_backend *backend = NULL;
    yvex_device_tensor *resident = NULL;
    unsigned char *arena = NULL, *registered = NULL;
    unsigned long long arena_bytes = 0ull;
    unsigned int invalid_indices[OMNI_ROWS] = {3u, 0u, 2u};
    char identity[65] = {0};
    int attached = 0, rc, cleanup_rc;
    yvex_error cleanup;
    if (!graph || !family || !family->omni_blocks_cuda) {
        yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "minimax-h3.omni-proof",
                       "the production MiniMax Omni backend is unavailable");
        return YVEX_ERR_UNSUPPORTED;
    }
    rc = graph->component_admit(
        "transformer", artifact, gguf, tensors, &admission, &admission_failure, err);
    yvex_materialization_options_default(&materialization_options);
    materialization_options.max_chunk_bytes = 64ull * 1024ull * 1024ull;
    if (rc == YVEX_OK)
        rc = yvex_materialization_plan_build(
            &plan, &admission, artifact, gguf, tensors, NULL, &materialization_options,
            &materialization_failure, err);
    if (rc == YVEX_OK)
        rc = yvex_materialization_session_open(
            &session, plan, artifact, &materialization_options, &materialization_failure, err);
    if (rc == YVEX_OK)
        rc = yvex_materialization_session_commit(session, &materialization_failure, err);
    if (rc == YVEX_OK)
        rc = weights_load(session, &arena, &arena_bytes, weights, identity, err);
    backend_options.kind = YVEX_BACKEND_KIND_CUDA;
    backend_options.memory_limit_bytes = 3ull * 1024ull * 1024ull * 1024ull;
    if (rc == YVEX_OK) rc = yvex_backend_open(&backend, &backend_options, err);
    descriptor.name = "minimax-h3-omni-block-zero-proof-residency";
    descriptor.dtype = YVEX_DTYPE_I8;
    descriptor.rank = 1u;
    descriptor.dims[0] = descriptor.bytes = arena_bytes;
    registered = arena;
    if (rc == YVEX_OK && (!backend->vtable || !backend->vtable->resident_alloc)) {
        yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "minimax-h3.omni-proof.residency",
                       "the CUDA backend cannot register selected Omni weights");
        rc = YVEX_ERR_UNSUPPORTED;
    }
    if (rc == YVEX_OK)
        rc = backend->vtable->resident_alloc(backend, &descriptor, &resident, &registered, err);
    if (rc == YVEX_OK && registered != arena) {
        yvex_error_set(err, YVEX_ERR_STATE, "minimax-h3.omni-proof.residency",
                       "CUDA registration changed the selected proof address");
        rc = YVEX_ERR_STATE;
    }
    if (rc == YVEX_OK) {
        rc = yvex_backend_resident_attach(backend, arena, arena_bytes, resident, 1ull, err);
        attached = rc == YVEX_OK;
    }
    if (rc == YVEX_OK) {
        yvex_minimax_h3_omni_result refused = {0};
        int refusal = family->omni_blocks_cuda(
            backend, weights, 0ull, identity, arena_bytes, hidden, temb, 1ull,
            block_positions, block_adaln_indices, OMNI_ROWS, output,
            OMNI_ROWS * OMNI_HIDDEN, &refused, err);
        if (refusal != YVEX_ERR_INVALID_ARG || refused.complete) {
            yvex_error_set(err, YVEX_ERR_STATE, "minimax-h3.omni-proof.refusal",
                           "zero-block execution did not fail closed");
            rc = YVEX_ERR_STATE;
        }
    }
    if (rc == YVEX_OK) {
        yvex_minimax_h3_omni_result refused = {0};
        int refusal = family->omni_blocks_cuda(
            backend, weights, 1ull, identity, arena_bytes, hidden, temb, 1ull,
            block_positions, invalid_indices, OMNI_ROWS, output,
            OMNI_ROWS * OMNI_HIDDEN, &refused, err);
        if (refusal != YVEX_ERR_BOUNDS || refused.complete) {
            yvex_error_set(err, YVEX_ERR_STATE, "minimax-h3.omni-proof.refusal",
                           "out-of-range AdaLN selection did not fail closed");
            rc = YVEX_ERR_STATE;
        }
    }
    if (rc == YVEX_OK) {
        yvex_minimax_h3_omni_result refused = {0};
        int refusal = family->omni_blocks_cuda(
            backend, weights, 1ull, identity, arena_bytes, hidden, temb, 1ull,
            block_positions, block_adaln_indices, OMNI_ROWS, output,
            OMNI_ROWS * OMNI_HIDDEN - 1ull, &refused, err);
        if (refusal != YVEX_ERR_INVALID_ARG || refused.complete) {
            yvex_error_set(err, YVEX_ERR_STATE, "minimax-h3.omni-proof.refusal",
                           "undersized output did not fail closed");
            rc = YVEX_ERR_STATE;
        }
    }
    if (rc == YVEX_OK)
        rc = family->omni_blocks_cuda(
            backend, weights, 1ull, identity, arena_bytes, hidden, temb, 1ull,
            block_positions, block_adaln_indices, OMNI_ROWS, output,
            OMNI_ROWS * OMNI_HIDDEN, result, err);
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

int main(int argc, char **argv)
{
    yvex_artifact_options options = {0};
    yvex_artifact *artifact = NULL;
    yvex_tensor_table *tensors = NULL;
    yvex_gguf *gguf = NULL;
    yvex_minimax_h3_omni_result result;
    yvex_error err;
    float *hidden = NULL, *temb = NULL, *output = NULL, *reference = NULL;
    unsigned long long values = OMNI_ROWS * OMNI_HIDDEN;
    int rc = YVEX_OK;
    if (argc != 6) {
        fprintf(stderr, "usage: minimax_h3_omni TRANSFORMER_GGUF INPUT_F32 "
                        "TEMB_F32 OUTPUT_F32 REFERENCE_F32\n");
        return 2;
    }
    hidden = malloc((size_t)values * sizeof(*hidden));
    temb = malloc(OMNI_TIME * sizeof(*temb));
    output = calloc((size_t)values, sizeof(*output));
    reference = malloc((size_t)values * sizeof(*reference));
    if (!hidden || !temb || !output || !reference ||
        !file_read_exact(argv[2], hidden, values) ||
        !file_read_exact(argv[3], temb, OMNI_TIME) ||
        !file_read_exact(argv[5], reference, values)) {
        fprintf(stderr, "omni_fixture=refused\n");
        rc = YVEX_ERR_IO;
        goto done;
    }
    options.path = argv[1];
    options.readonly = 1;
    yvex_error_clear(&err);
    rc = yvex_artifact_open(&artifact, &options, &err);
    if (rc == YVEX_OK) rc = yvex_gguf_open(&gguf, artifact, &err);
    if (rc == YVEX_OK) rc = yvex_tensor_table_from_gguf(&tensors, gguf, &err);
    if (rc == YVEX_OK)
        rc = execute_block(artifact, gguf, tensors, hidden, temb, output, &result, &err);
    if (rc == YVEX_OK && !output_write(argv[4], output, values)) {
        yvex_error_set(&err, YVEX_ERR_IO, "minimax-h3.omni-proof.output",
                       "Omni proof output could not be written completely");
        rc = YVEX_ERR_IO;
    }
    if (rc == YVEX_OK && !reference_compare(reference, output, values)) {
        yvex_error_set(&err, YVEX_ERR_FORMAT, "minimax-h3.omni-proof.oracle",
                       "YVEX Omni block differs from the independent BF16 oracle");
        rc = YVEX_ERR_FORMAT;
    }
    if (rc == YVEX_OK) {
        printf("omni_block=accepted rows=%llu blocks=%llu resident_bytes=%llu\n",
               result.packed_rows, result.block_count, result.resident_bytes);
        printf("kernel_launches=%llu h2d_bytes=%llu d2h_bytes=%llu device_bytes=%llu\n",
               result.kernel_launches, result.h2d_bytes, result.d2h_bytes,
               result.device_bytes);
        printf("residency_identity=%s\nexecution_identity=%s\n",
               result.residency_identity, result.execution_identity);
    } else if (artifact) {
        fprintf(stderr, "omni_block=refused where=%s message=%s\n",
                yvex_error_where(&err), yvex_error_message(&err));
    }
done:
    yvex_tensor_table_close(tensors);
    yvex_gguf_close(gguf);
    yvex_artifact_close(artifact);
    free(reference);
    free(output);
    free(temb);
    free(hidden);
    return rc == YVEX_OK ? 0 : 1;
}
