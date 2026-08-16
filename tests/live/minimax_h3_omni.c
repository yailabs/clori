/* Compare a selected MiniMax-H3 Omni block stack with an independent BF16 CUDA oracle. */
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
#include <yvex/internal/joint_transformer.h>

enum { OMNI_ROWS = 3u, OMNI_HIDDEN = 5376u, OMNI_TIME = 2688u, OMNI_BLOCKS = 50u };
/* Independent PyTorch CUDA BF16 execution bounds the composed block across
 * different legal reduction orders. */
static const double oracle_one_block_max_relative_l2 = 0.01;
static const double oracle_one_block_min_cosine = 0.9999;
/* A multi-row BF16 reduction can move one maximum-magnitude output by a binade ULP even
 * while the aggregate vector stays conformant; the paired L2 and cosine bounds prevent
 * this maximum-element allowance from admitting a structurally different result. */
static const double oracle_one_block_max_scaled_absolute = 0.01;
/* The complete stack admits the established aggregate BF16 envelope only after the strict
 * one-block contract above has ruled out a locally incorrect operation. */
static const double oracle_stack_max_relative_l2 = 0.04;
static const double oracle_stack_min_cosine = 0.9995;
static const double oracle_stack_max_scaled_absolute = 0.04;

static const char *const block_weight_suffixes[YVEX_TRANSFORMER_JOINT_BLOCK_WEIGHT_COUNT] = {
    "norm1.weight",
    "attn.qkv_proj.weight",
    "attn.q_norm.weight",
    "attn.k_norm.weight",
    "attn.out_proj.weight",
    "norm2.weight",
    "mlp.fc1.weight",
    "mlp.fc2.weight",
    "adaln_proj.linear.weight",
    "adaln_proj.linear.bias",
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
    yvex_minimax_h3_encoded_weight *weights, unsigned long long block_count,
    char identity[65], yvex_error *err)
{
    const yvex_materialized_tensor_binding **bindings = NULL;
    char (*names)[96] = NULL;
    yvex_materialization_failure failure;
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned char *arena;
    unsigned long long count, index, total = 0ull, cursor = 0ull;
    if (!weights || !block_count || block_count > OMNI_BLOCKS ||
        !yvex_core_u64_mul(block_count, YVEX_TRANSFORMER_JOINT_BLOCK_WEIGHT_COUNT, &count) ||
        count > SIZE_MAX / sizeof(*bindings) || count > SIZE_MAX / sizeof(*names) ||
        !(bindings = calloc((size_t)count, sizeof(*bindings))) ||
        !(names = calloc((size_t)count, sizeof(*names)))) {
        free(bindings);
        free(names);
        yvex_error_set(err, YVEX_ERR_NOMEM, "minimax-h3.omni-proof.binding",
                       "the selected block binding table could not be allocated");
        return YVEX_ERR_NOMEM;
    }
    for (index = 0ull; index < count; ++index) {
        unsigned long long block = index / YVEX_TRANSFORMER_JOINT_BLOCK_WEIGHT_COUNT;
        unsigned long long weight = index % YVEX_TRANSFORMER_JOINT_BLOCK_WEIGHT_COUNT;
        int length = snprintf(names[index], sizeof(names[index]), "blocks.%llu.%s",
                              block, block_weight_suffixes[weight]);
        if (length < 0 || (size_t)length >= sizeof(names[index])) {
            yvex_error_set(err, YVEX_ERR_BOUNDS, "minimax-h3.omni-proof.binding",
                           "a selected block tensor name exceeds the proof bound");
            goto failed_bindings;
        }
        bindings[index] = binding_find(session, names[index]);
        if (!bindings[index] || !bindings[index]->row_count ||
            !yvex_core_u64_add(total, bindings[index]->encoded_bytes, &total)) {
            yvex_error_set(err, YVEX_ERR_FORMAT, "minimax-h3.omni-proof.binding",
                           "the exact selected block weight set is unavailable");
            goto failed_bindings;
        }
    }
    if (!total || total > SIZE_MAX ||
        (arena = mmap(NULL, (size_t)total, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)) == MAP_FAILED) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "minimax-h3.omni-proof.arena",
                       "the selected block proof residency allocation failed");
        goto failed_bindings;
    }
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.minimax-h3.omni-block-stack.proof.v1") ||
        !yvex_sha256_update_u64(&hash, block_count))
        goto failed;
    for (index = 0ull; index < count; ++index) {
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
    free(names);
    free(bindings);
    return YVEX_OK;
failed:
    munmap(arena, (size_t)total);
    if (yvex_error_code(err) == YVEX_OK)
        yvex_error_set(err, YVEX_ERR_STATE, "minimax-h3.omni-proof.identity",
                       "the selected block proof identity could not be sealed");
failed_bindings:
    free(names);
    free(bindings);
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

static int file_read_u32_exact(
    const char *path, unsigned int *values, unsigned long long count)
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
                             unsigned long long count, unsigned long long block_count)
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
    if (block_count == 1ull)
        return relative_l2 <= oracle_one_block_max_relative_l2 &&
               cosine >= oracle_one_block_min_cosine &&
               scaled_absolute <= oracle_one_block_max_scaled_absolute;
    return relative_l2 <= oracle_stack_max_relative_l2 &&
           cosine >= oracle_stack_min_cosine &&
           scaled_absolute <= oracle_stack_max_scaled_absolute;
}

static int execute_block(
    const yvex_artifact *artifact, const yvex_gguf *gguf,
    const yvex_tensor_table *tensors, const float *hidden, const float *temb,
    const float *positions, const unsigned int *adaln_indices,
    unsigned long long rows, unsigned long long timesteps, unsigned long long block_count,
    float *output, yvex_minimax_h3_omni_result *result, yvex_error *err)
{
    const yvex_minimax_h3_graph_api *graph = yvex_graph_register_minimax_h3();
    const yvex_transformer_joint_recipe *recipe = graph ? graph->omni_recipe() : NULL;
    yvex_complete_artifact_admission admission;
    yvex_artifact_admission_failure admission_failure;
    yvex_materialization_options materialization_options;
    yvex_materialization_failure materialization_failure;
    yvex_materialization_plan *plan = NULL;
    yvex_materialization_session *session = NULL;
    yvex_minimax_h3_encoded_weight *weights = NULL;
    yvex_backend_options backend_options = {0};
    yvex_backend_tensor_desc descriptor = {0};
    yvex_backend *backend = NULL;
    yvex_device_tensor *resident = NULL;
    unsigned char *arena = NULL, *registered = NULL;
    unsigned long long arena_bytes = 0ull;
    unsigned int *invalid_indices = NULL;
    char identity[65] = {0};
    unsigned long long weight_count;
    int attached = 0, rc, cleanup_rc;
    yvex_error cleanup;
    if (!graph || !recipe) {
        yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "minimax-h3.omni-proof",
                       "the production MiniMax Omni backend is unavailable");
        return YVEX_ERR_UNSUPPORTED;
    }
    if (!block_count || block_count > OMNI_BLOCKS ||
        !yvex_core_u64_mul(block_count, YVEX_TRANSFORMER_JOINT_BLOCK_WEIGHT_COUNT,
                           &weight_count) ||
        weight_count > SIZE_MAX / sizeof(*weights) ||
        !(weights = calloc((size_t)weight_count, sizeof(*weights))) ||
        !(invalid_indices = malloc((size_t)rows * sizeof(*invalid_indices)))) {
        free(weights);
        yvex_error_set(err, YVEX_ERR_NOMEM, "minimax-h3.omni-proof",
                       "the bounded refusal fixture could not be allocated");
        return YVEX_ERR_NOMEM;
    }
    memcpy(invalid_indices, adaln_indices, (size_t)rows * sizeof(*invalid_indices));
    invalid_indices[0] = (unsigned int)(timesteps * 3ull);
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
        rc = weights_load(session, &arena, &arena_bytes, weights, block_count, identity, err);
    backend_options.kind = YVEX_BACKEND_KIND_CUDA;
    backend_options.memory_limit_bytes = 80ull * 1024ull * 1024ull * 1024ull;
    if (rc == YVEX_OK) rc = yvex_backend_open(&backend, &backend_options, err);
    descriptor.name = "minimax-h3-omni-block-zero-proof-residency";
    descriptor.dtype = YVEX_DTYPE_I8;
    descriptor.rank = 1u;
    descriptor.dims[0] = descriptor.bytes = arena_bytes;
    registered = arena;
    if (rc == YVEX_OK)
        rc = yvex_backend_resident_alloc(backend, &descriptor, &resident, &registered, err);
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
        int refusal = yvex_backend_transformer_joint_blocks_cuda(
            backend, recipe, weights, 0ull, identity, arena_bytes, hidden, temb, timesteps,
            positions, adaln_indices, rows, output, rows * OMNI_HIDDEN, &refused, err);
        if (refusal != YVEX_ERR_INVALID_ARG || refused.complete) {
            yvex_error_set(err, YVEX_ERR_STATE, "minimax-h3.omni-proof.refusal",
                           "zero-block execution did not fail closed");
            rc = YVEX_ERR_STATE;
        }
    }
    if (rc == YVEX_OK) {
        yvex_minimax_h3_omni_result refused = {0};
        int refusal = yvex_backend_transformer_joint_blocks_cuda(
            backend, recipe, weights, block_count, identity, arena_bytes, hidden, temb, timesteps,
            positions, invalid_indices, rows, output, rows * OMNI_HIDDEN, &refused, err);
        if (refusal != YVEX_ERR_BOUNDS || refused.complete) {
            yvex_error_set(err, YVEX_ERR_STATE, "minimax-h3.omni-proof.refusal",
                           "out-of-range AdaLN selection did not fail closed");
            rc = YVEX_ERR_STATE;
        }
    }
    if (rc == YVEX_OK) {
        yvex_minimax_h3_omni_result refused = {0};
        int refusal = yvex_backend_transformer_joint_blocks_cuda(
            backend, recipe, weights, block_count, identity, arena_bytes, hidden, temb, timesteps,
            positions, adaln_indices, rows, output, rows * OMNI_HIDDEN - 1ull, &refused, err);
        if (refusal != YVEX_ERR_INVALID_ARG || refused.complete) {
            yvex_error_set(err, YVEX_ERR_STATE, "minimax-h3.omni-proof.refusal",
                           "undersized output did not fail closed");
            rc = YVEX_ERR_STATE;
        }
    }
    if (rc == YVEX_OK)
        rc = yvex_backend_transformer_joint_blocks_cuda(
            backend, recipe, weights, block_count, identity, arena_bytes, hidden, temb, timesteps,
            positions, adaln_indices, rows, output, rows * OMNI_HIDDEN, result, err);
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
    free(invalid_indices);
    free(weights);
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
    float *hidden = NULL, *temb = NULL, *dynamic_positions = NULL;
    float *output = NULL, *reference = NULL;
    unsigned int *dynamic_adaln_indices = NULL;
    const float *positions = block_positions;
    const unsigned int *adaln_indices = block_adaln_indices;
    unsigned long long rows = OMNI_ROWS, timesteps = 1ull, values, temb_values;
    const char *blocks_text = getenv("YVEX_MINIMAX_H3_BLOCKS");
    char *blocks_end = NULL;
    unsigned long long block_count = blocks_text ? strtoull(blocks_text, &blocks_end, 10) : 1ull;
    const char *output_path, *reference_path;
    int dynamic = 0;
    int rc = YVEX_OK;
    if (argc != 6 && argc != 10) {
        fprintf(stderr, "usage: minimax_h3_omni TRANSFORMER_GGUF INPUT_F32 "
                        "TEMB_F32 OUTPUT_F32 REFERENCE_F32\n"
                        "   or: minimax_h3_omni TRANSFORMER_GGUF INPUT_F32 "
                        "TEMB_F32 POSITIONS_F32 ADALN_U32 OUTPUT_F32 REFERENCE_F32 "
                        "ROWS TIMESTEPS\n"
                        "Set YVEX_MINIMAX_H3_BLOCKS=1..50 to select the block stack.\n");
        return 2;
    }
    if ((blocks_text && (!blocks_end || *blocks_end)) || !block_count ||
        block_count > OMNI_BLOCKS)
        return 2;
    if (argc == 10) {
        char *rows_end = NULL, *timesteps_end = NULL;
        rows = strtoull(argv[8], &rows_end, 10);
        timesteps = strtoull(argv[9], &timesteps_end, 10);
        if (!rows_end || *rows_end || !rows || rows > 2048ull ||
            !timesteps_end || *timesteps_end || !timesteps || timesteps > 64ull)
            return 2;
        dynamic = 1;
    }
    if (!yvex_core_u64_mul(rows, OMNI_HIDDEN, &values) ||
        !yvex_core_u64_mul(timesteps, OMNI_TIME, &temb_values) ||
        values > SIZE_MAX / sizeof(float) || temb_values > SIZE_MAX / sizeof(float))
        return 2;
    hidden = malloc((size_t)values * sizeof(*hidden));
    temb = malloc((size_t)temb_values * sizeof(*temb));
    output = calloc((size_t)values, sizeof(*output));
    reference = malloc((size_t)values * sizeof(*reference));
    if (dynamic) {
        dynamic_positions = malloc((size_t)(rows * 3ull) * sizeof(*dynamic_positions));
        dynamic_adaln_indices =
            malloc((size_t)rows * sizeof(*dynamic_adaln_indices));
        positions = dynamic_positions;
        adaln_indices = dynamic_adaln_indices;
    }
    output_path = dynamic ? argv[6] : argv[4];
    reference_path = dynamic ? argv[7] : argv[5];
    if (!hidden || !temb || !output || !reference ||
        !file_read_exact(argv[2], hidden, values) ||
        !file_read_exact(argv[3], temb, temb_values) ||
        (dynamic && (!dynamic_positions || !dynamic_adaln_indices ||
                     !file_read_exact(argv[4], dynamic_positions, rows * 3ull) ||
                     !file_read_u32_exact(argv[5], dynamic_adaln_indices, rows))) ||
        !file_read_exact(reference_path, reference, values)) {
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
        rc = execute_block(artifact, gguf, tensors, hidden, temb, positions, adaln_indices,
                           rows, timesteps, block_count, output, &result, &err);
    if (rc == YVEX_OK && !output_write(output_path, output, values)) {
        yvex_error_set(&err, YVEX_ERR_IO, "minimax-h3.omni-proof.output",
                       "Omni proof output could not be written completely");
        rc = YVEX_ERR_IO;
    }
    if (rc == YVEX_OK && !reference_compare(reference, output, values, block_count)) {
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
    if (dynamic) {
        free(dynamic_adaln_indices);
        free(dynamic_positions);
    }
    return rc == YVEX_OK ? 0 : 1;
}
