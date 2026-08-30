/* Admit, identify, prepare, and release one exact joint-transformer execution contract. */
#include "src/backend/cuda/component_ops.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <yvex/internal/core.h>
#include <yvex/qtype.h>

enum {
    JOINT_HIDDEN = 5376u,
    JOINT_HEADS = 56u,
    JOINT_HEAD_DIM = 128u,
    JOINT_ATTENTION_WIDTH = 7168u,
    JOINT_FFN = 14336u,
    JOINT_TIME = 2688u,
    JOINT_ROTARY = 96u,
    JOINT_MODALITIES = 3u,
    JOINT_PARAMETERS = 6u,
    JOINT_BLOCKS = 50u,
    JOINT_MAX_TIMESTEPS = 64u,
    JOINT_IMPLEMENTATION_MAX_PACKED_ROWS = 131072u
};

static int joint_contract_refuse(yvex_error *err, yvex_status status,
                                 const char *stage, const char *message)
{
    yvex_error_set(err, status, stage, message);
    return status;
}

static int joint_linear_physical_supported(
    const yvex_transformer_linear_physical_plan *plan,
    const char *semantic_domain,
    const yvex_transformer_linear_requirement *requirement)
{
    yvex_error err;
    return plan && semantic_domain && requirement &&
           !strcmp(plan->semantic_domain, semantic_domain) &&
           plan->operation == requirement->operation &&
           plan->numeric_contract == requirement->publication_contract &&
           plan->source_dtype == requirement->source_dtype &&
           plan->bias == requirement->bias &&
           plan->input_width == requirement->input_width &&
           plan->output_width == requirement->output_width &&
           plan->backend == YVEX_BACKEND_KIND_CUDA &&
           yvex_transformer_linear_physical_validate(plan, &err) == YVEX_OK;
}

int yvex_cuda_joint_recipe_supported(
    const yvex_transformer_joint_recipe *recipe)
{
    return recipe && recipe->schema_version == YVEX_TRANSFORMER_JOINT_SCHEMA_V4 &&
           recipe->qkv_layout == YVEX_TRANSFORMER_QKV_LAYOUT_PER_HEAD_THREE &&
           recipe->swiglu_layout == YVEX_TRANSFORMER_SWIGLU_LAYOUT_GATE_THEN_UP &&
           recipe->identity_domain && recipe->identity_domain[0] &&
           recipe->hidden_width == JOINT_HIDDEN &&
           recipe->attention_heads == JOINT_HEADS &&
           recipe->head_dimension == JOINT_HEAD_DIM &&
           recipe->attention_width == JOINT_ATTENTION_WIDTH &&
           recipe->ffn_width == JOINT_FFN &&
           recipe->timestep_width == JOINT_TIME &&
           recipe->rotary_width == JOINT_ROTARY &&
           recipe->modality_count == JOINT_MODALITIES &&
           recipe->modulation_parameters == JOINT_PARAMETERS &&
           recipe->block_count == JOINT_BLOCKS &&
           recipe->refiner_block_count == 2ull &&
           recipe->maximum_timesteps == JOINT_MAX_TIMESTEPS &&
           recipe->maximum_packed_rows &&
           recipe->maximum_packed_rows <= JOINT_IMPLEMENTATION_MAX_PACKED_ROWS &&
           recipe->video_input_width == 96ull &&
           recipe->audio_input_width == 32ull &&
           recipe->condition_input_width == 5120ull &&
           recipe->video_output.operation ==
               YVEX_TRANSFORMER_LINEAR_OPERATION_JOINT_VIDEO_OUTPUT &&
           recipe->video_output.publication_contract ==
               YVEX_TRANSFORMER_LINEAR_NUMERIC_SOURCE_EXACT &&
           recipe->video_output.source_dtype == YVEX_DTYPE_F32 &&
           recipe->video_output.input_width == JOINT_HIDDEN &&
           recipe->video_output.output_width == 96ull &&
           recipe->video_output.bias == 1 &&
           recipe->audio_output.operation ==
               YVEX_TRANSFORMER_LINEAR_OPERATION_JOINT_AUDIO_OUTPUT &&
           recipe->audio_output.publication_contract ==
               YVEX_TRANSFORMER_LINEAR_NUMERIC_SOURCE_EXACT &&
           recipe->audio_output.source_dtype == YVEX_DTYPE_F32 &&
           recipe->audio_output.input_width == JOINT_HIDDEN &&
           recipe->audio_output.output_width == 32ull &&
           recipe->audio_output.bias == 1;
}

static int joint_external_weight_valid(
    const yvex_transformer_joint_encoded_weight *weight,
    unsigned int qtype, unsigned long long rows,
    unsigned long long width)
{
    unsigned long long row_bytes, bytes;
    return weight && weight->encoded && weight->qtype == qtype &&
           weight->row_count == rows && weight->row_width == width &&
           yvex_core_u64_mul(
               width, qtype == YVEX_GGUF_QTYPE_F32 ? 4ull : 2ull,
               &row_bytes) &&
           yvex_core_u64_mul(rows, row_bytes, &bytes) &&
           weight->row_bytes == row_bytes && weight->encoded_bytes == bytes;
}

int yvex_cuda_joint_external_valid(
    const yvex_transformer_joint_encoded_weight *weights,
    unsigned long long *bytes)
{
    static const unsigned int qtypes[YVEX_TRANSFORMER_JOINT_EXTERNAL_WEIGHT_COUNT] = {
        YVEX_GGUF_QTYPE_F32, YVEX_GGUF_QTYPE_F32, YVEX_GGUF_QTYPE_F32,
        YVEX_GGUF_QTYPE_F32, YVEX_GGUF_QTYPE_BF16, YVEX_GGUF_QTYPE_BF16,
        YVEX_GGUF_QTYPE_F32, YVEX_GGUF_QTYPE_F32, YVEX_GGUF_QTYPE_F32,
        YVEX_GGUF_QTYPE_F32, YVEX_GGUF_QTYPE_BF16, YVEX_GGUF_QTYPE_BF16,
        YVEX_GGUF_QTYPE_BF16, YVEX_GGUF_QTYPE_BF16, YVEX_GGUF_QTYPE_BF16,
        YVEX_GGUF_QTYPE_BF16, YVEX_GGUF_QTYPE_BF16, YVEX_GGUF_QTYPE_BF16,
        YVEX_GGUF_QTYPE_BF16, YVEX_GGUF_QTYPE_BF16, YVEX_GGUF_QTYPE_BF16,
        YVEX_GGUF_QTYPE_BF16, YVEX_GGUF_QTYPE_BF16, YVEX_GGUF_QTYPE_BF16,
        YVEX_GGUF_QTYPE_BF16, YVEX_GGUF_QTYPE_BF16, YVEX_GGUF_QTYPE_BF16,
        YVEX_GGUF_QTYPE_F32, YVEX_GGUF_QTYPE_BF16, YVEX_GGUF_QTYPE_BF16,
        YVEX_GGUF_QTYPE_BF16, YVEX_GGUF_QTYPE_F32, YVEX_GGUF_QTYPE_F32,
        YVEX_GGUF_QTYPE_F32, YVEX_GGUF_QTYPE_F32,
    };
    static const unsigned long long rows[YVEX_TRANSFORMER_JOINT_EXTERNAL_WEIGHT_COUNT] = {
        5376u, 1u, 5376u, 1u, 5376u, 1u, 5376u, 1u, 2688u, 1u,
        1u, 21504u, 1u, 1u, 5376u, 1u, 28672u, 5376u,
        1u, 21504u, 1u, 1u, 5376u, 1u, 28672u, 5376u, 1u,
        1u, 1u, 10752u, 1u, 96u, 1u, 32u, 1u,
    };
    static const unsigned long long widths[YVEX_TRANSFORMER_JOINT_EXTERNAL_WEIGHT_COUNT] = {
        32u, 5376u, 96u, 5376u, 5120u, 5376u, 256u, 5376u, 5376u, 2688u,
        5376u, 5376u, 128u, 128u, 7168u, 5376u, 5376u, 14336u,
        5376u, 5376u, 128u, 128u, 7168u, 5376u, 5376u, 14336u, 5376u,
        16u, 5376u, 2688u, 10752u, 5376u, 96u, 5376u, 32u,
    };
    unsigned long long index, next;
    if (bytes) *bytes = 0ull;
    if (!weights || !bytes) return 0;
    for (index = 0ull;
         index < YVEX_TRANSFORMER_JOINT_EXTERNAL_WEIGHT_COUNT; ++index) {
        if (!joint_external_weight_valid(
                weights + index, qtypes[index], rows[index], widths[index]) ||
            !yvex_core_u64_add(*bytes, weights[index].encoded_bytes, &next))
            return 0;
        *bytes = next;
    }
    return 1;
}

int yvex_cuda_joint_request_valid(
    const yvex_transformer_joint_request *request,
    unsigned long long *video_values, unsigned long long *audio_values,
    yvex_error *err)
{
    unsigned char *seen = NULL;
    unsigned long long kind, row, total;
    const unsigned int *indices[3];
    unsigned long long counts[3];
    if (!request || !yvex_cuda_joint_recipe_supported(request->recipe) ||
        !request->video || !request->audio || !request->conditioning ||
        !request->timesteps || !request->position_ids ||
        !request->video_indices || !request->audio_indices ||
        !request->text_indices || !request->timestep_indices ||
        !request->token_tags || !request->video_output ||
        !request->audio_output || !request->video_rows ||
        !request->audio_rows || !request->text_rows ||
        !request->timestep_count ||
        request->timestep_count > JOINT_MAX_TIMESTEPS ||
        !request->packed_rows ||
        request->packed_rows > request->recipe->maximum_packed_rows ||
        !request->block_count || request->block_count > JOINT_BLOCKS ||
        !joint_linear_physical_supported(
            &request->video_output_physical, request->recipe->identity_domain,
            &request->recipe->video_output) ||
        !joint_linear_physical_supported(
            &request->audio_output_physical, request->recipe->identity_domain,
            &request->recipe->audio_output) ||
        !yvex_core_u64_add(request->video_rows, request->audio_rows, &total) ||
        !yvex_core_u64_add(total, request->text_rows, &total) ||
        total != request->packed_rows ||
        !yvex_core_u64_mul(request->video_rows, 96ull, video_values) ||
        !yvex_core_u64_mul(request->audio_rows, 32ull, audio_values) ||
        *video_values > request->video_output_capacity ||
        *audio_values > request->audio_output_capacity)
        return joint_contract_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "cuda.transformer.joint.transformer.request",
            "one complete bounded packed joint request is required");
    seen = calloc((size_t)request->packed_rows, 1u);
    if (!seen)
        return joint_contract_refuse(
            err, YVEX_ERR_NOMEM, "cuda.transformer.joint.transformer.layout",
            "packed-row partition allocation failed");
    indices[0] = request->video_indices;
    counts[0] = request->video_rows;
    indices[1] = request->text_indices;
    counts[1] = request->text_rows;
    indices[2] = request->audio_indices;
    counts[2] = request->audio_rows;
    for (kind = 0ull; kind < 3ull; ++kind) {
        for (row = 0ull; row < counts[kind]; ++row) {
            unsigned int packed = indices[kind][row];
            if (packed >= request->packed_rows || seen[packed] ||
                request->token_tags[packed] >= request->recipe->modality_count) {
                free(seen);
                return joint_contract_refuse(
                    err, YVEX_ERR_FORMAT,
                    "cuda.transformer.joint.transformer.layout",
                    "packed modality indices must form one exact tagged partition");
            }
            seen[packed] = 1u;
        }
    }
    for (row = 0ull; row < request->packed_rows; ++row) {
        if (!seen[row] ||
            request->timestep_indices[row] >= request->timestep_count ||
            !isfinite(request->position_ids[row * 3ull]) ||
            !isfinite(request->position_ids[row * 3ull + 1ull]) ||
            !isfinite(request->position_ids[row * 3ull + 2ull])) {
            free(seen);
            return joint_contract_refuse(
                err, YVEX_ERR_FORMAT,
                "cuda.transformer.joint.transformer.layout",
                "packed rows require finite positions and admitted timesteps");
        }
    }
    for (row = 0ull; row < request->timestep_count; ++row) {
        if (!isfinite(request->timesteps[row]) ||
            request->timesteps[row] < 0.0f || request->timesteps[row] > 1.0f) {
            free(seen);
            return joint_contract_refuse(
                err, YVEX_ERR_FORMAT,
                "cuda.transformer.joint.transformer.timestep",
                "distinct timesteps must be finite values in [0,1]");
        }
    }
    free(seen);
    return YVEX_OK;
}

static int joint_hash_floats(yvex_sha256 *hash, const float *values,
                             unsigned long long count)
{
    unsigned long long index;
    for (index = 0ull; index < count; ++index) {
        uint32_t bits;
        memcpy(&bits, values + index, sizeof(bits));
        if (!yvex_sha256_update_u64(hash, bits)) return 0;
    }
    return 1;
}

int yvex_cuda_joint_execution_identity(
    const yvex_transformer_joint_request *request,
    const char *residency_identity, const char *block_identity,
    const float *video, unsigned long long video_values,
    const float *audio, unsigned long long audio_values,
    char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.transformer.joint.cuda.v1") ||
        !yvex_sha256_update_text(&hash, request->recipe->identity_domain) ||
        !yvex_sha256_update_u64(&hash, request->recipe->qkv_layout) ||
        !yvex_sha256_update_u64(&hash, request->recipe->swiglu_layout) ||
        !yvex_sha256_update_text(
            &hash, request->video_output_physical.physical_identity) ||
        !yvex_sha256_update_text(
            &hash, request->audio_output_physical.physical_identity) ||
        !yvex_sha256_update_text(&hash, residency_identity) ||
        !yvex_sha256_update_text(&hash, block_identity) ||
        !yvex_sha256_update_u64(&hash, request->video_rows) ||
        !yvex_sha256_update_u64(&hash, request->audio_rows) ||
        !yvex_sha256_update_u64(&hash, request->text_rows) ||
        !yvex_sha256_update_u64(&hash, request->timestep_count) ||
        !yvex_sha256_update_u64(&hash, request->packed_rows) ||
        !yvex_sha256_update_u64(&hash, request->block_count) ||
        !joint_hash_floats(
            &hash, request->video, request->video_rows * 96ull) ||
        !joint_hash_floats(
            &hash, request->audio, request->audio_rows * 32ull) ||
        !joint_hash_floats(
            &hash, request->conditioning, request->text_rows * 5120ull) ||
        !joint_hash_floats(
            &hash, request->timesteps, request->timestep_count) ||
        !joint_hash_floats(
            &hash, request->position_ids, request->packed_rows * 3ull))
        return 0;
    for (index = 0ull; index < request->video_rows; ++index)
        if (!yvex_sha256_update_u64(&hash, request->video_indices[index]))
            return 0;
    for (index = 0ull; index < request->audio_rows; ++index)
        if (!yvex_sha256_update_u64(&hash, request->audio_indices[index]))
            return 0;
    for (index = 0ull; index < request->text_rows; ++index)
        if (!yvex_sha256_update_u64(&hash, request->text_indices[index]))
            return 0;
    for (index = 0ull; index < request->packed_rows; ++index)
        if (!yvex_sha256_update_u64(&hash, request->timestep_indices[index]) ||
            !yvex_sha256_update_u64(&hash, request->token_tags[index]))
            return 0;
    if (!joint_hash_floats(&hash, video, video_values) ||
        !joint_hash_floats(&hash, audio, audio_values) ||
        !yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

static int joint_host_extent(unsigned long long rows,
                             unsigned long long width,
                             unsigned long long element_bytes,
                             unsigned long long *bytes)
{
    unsigned long long values;
    return yvex_core_u64_mul(rows, width, &values) &&
           yvex_core_u64_mul(values, element_bytes, bytes);
}

static int joint_host_plan(
    const yvex_transformer_joint_request *request,
    unsigned long long host[YVEX_CUDA_JOINT_HOST_COUNT], yvex_error *err)
{
#define HOST(slot_, rows_, width_, size_)                                      \
    if (!joint_host_extent((rows_), (width_), (size_), host + (slot_)))       \
        return joint_contract_refuse(                                          \
            err, YVEX_ERR_BOUNDS, "cuda.transformer.joint.prepare",          \
            "joint host arena geometry overflowed")
    HOST(YVEX_CUDA_JOINT_HOST_VIDEO_EMBED, request->video_rows,
         request->recipe->hidden_width, sizeof(float));
    HOST(YVEX_CUDA_JOINT_HOST_AUDIO_EMBED, request->audio_rows,
         request->recipe->hidden_width, sizeof(float));
    HOST(YVEX_CUDA_JOINT_HOST_TEXT_EMBED, request->text_rows,
         request->recipe->hidden_width, sizeof(float));
    HOST(YVEX_CUDA_JOINT_HOST_TEXT_REFINED, request->text_rows,
         request->recipe->hidden_width, sizeof(float));
    HOST(YVEX_CUDA_JOINT_HOST_PACKED, request->packed_rows,
         request->recipe->hidden_width, sizeof(float));
    HOST(YVEX_CUDA_JOINT_HOST_BLOCK_OUTPUT, request->packed_rows,
         request->recipe->hidden_width, sizeof(float));
    HOST(YVEX_CUDA_JOINT_HOST_BLOCK_STAGED, request->packed_rows,
         request->recipe->hidden_width, sizeof(float));
    HOST(YVEX_CUDA_JOINT_HOST_TIME_EMBED, request->recipe->maximum_timesteps,
         request->recipe->timestep_width, sizeof(float));
    HOST(YVEX_CUDA_JOINT_HOST_NORMALIZED, request->packed_rows,
         request->recipe->hidden_width, sizeof(float));
    HOST(YVEX_CUDA_JOINT_HOST_ALL_VIDEO, request->packed_rows,
         request->recipe->video_output.output_width, sizeof(float));
    HOST(YVEX_CUDA_JOINT_HOST_ALL_AUDIO, request->packed_rows,
         request->recipe->audio_output.output_width, sizeof(float));
    HOST(YVEX_CUDA_JOINT_HOST_STAGED_VIDEO, request->video_rows,
         request->recipe->video_output.output_width, sizeof(float));
    HOST(YVEX_CUDA_JOINT_HOST_STAGED_AUDIO, request->audio_rows,
         request->recipe->audio_output.output_width, sizeof(float));
    HOST(YVEX_CUDA_JOINT_HOST_ADALN, request->packed_rows, 1ull,
         sizeof(unsigned int));
    HOST(YVEX_CUDA_JOINT_HOST_ROPE_COSINE, request->packed_rows,
         request->recipe->rotary_width, sizeof(float));
    HOST(YVEX_CUDA_JOINT_HOST_ROPE_SINE, request->packed_rows,
         request->recipe->rotary_width, sizeof(float));
#undef HOST
    return YVEX_OK;
}

static unsigned long long joint_elapsed_ns(const struct timespec *start,
                                           const struct timespec *finish)
{
    unsigned long long seconds, nanoseconds;
    if (!start || !finish || finish->tv_sec < start->tv_sec ||
        (finish->tv_sec == start->tv_sec &&
         finish->tv_nsec < start->tv_nsec))
        return 0ull;
    seconds = (unsigned long long)(finish->tv_sec - start->tv_sec);
    if (finish->tv_nsec >= start->tv_nsec)
        nanoseconds = (unsigned long long)(finish->tv_nsec - start->tv_nsec);
    else {
        if (!seconds) return 0ull;
        seconds--;
        nanoseconds = 1000000000ull + (unsigned long long)finish->tv_nsec -
                      (unsigned long long)start->tv_nsec;
    }
    return seconds > (ULLONG_MAX - nanoseconds) / 1000000000ull
               ? ULLONG_MAX
               : seconds * 1000000000ull + nanoseconds;
}

int yvex_cuda_transformer_joint_prepare(
    yvex_backend *backend,
    const yvex_transformer_joint_encoded_weight *external_weights,
    const yvex_transformer_joint_encoded_weight *block_weights,
    const char *residency_identity, unsigned long long resident_bytes,
    const char *prepared_identity,
    const yvex_transformer_joint_request *request,
    yvex_transformer_joint_prepared **out,
    yvex_transformer_joint_prepared_summary *summary, yvex_error *err)
{
    yvex_transformer_joint_prepared *prepared = NULL;
    yvex_backend_tensor_desc device[YVEX_CUDA_JOINT_DEVICE_REGION_COUNT];
    yvex_cuda_execution_arena_plan arena_plan;
    yvex_cuda_execution_arena_summary arena_summary;
    unsigned long long host[YVEX_CUDA_JOINT_HOST_COUNT];
    unsigned long long external_bytes, block_bytes, required;
    unsigned long long video_values, audio_values;
    struct timespec started = {0}, finished = {0};
    int rc;
    if (out) *out = NULL;
    if (summary) memset(summary, 0, sizeof(*summary));
    rc = yvex_cuda_joint_request_valid(
        request, &video_values, &audio_values, err);
    if (rc != YVEX_OK) return rc;
    if (!backend || !out || !summary ||
        !yvex_sha256_hex_valid(residency_identity) ||
        !yvex_sha256_hex_valid(prepared_identity) ||
        !yvex_sha256_hex_valid(request->layout_identity) ||
        !yvex_sha256_hex_valid(request->condition_identity) ||
        !yvex_cuda_joint_external_valid(external_weights, &external_bytes) ||
        !yvex_cuda_joint_weights_validate(
            block_weights, request->block_count, &block_bytes) ||
        !yvex_core_u64_add(external_bytes, block_bytes, &required) ||
        resident_bytes < required)
        return joint_contract_refuse(
            err, YVEX_ERR_INVALID_ARG, "cuda.transformer.joint.prepare",
            "one identity-bound resident joint preparation is required");
    (void)clock_gettime(CLOCK_MONOTONIC, &started);
    rc = yvex_cuda_joint_device_plan(
        request, device, YVEX_CUDA_JOINT_DEVICE_REGION_COUNT, err);
    if (rc == YVEX_OK) rc = joint_host_plan(request, host, err);
    prepared = rc == YVEX_OK ? calloc(1u, sizeof(*prepared)) : NULL;
    if (rc == YVEX_OK && !prepared)
        rc = joint_contract_refuse(
            err, YVEX_ERR_NOMEM, "cuda.transformer.joint.prepare",
            "prepared execution metadata allocation failed");
    arena_plan = (yvex_cuda_execution_arena_plan){
        device, host, YVEX_CUDA_JOINT_DEVICE_REGION_COUNT,
        YVEX_CUDA_JOINT_HOST_COUNT};
    if (rc == YVEX_OK)
        rc = yvex_cuda_execution_arena_open(
            &prepared->arena, backend, &arena_plan, &arena_summary, err);
    if (rc == YVEX_OK) {
        prepared->summary.schema_version =
            YVEX_TRANSFORMER_JOINT_PREPARED_SCHEMA_V1;
        prepared->summary.host_arena_bytes = arena_summary.host_bytes;
        prepared->summary.device_arena_bytes = arena_summary.device_bytes;
        prepared->summary.allocation_count = arena_summary.allocation_count;
        prepared->backend = backend;
        yvex_core_text_copy(prepared->summary.identity,
                            sizeof(prepared->summary.identity),
                            prepared_identity);
        yvex_core_text_copy(prepared->residency_identity,
                            sizeof(prepared->residency_identity),
                            residency_identity);
        yvex_core_text_copy(prepared->layout_identity,
                            sizeof(prepared->layout_identity),
                            request->layout_identity);
        yvex_core_text_copy(prepared->condition_identity,
                            sizeof(prepared->condition_identity),
                            request->condition_identity);
        rc = yvex_cuda_joint_prepare_invariants(
            prepared, backend, external_weights, request, err);
    }
    (void)clock_gettime(CLOCK_MONOTONIC, &finished);
    if (rc == YVEX_OK) {
        prepared->summary.preparation_nanoseconds =
            joint_elapsed_ns(&started, &finished);
        *summary = prepared->summary;
        *out = prepared;
        yvex_error_clear(err);
        return YVEX_OK;
    }
    if (prepared) {
        yvex_error primary = err ? *err : (yvex_error){0}, cleanup;
        int cleanup_rc;
        yvex_error_clear(&cleanup);
        cleanup_rc = yvex_cuda_execution_arena_close(
            &prepared->arena, &cleanup);
        if (cleanup_rc == YVEX_OK) {
            free(prepared);
            if (err) *err = primary;
        } else {
            if (err) *err = cleanup;
            return cleanup_rc;
        }
    }
    return rc;
}

int yvex_cuda_transformer_joint_prepared_release(
    yvex_backend *backend, yvex_transformer_joint_prepared **prepared,
    yvex_error *err)
{
    yvex_transformer_joint_prepared *owned;
    int rc;
    if (!prepared || !*prepared) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    owned = *prepared;
    if (!backend || owned->backend != backend || owned->in_use)
        return joint_contract_refuse(
            err, YVEX_ERR_STATE,
            "cuda.transformer.joint.prepared.release",
            "an idle same-backend prepared execution resource is required");
    rc = yvex_cuda_execution_arena_close(&owned->arena, err);
    if (rc != YVEX_OK) return rc;
    memset(owned, 0, sizeof(*owned));
    free(owned);
    *prepared = NULL;
    yvex_error_clear(err);
    return YVEX_OK;
}
