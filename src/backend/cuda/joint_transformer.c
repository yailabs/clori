/* Execute one explicitly described joint-modality Transformer through CUDA primitives. */
#include "src/backend/cuda/private.h"
#include <yvex/backend.h>
#include <yvex/internal/backend.h>
#include <yvex/internal/component.h>
#include <yvex/internal/core.h>
#include <yvex/internal/joint_transformer.h>
#include <yvex/internal/quant_numeric.h>
#include <yvex/internal/transformer.h>
#include <yvex/qtype.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
enum { TEXT_IDENTITY_CAP = 65u };
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
    JOINT_MAX_PACKED_ROWS = 22016u
};
typedef enum {
    JOINT_DEVICE_HIDDEN = 0,
    JOINT_DEVICE_NORM,
    JOINT_DEVICE_NORM_WEIGHT,
    JOINT_DEVICE_TEMB, JOINT_DEVICE_TEMB_ACTIVATED,
    JOINT_DEVICE_MODULATION,
    JOINT_DEVICE_MODULATION_BIAS,
    JOINT_DEVICE_QKV,
    JOINT_DEVICE_QUERY,
    JOINT_DEVICE_KEY,
    JOINT_DEVICE_VALUE,
    JOINT_DEVICE_Q_NORM,
    JOINT_DEVICE_K_NORM,
    JOINT_DEVICE_COSINE,
    JOINT_DEVICE_SINE,
    JOINT_DEVICE_ATTENTION,
    JOINT_DEVICE_UPDATE,
    JOINT_DEVICE_FC1,
    JOINT_DEVICE_FF,
    JOINT_DEVICE_COUNT
} joint_device_slot;
typedef struct {
    yvex_backend *backend;
    const yvex_transformer_joint_recipe *recipe;
    const yvex_transformer_joint_encoded_weight *weights;
    yvex_device_tensor *device[JOINT_DEVICE_COUNT];
    const float *hidden, *temb, *positions, *inv_freq;
    const unsigned int *adaln_indices;
    unsigned long long rows, timesteps, values, output_bytes, device_bytes;
    unsigned long long block_count, block_index;
    yvex_transformer_joint_block_observer_fn observer;
    void *observer_context;
    yvex_transformer_joint_block_result facts;
} joint_run;
static const unsigned int text_zero_row = 0u;
static int conditioning_refuse(yvex_error *err, yvex_status status, const char *stage,
                               const char *message)
{
    yvex_error_set(err, status, stage, message);
    return status;
}
static float joint_bf16_value(float value)
{
    return yvex_quant_bf16_decode(yvex_quant_bf16_encode(value));
}

static int joint_recipe_supported(const yvex_transformer_joint_recipe *recipe)
{
    return recipe && recipe->schema_version == YVEX_TRANSFORMER_JOINT_SCHEMA_V1 &&
           recipe->qkv_layout == YVEX_TRANSFORMER_QKV_LAYOUT_PER_HEAD_THREE &&
           recipe->swiglu_layout == YVEX_TRANSFORMER_SWIGLU_LAYOUT_GATE_THEN_UP &&
           recipe->identity_domain && recipe->identity_domain[0] &&
           recipe->hidden_width == JOINT_HIDDEN && recipe->attention_heads == JOINT_HEADS &&
           recipe->head_dimension == JOINT_HEAD_DIM &&
           recipe->attention_width == JOINT_ATTENTION_WIDTH &&
           recipe->ffn_width == JOINT_FFN && recipe->timestep_width == JOINT_TIME &&
           recipe->rotary_width == JOINT_ROTARY && recipe->modality_count == JOINT_MODALITIES &&
           recipe->modulation_parameters == JOINT_PARAMETERS &&
           recipe->block_count == JOINT_BLOCKS && recipe->refiner_block_count == 2ull &&
           recipe->maximum_timesteps == JOINT_MAX_TIMESTEPS &&
           recipe->maximum_packed_rows == JOINT_MAX_PACKED_ROWS &&
           recipe->video_input_width == 96ull && recipe->audio_input_width == 32ull &&
           recipe->condition_input_width == 5120ull && recipe->video_output_width == 96ull &&
           recipe->audio_output_width == 32ull;
}
static int joint_facts_add(yvex_transformer_joint_block_result *total,
                          const yvex_backend_cuda_operation_facts *part)
{
    if (!total || !part || !part->compulsory_memory_facts_available) return 0;
    if (part->temporary_bytes > total->temporary_bytes)
        total->temporary_bytes = part->temporary_bytes;
    return yvex_core_u64_add(total->kernel_launches, part->kernel_launches,
                             &total->kernel_launches) &&
           yvex_core_u64_add(total->h2d_bytes, part->h2d_bytes, &total->h2d_bytes) &&
           yvex_core_u64_add(total->d2h_bytes, part->d2h_bytes, &total->d2h_bytes);
}

static const yvex_transformer_joint_encoded_weight *joint_weight(
    const joint_run *run, yvex_transformer_joint_weight_slot slot)
{
    return run->weights + run->block_index * YVEX_TRANSFORMER_JOINT_BLOCK_WEIGHT_COUNT + slot;
}

static int joint_weights_validate(const yvex_transformer_joint_encoded_weight *weights,
                                 unsigned long long block_count,
                                 unsigned long long *weight_bytes)
{
    static const unsigned long long rows[YVEX_TRANSFORMER_JOINT_BLOCK_WEIGHT_COUNT] = {
        1u, 3u * JOINT_ATTENTION_WIDTH, 1u, 1u, JOINT_HIDDEN,
        1u, 2u * JOINT_FFN, JOINT_HIDDEN, JOINT_PARAMETERS * JOINT_MODALITIES * JOINT_HIDDEN, 1u,
    };
    static const unsigned long long widths[YVEX_TRANSFORMER_JOINT_BLOCK_WEIGHT_COUNT] = {
        JOINT_HIDDEN, JOINT_HIDDEN, JOINT_HEAD_DIM, JOINT_HEAD_DIM, JOINT_ATTENTION_WIDTH,
        JOINT_HIDDEN, JOINT_HIDDEN, JOINT_FFN, JOINT_TIME,
        JOINT_PARAMETERS * JOINT_MODALITIES * JOINT_HIDDEN,
    };
    unsigned long long count, index;
    if (weight_bytes) *weight_bytes = 0ull;
    if (!weights || !block_count || block_count > JOINT_BLOCKS || !weight_bytes ||
        !yvex_core_u64_mul(block_count, YVEX_TRANSFORMER_JOINT_BLOCK_WEIGHT_COUNT, &count))
        return 0;
    for (index = 0ull; index < count; ++index) {
        const yvex_transformer_joint_encoded_weight *weight = weights + index;
        unsigned long long expected, slot = index % YVEX_TRANSFORMER_JOINT_BLOCK_WEIGHT_COUNT;
        if (!weight->encoded || weight->qtype != YVEX_GGUF_QTYPE_BF16 ||
            weight->row_count != rows[slot] || weight->row_width != widths[slot] ||
            !yvex_core_u64_mul(widths[slot], 2ull, &expected) ||
            weight->row_bytes != expected ||
            !yvex_core_u64_mul(rows[slot], expected, &expected) ||
            weight->encoded_bytes != expected ||
            !yvex_core_u64_add(*weight_bytes, expected, weight_bytes)) return 0;
    }
    return 1;
}

static int joint_validate(joint_run *run, const char *residency_identity,
                         unsigned long long resident_bytes, float *output,
                         unsigned long long output_capacity,
                         yvex_transformer_joint_block_result *result,
                         unsigned long long *weight_bytes, yvex_error *err)
{
    unsigned long long row;
    if (!run || !run->backend || !joint_recipe_supported(run->recipe) ||
        !joint_weights_validate(run->weights, run->block_count, weight_bytes) ||
        !yvex_sha256_hex_valid(residency_identity) || resident_bytes < *weight_bytes ||
        !run->hidden || !run->temb || !run->positions || !run->adaln_indices ||
        !run->rows || run->rows > JOINT_MAX_PACKED_ROWS || !run->timesteps ||
        run->timesteps > JOINT_MAX_TIMESTEPS || !output || !result ||
        !yvex_core_u64_mul(run->rows, JOINT_HIDDEN, &run->values) ||
        run->values > output_capacity ||
        !yvex_core_u64_mul(run->values, sizeof(float), &run->output_bytes) ||
        run->output_bytes > SIZE_MAX)
        return conditioning_refuse(
            err, YVEX_ERR_INVALID_ARG, "cuda.transformer.joint.joint.validate",
            "exact Omni BF16 block weights, packed inputs, residency, and output are required");
    for (row = 0ull; row < run->rows; ++row)
        if ((unsigned long long)run->adaln_indices[row] >= run->timesteps * JOINT_MODALITIES)
            return conditioning_refuse(err, YVEX_ERR_BOUNDS,
                                       "cuda.transformer.joint.joint.adaln-index",
                                       "packed row selects an unavailable timestep/modality table row");
    return YVEX_OK;
}

static int joint_tensor_allocate(joint_run *run, joint_device_slot slot,
                                const char *name, unsigned long long rows,
                                unsigned long long width, int rank_one, yvex_error *err)
{
    yvex_backend_tensor_desc descriptor = {0};
    unsigned long long elements, bytes, next;
    if (!run || slot >= JOINT_DEVICE_COUNT || !name || !rows || !width ||
        !yvex_core_u64_mul(rows, width, &elements) ||
        !yvex_core_u64_mul(elements, sizeof(float), &bytes) ||
        !yvex_core_u64_add(run->device_bytes, bytes, &next))
        return conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.transformer.joint.joint.allocate",
                                   "Omni activation allocation geometry overflowed");
    descriptor.name = name;
    descriptor.dtype = YVEX_DTYPE_F32;
    descriptor.rank = rank_one ? 1u : 2u;
    descriptor.dims[0] = rank_one ? width : rows;
    descriptor.dims[1] = rank_one ? 0ull : width;
    descriptor.bytes = bytes;
    if (yvex_backend_tensor_alloc(run->backend, &descriptor, &run->device[slot], err) != YVEX_OK)
        return yvex_error_code(err);
    run->device_bytes = next;
    return YVEX_OK;
}

static int joint_devices_prepare(joint_run *run, yvex_error *err)
{
    int rc;
#define JOINT_ALLOC(slot, name, rows, width, rank_one) \
    if (rc == YVEX_OK) rc = joint_tensor_allocate(run, slot, name, rows, width, rank_one, err)
    rc = joint_tensor_allocate(run, JOINT_DEVICE_HIDDEN, "joint-hidden",
                              run->rows, JOINT_HIDDEN, 0, err);
    JOINT_ALLOC(JOINT_DEVICE_NORM, "joint-norm", run->rows, JOINT_HIDDEN, 0);
    JOINT_ALLOC(JOINT_DEVICE_NORM_WEIGHT, "joint-norm-weight", 1ull, JOINT_HIDDEN, 1);
    JOINT_ALLOC(JOINT_DEVICE_TEMB, "joint-temb", run->timesteps, JOINT_TIME, 0);
    JOINT_ALLOC(JOINT_DEVICE_TEMB_ACTIVATED, "joint-temb-activated", run->timesteps, JOINT_TIME, 0);
    JOINT_ALLOC(JOINT_DEVICE_MODULATION, "joint-modulation", run->timesteps,
               JOINT_PARAMETERS * JOINT_MODALITIES * JOINT_HIDDEN, 0);
    JOINT_ALLOC(JOINT_DEVICE_MODULATION_BIAS, "joint-modulation-bias", 1ull,
               JOINT_PARAMETERS * JOINT_MODALITIES * JOINT_HIDDEN, 1);
    JOINT_ALLOC(JOINT_DEVICE_QKV, "joint-qkv", run->rows, 3ull * JOINT_ATTENTION_WIDTH, 0);
    JOINT_ALLOC(JOINT_DEVICE_QUERY, "joint-query", run->rows, JOINT_ATTENTION_WIDTH, 0);
    JOINT_ALLOC(JOINT_DEVICE_KEY, "joint-key", run->rows, JOINT_ATTENTION_WIDTH, 0);
    JOINT_ALLOC(JOINT_DEVICE_VALUE, "joint-value", run->rows, JOINT_ATTENTION_WIDTH, 0);
    JOINT_ALLOC(JOINT_DEVICE_Q_NORM, "joint-q-norm", 1ull, JOINT_HEAD_DIM, 1);
    JOINT_ALLOC(JOINT_DEVICE_K_NORM, "joint-k-norm", 1ull, JOINT_HEAD_DIM, 1);
    JOINT_ALLOC(JOINT_DEVICE_COSINE, "joint-cosine", run->rows, JOINT_ROTARY, 0);
    JOINT_ALLOC(JOINT_DEVICE_SINE, "joint-sine", run->rows, JOINT_ROTARY, 0);
    JOINT_ALLOC(JOINT_DEVICE_ATTENTION, "joint-attention", run->rows, JOINT_ATTENTION_WIDTH, 0);
    JOINT_ALLOC(JOINT_DEVICE_UPDATE, "joint-update", run->rows, JOINT_HIDDEN, 0);
    JOINT_ALLOC(JOINT_DEVICE_FC1, "joint-fc1", run->rows, 2ull * JOINT_FFN, 0);
    JOINT_ALLOC(JOINT_DEVICE_FF, "joint-ff", run->rows, JOINT_FFN, 0);
#undef JOINT_ALLOC
    return rc;
}
static int joint_weight_gather(joint_run *run, yvex_transformer_joint_weight_slot slot,
                              yvex_device_tensor *output, yvex_error *err)
{
    const yvex_transformer_joint_encoded_weight *weight = joint_weight(run, slot);
    yvex_backend_cuda_operation_facts facts;
    int rc = yvex_backend_cuda_encoded_gather(
        run->backend, weight->encoded, weight->encoded_bytes, weight->qtype,
        weight->row_count, weight->row_width, weight->row_bytes,
        &text_zero_row, 1ull, output, &facts, err);
    if (rc == YVEX_OK && !joint_facts_add(&run->facts, &facts))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.transformer.joint.joint.facts",
                                 "Omni gather accounting overflowed");
    return rc;
}

static int joint_weight_project(joint_run *run, yvex_transformer_joint_weight_slot slot,
                               unsigned long long rows, const yvex_device_tensor *input,
                               yvex_device_tensor *output, yvex_error *err)
{
    const yvex_transformer_joint_encoded_weight *weight = joint_weight(run, slot);
    yvex_backend_cuda_operation_facts facts;
    int rc = yvex_backend_cuda_encoded_matvec(
        run->backend, weight->encoded, weight->encoded_bytes, weight->qtype,
        weight->row_count, weight->row_width, weight->row_bytes, rows,
        input, NULL, 0ull, NULL, output, 0, &facts, err);
    if (rc == YVEX_OK && !joint_facts_add(&run->facts, &facts))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.transformer.joint.joint.facts",
                                 "Omni projection accounting overflowed");
    return rc;
}

static int joint_round(joint_run *run, joint_device_slot slot,
                      unsigned long long count, yvex_error *err)
{
    yvex_backend_cuda_operation_facts facts;
    int rc = yvex_cuda_transformer_bf16_round(
        run->backend, run->device[slot], count, &facts, err);
    if (rc == YVEX_OK && !joint_facts_add(&run->facts, &facts))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.transformer.joint.joint.facts",
                                 "Omni rounding accounting overflowed");
    return rc;
}

static int joint_norm(joint_run *run, joint_device_slot input,
                     yvex_transformer_joint_weight_slot weight,
                     joint_device_slot weight_device, joint_device_slot output,
                     unsigned long long rows, unsigned long long width,
                     float epsilon, yvex_error *err)
{
    yvex_backend_cuda_operation_facts facts;
    int rc = joint_weight_gather(run, weight, run->device[weight_device], err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_rms_norm_bf16(
            run->backend, run->device[input], run->device[weight_device],
            run->device[output], rows, width, epsilon, &facts, err);
    if (rc == YVEX_OK && !joint_facts_add(&run->facts, &facts))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.transformer.joint.joint.facts",
                                 "Omni normalization accounting overflowed");
    return rc;
}

static int joint_upload_inputs(joint_run *run, yvex_error *err)
{
    unsigned long long temb_values, temb_bytes;
    int rc;
    if (!yvex_core_u64_mul(run->timesteps, JOINT_TIME, &temb_values) ||
        !yvex_core_u64_mul(temb_values, sizeof(float), &temb_bytes))
        return conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.transformer.joint.joint.input",
                                   "Omni packed input byte geometry overflowed");
    rc = yvex_backend_tensor_write(run->backend, run->device[JOINT_DEVICE_HIDDEN],
                                   run->hidden, run->output_bytes, err);
    if (rc == YVEX_OK)
        rc = yvex_backend_tensor_write(run->backend, run->device[JOINT_DEVICE_TEMB],
                                       run->temb, temb_bytes, err);
    if (rc == YVEX_OK &&
        (!yvex_core_u64_add(run->facts.h2d_bytes, run->output_bytes,
                            &run->facts.h2d_bytes) ||
         !yvex_core_u64_add(run->facts.h2d_bytes, temb_bytes, &run->facts.h2d_bytes)))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.transformer.joint.joint.facts",
                                 "Omni input upload accounting overflowed");
    return rc;
}

static int joint_rope_tables(joint_run *run, yvex_error *err)
{
    float *cosines = NULL, *sines = NULL;
    unsigned long long row, axis, frequency_index, half_index, elements, bytes;
    int rc = YVEX_OK;
    if (!yvex_core_u64_mul(run->rows, JOINT_ROTARY, &elements) ||
        !yvex_core_u64_mul(elements, sizeof(float), &bytes) || bytes > SIZE_MAX)
        return conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.transformer.joint.joint.rope",
                                   "Omni MM-RoPE table geometry overflowed");
    cosines = (float *)malloc((size_t)bytes);
    sines = (float *)malloc((size_t)bytes);
    if (!cosines || !sines)
        rc = conditioning_refuse(err, YVEX_ERR_NOMEM, "cuda.transformer.joint.joint.rope",
                                 "Omni MM-RoPE table allocation failed");
    for (row = 0ull; rc == YVEX_OK && row < run->rows; ++row) {
        for (axis = 0ull; axis < 3ull; ++axis) {
            for (frequency_index = 0ull; frequency_index < 16ull; ++frequency_index) {
                float frequency = run->inv_freq ? run->inv_freq[frequency_index]
                                                : powf(10000.0f, -(float)(2ull * frequency_index) / 32.0f);
                float angle = run->positions[row * 3ull + axis] * frequency;
                unsigned long long coordinate = axis * 16ull + frequency_index;
                for (half_index = 0ull; half_index < 2ull; ++half_index) {
                    unsigned long long index =
                        row * JOINT_ROTARY + half_index * (JOINT_ROTARY / 2ull) + coordinate;
                    cosines[index] = joint_bf16_value(cosf(angle));
                    sines[index] = joint_bf16_value(sinf(angle));
                }
            }
        }
    }
    if (rc == YVEX_OK)
        rc = yvex_backend_tensor_write(run->backend, run->device[JOINT_DEVICE_COSINE],
                                       cosines, bytes, err);
    if (rc == YVEX_OK)
        rc = yvex_backend_tensor_write(run->backend, run->device[JOINT_DEVICE_SINE],
                                       sines, bytes, err);
    if (rc == YVEX_OK &&
        (!yvex_core_u64_add(run->facts.h2d_bytes, bytes, &run->facts.h2d_bytes) ||
         !yvex_core_u64_add(run->facts.h2d_bytes, bytes, &run->facts.h2d_bytes)))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.transformer.joint.joint.facts",
                                 "Omni MM-RoPE upload accounting overflowed");
    free(sines);
    free(cosines);
    return rc;
}

static int joint_modulation(joint_run *run, yvex_error *err)
{
    yvex_backend_cuda_operation_facts facts;
    unsigned long long temb_values = run->timesteps * JOINT_TIME;
    unsigned long long table_width = JOINT_PARAMETERS * JOINT_MODALITIES * JOINT_HIDDEN;
    int rc = yvex_cuda_transformer_silu(
        run->backend, run->device[JOINT_DEVICE_TEMB], run->device[JOINT_DEVICE_TEMB_ACTIVATED],
        temb_values, 1, &facts, err);
    if (rc == YVEX_OK && !joint_facts_add(&run->facts, &facts))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.transformer.joint.joint.facts",
                                 "Omni timestep activation accounting overflowed");
    if (rc == YVEX_OK)
        rc = joint_weight_project(run, YVEX_TRANSFORMER_JOINT_ADALN_WEIGHT, run->timesteps,
                                 run->device[JOINT_DEVICE_TEMB_ACTIVATED],
                                 run->device[JOINT_DEVICE_MODULATION], err);
    /* The source BF16 linear rounds after its bias epilogue. Keeping the GEMM result in
     * F32 until the broadcast bias kernel avoids a second, coherently accumulating round. */
    if (rc == YVEX_OK)
        rc = joint_weight_gather(run, YVEX_TRANSFORMER_JOINT_ADALN_BIAS,
                                run->device[JOINT_DEVICE_MODULATION_BIAS], err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_bias(
            run->backend, run->device[JOINT_DEVICE_MODULATION],
            run->device[JOINT_DEVICE_MODULATION_BIAS],
            run->device[JOINT_DEVICE_MODULATION], run->timesteps, table_width,
            1, &facts, err);
    if (rc == YVEX_OK && !joint_facts_add(&run->facts, &facts))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.transformer.joint.joint.facts",
                                 "Omni AdaLN bias accounting overflowed");
    return rc;
}

static int joint_modulate(joint_run *run, unsigned int shift_slot,
                         unsigned int scale_slot, yvex_error *err)
{
    yvex_backend_cuda_operation_facts facts;
    int rc = yvex_cuda_transformer_modulate_bf16(
        run->backend, run->device[JOINT_DEVICE_NORM],
        run->device[JOINT_DEVICE_MODULATION], run->adaln_indices,
        run->device[JOINT_DEVICE_NORM], run->rows, JOINT_HIDDEN,
        run->timesteps * JOINT_MODALITIES, JOINT_PARAMETERS,
        shift_slot, scale_slot, &facts, err);
    if (rc == YVEX_OK && !joint_facts_add(&run->facts, &facts))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.transformer.joint.joint.facts",
                                 "Omni modulation accounting overflowed");
    return rc;
}

static int joint_residual(joint_run *run, unsigned int gate_slot, yvex_error *err)
{
    yvex_backend_cuda_operation_facts facts;
    int rc = yvex_cuda_transformer_gated_residual_bf16(
        run->backend, run->device[JOINT_DEVICE_HIDDEN],
        run->device[JOINT_DEVICE_MODULATION], run->adaln_indices,
        run->device[JOINT_DEVICE_UPDATE], run->device[JOINT_DEVICE_HIDDEN],
        run->rows, JOINT_HIDDEN, run->timesteps * JOINT_MODALITIES,
        JOINT_PARAMETERS, gate_slot, &facts, err);
    if (rc == YVEX_OK && !joint_facts_add(&run->facts, &facts))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.transformer.joint.joint.facts",
                                 "Omni gated residual accounting overflowed");
    return rc;
}

static int joint_attention(joint_run *run, yvex_error *err)
{
    yvex_backend_cuda_operation_facts facts;
    unsigned long long attention_values = run->rows * JOINT_ATTENTION_WIDTH;
    int rc = joint_weight_project(run, YVEX_TRANSFORMER_JOINT_QKV, run->rows,
                                 run->device[JOINT_DEVICE_NORM],
                                 run->device[JOINT_DEVICE_QKV], err);
    if (rc == YVEX_OK) rc = joint_round(run, JOINT_DEVICE_QKV, 3ull * attention_values, err);
    /* The released checkpoint groups Q/K/V rows inside each attention head. */
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_split_interleaved_three(
            run->backend, run->device[JOINT_DEVICE_QKV],
            run->device[JOINT_DEVICE_QUERY], run->device[JOINT_DEVICE_KEY],
            run->device[JOINT_DEVICE_VALUE], run->rows, JOINT_HEADS,
            JOINT_HEAD_DIM, &facts, err);
    if (rc == YVEX_OK && !joint_facts_add(&run->facts, &facts))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.transformer.joint.joint.facts",
                                 "Omni QKV split accounting overflowed");
    if (rc == YVEX_OK)
        rc = joint_norm(run, JOINT_DEVICE_QUERY, YVEX_TRANSFORMER_JOINT_Q_NORM,
                       JOINT_DEVICE_Q_NORM, JOINT_DEVICE_QUERY,
                       run->rows * JOINT_HEADS, JOINT_HEAD_DIM, 1.0e-5f, err);
    if (rc == YVEX_OK)
        rc = joint_norm(run, JOINT_DEVICE_KEY, YVEX_TRANSFORMER_JOINT_K_NORM,
                       JOINT_DEVICE_K_NORM, JOINT_DEVICE_KEY,
                       run->rows * JOINT_HEADS, JOINT_HEAD_DIM, 1.0e-5f, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_rotary_half(
            run->backend, run->device[JOINT_DEVICE_QUERY], run->device[JOINT_DEVICE_COSINE],
            run->device[JOINT_DEVICE_SINE], run->rows, JOINT_HEADS, JOINT_HEAD_DIM,
            JOINT_ROTARY, &facts, err);
    if (rc == YVEX_OK && !joint_facts_add(&run->facts, &facts))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.transformer.joint.joint.facts",
                                 "Omni query MM-RoPE accounting overflowed");
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_rotary_half(
            run->backend, run->device[JOINT_DEVICE_KEY], run->device[JOINT_DEVICE_COSINE],
            run->device[JOINT_DEVICE_SINE], run->rows, JOINT_HEADS, JOINT_HEAD_DIM,
            JOINT_ROTARY, &facts, err);
    if (rc == YVEX_OK && !joint_facts_add(&run->facts, &facts))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.transformer.joint.joint.facts",
                                 "Omni key MM-RoPE accounting overflowed");
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_gqa(
            run->backend, run->device[JOINT_DEVICE_QUERY], run->device[JOINT_DEVICE_KEY],
            run->device[JOINT_DEVICE_VALUE], run->device[JOINT_DEVICE_ATTENTION],
            run->rows, JOINT_HEADS, JOINT_HEADS, JOINT_HEAD_DIM, 0, &facts, err);
    if (rc == YVEX_OK && !joint_facts_add(&run->facts, &facts))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.transformer.joint.joint.facts",
                                 "Omni full-attention accounting overflowed");
    if (rc == YVEX_OK) rc = joint_round(run, JOINT_DEVICE_ATTENTION, attention_values, err);
    if (rc == YVEX_OK)
        rc = joint_weight_project(run, YVEX_TRANSFORMER_JOINT_ATTENTION_OUT, run->rows,
                                 run->device[JOINT_DEVICE_ATTENTION],
                                 run->device[JOINT_DEVICE_UPDATE], err);
    if (rc == YVEX_OK) rc = joint_round(run, JOINT_DEVICE_UPDATE, run->values, err);
    return rc;
}

static int joint_mlp(joint_run *run, yvex_error *err)
{
    yvex_backend_cuda_operation_facts facts;
    unsigned long long ffn_values = run->rows * JOINT_FFN;
    int rc = joint_weight_project(run, YVEX_TRANSFORMER_JOINT_FC1, run->rows,
                                 run->device[JOINT_DEVICE_NORM],
                                 run->device[JOINT_DEVICE_FC1], err);
    if (rc == YVEX_OK) rc = joint_round(run, JOINT_DEVICE_FC1, 2ull * ffn_values, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_swiglu_split_bf16(
            run->backend, run->device[JOINT_DEVICE_FC1], run->device[JOINT_DEVICE_FF],
            run->rows, JOINT_FFN, &facts, err);
    if (rc == YVEX_OK && !joint_facts_add(&run->facts, &facts))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.transformer.joint.joint.facts",
                                 "Omni SwiGLU accounting overflowed");
    if (rc == YVEX_OK)
        rc = joint_weight_project(run, YVEX_TRANSFORMER_JOINT_FC2, run->rows,
                                 run->device[JOINT_DEVICE_FF],
                                 run->device[JOINT_DEVICE_UPDATE], err);
    if (rc == YVEX_OK) rc = joint_round(run, JOINT_DEVICE_UPDATE, run->values, err);
    return rc;
}

static int joint_block(joint_run *run, yvex_error *err)
{
    int rc = joint_modulation(run, err);
    if (rc == YVEX_OK)
        rc = joint_norm(run, JOINT_DEVICE_HIDDEN, YVEX_TRANSFORMER_JOINT_NORM1,
                       JOINT_DEVICE_NORM_WEIGHT, JOINT_DEVICE_NORM,
                       run->rows, JOINT_HIDDEN, 1.0e-5f, err);
    if (rc == YVEX_OK) rc = joint_modulate(run, 0u, 1u, err);
    if (rc == YVEX_OK) rc = joint_attention(run, err);
    if (rc == YVEX_OK) rc = joint_residual(run, 2u, err);
    if (rc == YVEX_OK)
        rc = joint_norm(run, JOINT_DEVICE_HIDDEN, YVEX_TRANSFORMER_JOINT_NORM2,
                       JOINT_DEVICE_NORM_WEIGHT, JOINT_DEVICE_NORM,
                       run->rows, JOINT_HIDDEN, 1.0e-5f, err);
    if (rc == YVEX_OK) rc = joint_modulate(run, 3u, 4u, err);
    if (rc == YVEX_OK) rc = joint_mlp(run, err);
    if (rc == YVEX_OK) rc = joint_residual(run, 5u, err);
    return rc;
}

static int joint_observe(joint_run *run, float *values, yvex_error *err)
{
    yvex_transformer_joint_block_observation observation;
    int rc;
    if (!run->observer) return YVEX_OK;
    rc = yvex_backend_tensor_read(run->backend, run->device[JOINT_DEVICE_HIDDEN],
                                  values, run->output_bytes, err);
    if (rc == YVEX_OK &&
        !yvex_core_u64_add(run->facts.d2h_bytes, run->output_bytes,
                           &run->facts.d2h_bytes))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.transformer.joint.observe",
                                 "Omni observation accounting overflowed");
    observation.completed_blocks = run->block_index + 1ull;
    observation.packed_rows = run->rows;
    observation.hidden_width = JOINT_HIDDEN;
    observation.value_count = run->values;
    observation.values = values;
    if (rc == YVEX_OK)
        rc = run->observer(run->observer_context, &observation, err);
    if (rc != YVEX_OK && err && yvex_error_code(err) == YVEX_OK)
        yvex_error_set(err, rc, "cuda.transformer.joint.observe",
                       "Omni block observer refused the execution");
    return rc;
}

static int joint_compute(joint_run *run, float *staged, yvex_error *err)
{
    float *observed = NULL;
    int rc = joint_devices_prepare(run, err);
    if (rc == YVEX_OK && run->observer &&
        !(observed = (float *)malloc((size_t)run->output_bytes)))
        rc = conditioning_refuse(err, YVEX_ERR_NOMEM, "cuda.transformer.joint.observe",
                                 "bounded Omni observation allocation failed");
    if (rc == YVEX_OK) rc = joint_upload_inputs(run, err);
    if (rc == YVEX_OK) rc = joint_rope_tables(run, err);
    for (run->block_index = 0ull;
         rc == YVEX_OK && run->block_index < run->block_count; ++run->block_index) {
        rc = joint_block(run, err);
        if (rc == YVEX_OK) rc = joint_observe(run, observed, err);
        if (rc != YVEX_OK && err) {
            yvex_error cause = *err;
            yvex_error_setf(err, cause.code, cause.where,
                            "Omni block %llu failed: %s", run->block_index,
                            cause.message);
        }
    }
    if (rc == YVEX_OK)
        rc = yvex_backend_tensor_read(run->backend, run->device[JOINT_DEVICE_HIDDEN],
                                      staged, run->output_bytes, err);
    if (rc == YVEX_OK &&
        !yvex_core_u64_add(run->facts.d2h_bytes, run->output_bytes, &run->facts.d2h_bytes))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.transformer.joint.joint.facts",
                                 "Omni output accounting overflowed");
    free(observed);
    return rc;
}

static int joint_devices_release(joint_run *run, int rc, yvex_error *err)
{
    int slot;
    for (slot = JOINT_DEVICE_COUNT - 1; slot >= 0; --slot) {
        yvex_error cleanup;
        int cleanup_rc;
        if (!run->device[slot]) continue;
        yvex_error_clear(&cleanup);
        cleanup_rc = yvex_backend_tensor_release(run->backend, &run->device[slot], &cleanup);
        if (cleanup_rc != YVEX_OK) {
            rc = cleanup_rc;
            if (err) *err = cleanup;
        }
    }
    return rc;
}

static int joint_identity(const joint_run *run, const char *residency_identity,
                         const float *output, char identity[TEXT_IDENTITY_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index, temb_values = run->timesteps * JOINT_TIME;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.transformer.joint-block-stack.cuda.v1") ||
        !yvex_sha256_update_text(&hash, run->recipe->identity_domain) ||
        !yvex_sha256_update_u64(&hash, run->recipe->qkv_layout) ||
        !yvex_sha256_update_u64(&hash, run->recipe->swiglu_layout) ||
        !yvex_sha256_update_text(&hash, residency_identity) ||
        !yvex_sha256_update_u64(&hash, run->block_count) ||
        !yvex_sha256_update_u64(&hash, run->rows) ||
        !yvex_sha256_update_u64(&hash, run->timesteps)) return 0;
#define HASH_FLOATS(values, count) \
    for (index = 0ull; index < (count); ++index) { \
        uint32_t bits; \
        memcpy(&bits, (values) + index, sizeof(bits)); \
        if (!yvex_sha256_update_u64(&hash, bits)) return 0; \
    }
    HASH_FLOATS(run->positions, run->rows * 3ull)
    HASH_FLOATS(run->temb, temb_values)
    HASH_FLOATS(output, run->values)
#undef HASH_FLOATS
    for (index = 0ull; index < run->rows; ++index)
        if (!yvex_sha256_update_u64(&hash, run->adaln_indices[index])) return 0;
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, identity);
    return 1;
}

static int joint_blocks_execute(
    yvex_backend *backend, const yvex_transformer_joint_recipe *recipe,
    const yvex_transformer_joint_encoded_weight *weights,
    unsigned long long block_count, const char *residency_identity,
    unsigned long long resident_bytes, const float *hidden, const float *temb,
    unsigned long long timestep_count, const float *position_ids,
    const unsigned int *adaln_indices, unsigned long long packed_rows,
    float *output, unsigned long long output_capacity,
    yvex_transformer_joint_block_result *result, const float *inv_freq,
    yvex_transformer_joint_block_observer_fn observer, void *observer_context,
    yvex_error *err)
{
    joint_run run = {0};
    yvex_transformer_joint_block_result published = {0};
    float *staged = NULL;
    unsigned long long weight_bytes = 0ull;
    int rc;
    if (result) memset(result, 0, sizeof(*result));
    run.backend = backend;
    run.recipe = recipe;
    run.weights = weights;
    run.hidden = hidden;
    run.temb = temb;
    run.positions = position_ids;
    run.inv_freq = inv_freq;
    run.adaln_indices = adaln_indices;
    run.rows = packed_rows;
    run.timesteps = timestep_count;
    run.block_count = block_count;
    run.observer = observer;
    run.observer_context = observer_context;
    rc = joint_validate(&run, residency_identity, resident_bytes, output,
                       output_capacity, result, &weight_bytes, err);
    if (rc == YVEX_OK) {
        staged = (float *)malloc((size_t)run.output_bytes);
        if (!staged)
            rc = conditioning_refuse(err, YVEX_ERR_NOMEM, "cuda.transformer.joint.joint.output",
                                     "transactional Omni output allocation failed");
    }
    if (rc == YVEX_OK) rc = joint_compute(&run, staged, err);
    if (rc == YVEX_OK && !joint_identity(&run, residency_identity, staged,
                                        published.execution_identity))
        rc = conditioning_refuse(err, YVEX_ERR_STATE, "cuda.transformer.joint.joint.identity",
                                 "Omni execution identity could not be sealed");
    rc = joint_devices_release(&run, rc, err);
    if (rc == YVEX_OK) {
        memcpy(output, staged, (size_t)run.output_bytes);
        published.packed_rows = packed_rows;
        published.block_count = block_count;
        published.resident_bytes = resident_bytes;
        published.kernel_launches = run.facts.kernel_launches;
        published.h2d_bytes = run.facts.h2d_bytes;
        published.d2h_bytes = run.facts.d2h_bytes;
        if (!yvex_core_u64_add(run.device_bytes, run.facts.temporary_bytes,
                               &published.device_bytes)) {
            free(staged);
            return conditioning_refuse(err, YVEX_ERR_BOUNDS,
                "cuda.transformer.joint.joint.facts", "peak device-byte accounting overflowed");
        }
        memcpy(published.residency_identity, residency_identity,
               sizeof(published.residency_identity));
        published.complete = 1;
        *result = published;
        yvex_error_clear(err);
    }
    free(staged);
    return rc;
}
int yvex_backend_transformer_joint_blocks_cuda(
    yvex_backend *backend, const yvex_transformer_joint_recipe *recipe,
    const yvex_transformer_joint_encoded_weight *weights, unsigned long long block_count,
    const char *residency_identity, unsigned long long resident_bytes,
    const float *hidden, const float *temb, unsigned long long timestep_count,
    const float *position_ids, const unsigned int *adaln_indices,
    unsigned long long packed_rows, float *output, unsigned long long output_capacity,
    yvex_transformer_joint_block_result *result, yvex_error *err)
{
    return joint_blocks_execute(backend, recipe, weights, block_count, residency_identity,
        resident_bytes, hidden, temb, timestep_count, position_ids, adaln_indices, packed_rows, output,
        output_capacity, result, NULL, NULL, NULL, err);
}

typedef enum {
    REFINER_HIDDEN = 0, REFINER_NORM, REFINER_NORM_WEIGHT, REFINER_QKV,
    REFINER_QUERY, REFINER_KEY, REFINER_VALUE, REFINER_Q_NORM,
    REFINER_K_NORM, REFINER_ATTENTION, REFINER_UPDATE, REFINER_FC1,
    REFINER_FF, REFINER_DEVICE_COUNT
} refiner_device_slot;

typedef struct {
    yvex_backend *backend;
    const yvex_transformer_joint_encoded_weight *weights;
    yvex_device_tensor *device[REFINER_DEVICE_COUNT];
    unsigned long long rows, values, output_bytes, device_bytes;
    yvex_transformer_joint_result *facts;
} refiner_run;

static int transformer_facts_add(yvex_transformer_joint_result *total,
                                 const yvex_backend_cuda_operation_facts *part)
{
    return total && part && part->compulsory_memory_facts_available &&
           yvex_core_u64_add(total->kernel_launches, part->kernel_launches,
                             &total->kernel_launches) &&
           yvex_core_u64_add(total->h2d_bytes, part->h2d_bytes, &total->h2d_bytes) &&
           yvex_core_u64_add(total->d2h_bytes, part->d2h_bytes, &total->d2h_bytes);
}
static int transformer_fact_bytes(unsigned long long *total, unsigned long long bytes,
                                  yvex_error *err)
{
    if (yvex_core_u64_add(*total, bytes, total)) return YVEX_OK;
    return conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.transformer.joint.transformer.facts",
                               "transformer transfer accounting overflowed");
}
static int transformer_weight_valid(const yvex_transformer_joint_encoded_weight *weight,
                                    unsigned int qtype, unsigned long long rows,
                                    unsigned long long width)
{
    unsigned long long row_bytes, bytes;
    return weight && weight->encoded && weight->qtype == qtype &&
           weight->row_count == rows && weight->row_width == width &&
           yvex_core_u64_mul(width, qtype == YVEX_GGUF_QTYPE_F32 ? 4ull : 2ull,
                             &row_bytes) &&
           yvex_core_u64_mul(rows, row_bytes, &bytes) && weight->row_bytes == row_bytes &&
           weight->encoded_bytes == bytes;
}

static int transformer_external_valid(const yvex_transformer_joint_encoded_weight *weights,
                                      unsigned long long *bytes)
{
    static const unsigned int qtypes[YVEX_TRANSFORMER_JOINT_EXTERNAL_WEIGHT_COUNT] = {
        YVEX_GGUF_QTYPE_F32, YVEX_GGUF_QTYPE_F32, YVEX_GGUF_QTYPE_F32, YVEX_GGUF_QTYPE_F32,
        YVEX_GGUF_QTYPE_BF16, YVEX_GGUF_QTYPE_BF16, YVEX_GGUF_QTYPE_F32, YVEX_GGUF_QTYPE_F32,
        YVEX_GGUF_QTYPE_F32, YVEX_GGUF_QTYPE_F32,
        YVEX_GGUF_QTYPE_BF16, YVEX_GGUF_QTYPE_BF16, YVEX_GGUF_QTYPE_BF16,
        YVEX_GGUF_QTYPE_BF16, YVEX_GGUF_QTYPE_BF16, YVEX_GGUF_QTYPE_BF16,
        YVEX_GGUF_QTYPE_BF16, YVEX_GGUF_QTYPE_BF16, YVEX_GGUF_QTYPE_BF16,
        YVEX_GGUF_QTYPE_BF16, YVEX_GGUF_QTYPE_BF16, YVEX_GGUF_QTYPE_BF16,
        YVEX_GGUF_QTYPE_BF16, YVEX_GGUF_QTYPE_BF16, YVEX_GGUF_QTYPE_BF16,
        YVEX_GGUF_QTYPE_BF16, YVEX_GGUF_QTYPE_BF16, YVEX_GGUF_QTYPE_F32,
        YVEX_GGUF_QTYPE_BF16, YVEX_GGUF_QTYPE_BF16, YVEX_GGUF_QTYPE_BF16,
        YVEX_GGUF_QTYPE_F32, YVEX_GGUF_QTYPE_F32, YVEX_GGUF_QTYPE_F32, YVEX_GGUF_QTYPE_F32,
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
    for (index = 0ull; index < YVEX_TRANSFORMER_JOINT_EXTERNAL_WEIGHT_COUNT; ++index) {
        if (!transformer_weight_valid(weights + index, qtypes[index], rows[index], widths[index]) ||
            !yvex_core_u64_add(*bytes, weights[index].encoded_bytes, &next)) return 0;
        *bytes = next;
    }
    return 1;
}

static int transformer_tensor(yvex_backend *backend, const char *name,
                              unsigned long long rows, unsigned long long width,
                              yvex_device_tensor **out, unsigned long long *bytes,
                              yvex_error *err)
{
    yvex_backend_tensor_desc descriptor = {0};
    unsigned long long elements, allocation, next;
    if (!backend || !name || !rows || !width || !out || !bytes ||
        !yvex_core_u64_mul(rows, width, &elements) ||
        !yvex_core_u64_mul(elements, sizeof(float), &allocation) ||
        !yvex_core_u64_add(*bytes, allocation, &next))
        return conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.transformer.joint.transformer.allocate",
                                   "transformer activation geometry overflowed");
    descriptor.name = name;
    descriptor.dtype = YVEX_DTYPE_F32;
    descriptor.rank = rows == 1ull ? 1u : 2u;
    descriptor.dims[0] = rows == 1ull ? width : rows;
    descriptor.dims[1] = rows == 1ull ? 0ull : width;
    descriptor.bytes = allocation;
    if (yvex_backend_tensor_alloc(backend, &descriptor, out, err) != YVEX_OK)
        return yvex_error_code(err);
    *bytes = next;
    return YVEX_OK;
}

static int transformer_devices_release(yvex_backend *backend, yvex_device_tensor **device,
                                       unsigned int count, int rc, yvex_error *err)
{
    while (count) {
        yvex_error cleanup;
        int cleanup_rc;
        --count;
        if (!device[count]) continue;
        yvex_error_clear(&cleanup);
        cleanup_rc = yvex_backend_tensor_release(backend, &device[count], &cleanup);
        if (cleanup_rc != YVEX_OK) {
            rc = cleanup_rc;
            if (err) *err = cleanup;
        }
    }
    return rc;
}

static int transformer_gather(yvex_backend *backend,
                              const yvex_transformer_joint_encoded_weight *weight,
                              yvex_device_tensor *output,
                              yvex_transformer_joint_result *facts,
                              yvex_error *err)
{
    yvex_backend_cuda_operation_facts part;
    int rc = yvex_backend_cuda_encoded_gather(
        backend, weight->encoded, weight->encoded_bytes, weight->qtype,
        weight->row_count, weight->row_width, weight->row_bytes,
        &text_zero_row, 1ull, output, &part, err);
    if (rc == YVEX_OK && !transformer_facts_add(facts, &part))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.transformer.joint.transformer.facts",
                                 "transformer gather accounting overflowed");
    return rc;
}

static int transformer_project(yvex_backend *backend,
                               const yvex_transformer_joint_encoded_weight *weight,
                               unsigned long long rows, const yvex_device_tensor *input,
                               const yvex_device_tensor *additive, yvex_device_tensor *output,
                               yvex_transformer_joint_result *facts,
                               yvex_error *err)
{
    yvex_backend_cuda_operation_facts part;
    int rc = yvex_backend_cuda_encoded_matvec(
        backend, weight->encoded, weight->encoded_bytes, weight->qtype,
        weight->row_count, weight->row_width, weight->row_bytes, rows,
        input, NULL, 0ull, additive, output, 0, &part, err);
    if (rc == YVEX_OK && !transformer_facts_add(facts, &part))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.transformer.joint.transformer.facts",
                                 "transformer projection accounting overflowed");
    return rc;
}

static int transformer_round(yvex_backend *backend, yvex_device_tensor *tensor,
                             unsigned long long values,
                             yvex_transformer_joint_result *facts,
                             yvex_error *err)
{
    yvex_backend_cuda_operation_facts part;
    int rc = yvex_cuda_transformer_bf16_round(backend, tensor, values, &part, err);
    if (rc == YVEX_OK && !transformer_facts_add(facts, &part))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.transformer.joint.transformer.facts",
                                 "transformer rounding accounting overflowed");
    return rc;
}

static int transformer_linear_host(
    yvex_backend *backend, const yvex_transformer_joint_encoded_weight *weight,
    const yvex_transformer_joint_encoded_weight *bias, const float *input,
    unsigned long long rows, float *output, int bf16_output,
    yvex_transformer_joint_result *facts, yvex_error *err)
{
    enum { INPUT = 0, OUTPUT, BIAS, COUNT };
    yvex_device_tensor *device[COUNT] = {0};
    yvex_backend_cuda_operation_facts part;
    unsigned long long device_bytes = 0ull, input_values, input_bytes, output_values, output_bytes;
    int rc;
    if (!yvex_core_u64_mul(rows, weight->row_width, &input_values) ||
        !yvex_core_u64_mul(input_values, sizeof(float), &input_bytes) ||
        !yvex_core_u64_mul(rows, weight->row_count, &output_values) ||
        !yvex_core_u64_mul(output_values, sizeof(float), &output_bytes))
        return conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.transformer.joint.transformer.linear",
                                   "transformer projection byte geometry overflowed");
    rc = transformer_tensor(backend, "transformer-linear-input", rows, weight->row_width,
                            &device[INPUT], &device_bytes, err);
    if (rc == YVEX_OK)
        rc = transformer_tensor(backend, "transformer-linear-output", rows, weight->row_count,
                                &device[OUTPUT], &device_bytes, err);
    if (rc == YVEX_OK)
        rc = transformer_tensor(backend, "transformer-linear-bias", 1ull, weight->row_count,
                                &device[BIAS], &device_bytes, err);
    if (rc == YVEX_OK)
        rc = yvex_backend_tensor_write(backend, device[INPUT], input, input_bytes, err);
    if (rc == YVEX_OK) rc = transformer_fact_bytes(&facts->h2d_bytes, input_bytes, err);
    if (rc == YVEX_OK)
        rc = transformer_project(backend, weight, rows, device[INPUT], NULL,
                                 device[OUTPUT], facts, err);
    if (rc == YVEX_OK) rc = transformer_gather(backend, bias, device[BIAS], facts, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_bias(backend, device[OUTPUT], device[BIAS], device[OUTPUT],
                                        rows, weight->row_count, bf16_output, &part, err);
    if (rc == YVEX_OK && !transformer_facts_add(facts, &part))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.transformer.joint.transformer.facts",
                                 "transformer bias accounting overflowed");
    if (rc == YVEX_OK)
        rc = yvex_backend_tensor_read(backend, device[OUTPUT], output, output_bytes, err);
    if (rc == YVEX_OK) rc = transformer_fact_bytes(&facts->d2h_bytes, output_bytes, err);
    if (device_bytes > facts->device_bytes) facts->device_bytes = device_bytes;
    return transformer_devices_release(backend, device, COUNT, rc, err);
}

static const yvex_transformer_joint_encoded_weight *refiner_weight(
    const refiner_run *run, unsigned long long block, unsigned long long slot)
{
    return run->weights + block * 8ull + slot;
}

static int refiner_norm(refiner_run *run, refiner_device_slot input,
                        const yvex_transformer_joint_encoded_weight *weight,
                        refiner_device_slot weight_device, refiner_device_slot output,
                        unsigned long long rows, unsigned long long width, yvex_error *err)
{
    yvex_backend_cuda_operation_facts part;
    int rc = transformer_gather(run->backend, weight, run->device[weight_device], run->facts, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_rms_norm_bf16(
            run->backend, run->device[input], run->device[weight_device], run->device[output],
            rows, width, 1.0e-5f, &part, err);
    if (rc == YVEX_OK && !transformer_facts_add(run->facts, &part))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.transformer.joint.refiner.facts",
                                 "token-refiner norm accounting overflowed");
    return rc;
}

static int refiner_devices_prepare(refiner_run *run, yvex_error *err)
{
    int rc;
#define REFINE_ALLOC(slot, name, rows, width) \
    if (rc == YVEX_OK) rc = transformer_tensor(run->backend, name, rows, width, \
                                                &run->device[slot], &run->device_bytes, err)
    rc = transformer_tensor(run->backend, "refiner-hidden", run->rows, JOINT_HIDDEN,
                            &run->device[REFINER_HIDDEN], &run->device_bytes, err);
    REFINE_ALLOC(REFINER_NORM, "refiner-norm", run->rows, JOINT_HIDDEN);
    REFINE_ALLOC(REFINER_NORM_WEIGHT, "refiner-norm-weight", 1ull, JOINT_HIDDEN);
    REFINE_ALLOC(REFINER_QKV, "refiner-qkv", run->rows, 3ull * JOINT_ATTENTION_WIDTH);
    REFINE_ALLOC(REFINER_QUERY, "refiner-query", run->rows, JOINT_ATTENTION_WIDTH);
    REFINE_ALLOC(REFINER_KEY, "refiner-key", run->rows, JOINT_ATTENTION_WIDTH);
    REFINE_ALLOC(REFINER_VALUE, "refiner-value", run->rows, JOINT_ATTENTION_WIDTH);
    REFINE_ALLOC(REFINER_Q_NORM, "refiner-q-norm", 1ull, JOINT_HEAD_DIM);
    REFINE_ALLOC(REFINER_K_NORM, "refiner-k-norm", 1ull, JOINT_HEAD_DIM);
    REFINE_ALLOC(REFINER_ATTENTION, "refiner-attention", run->rows, JOINT_ATTENTION_WIDTH);
    REFINE_ALLOC(REFINER_UPDATE, "refiner-update", run->rows, JOINT_HIDDEN);
    REFINE_ALLOC(REFINER_FC1, "refiner-fc1", run->rows, 2ull * JOINT_FFN);
    REFINE_ALLOC(REFINER_FF, "refiner-ff", run->rows, JOINT_FFN);
#undef REFINE_ALLOC
    return rc;
}

static int refiner_block(refiner_run *run, unsigned long long block, yvex_error *err)
{
    const yvex_transformer_joint_encoded_weight *weight = refiner_weight(run, block, 0ull);
    yvex_backend_cuda_operation_facts part;
    unsigned long long attention_values = run->rows * JOINT_ATTENTION_WIDTH;
    unsigned long long ffn_values = run->rows * JOINT_FFN;
    int rc = refiner_norm(run, REFINER_HIDDEN, weight, REFINER_NORM_WEIGHT,
                          REFINER_NORM, run->rows, JOINT_HIDDEN, err);
    if (rc == YVEX_OK)
        rc = transformer_project(run->backend, weight + 1, run->rows,
                                 run->device[REFINER_NORM], NULL,
                                 run->device[REFINER_QKV], run->facts, err);
    if (rc == YVEX_OK)
        rc = transformer_round(run->backend, run->device[REFINER_QKV],
                               3ull * attention_values, run->facts, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_split_interleaved_three(
            run->backend, run->device[REFINER_QKV], run->device[REFINER_QUERY],
            run->device[REFINER_KEY], run->device[REFINER_VALUE], run->rows,
            JOINT_HEADS, JOINT_HEAD_DIM, &part, err);
    if (rc == YVEX_OK && !transformer_facts_add(run->facts, &part)) rc = YVEX_ERR_BOUNDS;
    if (rc == YVEX_OK)
        rc = refiner_norm(run, REFINER_QUERY, weight + 2, REFINER_Q_NORM,
                          REFINER_QUERY, run->rows * JOINT_HEADS, JOINT_HEAD_DIM, err);
    if (rc == YVEX_OK)
        rc = refiner_norm(run, REFINER_KEY, weight + 3, REFINER_K_NORM,
                          REFINER_KEY, run->rows * JOINT_HEADS, JOINT_HEAD_DIM, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_gqa(
            run->backend, run->device[REFINER_QUERY], run->device[REFINER_KEY],
            run->device[REFINER_VALUE], run->device[REFINER_ATTENTION], run->rows,
            JOINT_HEADS, JOINT_HEADS, JOINT_HEAD_DIM, 0, &part, err);
    if (rc == YVEX_OK && !transformer_facts_add(run->facts, &part)) rc = YVEX_ERR_BOUNDS;
    if (rc == YVEX_OK)
        rc = transformer_round(run->backend, run->device[REFINER_ATTENTION],
                               attention_values, run->facts, err);
    if (rc == YVEX_OK)
        rc = transformer_project(run->backend, weight + 4, run->rows,
                                 run->device[REFINER_ATTENTION], run->device[REFINER_HIDDEN],
                                 run->device[REFINER_UPDATE], run->facts, err);
    if (rc == YVEX_OK)
        rc = transformer_round(run->backend, run->device[REFINER_UPDATE], run->values,
                               run->facts, err);
    if (rc == YVEX_OK)
        rc = refiner_norm(run, REFINER_UPDATE, weight + 5, REFINER_NORM_WEIGHT,
                          REFINER_NORM, run->rows, JOINT_HIDDEN, err);
    if (rc == YVEX_OK)
        rc = transformer_project(run->backend, weight + 6, run->rows,
                                 run->device[REFINER_NORM], NULL,
                                 run->device[REFINER_FC1], run->facts, err);
    if (rc == YVEX_OK)
        rc = transformer_round(run->backend, run->device[REFINER_FC1],
                               2ull * ffn_values, run->facts, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_swiglu_split_bf16(
            run->backend, run->device[REFINER_FC1], run->device[REFINER_FF],
            run->rows, JOINT_FFN, &part, err);
    if (rc == YVEX_OK && !transformer_facts_add(run->facts, &part)) rc = YVEX_ERR_BOUNDS;
    if (rc == YVEX_OK)
        rc = transformer_project(run->backend, weight + 7, run->rows,
                                 run->device[REFINER_FF], run->device[REFINER_UPDATE],
                                 run->device[REFINER_HIDDEN], run->facts, err);
    if (rc == YVEX_OK)
        rc = transformer_round(run->backend, run->device[REFINER_HIDDEN], run->values,
                               run->facts, err);
    if (rc == YVEX_ERR_BOUNDS && yvex_error_code(err) == YVEX_OK)
        rc = conditioning_refuse(err, rc, "cuda.transformer.joint.refiner.facts",
                                 "token-refiner accounting overflowed");
    return rc;
}

static int transformer_refine(yvex_backend *backend,
                              const yvex_transformer_joint_encoded_weight *weights,
                              const float *input, unsigned long long rows, float *output,
                              yvex_transformer_joint_result *facts,
                              yvex_error *err)
{
    refiner_run run = {0};
    unsigned long long block, input_bytes;
    int rc;
    run.backend = backend;
    run.weights = weights;
    run.rows = rows;
    run.facts = facts;
    if (!yvex_core_u64_mul(rows, JOINT_HIDDEN, &run.values) ||
        !yvex_core_u64_mul(run.values, sizeof(float), &run.output_bytes))
        return conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.transformer.joint.refiner",
                                   "token-refiner geometry overflowed");
    input_bytes = run.output_bytes;
    rc = refiner_devices_prepare(&run, err);
    if (rc == YVEX_OK)
        rc = yvex_backend_tensor_write(backend, run.device[REFINER_HIDDEN], input,
                                       input_bytes, err);
    if (rc == YVEX_OK) rc = transformer_fact_bytes(&facts->h2d_bytes, input_bytes, err);
    for (block = 0ull; rc == YVEX_OK && block < 2ull; ++block)
        rc = refiner_block(&run, block, err);
    if (rc == YVEX_OK)
        rc = refiner_norm(&run, REFINER_HIDDEN, weights + 16,
                          REFINER_NORM_WEIGHT, REFINER_NORM,
                          rows, JOINT_HIDDEN, err);
    if (rc == YVEX_OK)
        rc = yvex_backend_tensor_read(backend, run.device[REFINER_NORM], output,
                                      run.output_bytes, err);
    if (rc == YVEX_OK) rc = transformer_fact_bytes(&facts->d2h_bytes, run.output_bytes, err);
    if (run.device_bytes > facts->device_bytes) facts->device_bytes = run.device_bytes;
    return transformer_devices_release(backend, run.device, REFINER_DEVICE_COUNT, rc, err);
}

static int transformer_time_embed(
    yvex_backend *backend, const yvex_transformer_joint_encoded_weight *weights,
    const float *timesteps, unsigned long long rows, float *output,
    yvex_transformer_joint_result *facts, yvex_error *err)
{
    enum { INPUT = 0, HIDDEN, OUTPUT, BIAS_IN, BIAS_OUT, COUNT };
    yvex_device_tensor *device[COUNT] = {0};
    yvex_backend_cuda_operation_facts part;
    float *embedding = NULL;
    unsigned long long row, lane, input_values, input_bytes, output_bytes;
    unsigned long long device_bytes = 0ull;
    int rc = YVEX_OK;
    if (!yvex_core_u64_mul(rows, 256ull, &input_values) ||
        !yvex_core_u64_mul(input_values, sizeof(float), &input_bytes) ||
        !yvex_core_u64_mul(rows * JOINT_TIME, sizeof(float), &output_bytes) ||
        input_bytes > SIZE_MAX || !(embedding = (float *)malloc((size_t)input_bytes)))
        return conditioning_refuse(err, YVEX_ERR_NOMEM, "cuda.transformer.joint.time-embed",
                                   "bounded timestep embedding allocation failed");
    for (row = 0ull; row < rows; ++row)
        for (lane = 0ull; lane < 128ull; ++lane) {
            double angle = timesteps[row] * exp(-log(10000.0) * (double)lane / 128.0);
            embedding[row * 256ull + lane] = (float)cos(angle);
            embedding[row * 256ull + 128ull + lane] = (float)sin(angle);
        }
    rc = transformer_tensor(backend, "time-input", rows, 256ull, &device[INPUT],
                            &device_bytes, err);
    if (rc == YVEX_OK)
        rc = transformer_tensor(backend, "time-hidden", rows, 5376ull, &device[HIDDEN],
                                &device_bytes, err);
    if (rc == YVEX_OK)
        rc = transformer_tensor(backend, "time-output", rows, JOINT_TIME, &device[OUTPUT],
                                &device_bytes, err);
    if (rc == YVEX_OK)
        rc = transformer_tensor(backend, "time-bias-in", 1ull, 5376ull, &device[BIAS_IN],
                                &device_bytes, err);
    if (rc == YVEX_OK)
        rc = transformer_tensor(backend, "time-bias-out", 1ull, JOINT_TIME,
                                &device[BIAS_OUT], &device_bytes, err);
    if (rc == YVEX_OK)
        rc = yvex_backend_tensor_write(backend, device[INPUT], embedding, input_bytes, err);
    if (rc == YVEX_OK) rc = transformer_fact_bytes(&facts->h2d_bytes, input_bytes, err);
    if (rc == YVEX_OK)
        rc = transformer_project(backend, weights + YVEX_TRANSFORMER_JOINT_TIME_IN_WEIGHT,
                                 rows, device[INPUT], NULL, device[HIDDEN], facts, err);
    if (rc == YVEX_OK)
        rc = transformer_gather(backend, weights + YVEX_TRANSFORMER_JOINT_TIME_IN_BIAS,
                                device[BIAS_IN], facts, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_bias(backend, device[HIDDEN], device[BIAS_IN], device[HIDDEN],
                                        rows, 5376ull, 0, &part, err);
    if (rc == YVEX_OK && !transformer_facts_add(facts, &part)) rc = YVEX_ERR_BOUNDS;
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_silu(backend, device[HIDDEN], device[HIDDEN], rows * 5376ull,
                                        0, &part, err);
    if (rc == YVEX_OK && !transformer_facts_add(facts, &part)) rc = YVEX_ERR_BOUNDS;
    if (rc == YVEX_OK)
        rc = transformer_project(backend, weights + YVEX_TRANSFORMER_JOINT_TIME_OUT_WEIGHT,
                                 rows, device[HIDDEN], NULL, device[OUTPUT], facts, err);
    if (rc == YVEX_OK)
        rc = transformer_gather(backend, weights + YVEX_TRANSFORMER_JOINT_TIME_OUT_BIAS,
                                device[BIAS_OUT], facts, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_bias(backend, device[OUTPUT], device[BIAS_OUT], device[OUTPUT],
                                        rows, JOINT_TIME, 0, &part, err);
    if (rc == YVEX_OK && !transformer_facts_add(facts, &part)) rc = YVEX_ERR_BOUNDS;
    if (rc == YVEX_OK)
        rc = yvex_backend_tensor_read(backend, device[OUTPUT], output, output_bytes, err);
    if (rc == YVEX_OK) rc = transformer_fact_bytes(&facts->d2h_bytes, output_bytes, err);
    if (rc == YVEX_ERR_BOUNDS && yvex_error_code(err) == YVEX_OK)
        rc = conditioning_refuse(err, rc, "cuda.transformer.joint.time-embed.facts",
                                 "timestep embedding accounting overflowed");
    if (device_bytes > facts->device_bytes) facts->device_bytes = device_bytes;
    free(embedding);
    return transformer_devices_release(backend, device, COUNT, rc, err);
}

static int transformer_final_norm(
    yvex_backend *backend, const yvex_transformer_joint_encoded_weight *weights,
    const float *hidden, const float *temb, const unsigned int *timestep_indices,
    unsigned long long rows, unsigned long long timesteps, float *output,
    yvex_transformer_joint_result *facts, yvex_error *err)
{
    enum { HIDDEN = 0, NORM, NORM_WEIGHT, TEMB, TABLE, BIAS, COUNT };
    yvex_device_tensor *device[COUNT] = {0};
    yvex_backend_cuda_operation_facts part;
    unsigned long long device_bytes = 0ull, hidden_bytes = rows * JOINT_HIDDEN * 4ull;
    unsigned long long temb_bytes = timesteps * JOINT_TIME * 4ull;
    int rc = transformer_tensor(backend, "final-hidden", rows, JOINT_HIDDEN,
                                &device[HIDDEN], &device_bytes, err);
    if (rc == YVEX_OK) rc = transformer_tensor(backend, "final-norm", rows, JOINT_HIDDEN,
                                                &device[NORM], &device_bytes, err);
    if (rc == YVEX_OK) rc = transformer_tensor(backend, "final-norm-weight", 1ull, JOINT_HIDDEN,
                                                &device[NORM_WEIGHT], &device_bytes, err);
    if (rc == YVEX_OK) rc = transformer_tensor(backend, "final-temb", timesteps, JOINT_TIME,
                                                &device[TEMB], &device_bytes, err);
    if (rc == YVEX_OK) rc = transformer_tensor(backend, "final-table", timesteps,
                                                2ull * JOINT_HIDDEN, &device[TABLE],
                                                &device_bytes, err);
    if (rc == YVEX_OK) rc = transformer_tensor(backend, "final-bias", 1ull,
                                                2ull * JOINT_HIDDEN, &device[BIAS],
                                                &device_bytes, err);
    if (rc == YVEX_OK) rc = yvex_backend_tensor_write(backend, device[HIDDEN], hidden,
                                                       hidden_bytes, err);
    if (rc == YVEX_OK) rc = yvex_backend_tensor_write(backend, device[TEMB], temb,
                                                       temb_bytes, err);
    if (rc == YVEX_OK) rc = transformer_fact_bytes(&facts->h2d_bytes, hidden_bytes, err);
    if (rc == YVEX_OK) rc = transformer_fact_bytes(&facts->h2d_bytes, temb_bytes, err);
    if (rc == YVEX_OK)
        rc = transformer_gather(backend, weights + YVEX_TRANSFORMER_JOINT_FINAL_NORM,
                                device[NORM_WEIGHT], facts, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_rms_norm_bf16(backend, device[HIDDEN], device[NORM_WEIGHT],
                                                  device[NORM], rows, JOINT_HIDDEN, 1.0e-5f,
                                                  &part, err);
    if (rc == YVEX_OK && !transformer_facts_add(facts, &part)) rc = YVEX_ERR_BOUNDS;
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_silu(backend, device[TEMB], device[TEMB],
                                        timesteps * JOINT_TIME, 0, &part, err);
    if (rc == YVEX_OK && !transformer_facts_add(facts, &part)) rc = YVEX_ERR_BOUNDS;
    if (rc == YVEX_OK)
        rc = transformer_project(backend, weights + YVEX_TRANSFORMER_JOINT_FINAL_ADALN_WEIGHT,
                                 timesteps, device[TEMB], NULL, device[TABLE], facts, err);
    if (rc == YVEX_OK)
        rc = transformer_gather(backend, weights + YVEX_TRANSFORMER_JOINT_FINAL_ADALN_BIAS,
                                device[BIAS], facts, err);
    /* Match the BF16 linear epilogue: the projection remains F32 until bias is added, then
       the complete result is rounded once. Pre-rounding introduces a second coherent error. */
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_bias(backend, device[TABLE], device[BIAS], device[TABLE],
                                        timesteps, 2ull * JOINT_HIDDEN, 1, &part, err);
    if (rc == YVEX_OK && !transformer_facts_add(facts, &part)) rc = YVEX_ERR_BOUNDS;
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_modulate_bf16(
            backend, device[NORM], device[TABLE], timestep_indices, device[NORM], rows,
            JOINT_HIDDEN, timesteps, 2ull, 0u, 1u, &part, err);
    if (rc == YVEX_OK && !transformer_facts_add(facts, &part)) rc = YVEX_ERR_BOUNDS;
    if (rc == YVEX_OK)
        rc = yvex_backend_tensor_read(backend, device[NORM], output, hidden_bytes, err);
    if (rc == YVEX_OK) rc = transformer_fact_bytes(&facts->d2h_bytes, hidden_bytes, err);
    if (rc == YVEX_ERR_BOUNDS && yvex_error_code(err) == YVEX_OK)
        rc = conditioning_refuse(err, rc, "cuda.transformer.joint.final.facts",
                                 "final normalization accounting overflowed");
    if (device_bytes > facts->device_bytes) facts->device_bytes = device_bytes;
    return transformer_devices_release(backend, device, COUNT, rc, err);
}

static int transformer_request_valid(
    const yvex_transformer_joint_request *request,
    unsigned long long *video_values, unsigned long long *audio_values,
    yvex_error *err)
{
    unsigned char seen[JOINT_MAX_PACKED_ROWS] = {0};
    unsigned long long kind, row, total;
    const unsigned int *indices[3];
    unsigned long long counts[3];
    if (!request || !joint_recipe_supported(request->recipe) ||
        !request->video || !request->audio || !request->conditioning ||
        !request->timesteps || !request->position_ids || !request->video_indices ||
        !request->audio_indices || !request->text_indices || !request->timestep_indices ||
        !request->token_tags || !request->video_output || !request->audio_output ||
        !request->video_rows || !request->audio_rows || !request->text_rows ||
        !request->timestep_count || request->timestep_count > JOINT_MAX_TIMESTEPS ||
        !request->packed_rows ||
        request->packed_rows > JOINT_MAX_PACKED_ROWS ||
        !request->block_count || request->block_count > JOINT_BLOCKS ||
        !yvex_core_u64_add(request->video_rows, request->audio_rows, &total) ||
        !yvex_core_u64_add(total, request->text_rows, &total) || total != request->packed_rows ||
        !yvex_core_u64_mul(request->video_rows, 96ull, video_values) ||
        !yvex_core_u64_mul(request->audio_rows, 32ull, audio_values) ||
        *video_values > request->video_output_capacity ||
        *audio_values > request->audio_output_capacity)
        return conditioning_refuse(err, YVEX_ERR_INVALID_ARG, "cuda.transformer.joint.transformer.request",
                                   "one complete bounded packed FL2VA request is required");
    indices[0] = request->video_indices; counts[0] = request->video_rows;
    indices[1] = request->text_indices; counts[1] = request->text_rows;
    indices[2] = request->audio_indices; counts[2] = request->audio_rows;
    for (kind = 0ull; kind < 3ull; ++kind)
        for (row = 0ull; row < counts[kind]; ++row) {
            unsigned int packed = indices[kind][row];
            if (packed >= request->packed_rows || seen[packed] || request->token_tags[packed] != kind)
                return conditioning_refuse(err, YVEX_ERR_FORMAT,
                                           "cuda.transformer.joint.transformer.layout",
                                           "packed modality indices must form one exact tagged partition");
            seen[packed] = 1u;
        }
    for (row = 0ull; row < request->packed_rows; ++row)
        if (!seen[row] || request->timestep_indices[row] >= request->timestep_count ||
            !isfinite(request->position_ids[row * 3ull]) ||
            !isfinite(request->position_ids[row * 3ull + 1ull]) ||
            !isfinite(request->position_ids[row * 3ull + 2ull]))
            return conditioning_refuse(err, YVEX_ERR_FORMAT, "cuda.transformer.joint.transformer.layout",
                                       "packed rows require finite positions and admitted timesteps");
    for (row = 0ull; row < request->timestep_count; ++row)
        if (!isfinite(request->timesteps[row]) || request->timesteps[row] < 0.0f ||
            request->timesteps[row] > 1.0f)
            return conditioning_refuse(err, YVEX_ERR_FORMAT, "cuda.transformer.joint.transformer.timestep",
                                       "distinct timesteps must be finite values in [0,1]");
    return YVEX_OK;
}

static int transformer_hash_floats(yvex_sha256 *hash, const float *values,
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

static int transformer_execution_identity(
    const yvex_transformer_joint_request *request,
    const char *residency_identity, const char *block_identity,
    const float *video, unsigned long long video_values,
    const float *audio, unsigned long long audio_values, char output[65])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.transformer.joint.cuda.v1") ||
        !yvex_sha256_update_text(&hash, request->recipe->identity_domain) ||
        !yvex_sha256_update_u64(&hash, request->recipe->qkv_layout) ||
        !yvex_sha256_update_u64(&hash, request->recipe->swiglu_layout) ||
        !yvex_sha256_update_text(&hash, residency_identity) ||
        !yvex_sha256_update_text(&hash, block_identity) ||
        !yvex_sha256_update_u64(&hash, request->video_rows) ||
        !yvex_sha256_update_u64(&hash, request->audio_rows) ||
        !yvex_sha256_update_u64(&hash, request->text_rows) ||
        !yvex_sha256_update_u64(&hash, request->timestep_count) ||
        !yvex_sha256_update_u64(&hash, request->packed_rows) ||
        !yvex_sha256_update_u64(&hash, request->block_count) ||
        !transformer_hash_floats(&hash, request->video, request->video_rows * 96ull) ||
        !transformer_hash_floats(&hash, request->audio, request->audio_rows * 32ull) ||
        !transformer_hash_floats(&hash, request->conditioning,
                                 request->text_rows * 5120ull) ||
        !transformer_hash_floats(&hash, request->timesteps, request->timestep_count) ||
        !transformer_hash_floats(&hash, request->position_ids,
                                 request->packed_rows * 3ull)) return 0;
    for (index = 0ull; index < request->video_rows; ++index)
        if (!yvex_sha256_update_u64(&hash, request->video_indices[index])) return 0;
    for (index = 0ull; index < request->audio_rows; ++index)
        if (!yvex_sha256_update_u64(&hash, request->audio_indices[index])) return 0;
    for (index = 0ull; index < request->text_rows; ++index)
        if (!yvex_sha256_update_u64(&hash, request->text_indices[index])) return 0;
    for (index = 0ull; index < request->packed_rows; ++index)
        if (!yvex_sha256_update_u64(&hash, request->timestep_indices[index]) ||
            !yvex_sha256_update_u64(&hash, request->token_tags[index])) return 0;
    if (!transformer_hash_floats(&hash, video, video_values) ||
        !transformer_hash_floats(&hash, audio, audio_values) ||
        !yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

int yvex_backend_transformer_joint_cuda(
    yvex_backend *backend, const yvex_transformer_joint_encoded_weight *external_weights,
    const yvex_transformer_joint_encoded_weight *block_weights, const char *residency_identity,
    unsigned long long resident_bytes, const yvex_transformer_joint_request *request,
    yvex_transformer_joint_result *result, yvex_error *err)
{
    yvex_transformer_joint_result published = {0};
    yvex_transformer_joint_block_result blocks = {0};
    float *video_embed = NULL, *audio_embed = NULL, *text_embed = NULL, *text_refined = NULL;
    float *packed = NULL, *block_output = NULL, *temb = NULL, *normalized = NULL;
    float *all_video = NULL, *all_audio = NULL, *staged_video = NULL, *staged_audio = NULL;
    unsigned int *adaln = NULL;
    unsigned long long external_bytes = 0ull, block_bytes = 0ull, required_bytes;
    unsigned long long video_values = 0ull, audio_values = 0ull, hidden_values, row, lane;
    int rc;
    if (result) memset(result, 0, sizeof(*result));
    rc = transformer_request_valid(request, &video_values, &audio_values, err);
    if (rc == YVEX_OK &&
        (!backend || !result || !yvex_sha256_hex_valid(residency_identity) ||
         !transformer_external_valid(external_weights, &external_bytes) ||
         !joint_weights_validate(block_weights, request->block_count, &block_bytes) ||
         !yvex_core_u64_add(external_bytes, block_bytes, &required_bytes) ||
         resident_bytes < required_bytes ||
         !yvex_core_u64_mul(request->packed_rows, JOINT_HIDDEN, &hidden_values) ||
         hidden_values > SIZE_MAX / sizeof(float)))
        rc = conditioning_refuse(err, YVEX_ERR_INVALID_ARG, "cuda.transformer.joint.transformer",
                                 "exact resident Transformer weights and output state are required");
#define HOST_FLOATS(target, count) \
    if (rc == YVEX_OK && (!(target = (float *)malloc((size_t)(count) * sizeof(float))))) \
        rc = conditioning_refuse(err, YVEX_ERR_NOMEM, "cuda.transformer.joint.transformer.host", \
                                 "transactional Transformer host allocation failed")
    HOST_FLOATS(video_embed, request ? request->video_rows * JOINT_HIDDEN : 0ull);
    HOST_FLOATS(audio_embed, request ? request->audio_rows * JOINT_HIDDEN : 0ull);
    HOST_FLOATS(text_embed, request ? request->text_rows * JOINT_HIDDEN : 0ull);
    HOST_FLOATS(text_refined, request ? request->text_rows * JOINT_HIDDEN : 0ull);
    HOST_FLOATS(packed, hidden_values); HOST_FLOATS(block_output, hidden_values);
    HOST_FLOATS(temb, request ? request->timestep_count * JOINT_TIME : 0ull);
    HOST_FLOATS(normalized, hidden_values);
    HOST_FLOATS(all_video, request ? request->packed_rows * 96ull : 0ull);
    HOST_FLOATS(all_audio, request ? request->packed_rows * 32ull : 0ull);
    HOST_FLOATS(staged_video, video_values); HOST_FLOATS(staged_audio, audio_values);
#undef HOST_FLOATS
    if (rc == YVEX_OK && !(adaln = (unsigned int *)malloc(
                               (size_t)request->packed_rows * sizeof(*adaln))))
        rc = conditioning_refuse(err, YVEX_ERR_NOMEM, "cuda.transformer.joint.transformer.host",
                                 "packed AdaLN selection allocation failed");
    if (rc == YVEX_OK)
        rc = transformer_linear_host(backend, external_weights + YVEX_TRANSFORMER_JOINT_VIDEO_WEIGHT,
                                     external_weights + YVEX_TRANSFORMER_JOINT_VIDEO_BIAS,
                                     request->video, request->video_rows, video_embed, 0,
                                     &published, err);
    if (rc == YVEX_OK)
        rc = transformer_linear_host(backend, external_weights + YVEX_TRANSFORMER_JOINT_AUDIO_WEIGHT,
                                     external_weights + YVEX_TRANSFORMER_JOINT_AUDIO_BIAS,
                                     request->audio, request->audio_rows, audio_embed, 0,
                                     &published, err);
    if (rc == YVEX_OK)
        rc = transformer_linear_host(backend, external_weights + YVEX_TRANSFORMER_JOINT_CONDITION_WEIGHT,
                                     external_weights + YVEX_TRANSFORMER_JOINT_CONDITION_BIAS,
                                     request->conditioning, request->text_rows, text_embed, 1,
                                     &published, err);
    if (rc == YVEX_OK)
        rc = transformer_refine(backend, external_weights + YVEX_TRANSFORMER_JOINT_REFINER_WEIGHTS,
                                text_embed, request->text_rows, text_refined, &published, err);
    if (rc == YVEX_OK)
        rc = transformer_time_embed(backend, external_weights, request->timesteps,
                                    request->timestep_count, temb, &published, err);
    if (rc == YVEX_OK) memset(packed, 0, (size_t)hidden_values * sizeof(float));
    for (row = 0ull; rc == YVEX_OK && row < request->video_rows; ++row)
        for (lane = 0ull; lane < JOINT_HIDDEN; ++lane)
            packed[request->video_indices[row] * JOINT_HIDDEN + lane] =
                joint_bf16_value(video_embed[row * JOINT_HIDDEN + lane]);
    for (row = 0ull; rc == YVEX_OK && row < request->audio_rows; ++row)
        for (lane = 0ull; lane < JOINT_HIDDEN; ++lane)
            packed[request->audio_indices[row] * JOINT_HIDDEN + lane] =
                joint_bf16_value(audio_embed[row * JOINT_HIDDEN + lane]);
    for (row = 0ull; rc == YVEX_OK && row < request->text_rows; ++row)
        memcpy(packed + request->text_indices[row] * JOINT_HIDDEN,
               text_refined + row * JOINT_HIDDEN, JOINT_HIDDEN * sizeof(float));
    for (row = 0ull; rc == YVEX_OK && row < request->packed_rows; ++row)
        adaln[row] = request->timestep_indices[row] * JOINT_MODALITIES + request->token_tags[row];
    if (rc == YVEX_OK)
        rc = joint_blocks_execute(backend, request->recipe, block_weights, request->block_count,
            residency_identity, resident_bytes, packed, temb, request->timestep_count,
            request->position_ids, adaln,
            request->packed_rows, block_output, hidden_values, &blocks,
            (const float *)external_weights[YVEX_TRANSFORMER_JOINT_ROPE_INV_FREQ].encoded,
            request->block_observer, request->block_observer_context, err);
    if (rc == YVEX_OK &&
        (!yvex_core_u64_add(published.kernel_launches, blocks.kernel_launches,
                            &published.kernel_launches) ||
         !yvex_core_u64_add(published.h2d_bytes, blocks.h2d_bytes, &published.h2d_bytes) ||
         !yvex_core_u64_add(published.d2h_bytes, blocks.d2h_bytes, &published.d2h_bytes)))
        rc = conditioning_refuse(err, YVEX_ERR_BOUNDS, "cuda.transformer.joint.transformer.facts",
                                 "block-stack accounting overflowed");
    if (blocks.device_bytes > published.device_bytes) published.device_bytes = blocks.device_bytes;
    if (rc == YVEX_OK)
        rc = transformer_final_norm(backend, external_weights, block_output, temb,
                                    request->timestep_indices, request->packed_rows,
                                    request->timestep_count, normalized, &published, err);
    if (rc == YVEX_OK)
        rc = transformer_linear_host(backend, external_weights + YVEX_TRANSFORMER_JOINT_VIDEO_OUT_WEIGHT,
                                     external_weights + YVEX_TRANSFORMER_JOINT_VIDEO_OUT_BIAS,
                                     normalized, request->packed_rows, all_video, 0, &published, err);
    if (rc == YVEX_OK)
        rc = transformer_linear_host(backend, external_weights + YVEX_TRANSFORMER_JOINT_AUDIO_OUT_WEIGHT,
                                     external_weights + YVEX_TRANSFORMER_JOINT_AUDIO_OUT_BIAS,
                                     normalized, request->packed_rows, all_audio, 0, &published, err);
    for (row = 0ull; rc == YVEX_OK && row < request->video_rows; ++row)
        memcpy(staged_video + row * 96ull, all_video + request->video_indices[row] * 96ull,
               96ull * sizeof(float));
    for (row = 0ull; rc == YVEX_OK && row < request->audio_rows; ++row)
        memcpy(staged_audio + row * 32ull, all_audio + request->audio_indices[row] * 32ull,
               32ull * sizeof(float));
    if (rc == YVEX_OK && !transformer_execution_identity(
            request, residency_identity, blocks.execution_identity, staged_video, video_values,
            staged_audio, audio_values, published.execution_identity))
        rc = conditioning_refuse(err, YVEX_ERR_STATE, "cuda.transformer.joint.transformer.identity",
                                 "Transformer execution identity could not be sealed");
    if (rc == YVEX_OK) {
        memcpy(request->video_output, staged_video, (size_t)video_values * sizeof(float));
        memcpy(request->audio_output, staged_audio, (size_t)audio_values * sizeof(float));
        published.video_rows = request->video_rows; published.audio_rows = request->audio_rows;
        published.text_rows = request->text_rows; published.packed_rows = request->packed_rows;
        published.block_count = request->block_count; published.resident_bytes = resident_bytes;
        memcpy(published.residency_identity, residency_identity, 65u);
        published.complete = 1; *result = published; yvex_error_clear(err);
    }
    free(adaln); free(staged_audio); free(staged_video); free(all_audio); free(all_video);
    free(normalized); free(temb); free(block_output); free(packed); free(text_refined);
    free(text_embed); free(audio_embed); free(video_embed);
    return rc;
}
