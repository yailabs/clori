/* Execute a source-described patch vision tower with reusable CUDA transformer primitives. */
#include "src/backend/cuda/private.h"
#include "src/backend/cuda/transformer_ops.h"

#include <yvex/internal/multimodal.h>
#include <yvex/internal/quant_numeric.h>

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    VISION_INPUT = 0, VISION_HIDDEN, VISION_NORM, VISION_SMALL_WEIGHT,
    VISION_SMALL_BIAS, VISION_LARGE_WEIGHT, VISION_LARGE_BIAS, VISION_QKV,
    VISION_QUERY, VISION_KEY, VISION_VALUE, VISION_COSINE, VISION_SINE,
    VISION_ATTENTION, VISION_UPDATE, VISION_FFN, VISION_MERGE_INPUT,
    VISION_MERGE_NORM, VISION_MERGE_FC, VISION_MERGE_OUTPUT, VISION_DEVICE_COUNT
} vision_device_slot;

typedef struct {
    yvex_backend *backend;
    const yvex_backend_vision_request *execution;
    const yvex_vision_request *request;
    const yvex_vision_recipe *recipe;
    yvex_device_tensor *device[VISION_DEVICE_COUNT];
    unsigned long long rows, merged_rows, patch_width, image_index, device_bytes;
    yvex_vision_result facts;
} vision_run;

static int vision_refuse(yvex_error *err, yvex_status status, const char *stage,
                         const char *message)
{
    yvex_error_set(err, status, stage, message);
    return status;
}

static int vision_facts_add(vision_run *run, const yvex_backend_operation_facts *part)
{
    return run && part && part->compulsory_memory_facts_available &&
           yvex_core_u64_add(run->facts.kernel_launches, part->kernel_launches,
                             &run->facts.kernel_launches) &&
           yvex_core_u64_add(run->facts.h2d_bytes, part->h2d_bytes,
                             &run->facts.h2d_bytes) &&
           yvex_core_u64_add(run->facts.d2h_bytes, part->d2h_bytes,
                             &run->facts.d2h_bytes);
}

static unsigned long long vision_block_base(unsigned long long layer)
{
    return YVEX_VISION_EXTERNAL_WEIGHT_COUNT + layer * YVEX_VISION_BLOCK_WEIGHT_COUNT;
}

static unsigned long long vision_merger_base(const vision_run *run, unsigned long long merger)
{
    return YVEX_VISION_EXTERNAL_WEIGHT_COUNT +
           run->recipe->layer_count * YVEX_VISION_BLOCK_WEIGHT_COUNT +
           merger * YVEX_VISION_MERGER_WEIGHT_COUNT;
}

static const yvex_component_encoded_weight *vision_weight(
    const vision_run *run, unsigned long long index)
{
    return index < run->execution->weight_count ? run->execution->weights + index : NULL;
}

static int vision_tensor_open(vision_run *run, vision_device_slot slot, const char *name,
                              unsigned long long rows, unsigned long long width,
                              yvex_error *err)
{
    yvex_backend_tensor_desc descriptor = {0};
    unsigned long long values, bytes, next;
    if (!run || slot >= VISION_DEVICE_COUNT || !rows || !width ||
        !yvex_core_u64_mul(rows, width, &values) ||
        !yvex_core_u64_mul(values, sizeof(float), &bytes) ||
        !yvex_core_u64_add(run->device_bytes, bytes, &next))
        return vision_refuse(err, YVEX_ERR_BOUNDS, "cuda.vision.allocate",
                             "vision activation geometry overflowed");
    descriptor.name = name; descriptor.dtype = YVEX_DTYPE_F32;
    descriptor.rank = rows == 1ull ? 1u : 2u;
    descriptor.dims[0] = rows == 1ull ? width : rows;
    descriptor.dims[1] = rows == 1ull ? 0ull : width;
    descriptor.bytes = bytes;
    if (yvex_backend_tensor_alloc(run->backend, &descriptor, &run->device[slot], err) != YVEX_OK)
        return yvex_error_code(err);
    run->device_bytes = next;
    return YVEX_OK;
}

static int vision_devices_open(vision_run *run, yvex_error *err)
{
    const yvex_vision_recipe *r = run->recipe;
    unsigned long long merge_width, qkv_width;
    int rc;
    if (!yvex_core_u64_mul(r->hidden_width, r->merge * r->merge, &merge_width) ||
        !yvex_core_u64_mul(r->hidden_width, 3ull, &qkv_width))
        return vision_refuse(err, YVEX_ERR_BOUNDS, "cuda.vision.allocate",
                             "vision derived geometry overflowed");
#define OPEN(slot, name, rows, width) \
    if (rc == YVEX_OK) rc = vision_tensor_open(run, slot, name, rows, width, err)
    rc = vision_tensor_open(run, VISION_INPUT, "vision-patches", run->rows,
                            run->patch_width, err);
    OPEN(VISION_HIDDEN, "vision-hidden", run->rows, r->hidden_width);
    OPEN(VISION_NORM, "vision-norm", run->rows, r->hidden_width);
    OPEN(VISION_SMALL_WEIGHT, "vision-small-weight", 1ull, r->hidden_width);
    OPEN(VISION_SMALL_BIAS, "vision-small-bias", 1ull, r->hidden_width);
    OPEN(VISION_LARGE_WEIGHT, "vision-large-weight", 1ull, merge_width);
    OPEN(VISION_LARGE_BIAS, "vision-large-bias", 1ull, merge_width);
    OPEN(VISION_QKV, "vision-qkv", run->rows, qkv_width);
    OPEN(VISION_QUERY, "vision-query", run->rows, r->hidden_width);
    OPEN(VISION_KEY, "vision-key", run->rows, r->hidden_width);
    OPEN(VISION_VALUE, "vision-value", run->rows, r->hidden_width);
    OPEN(VISION_COSINE, "vision-cosine", run->rows, r->head_dimension);
    OPEN(VISION_SINE, "vision-sine", run->rows, r->head_dimension);
    OPEN(VISION_ATTENTION, "vision-attention", run->rows, r->hidden_width);
    OPEN(VISION_UPDATE, "vision-update", run->rows, r->hidden_width);
    OPEN(VISION_FFN, "vision-ffn", run->rows, r->ffn_width);
    OPEN(VISION_MERGE_INPUT, "vision-merge-input", run->merged_rows, merge_width);
    OPEN(VISION_MERGE_NORM, "vision-merge-norm", run->merged_rows, merge_width);
    OPEN(VISION_MERGE_FC, "vision-merge-fc", run->merged_rows, merge_width);
    OPEN(VISION_MERGE_OUTPUT, "vision-merge-output", run->merged_rows, r->output_width);
#undef OPEN
    return rc;
}

static int vision_devices_close(vision_run *run, int rc, yvex_error *err)
{
    int slot;
    for (slot = VISION_DEVICE_COUNT - 1; slot >= 0; --slot) {
        yvex_error cleanup;
        int cleanup_rc;
        yvex_error_clear(&cleanup);
        cleanup_rc = yvex_backend_tensor_release(run->backend, &run->device[slot], &cleanup);
        if (cleanup_rc != YVEX_OK) { rc = cleanup_rc; if (err) *err = cleanup; }
    }
    return rc;
}

static int vision_gather(vision_run *run, const yvex_component_encoded_weight *weight,
                         yvex_device_tensor *output, yvex_error *err)
{
    static const unsigned int row = 0u;
    yvex_backend_operation_facts facts = {0};
    int rc;
    if (!weight)
        return vision_refuse(err, YVEX_ERR_FORMAT, "cuda.vision.weight",
                             "vision weight binding is absent");
    rc = yvex_backend_encoded_gather(
        run->backend, weight->encoded, weight->encoded_bytes, weight->qtype,
        weight->row_count, weight->row_width, weight->row_bytes,
        &row, 1ull, output, &facts, err);
    if (rc == YVEX_OK && !vision_facts_add(run, &facts))
        rc = vision_refuse(err, YVEX_ERR_BOUNDS, "cuda.vision.facts",
                           "vision gather accounting overflowed");
    return rc;
}

static int vision_linear(vision_run *run, const yvex_component_encoded_weight *weight,
                         const yvex_component_encoded_weight *bias,
                         unsigned long long rows, unsigned long long input_width,
                         unsigned long long output_width, const yvex_device_tensor *input,
                         yvex_device_tensor *output, yvex_error *err)
{
    yvex_backend_operation_facts facts = {0};
    int rc;
    if (!weight || !bias || weight->qtype != YVEX_GGUF_QTYPE_BF16 ||
        bias->qtype != YVEX_GGUF_QTYPE_BF16)
        return vision_refuse(err, YVEX_ERR_FORMAT, "cuda.vision.linear",
                             "vision linear requires BF16 source weights and bias");
    rc = yvex_cuda_transformer_linear_bf16(
        run->backend, weight->encoded, weight->encoded_bytes,
        bias->encoded, bias->encoded_bytes, output_width, input_width, rows,
        input, output, &facts, err);
    if (rc == YVEX_OK && !vision_facts_add(run, &facts))
        rc = vision_refuse(err, YVEX_ERR_BOUNDS, "cuda.vision.facts",
                           "vision linear accounting overflowed");
    return rc;
}

static int vision_round(vision_run *run, yvex_device_tensor *tensor,
                        unsigned long long values, yvex_error *err)
{
    yvex_backend_operation_facts facts = {0};
    int rc = yvex_cuda_transformer_bf16_round(run->backend, tensor, values, &facts, err);
    if (rc == YVEX_OK && !vision_facts_add(run, &facts))
        rc = vision_refuse(err, YVEX_ERR_BOUNDS, "cuda.vision.facts",
                           "vision rounding accounting overflowed");
    return rc;
}

static int vision_observe(
    vision_run *run, yvex_vision_observation_stage stage,
    unsigned long long index, const yvex_device_tensor *tensor,
    unsigned long long rows, unsigned long long width, yvex_error *err)
{
    float *host;
    unsigned long long values, bytes;
    int rc;
    if (!run->request->observe) return YVEX_OK;
    if (!tensor || !rows || !width || !yvex_core_u64_mul(rows, width, &values) ||
        !yvex_core_u64_mul(values, sizeof(float), &bytes) || bytes > SIZE_MAX ||
        !(host = malloc((size_t)bytes)))
        return vision_refuse(err, YVEX_ERR_NOMEM, "cuda.vision.observe",
                             "vision observation staging allocation failed");
    rc = yvex_backend_tensor_read(run->backend, tensor, host, bytes, err);
    if (rc == YVEX_OK &&
        !yvex_core_u64_add(run->facts.d2h_bytes, bytes, &run->facts.d2h_bytes))
        rc = vision_refuse(err, YVEX_ERR_BOUNDS, "cuda.vision.observe",
                           "vision observation accounting overflowed");
    if (rc == YVEX_OK)
        rc = run->request->observe(
            run->request->observer_context, stage, index, host, rows,
            width, err);
    free(host);
    return rc;
}

static int vision_norm(vision_run *run, const yvex_device_tensor *input,
                       yvex_device_tensor *output, const yvex_component_encoded_weight *weight,
                       const yvex_component_encoded_weight *bias, unsigned long long rows,
                       unsigned long long width, int large, yvex_error *err)
{
    yvex_backend_operation_facts facts = {0};
    yvex_device_tensor *device_weight = run->device[large ? VISION_LARGE_WEIGHT
                                                          : VISION_SMALL_WEIGHT];
    yvex_device_tensor *device_bias = run->device[large ? VISION_LARGE_BIAS
                                                        : VISION_SMALL_BIAS];
    int rc = vision_gather(run, weight, device_weight, err);
    if (rc == YVEX_OK) rc = vision_gather(run, bias, device_bias, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_layer_norm_f32(
            run->backend, input, device_weight, device_bias, output,
            rows, width, run->recipe->normalization_epsilon, &facts, err);
    if (rc == YVEX_OK && !vision_facts_add(run, &facts))
        rc = vision_refuse(err, YVEX_ERR_BOUNDS, "cuda.vision.facts",
                           "vision normalization accounting overflowed");
    if (rc == YVEX_OK) rc = vision_round(run, output, rows * width, err);
    return rc;
}

static unsigned long long vision_patch_row(
    const vision_run *run, unsigned long long row, unsigned long long *height,
    unsigned long long *width)
{
    unsigned long long merge = run->recipe->merge;
    unsigned long long blocks_width = run->request->grid_width / merge;
    unsigned long long in_column = row % merge;
    unsigned long long in_row = (row / merge) % merge;
    unsigned long long block_column = (row / (merge * merge)) % blocks_width;
    unsigned long long block_row = row / (merge * merge * blocks_width);
    *height = block_row * merge + in_row;
    *width = block_column * merge + in_column;
    return *height * run->request->grid_width + *width;
}

static float vision_bf16(float value)
{
    return yvex_quant_bf16_decode(yvex_quant_bf16_encode(value));
}

static int vision_position_add(vision_run *run, yvex_error *err)
{
    const yvex_component_encoded_weight *position =
        vision_weight(run, YVEX_VISION_POSITION_WEIGHT);
    float *host;
    unsigned long long row, column, bytes, values;
    int rc;
    if (!position || position->qtype != YVEX_GGUF_QTYPE_BF16 ||
        position->row_count != run->recipe->position_grid_side *
                                   run->recipe->position_grid_side ||
        position->row_width != run->recipe->hidden_width ||
        !yvex_core_u64_mul(run->rows, run->recipe->hidden_width, &values) ||
        !yvex_core_u64_mul(values, sizeof(float), &bytes) || bytes > SIZE_MAX)
        return vision_refuse(err, YVEX_ERR_FORMAT, "cuda.vision.position",
                             "vision position table does not match its source grid");
    host = (float *)malloc((size_t)bytes);
    if (!host)
        return vision_refuse(err, YVEX_ERR_NOMEM, "cuda.vision.position",
                             "vision position staging allocation failed");
    rc = yvex_backend_tensor_read(run->backend, run->device[VISION_HIDDEN], host, bytes, err);
    for (row = 0ull; rc == YVEX_OK && row < run->rows; ++row) {
        unsigned long long h, w, h0, h1, w0, w1;
        double hs, ws, hf, wf;
        (void)vision_patch_row(run, row, &h, &w);
        hs = run->request->grid_height == 1ull ? 0.0 :
             (double)h * (double)(run->recipe->position_grid_side - 1ull) /
             (double)(run->request->grid_height - 1ull);
        ws = run->request->grid_width == 1ull ? 0.0 :
             (double)w * (double)(run->recipe->position_grid_side - 1ull) /
             (double)(run->request->grid_width - 1ull);
        h0 = (unsigned long long)floor(hs); w0 = (unsigned long long)floor(ws);
        h1 = h0 + 1ull < run->recipe->position_grid_side ? h0 + 1ull : h0;
        w1 = w0 + 1ull < run->recipe->position_grid_side ? w0 + 1ull : w0;
        hf = hs - floor(hs); wf = ws - floor(ws);
        for (column = 0ull; column < run->recipe->hidden_width; ++column) {
            const uint16_t *table = (const uint16_t *)position->encoded;
            unsigned long long side = run->recipe->position_grid_side;
            float p00 = yvex_quant_bf16_decode(table[(h0 * side + w0) * position->row_width + column]);
            float p01 = yvex_quant_bf16_decode(table[(h0 * side + w1) * position->row_width + column]);
            float p10 = yvex_quant_bf16_decode(table[(h1 * side + w0) * position->row_width + column]);
            float p11 = yvex_quant_bf16_decode(table[(h1 * side + w1) * position->row_width + column]);
            float top = (float)((1.0 - wf) * p00 + wf * p01);
            float bottom = (float)((1.0 - wf) * p10 + wf * p11);
            float interpolated = vision_bf16((float)((1.0 - hf) * top + hf * bottom));
            unsigned long long index = row * run->recipe->hidden_width + column;
            host[index] = vision_bf16(host[index] + interpolated);
        }
    }
    if (rc == YVEX_OK)
        rc = yvex_backend_tensor_write(run->backend, run->device[VISION_HIDDEN], host, bytes, err);
    if (rc == YVEX_OK &&
        (!yvex_core_u64_add(run->facts.d2h_bytes, bytes, &run->facts.d2h_bytes) ||
         !yvex_core_u64_add(run->facts.h2d_bytes, bytes, &run->facts.h2d_bytes)))
        rc = vision_refuse(err, YVEX_ERR_BOUNDS, "cuda.vision.facts",
                           "vision position transfer accounting overflowed");
    free(host);
    return rc;
}

static int vision_rope_tables(vision_run *run, yvex_error *err)
{
    float *cosines, *sines;
    unsigned long long row, lane, elements, bytes;
    int rc = YVEX_OK;
    if (!yvex_core_u64_mul(run->rows, run->recipe->head_dimension, &elements) ||
        !yvex_core_u64_mul(elements, sizeof(float), &bytes) || bytes > SIZE_MAX)
        return vision_refuse(err, YVEX_ERR_BOUNDS, "cuda.vision.rope",
                             "vision rotary table geometry overflowed");
    cosines = (float *)malloc((size_t)bytes); sines = (float *)malloc((size_t)bytes);
    if (!cosines || !sines)
        rc = vision_refuse(err, YVEX_ERR_NOMEM, "cuda.vision.rope",
                           "vision rotary table allocation failed");
    for (row = 0ull; rc == YVEX_OK && row < run->rows; ++row) {
        unsigned long long h, w;
        (void)vision_patch_row(run, row, &h, &w);
        for (lane = 0ull; lane < run->recipe->head_dimension / 2ull; ++lane) {
            unsigned long long axis_lane = lane % (run->recipe->head_dimension / 4ull);
            unsigned long long position = lane < run->recipe->head_dimension / 4ull ? h : w;
            float exponent = -(float)(2ull * axis_lane) /
                             (float)(run->recipe->head_dimension / 2ull);
            float frequency = powf((float)run->recipe->rope_theta, exponent);
            float angle = (float)position * frequency;
            float cosine = cosf(angle);
            float sine = sinf(angle);
            cosines[row * run->recipe->head_dimension + lane] = cosine;
            sines[row * run->recipe->head_dimension + lane] = sine;
            cosines[row * run->recipe->head_dimension +
                    run->recipe->head_dimension / 2ull + lane] = cosine;
            sines[row * run->recipe->head_dimension +
                  run->recipe->head_dimension / 2ull + lane] = sine;
        }
    }
    if (rc == YVEX_OK)
        rc = yvex_backend_tensor_write(run->backend, run->device[VISION_COSINE], cosines, bytes, err);
    if (rc == YVEX_OK)
        rc = yvex_backend_tensor_write(run->backend, run->device[VISION_SINE], sines, bytes, err);
    if (rc == YVEX_OK &&
        (!yvex_core_u64_add(run->facts.h2d_bytes, bytes, &run->facts.h2d_bytes) ||
         !yvex_core_u64_add(run->facts.h2d_bytes, bytes, &run->facts.h2d_bytes)))
        rc = vision_refuse(err, YVEX_ERR_BOUNDS, "cuda.vision.facts",
                           "vision rotary transfer accounting overflowed");
    free(sines); free(cosines);
    return rc;
}

static int vision_block_attention(vision_run *run, unsigned long long layer,
                                  yvex_error *err)
{
    unsigned long long base = vision_block_base(layer);
    yvex_backend_operation_facts facts = {0};
    int rc = vision_norm(
        run, run->device[VISION_HIDDEN], run->device[VISION_NORM],
        vision_weight(run, base + YVEX_VISION_NORM1_WEIGHT),
        vision_weight(run, base + YVEX_VISION_NORM1_BIAS),
        run->rows, run->recipe->hidden_width, 0, err);
    if (rc == YVEX_OK && layer == 0ull)
        rc = vision_observe(run, YVEX_VISION_OBSERVE_NORM1, layer,
                            run->device[VISION_NORM], run->rows,
                            run->recipe->hidden_width, err);
    if (rc == YVEX_OK)
        rc = vision_linear(
            run, vision_weight(run, base + YVEX_VISION_QKV_WEIGHT),
            vision_weight(run, base + YVEX_VISION_QKV_BIAS), run->rows,
            run->recipe->hidden_width, run->recipe->hidden_width * 3ull,
            run->device[VISION_NORM], run->device[VISION_QKV], err);
    if (rc == YVEX_OK && layer == 0ull)
        rc = vision_observe(run, YVEX_VISION_OBSERVE_QKV, layer,
                            run->device[VISION_QKV], run->rows,
                            run->recipe->hidden_width * 3ull, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_split_three(
            run->backend, run->device[VISION_QKV], run->device[VISION_QUERY],
            run->device[VISION_KEY], run->device[VISION_VALUE], run->rows,
            run->recipe->hidden_width, &facts, err);
    if (rc == YVEX_OK && !vision_facts_add(run, &facts)) rc = YVEX_ERR_BOUNDS;
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_rotary_half_f32(
            run->backend, run->device[VISION_QUERY], run->device[VISION_COSINE],
            run->device[VISION_SINE], run->rows, run->recipe->heads,
            run->recipe->head_dimension, run->recipe->head_dimension, &facts, err);
    if (rc == YVEX_OK && !vision_facts_add(run, &facts)) rc = YVEX_ERR_BOUNDS;
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_rotary_half_f32(
            run->backend, run->device[VISION_KEY], run->device[VISION_COSINE],
            run->device[VISION_SINE], run->rows, run->recipe->heads,
            run->recipe->head_dimension, run->recipe->head_dimension, &facts, err);
    if (rc == YVEX_OK && !vision_facts_add(run, &facts)) rc = YVEX_ERR_BOUNDS;
    if (rc == YVEX_OK) rc = vision_round(run, run->device[VISION_QUERY],
                                          run->rows * run->recipe->hidden_width, err);
    if (rc == YVEX_OK) rc = vision_round(run, run->device[VISION_KEY],
                                          run->rows * run->recipe->hidden_width, err);
    if (rc == YVEX_OK && layer == 0ull)
        rc = vision_observe(run, YVEX_VISION_OBSERVE_QUERY, layer,
                            run->device[VISION_QUERY], run->rows,
                            run->recipe->hidden_width, err);
    if (rc == YVEX_OK && layer == 0ull)
        rc = vision_observe(run, YVEX_VISION_OBSERVE_KEY, layer,
                            run->device[VISION_KEY], run->rows,
                            run->recipe->hidden_width, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_gqa(
            run->backend, run->device[VISION_QUERY], run->device[VISION_KEY],
            run->device[VISION_VALUE], run->device[VISION_ATTENTION], run->rows,
            run->recipe->heads, run->recipe->heads, run->recipe->head_dimension,
            0, &facts, err);
    if (rc == YVEX_OK && !vision_facts_add(run, &facts)) rc = YVEX_ERR_BOUNDS;
    if (rc == YVEX_OK) rc = vision_round(run, run->device[VISION_ATTENTION],
                                          run->rows * run->recipe->hidden_width, err);
    if (rc == YVEX_OK && layer == 0ull)
        rc = vision_observe(run, YVEX_VISION_OBSERVE_ATTENTION, layer,
                            run->device[VISION_ATTENTION], run->rows,
                            run->recipe->hidden_width, err);
    if (rc == YVEX_OK)
        rc = vision_linear(
            run, vision_weight(run, base + YVEX_VISION_ATTENTION_WEIGHT),
            vision_weight(run, base + YVEX_VISION_ATTENTION_BIAS), run->rows,
            run->recipe->hidden_width, run->recipe->hidden_width,
            run->device[VISION_ATTENTION], run->device[VISION_UPDATE], err);
    if (rc == YVEX_OK && layer == 0ull)
        rc = vision_observe(run, YVEX_VISION_OBSERVE_ATTENTION_PROJECTION, layer,
                            run->device[VISION_UPDATE], run->rows,
                            run->recipe->hidden_width, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_add_bf16(
            run->backend, run->device[VISION_HIDDEN], run->device[VISION_UPDATE],
            run->device[VISION_HIDDEN], run->rows, run->recipe->hidden_width, &facts, err);
    if (rc == YVEX_OK && !vision_facts_add(run, &facts)) rc = YVEX_ERR_BOUNDS;
    if (rc == YVEX_ERR_BOUNDS)
        return vision_refuse(err, rc, "cuda.vision.facts",
                             "vision attention accounting overflowed");
    return rc;
}

static int vision_block_mlp(vision_run *run, unsigned long long layer, yvex_error *err)
{
    unsigned long long base = vision_block_base(layer);
    yvex_backend_operation_facts facts = {0};
    int rc = vision_norm(
        run, run->device[VISION_HIDDEN], run->device[VISION_NORM],
        vision_weight(run, base + YVEX_VISION_NORM2_WEIGHT),
        vision_weight(run, base + YVEX_VISION_NORM2_BIAS),
        run->rows, run->recipe->hidden_width, 0, err);
    if (rc == YVEX_OK && layer == 0ull)
        rc = vision_observe(run, YVEX_VISION_OBSERVE_NORM2, layer,
                            run->device[VISION_NORM], run->rows,
                            run->recipe->hidden_width, err);
    if (rc == YVEX_OK)
        rc = vision_linear(
            run, vision_weight(run, base + YVEX_VISION_FF1_WEIGHT),
            vision_weight(run, base + YVEX_VISION_FF1_BIAS), run->rows,
            run->recipe->hidden_width, run->recipe->ffn_width,
            run->device[VISION_NORM], run->device[VISION_FFN], err);
    if (rc == YVEX_OK && layer == 0ull)
        rc = vision_observe(run, YVEX_VISION_OBSERVE_FF1, layer,
                            run->device[VISION_FFN], run->rows,
                            run->recipe->ffn_width, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_gelu(
            run->backend, run->device[VISION_FFN], run->device[VISION_FFN],
            run->rows * run->recipe->ffn_width, 1, 1, &facts, err);
    if (rc == YVEX_OK && layer == 0ull)
        rc = vision_observe(run, YVEX_VISION_OBSERVE_GELU, layer,
                            run->device[VISION_FFN], run->rows,
                            run->recipe->ffn_width, err);
    if (rc == YVEX_OK && !vision_facts_add(run, &facts)) rc = YVEX_ERR_BOUNDS;
    if (rc == YVEX_OK)
        rc = vision_linear(
            run, vision_weight(run, base + YVEX_VISION_FF2_WEIGHT),
            vision_weight(run, base + YVEX_VISION_FF2_BIAS), run->rows,
            run->recipe->ffn_width, run->recipe->hidden_width,
            run->device[VISION_FFN], run->device[VISION_UPDATE], err);
    if (rc == YVEX_OK && layer == 0ull)
        rc = vision_observe(run, YVEX_VISION_OBSERVE_FF2, layer,
                            run->device[VISION_UPDATE], run->rows,
                            run->recipe->hidden_width, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_add_bf16(
            run->backend, run->device[VISION_HIDDEN], run->device[VISION_UPDATE],
            run->device[VISION_HIDDEN], run->rows, run->recipe->hidden_width, &facts, err);
    if (rc == YVEX_OK && !vision_facts_add(run, &facts)) rc = YVEX_ERR_BOUNDS;
    if (rc == YVEX_ERR_BOUNDS)
        return vision_refuse(err, rc, "cuda.vision.facts",
                             "vision MLP accounting overflowed");
    return rc;
}

static int vision_merge_prepare(vision_run *run, int postshuffle_norm,
                                const yvex_component_encoded_weight *norm_weight,
                                const yvex_component_encoded_weight *norm_bias,
                                yvex_error *err)
{
    unsigned long long hidden_values, hidden_bytes;
    float *host;
    int rc;
    if (!yvex_core_u64_mul(run->rows, run->recipe->hidden_width, &hidden_values) ||
        !yvex_core_u64_mul(hidden_values, sizeof(float), &hidden_bytes) ||
        hidden_bytes > SIZE_MAX)
        return vision_refuse(err, YVEX_ERR_BOUNDS, "cuda.vision.merge",
                             "vision merger input geometry overflowed");
    if (!postshuffle_norm)
        return vision_norm(run, run->device[VISION_HIDDEN], run->device[VISION_MERGE_INPUT],
                           norm_weight, norm_bias, run->rows,
                           run->recipe->hidden_width, 0, err);
    host = (float *)malloc((size_t)hidden_bytes);
    if (!host)
        return vision_refuse(err, YVEX_ERR_NOMEM, "cuda.vision.merge",
                             "vision merger staging allocation failed");
    rc = yvex_backend_tensor_read(run->backend, run->device[VISION_HIDDEN],
                                  host, hidden_bytes, err);
    if (rc == YVEX_OK)
        rc = yvex_backend_tensor_write(run->backend, run->device[VISION_MERGE_INPUT],
                                       host, hidden_bytes, err);
    if (rc == YVEX_OK &&
        (!yvex_core_u64_add(run->facts.d2h_bytes, hidden_bytes, &run->facts.d2h_bytes) ||
         !yvex_core_u64_add(run->facts.h2d_bytes, hidden_bytes, &run->facts.h2d_bytes)))
        rc = vision_refuse(err, YVEX_ERR_BOUNDS, "cuda.vision.facts",
                           "vision merger transfer accounting overflowed");
    free(host);
    if (rc == YVEX_OK)
        rc = vision_norm(run, run->device[VISION_MERGE_INPUT],
                         run->device[VISION_MERGE_NORM], norm_weight, norm_bias,
                         run->merged_rows, run->recipe->hidden_width *
                                               run->recipe->merge * run->recipe->merge,
                         1, err);
    return rc;
}

static int vision_merge(vision_run *run, unsigned long long merger, int postshuffle_norm,
                        float *output, yvex_error *err)
{
    unsigned long long base = vision_merger_base(run, merger);
    unsigned long long merge_width = run->recipe->hidden_width *
                                     run->recipe->merge * run->recipe->merge;
    unsigned long long output_values, output_bytes;
    yvex_backend_operation_facts facts = {0};
    const yvex_device_tensor *fc_input = postshuffle_norm
                                            ? run->device[VISION_MERGE_NORM]
                                            : run->device[VISION_MERGE_INPUT];
    int rc = vision_merge_prepare(
        run, postshuffle_norm,
        vision_weight(run, base + YVEX_VISION_MERGER_NORM_WEIGHT),
        vision_weight(run, base + YVEX_VISION_MERGER_NORM_BIAS), err);
    if (rc == YVEX_OK)
        rc = vision_linear(
            run, vision_weight(run, base + YVEX_VISION_MERGER_FC1_WEIGHT),
            vision_weight(run, base + YVEX_VISION_MERGER_FC1_BIAS),
            run->merged_rows, merge_width, merge_width,
            fc_input, run->device[VISION_MERGE_FC], err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_transformer_gelu(
            run->backend, run->device[VISION_MERGE_FC], run->device[VISION_MERGE_FC],
            run->merged_rows * merge_width, 0, 1, &facts, err);
    if (rc == YVEX_OK && !vision_facts_add(run, &facts)) rc = YVEX_ERR_BOUNDS;
    if (rc == YVEX_OK)
        rc = vision_linear(
            run, vision_weight(run, base + YVEX_VISION_MERGER_FC2_WEIGHT),
            vision_weight(run, base + YVEX_VISION_MERGER_FC2_BIAS),
            run->merged_rows, merge_width, run->recipe->output_width,
            run->device[VISION_MERGE_FC], run->device[VISION_MERGE_OUTPUT], err);
    if (!yvex_core_u64_mul(run->merged_rows, run->recipe->output_width, &output_values) ||
        !yvex_core_u64_mul(output_values, sizeof(float), &output_bytes))
        rc = YVEX_ERR_BOUNDS;
    if (rc == YVEX_OK)
        rc = yvex_backend_tensor_read(run->backend, run->device[VISION_MERGE_OUTPUT],
                                      output, output_bytes, err);
    if (rc == YVEX_OK &&
        !yvex_core_u64_add(run->facts.d2h_bytes, output_bytes, &run->facts.d2h_bytes))
        rc = YVEX_ERR_BOUNDS;
    if (rc == YVEX_ERR_BOUNDS)
        return vision_refuse(err, rc, "cuda.vision.merge",
                             "vision merger output accounting overflowed");
    return rc;
}

static int vision_patch_execute(vision_run *run, yvex_error *err)
{
    unsigned long long input_values, input_bytes;
    int rc;
    if (!yvex_core_u64_mul(run->rows, run->patch_width, &input_values) ||
        !yvex_core_u64_mul(input_values, sizeof(float), &input_bytes))
        return vision_refuse(err, YVEX_ERR_BOUNDS, "cuda.vision.patch",
                             "vision patch input geometry overflowed");
    rc = yvex_backend_tensor_write(
        run->backend, run->device[VISION_INPUT],
        run->request->patches + run->image_index * input_values, input_bytes, err);
    if (rc == YVEX_OK &&
        !yvex_core_u64_add(run->facts.h2d_bytes, input_bytes, &run->facts.h2d_bytes))
        rc = vision_refuse(err, YVEX_ERR_BOUNDS, "cuda.vision.facts",
                           "vision patch upload accounting overflowed");
    if (rc == YVEX_OK)
        rc = vision_linear(
            run, vision_weight(run, YVEX_VISION_PATCH_WEIGHT),
            vision_weight(run, YVEX_VISION_PATCH_BIAS), run->rows, run->patch_width,
            run->recipe->hidden_width, run->device[VISION_INPUT],
            run->device[VISION_HIDDEN], err);
    if (rc == YVEX_OK)
        rc = vision_observe(run, YVEX_VISION_OBSERVE_PATCH, 0ull,
                            run->device[VISION_HIDDEN], run->rows,
                            run->recipe->hidden_width, err);
    if (rc == YVEX_OK) rc = vision_position_add(run, err);
    if (rc == YVEX_OK)
        rc = vision_observe(run, YVEX_VISION_OBSERVE_POSITION, 0ull,
                            run->device[VISION_HIDDEN], run->rows,
                            run->recipe->hidden_width, err);
    if (rc == YVEX_OK) rc = vision_rope_tables(run, err);
    return rc;
}

static int vision_image_execute(vision_run *run, yvex_error *err)
{
    unsigned long long layer, deepstack = 0ull;
    unsigned long long merged_values = run->merged_rows * run->recipe->output_width;
    float *merged = run->request->merged + run->image_index * merged_values;
    float *deep = run->request->deepstack +
                  run->image_index * run->recipe->deepstack_layer_count * merged_values;
    int rc = vision_devices_open(run, err);
    if (rc == YVEX_OK) rc = vision_patch_execute(run, err);
    for (layer = 0ull; rc == YVEX_OK && layer < run->recipe->layer_count; ++layer) {
        if (run->request->cancel_requested &&
            run->request->cancel_requested(run->request->cancel_context))
            rc = vision_refuse(err, YVEX_ERR_CANCELLED, "cuda.vision.cancel",
                               "vision execution was cancelled");
        if (rc == YVEX_OK) rc = vision_block_attention(run, layer, err);
        if (rc == YVEX_OK) rc = vision_block_mlp(run, layer, err);
        if (rc == YVEX_OK)
            rc = vision_observe(run, YVEX_VISION_OBSERVE_BLOCK, layer,
                                run->device[VISION_HIDDEN], run->rows,
                                run->recipe->hidden_width, err);
        if (rc == YVEX_OK && deepstack < run->recipe->deepstack_layer_count &&
            layer == run->recipe->deepstack_layers[deepstack]) {
            rc = vision_merge(run, 1ull + deepstack, 1,
                              deep + deepstack * merged_values, err);
            ++deepstack;
        }
    }
    if (rc == YVEX_OK && deepstack != run->recipe->deepstack_layer_count)
        rc = vision_refuse(err, YVEX_ERR_STATE, "cuda.vision.deepstack",
                           "vision tower did not publish every deep-stack layer");
    if (rc == YVEX_OK) rc = vision_merge(run, 0ull, 0, merged, err);
    return vision_devices_close(run, rc, err);
}

static int vision_request_validate(const yvex_backend_vision_request *execution,
                                   unsigned long long *weight_count,
                                   unsigned long long *merged_rows, yvex_error *err)
{
    const yvex_vision_request *request = execution ? execution->request : NULL;
    const yvex_vision_recipe *r = request ? request->recipe : NULL;
    unsigned long long patches_per_image, patch_width, merged_values, deep_values;
    if (!execution || !request || !r || r->schema_version != YVEX_VISION_RECIPE_SCHEMA_V1 ||
        !r->semantic_identity || !yvex_sha256_hex_valid(r->semantic_identity) ||
        !request->patches || !request->image_count || !request->grid_height ||
        !request->grid_width || request->grid_height % r->merge ||
        request->grid_width % r->merge || !request->merged || !request->deepstack ||
        !execution->weights || !yvex_sha256_hex_valid(execution->residency_identity) ||
        !r->patch_channels || !r->temporal_patch || !r->patch_height || !r->patch_width ||
        !r->hidden_width || !r->ffn_width || !r->heads || !r->head_dimension ||
        r->heads * r->head_dimension != r->hidden_width || !r->layer_count ||
        !r->merge || r->deepstack_layer_count != 3ull || !r->output_width ||
        !r->position_grid_side || !r->rope_theta || r->normalization_epsilon <= 0.0f ||
        !yvex_core_u64_mul(request->grid_height, request->grid_width, &patches_per_image) ||
        !yvex_core_u64_mul(patches_per_image, request->image_count, &patch_width) ||
        request->patch_rows != patch_width ||
        !yvex_core_u64_mul(r->patch_channels * r->temporal_patch,
                           r->patch_height * r->patch_width, &patch_width) ||
        !yvex_core_u64_mul(request->patch_rows, patch_width, &merged_values) ||
        request->patch_capacity < merged_values ||
        !yvex_core_u64_mul(patches_per_image / (r->merge * r->merge),
                           request->image_count, merged_rows) ||
        !yvex_core_u64_mul(*merged_rows, r->output_width, &merged_values) ||
        request->merged_capacity < merged_values ||
        !yvex_core_u64_mul(merged_values, r->deepstack_layer_count, &deep_values) ||
        request->deepstack_capacity < deep_values ||
        !yvex_core_u64_mul(r->layer_count, YVEX_VISION_BLOCK_WEIGHT_COUNT, weight_count) ||
        !yvex_core_u64_add(*weight_count, YVEX_VISION_EXTERNAL_WEIGHT_COUNT, weight_count) ||
        !yvex_core_u64_add(*weight_count,
                           (1ull + r->deepstack_layer_count) *
                               YVEX_VISION_MERGER_WEIGHT_COUNT,
                           weight_count) || execution->weight_count != *weight_count)
        return vision_refuse(err, YVEX_ERR_INVALID_ARG, "cuda.vision.validate",
                             "one complete source-described vision tower is required");
    return YVEX_OK;
}

static int vision_identity(const vision_run *run, char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index, merged_values, deep_values;
    if (!yvex_core_u64_mul(run->request->image_count * run->merged_rows,
                           run->recipe->output_width, &merged_values) ||
        !yvex_core_u64_mul(merged_values, run->recipe->deepstack_layer_count, &deep_values))
        return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.backend.vision.v1") ||
        !yvex_sha256_update_text(&hash, run->recipe->semantic_identity) ||
        !yvex_sha256_update_text(&hash, run->execution->residency_identity) ||
        !yvex_sha256_update_u64(&hash, run->request->image_count) ||
        !yvex_sha256_update_u64(&hash, run->request->grid_height) ||
        !yvex_sha256_update_u64(&hash, run->request->grid_width)) return 0;
    for (index = 0ull; index < merged_values; ++index) {
        uint32_t bits; memcpy(&bits, run->request->merged + index, sizeof(bits));
        if (!yvex_sha256_update_u64(&hash, bits)) return 0;
    }
    for (index = 0ull; index < deep_values; ++index) {
        uint32_t bits; memcpy(&bits, run->request->deepstack + index, sizeof(bits));
        if (!yvex_sha256_update_u64(&hash, bits)) return 0;
    }
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

int yvex_backend_vision_execute(
    yvex_backend *backend, const yvex_backend_vision_request *execution,
    yvex_vision_result *result, yvex_error *err)
{
    vision_run run = {0};
    unsigned long long weight_count, total_merged_rows, image;
    int rc;
    if (result) memset(result, 0, sizeof(*result));
    if (!backend || !result || yvex_backend_kind_of(backend) != YVEX_BACKEND_KIND_CUDA)
        return vision_refuse(err, YVEX_ERR_INVALID_ARG, "cuda.vision",
                             "one CUDA backend and result are required");
    rc = vision_request_validate(execution, &weight_count, &total_merged_rows, err);
    if (rc != YVEX_OK) return rc;
    run.backend = backend; run.execution = execution; run.request = execution->request;
    run.recipe = run.request->recipe;
    run.rows = run.request->grid_height * run.request->grid_width;
    run.merged_rows = run.rows / (run.recipe->merge * run.recipe->merge);
    run.patch_width = run.recipe->patch_channels * run.recipe->temporal_patch *
                      run.recipe->patch_height * run.recipe->patch_width;
    for (image = 0ull; rc == YVEX_OK && image < run.request->image_count; ++image) {
        run.image_index = image;
        rc = vision_image_execute(&run, err);
    }
    if (rc == YVEX_OK && !vision_identity(&run, run.facts.execution_identity))
        rc = vision_refuse(err, YVEX_ERR_STATE, "cuda.vision.identity",
                           "vision execution identity could not be sealed");
    if (rc == YVEX_OK) {
        run.facts.patch_rows = run.request->patch_rows;
        run.facts.merged_rows = total_merged_rows;
        run.facts.hidden_width = run.recipe->hidden_width;
        run.facts.output_width = run.recipe->output_width;
        run.facts.layer_count = run.recipe->layer_count;
        run.facts.device_bytes = run.device_bytes;
        yvex_core_text_copy(run.facts.residency_identity,
                            sizeof(run.facts.residency_identity),
                            execution->residency_identity);
        run.facts.complete = 1; *result = run.facts; yvex_error_clear(err);
    }
    return rc;
}
