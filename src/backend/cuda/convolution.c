/* Execute multistage alias-free Conv1D decoders over admitted resident F32 weights. */
#include "src/backend/cuda/private.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yvex/internal/component.h>
#include <yvex/internal/convolution.h>
#include <yvex/qtype.h>

#define CONVOLUTION_BLOCK 256u

typedef struct {
    yvex_backend *backend;
    const yvex_alias_decoder_request *request;
    yvex_alias_decoder_result *result;
    unsigned long long live_device_bytes, peak_device_bytes;
} alias_decoder_run;

static int decoder_refuse(yvex_error *err, yvex_status status,
                          const char *stage, const char *message)
{
    yvex_error_set(err, status, stage, message);
    return status;
}

int yvex_alias_decoder_template_name(
    void *context, yvex_alias_decoder_weight_role role,
    unsigned long long stage, unsigned long long block,
    unsigned long long layer, char output[256], yvex_error *err)
{
    const yvex_alias_decoder_name_templates *templates =
        (const yvex_alias_decoder_name_templates *)context;
    int length = -1;
    unsigned long long residual = stage * 3ull + block;
    if (!templates || role >= YVEX_ALIAS_DECODER_WEIGHT_ROLE_COUNT || !output ||
        !templates->input_projection || !templates->pre_convolution ||
        !templates->upsamples || !templates->residual_blocks ||
        !templates->post_activation || !templates->post_convolution) {
        return decoder_refuse(err, YVEX_ERR_INVALID_ARG,
                              "cuda.alias-decoder.weight-name",
                              "alias decoder weight-name template is incomplete");
    }
    switch (role) {
    case YVEX_ALIAS_DECODER_INPUT_WEIGHT:
        length = snprintf(output, 256u, "%s.weight", templates->input_projection); break;
    case YVEX_ALIAS_DECODER_INPUT_BIAS:
        length = snprintf(output, 256u, "%s.bias", templates->input_projection); break;
    case YVEX_ALIAS_DECODER_PRE_WEIGHT:
        length = snprintf(output, 256u, "%s.weight_v", templates->pre_convolution); break;
    case YVEX_ALIAS_DECODER_PRE_GAIN:
        length = snprintf(output, 256u, "%s.weight_g", templates->pre_convolution); break;
    case YVEX_ALIAS_DECODER_PRE_BIAS:
        length = snprintf(output, 256u, "%s.bias", templates->pre_convolution); break;
    case YVEX_ALIAS_DECODER_UP_WEIGHT:
        length = snprintf(output, 256u, "%s.%llu.0.weight_v", templates->upsamples, stage); break;
    case YVEX_ALIAS_DECODER_UP_GAIN:
        length = snprintf(output, 256u, "%s.%llu.0.weight_g", templates->upsamples, stage); break;
    case YVEX_ALIAS_DECODER_UP_BIAS:
        length = snprintf(output, 256u, "%s.%llu.0.bias", templates->upsamples, stage); break;
    case YVEX_ALIAS_DECODER_ACT_ALPHA:
        length = snprintf(output, 256u, "%s.%llu.activations.%llu.act.alpha",
                          templates->residual_blocks, residual, layer); break;
    case YVEX_ALIAS_DECODER_ACT_BETA:
        length = snprintf(output, 256u, "%s.%llu.activations.%llu.act.beta",
                          templates->residual_blocks, residual, layer); break;
    case YVEX_ALIAS_DECODER_ACT_UP_FILTER:
        length = snprintf(output, 256u, "%s.%llu.activations.%llu.upsample.filter",
                          templates->residual_blocks, residual, layer); break;
    case YVEX_ALIAS_DECODER_ACT_DOWN_FILTER:
        length = snprintf(output, 256u, "%s.%llu.activations.%llu.downsample.lowpass.filter",
                          templates->residual_blocks, residual, layer); break;
    case YVEX_ALIAS_DECODER_RES1_WEIGHT:
        length = snprintf(output, 256u, "%s.%llu.convs1.%llu.weight_v",
                          templates->residual_blocks, residual, layer); break;
    case YVEX_ALIAS_DECODER_RES1_GAIN:
        length = snprintf(output, 256u, "%s.%llu.convs1.%llu.weight_g",
                          templates->residual_blocks, residual, layer); break;
    case YVEX_ALIAS_DECODER_RES1_BIAS:
        length = snprintf(output, 256u, "%s.%llu.convs1.%llu.bias",
                          templates->residual_blocks, residual, layer); break;
    case YVEX_ALIAS_DECODER_RES2_WEIGHT:
        length = snprintf(output, 256u, "%s.%llu.convs2.%llu.weight_v",
                          templates->residual_blocks, residual, layer); break;
    case YVEX_ALIAS_DECODER_RES2_GAIN:
        length = snprintf(output, 256u, "%s.%llu.convs2.%llu.weight_g",
                          templates->residual_blocks, residual, layer); break;
    case YVEX_ALIAS_DECODER_RES2_BIAS:
        length = snprintf(output, 256u, "%s.%llu.convs2.%llu.bias",
                          templates->residual_blocks, residual, layer); break;
    case YVEX_ALIAS_DECODER_POST_ACT_ALPHA:
        length = snprintf(output, 256u, "%s.act.alpha", templates->post_activation); break;
    case YVEX_ALIAS_DECODER_POST_ACT_BETA:
        length = snprintf(output, 256u, "%s.act.beta", templates->post_activation); break;
    case YVEX_ALIAS_DECODER_POST_ACT_UP_FILTER:
        length = snprintf(output, 256u, "%s.upsample.filter", templates->post_activation); break;
    case YVEX_ALIAS_DECODER_POST_ACT_DOWN_FILTER:
        length = snprintf(output, 256u, "%s.downsample.lowpass.filter",
                          templates->post_activation); break;
    case YVEX_ALIAS_DECODER_POST_WEIGHT:
        length = snprintf(output, 256u, "%s.weight_v", templates->post_convolution); break;
    case YVEX_ALIAS_DECODER_POST_GAIN:
        length = snprintf(output, 256u, "%s.weight_g", templates->post_convolution); break;
    default: break;
    }
    if (length < 0 || length >= 256)
        return decoder_refuse(err, YVEX_ERR_BOUNDS,
                              "cuda.alias-decoder.weight-name",
                              "alias decoder weight name exceeded its identity bound");
    return YVEX_OK;
}

static int decoder_grid(unsigned long long tasks, unsigned int *grid, yvex_error *err)
{
    if (!tasks || tasks > (unsigned long long)UINT_MAX * CONVOLUTION_BLOCK)
        return decoder_refuse(err, YVEX_ERR_BOUNDS, "cuda.alias-decoder.launch",
                              "alias decoder launch geometry exceeds the CUDA grid bound");
    *grid = (unsigned int)((tasks + CONVOLUTION_BLOCK - 1ull) / CONVOLUTION_BLOCK);
    return YVEX_OK;
}

static int decoder_launch(alias_decoder_run *run, CUfunction function,
                          unsigned long long tasks, void **parameters,
                          const char *stage, yvex_error *err)
{
    unsigned int grid;
    int rc = decoder_grid(tasks, &grid, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_launch(run->backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
                              function, grid, CONVOLUTION_BLOCK, 0u,
                              parameters, stage, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_synchronize(run->backend,
                                   YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
                                   stage, err);
    if (rc == YVEX_OK) run->result->kernel_launches++;
    return rc;
}

static int decoder_output_length(const yvex_convolution_1d_geometry *geometry,
                                 unsigned long long *output_length, yvex_error *err)
{
    unsigned long long dilated, result;
    if (!geometry || !output_length || !geometry->batch ||
        !geometry->input_channels || !geometry->output_channels ||
        !geometry->input_length || !geometry->kernel_size ||
        !geometry->stride || !geometry->dilation ||
        !yvex_core_u64_mul(geometry->kernel_size - 1ull,
                           geometry->dilation, &dilated) ||
        !yvex_core_u64_add(dilated, 1ull, &dilated))
        return decoder_refuse(err, YVEX_ERR_INVALID_ARG,
                              "cuda.alias-decoder.geometry",
                              "alias decoder Conv1D geometry is invalid");
    if (geometry->transposed) {
        unsigned long long expanded, removed;
        if (geometry->output_padding >= geometry->stride ||
            !yvex_core_u64_mul(geometry->input_length - 1ull,
                               geometry->stride, &expanded) ||
            !yvex_core_u64_mul(geometry->padding, 2ull, &removed) ||
            !yvex_core_u64_add(expanded, dilated, &result) ||
            !yvex_core_u64_add(result, geometry->output_padding, &result) ||
            result <= removed)
            return decoder_refuse(err, YVEX_ERR_BOUNDS,
                                  "cuda.alias-decoder.geometry",
                                  "transposed Conv1D output geometry is invalid");
        result -= removed;
    } else {
        unsigned long long added, padded;
        if (geometry->output_padding ||
            !yvex_core_u64_mul(geometry->padding, 2ull, &added) ||
            !yvex_core_u64_add(geometry->input_length, added, &padded) ||
            padded < dilated)
            return decoder_refuse(err, YVEX_ERR_BOUNDS,
                                  "cuda.alias-decoder.geometry",
                                  "ordinary Conv1D output geometry is invalid");
        result = (padded - dilated) / geometry->stride + 1ull;
    }
    *output_length = result;
    return YVEX_OK;
}

static int decoder_tensor_open(alias_decoder_run *run, const char *name,
                               unsigned long long batch, unsigned long long channels,
                               unsigned long long length, yvex_device_tensor **tensor,
                               yvex_error *err)
{
    yvex_backend_tensor_desc descriptor = {0};
    unsigned long long values, bytes, live;
    if (!run || !name || !batch || !channels || !length || !tensor ||
        !yvex_core_u64_mul(batch, channels, &values) ||
        !yvex_core_u64_mul(values, length, &values) ||
        !yvex_core_u64_mul(values, sizeof(float), &bytes) ||
        !yvex_core_u64_add(run->live_device_bytes, bytes, &live))
        return decoder_refuse(err, YVEX_ERR_BOUNDS, "cuda.alias-decoder.allocate",
                              "alias decoder activation geometry overflowed");
    descriptor.name = name;
    descriptor.dtype = YVEX_DTYPE_F32;
    descriptor.rank = 3u;
    descriptor.dims[0] = batch;
    descriptor.dims[1] = channels;
    descriptor.dims[2] = length;
    descriptor.bytes = bytes;
    if (yvex_backend_tensor_alloc(run->backend, &descriptor, tensor, err) != YVEX_OK)
        return yvex_error_code(err);
    run->live_device_bytes = live;
    if (live > run->peak_device_bytes) run->peak_device_bytes = live;
    return YVEX_OK;
}

static int decoder_tensor_close(alias_decoder_run *run, yvex_device_tensor **tensor,
                                int rc, yvex_error *err)
{
    unsigned long long bytes;
    yvex_error cleanup;
    int cleanup_rc;
    if (!tensor || !*tensor) return rc;
    bytes = (*tensor)->bytes;
    yvex_error_clear(&cleanup);
    cleanup_rc = yvex_backend_tensor_release(run->backend, tensor, &cleanup);
    if (bytes <= run->live_device_bytes) run->live_device_bytes -= bytes;
    if (cleanup_rc != YVEX_OK) {
        if (err) *err = cleanup;
        return cleanup_rc;
    }
    return rc;
}

static int decoder_weight(alias_decoder_run *run,
                          yvex_alias_decoder_weight_role role,
                          unsigned long long stage, unsigned long long block,
                          unsigned long long layer, unsigned long long rows,
                          unsigned long long width, CUdeviceptr *address,
                          yvex_error *err)
{
    yvex_convolution_weight weight = {0};
    unsigned long long row_bytes, bytes;
    char name[256];
    int rc;
    if (!run->request->weight_name || !run->request->weight_bind ||
        !rows || !width || !address ||
        !yvex_core_u64_mul(width, sizeof(float), &row_bytes) ||
        !yvex_core_u64_mul(rows, row_bytes, &bytes))
        return decoder_refuse(err, YVEX_ERR_INVALID_ARG,
                              "cuda.alias-decoder.weight",
                              "alias decoder weight request is invalid");
    rc = run->request->weight_name(run->request->weight_name_context, role,
                                   stage, block, layer, name, err);
    if (rc == YVEX_OK)
        rc = run->request->weight_bind(run->request->weight_bind_context, name,
                                      rows, width, &weight, err);
    if (rc != YVEX_OK) return rc;
    if (!weight.encoded || weight.qtype != YVEX_GGUF_QTYPE_F32 ||
        weight.encoded_bytes != bytes ||
        yvex_backend_resident_resolve(run->backend, weight.encoded,
                                      weight.encoded_bytes, address) !=
            YVEX_BACKEND_RESIDENT_HIT)
        return decoder_refuse(err, YVEX_ERR_FORMAT,
                              "cuda.alias-decoder.weight",
                              "alias decoder requires an exact resident F32 weight");
    run->result->weight_bindings++;
    return YVEX_OK;
}

static int decoder_scale(alias_decoder_run *run, CUdeviceptr weight,
                         CUdeviceptr gain, unsigned long long rows,
                         unsigned long long row_width,
                         yvex_device_tensor **scale, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(run->backend);
    CUdeviceptr scale_address;
    int rc = decoder_tensor_open(run, "conv-weight-scale", 1ull, 1ull,
                                 rows, scale, err);
    if (rc != YVEX_OK) return rc;
    scale_address = yvex_cuda_tensor_ptr(*scale);
    {
        void *parameters[] = {&weight, &gain, &scale_address, &rows, &row_width};
        rc = decoder_launch(run, state ? state->conv_scale_function : NULL,
                            rows, parameters, "cuda.alias-decoder.weight-scale", err);
    }
    return rc;
}

static int decoder_convolution(
    alias_decoder_run *run, yvex_alias_decoder_weight_role weight_role,
    yvex_alias_decoder_weight_role gain_role,
    yvex_alias_decoder_weight_role bias_role, int normalized, int biased,
    unsigned long long stage, unsigned long long block, unsigned long long layer,
    const yvex_convolution_1d_geometry *geometry,
    const yvex_device_tensor *input, yvex_device_tensor *output,
    yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(run->backend);
    yvex_device_tensor *scale = NULL;
    CUdeviceptr input_address, weight = 0ull, gain = 0ull, bias = 0ull;
    CUdeviceptr scale_address = 0ull, output_address;
    unsigned long long batch, input_channels, output_channels, input_length;
    unsigned long long kernel_size, stride, dilation, padding;
    unsigned long long output_length, weight_rows, weight_width, tasks;
    CUfunction function;
    int rc;
    if (!geometry || !input || !output ||
        decoder_output_length(geometry, &output_length, err) != YVEX_OK)
        return yvex_error_code(err);
    weight_rows = geometry->transposed ? geometry->input_channels :
                                           geometry->output_channels;
    if (!yvex_core_u64_mul(geometry->transposed ? geometry->output_channels :
                                                    geometry->input_channels,
                           geometry->kernel_size, &weight_width) ||
        !yvex_core_u64_mul(geometry->batch, geometry->output_channels, &tasks) ||
        !yvex_core_u64_mul(tasks, output_length, &tasks))
        return decoder_refuse(err, YVEX_ERR_BOUNDS,
                              "cuda.alias-decoder.convolution",
                              "alias decoder convolution extent overflowed");
    rc = decoder_weight(run, weight_role, stage, block, layer,
                        weight_rows, weight_width, &weight, err);
    if (normalized && rc == YVEX_OK)
        rc = decoder_weight(run, gain_role, stage, block, layer,
                            weight_rows, 1ull, &gain, err);
    if (biased && rc == YVEX_OK)
        rc = decoder_weight(run, bias_role, stage, block, layer,
                            geometry->output_channels, 1ull, &bias, err);
    if (normalized && rc == YVEX_OK)
        rc = decoder_scale(run, weight, gain, weight_rows,
                           weight_width, &scale, err);
    if (rc == YVEX_OK) {
        input_address = yvex_cuda_tensor_ptr(input);
        output_address = yvex_cuda_tensor_ptr(output);
        if (scale) scale_address = yvex_cuda_tensor_ptr(scale);
        batch = geometry->batch;
        input_channels = geometry->input_channels;
        output_channels = geometry->output_channels;
        input_length = geometry->input_length;
        kernel_size = geometry->kernel_size;
        stride = geometry->stride;
        dilation = geometry->dilation;
        padding = geometry->padding;
        function = geometry->transposed
                       ? (state ? state->conv1d_transposed_function : NULL)
                       : (state ? state->conv1d_function : NULL);
        void *parameters[] = {
            &input_address, &weight, &bias, &scale_address, &output_address,
            &batch, &input_channels, &output_channels, &input_length,
            &output_length, &kernel_size, &stride, &dilation, &padding,
        };
        rc = decoder_launch(run, function, tasks, parameters,
                            "cuda.alias-decoder.convolution", err);
    }
    return decoder_tensor_close(run, &scale, rc, err);
}

static int decoder_activation(
    alias_decoder_run *run, yvex_alias_decoder_weight_role alpha_role,
    yvex_alias_decoder_weight_role beta_role,
    yvex_alias_decoder_weight_role up_role,
    yvex_alias_decoder_weight_role down_role,
    unsigned long long stage, unsigned long long block, unsigned long long layer,
    unsigned long long batch, unsigned long long channels,
    unsigned long long length, const yvex_device_tensor *input,
    yvex_device_tensor *output, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(run->backend);
    yvex_device_tensor *scratch = NULL;
    CUdeviceptr input_address, alpha = 0ull, beta = 0ull, up = 0ull, down = 0ull;
    CUdeviceptr scratch_address, output_address;
    unsigned long long groups, doubled, up_tasks, down_tasks;
    int rc;
    if (!yvex_core_u64_mul(batch, channels, &groups) ||
        !yvex_core_u64_mul(length, 2ull, &doubled) ||
        !yvex_core_u64_mul(groups, doubled, &up_tasks) ||
        !yvex_core_u64_mul(groups, length, &down_tasks))
        return decoder_refuse(err, YVEX_ERR_BOUNDS,
                              "cuda.alias-decoder.activation",
                              "alias-free activation extent overflowed");
    rc = decoder_weight(run, alpha_role, stage, block, layer,
                        channels, 1ull, &alpha, err);
    if (rc == YVEX_OK)
        rc = decoder_weight(run, beta_role, stage, block, layer,
                            channels, 1ull, &beta, err);
    if (rc == YVEX_OK)
        rc = decoder_weight(run, up_role, stage, block, layer, 1ull, 12ull, &up, err);
    if (rc == YVEX_OK)
        rc = decoder_weight(run, down_role, stage, block, layer, 1ull, 12ull, &down, err);
    if (rc == YVEX_OK)
        rc = decoder_tensor_open(run, "alias-upsampled", batch, channels,
                                 doubled, &scratch, err);
    if (rc == YVEX_OK) {
        input_address = yvex_cuda_tensor_ptr(input);
        scratch_address = yvex_cuda_tensor_ptr(scratch);
        void *parameters[] = {
            &input_address, &alpha, &beta, &up, &scratch_address,
            &groups, &channels, &length,
        };
        rc = decoder_launch(run, state ? state->alias_up_function : NULL,
                            up_tasks, parameters, "cuda.alias-decoder.alias-up", err);
    }
    if (rc == YVEX_OK) {
        scratch_address = yvex_cuda_tensor_ptr(scratch);
        output_address = yvex_cuda_tensor_ptr(output);
        void *parameters[] = {
            &scratch_address, &down, &output_address, &groups, &length,
        };
        rc = decoder_launch(run, state ? state->alias_down_function : NULL,
                            down_tasks, parameters, "cuda.alias-decoder.alias-down", err);
    }
    return decoder_tensor_close(run, &scratch, rc, err);
}

static int decoder_update(alias_decoder_run *run, yvex_device_tensor *destination,
                          const yvex_device_tensor *source, float destination_scale,
                          float source_scale, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(run->backend);
    CUdeviceptr destination_address, source_address;
    unsigned long long count;
    if (!destination || !source || destination->bytes != source->bytes ||
        destination->bytes % sizeof(float))
        return decoder_refuse(err, YVEX_ERR_FORMAT, "cuda.alias-decoder.update",
                              "alias decoder update tensors differ in extent");
    count = destination->bytes / sizeof(float);
    destination_address = yvex_cuda_tensor_ptr(destination);
    source_address = yvex_cuda_tensor_ptr(source);
    {
        void *parameters[] = {
            &destination_address, &source_address, &count,
            &destination_scale, &source_scale,
        };
        return decoder_launch(run, state ? state->vector_update_function : NULL,
                              count, parameters, "cuda.alias-decoder.update", err);
    }
}

static int decoder_cancel(alias_decoder_run *run, yvex_error *err)
{
    if (run->request->cancel_requested &&
        run->request->cancel_requested(run->request->cancel_context))
        return decoder_refuse(err, YVEX_ERR_CANCELLED, "cuda.alias-decoder.cancel",
                              "alias decoder execution was cancelled between layers");
    return YVEX_OK;
}

static int decoder_residual_block(alias_decoder_run *run,
                                  unsigned long long stage,
                                  unsigned long long block,
                                  unsigned long long batch,
                                  unsigned long long channels,
                                  unsigned long long length,
                                  const yvex_device_tensor *input,
                                  yvex_device_tensor *output, yvex_error *err)
{
    const yvex_alias_decoder_recipe *recipe = run->request->recipe;
    yvex_device_tensor *activation = NULL, *convolution = NULL;
    yvex_convolution_1d_geometry geometry = {0};
    unsigned long long layer;
    int rc = yvex_backend_tensor_copy(run->backend, output, input, err);
    if (rc == YVEX_OK) {
        run->result->kernel_launches++;
        rc = decoder_tensor_open(run, "residual-activation", batch, channels,
                                 length, &activation, err);
    }
    if (rc == YVEX_OK)
        rc = decoder_tensor_open(run, "residual-convolution", batch, channels,
                                 length, &convolution, err);
    geometry.batch = batch;
    geometry.input_channels = channels;
    geometry.output_channels = channels;
    geometry.input_length = length;
    geometry.kernel_size = recipe->residual_kernels[block];
    geometry.stride = 1ull;
    geometry.dilation = 1ull;
    for (layer = 0ull; rc == YVEX_OK && layer < recipe->residual_layers; ++layer) {
        rc = decoder_cancel(run, err);
        if (rc == YVEX_OK)
            rc = decoder_activation(
                run, YVEX_ALIAS_DECODER_ACT_ALPHA, YVEX_ALIAS_DECODER_ACT_BETA,
                YVEX_ALIAS_DECODER_ACT_UP_FILTER, YVEX_ALIAS_DECODER_ACT_DOWN_FILTER,
                stage, block, layer * 2ull, batch, channels, length,
                output, activation, err);
        geometry.dilation = recipe->residual_dilations[layer];
        geometry.padding = ((geometry.kernel_size - 1ull) * geometry.dilation) / 2ull;
        if (rc == YVEX_OK)
            rc = decoder_convolution(
                run, YVEX_ALIAS_DECODER_RES1_WEIGHT,
                YVEX_ALIAS_DECODER_RES1_GAIN, YVEX_ALIAS_DECODER_RES1_BIAS,
                1, 1, stage, block, layer, &geometry,
                activation, convolution, err);
        if (rc == YVEX_OK)
            rc = decoder_activation(
                run, YVEX_ALIAS_DECODER_ACT_ALPHA, YVEX_ALIAS_DECODER_ACT_BETA,
                YVEX_ALIAS_DECODER_ACT_UP_FILTER, YVEX_ALIAS_DECODER_ACT_DOWN_FILTER,
                stage, block, layer * 2ull + 1ull, batch, channels, length,
                convolution, activation, err);
        geometry.dilation = 1ull;
        geometry.padding = (geometry.kernel_size - 1ull) / 2ull;
        if (rc == YVEX_OK)
            rc = decoder_convolution(
                run, YVEX_ALIAS_DECODER_RES2_WEIGHT,
                YVEX_ALIAS_DECODER_RES2_GAIN, YVEX_ALIAS_DECODER_RES2_BIAS,
                1, 1, stage, block, layer, &geometry,
                activation, convolution, err);
        if (rc == YVEX_OK)
            rc = decoder_update(run, output, convolution, 1.0f, 1.0f, err);
    }
    rc = decoder_tensor_close(run, &convolution, rc, err);
    return decoder_tensor_close(run, &activation, rc, err);
}

static int decoder_stage(alias_decoder_run *run, unsigned long long stage,
                         unsigned long long batch, unsigned long long input_channels,
                         unsigned long long input_length, yvex_device_tensor *input,
                         yvex_device_tensor **output, unsigned long long *output_channels,
                         unsigned long long *output_length, yvex_error *err)
{
    const yvex_alias_decoder_recipe *recipe = run->request->recipe;
    yvex_device_tensor *upsampled = NULL, *sum = NULL, *block = NULL;
    yvex_convolution_1d_geometry geometry = {0};
    unsigned long long residual;
    int rc;
    *output = NULL;
    *output_channels = input_channels / 2ull;
    geometry.batch = batch;
    geometry.input_channels = input_channels;
    geometry.output_channels = *output_channels;
    geometry.input_length = input_length;
    geometry.kernel_size = recipe->upsample_kernels[stage];
    geometry.stride = recipe->rates[stage];
    geometry.dilation = 1ull;
    geometry.padding = (geometry.kernel_size - geometry.stride) / 2ull;
    geometry.transposed = 1;
    rc = decoder_output_length(&geometry, output_length, err);
    if (rc == YVEX_OK)
        rc = decoder_tensor_open(run, "stage-upsampled", batch, *output_channels,
                                 *output_length, &upsampled, err);
    if (rc == YVEX_OK)
        rc = decoder_tensor_open(run, "stage-sum", batch, *output_channels,
                                 *output_length, &sum, err);
    if (rc == YVEX_OK)
        rc = decoder_tensor_open(run, "stage-block", batch, *output_channels,
                                 *output_length, &block, err);
    if (rc == YVEX_OK)
        rc = decoder_convolution(
            run, YVEX_ALIAS_DECODER_UP_WEIGHT, YVEX_ALIAS_DECODER_UP_GAIN,
            YVEX_ALIAS_DECODER_UP_BIAS, 1, 1, stage, 0ull, 0ull,
            &geometry, input, upsampled, err);
    for (residual = 0ull; rc == YVEX_OK &&
         residual < recipe->residual_blocks; ++residual) {
        rc = decoder_cancel(run, err);
        if (rc == YVEX_OK)
            rc = decoder_residual_block(run, stage, residual, batch,
                                        *output_channels, *output_length,
                                        upsampled, block, err);
        if (rc == YVEX_OK && residual == 0ull)
            rc = yvex_backend_tensor_copy(run->backend, sum, block, err);
        else if (rc == YVEX_OK)
            rc = decoder_update(run, sum, block, 1.0f, 1.0f, err);
        if (rc == YVEX_OK && residual == 0ull) run->result->kernel_launches++;
    }
    if (rc == YVEX_OK)
        rc = decoder_update(run, sum, sum,
                            1.0f / (float)recipe->residual_blocks, 0.0f, err);
    if (rc == YVEX_OK) {
        *output = sum;
        sum = NULL;
    }
    rc = decoder_tensor_close(run, &block, rc, err);
    rc = decoder_tensor_close(run, &sum, rc, err);
    return decoder_tensor_close(run, &upsampled, rc, err);
}

static int decoder_recipe_valid(const yvex_alias_decoder_request *request,
                                unsigned long long *output_length)
{
    const yvex_alias_decoder_recipe *recipe;
    unsigned long long channels, length, stage, values;
    if (!request || !(recipe = request->recipe) || !request->input ||
        !request->output || !request->batch || !request->input_length ||
        !recipe->input_channels || !recipe->projection_channels ||
        !recipe->input_kernel || !recipe->pre_channels || !recipe->pre_kernel ||
        !(recipe->pre_kernel & 1ull) || !recipe->stage_count ||
        recipe->stage_count > YVEX_ALIAS_DECODER_MAX_STAGES ||
        !recipe->residual_blocks ||
        recipe->residual_blocks > YVEX_ALIAS_DECODER_MAX_RESBLOCKS ||
        !recipe->residual_layers ||
        recipe->residual_layers > YVEX_ALIAS_DECODER_MAX_LAYERS ||
        !recipe->final_channels || !recipe->final_kernel ||
        !request->weight_name || !request->weight_bind ||
        !yvex_core_u64_mul(request->batch, recipe->input_channels, &values) ||
        !yvex_core_u64_mul(values, request->input_length, &values) ||
        values != request->input_count) return 0;
    channels = recipe->pre_channels;
    length = request->input_length;
    for (stage = 0ull; stage < recipe->stage_count; ++stage) {
        unsigned long long kernel = recipe->upsample_kernels[stage];
        unsigned long long rate = recipe->rates[stage];
        unsigned long long block;
        if (!rate || !kernel || kernel < rate || ((kernel - rate) & 1ull) ||
            channels < 2ull) return 0;
        for (block = 0ull; block < recipe->residual_blocks; ++block)
            if (!recipe->residual_kernels[block] ||
                !(recipe->residual_kernels[block] & 1ull)) return 0;
        if (length > (ULLONG_MAX - kernel) / rate) return 0;
        length = (length - 1ull) * rate + kernel - (kernel - rate);
        channels /= 2ull;
    }
    for (stage = 0ull; stage < recipe->residual_layers; ++stage)
        if (!recipe->residual_dilations[stage]) return 0;
    if (!yvex_core_u64_mul(request->batch, recipe->final_channels, &values) ||
        !yvex_core_u64_mul(values, length, &values) ||
        request->output_capacity < values) return 0;
    *output_length = length;
    return 1;
}

static int decoder_output(alias_decoder_run *run, yvex_device_tensor *current,
                          unsigned long long channels, unsigned long long length,
                          float *staged, yvex_error *err)
{
    const yvex_alias_decoder_recipe *recipe = run->request->recipe;
    yvex_cuda_backend_state *state = yvex_cuda_state(run->backend);
    yvex_device_tensor *activation = NULL, *output = NULL;
    yvex_convolution_1d_geometry geometry = {0};
    CUdeviceptr output_address;
    unsigned long long values;
    float lower = -1.0f, upper = 1.0f;
    int rc = decoder_tensor_open(run, "post-activation", run->request->batch,
                                 channels, length, &activation, err);
    if (rc == YVEX_OK)
        rc = decoder_activation(
            run, YVEX_ALIAS_DECODER_POST_ACT_ALPHA,
            YVEX_ALIAS_DECODER_POST_ACT_BETA,
            YVEX_ALIAS_DECODER_POST_ACT_UP_FILTER,
            YVEX_ALIAS_DECODER_POST_ACT_DOWN_FILTER,
            0ull, 0ull, 0ull, run->request->batch, channels, length,
            current, activation, err);
    if (rc == YVEX_OK)
        rc = decoder_tensor_open(run, "decoder-output", run->request->batch,
                                 recipe->final_channels, length, &output, err);
    geometry.batch = run->request->batch;
    geometry.input_channels = channels;
    geometry.output_channels = recipe->final_channels;
    geometry.input_length = length;
    geometry.kernel_size = recipe->final_kernel;
    geometry.stride = 1ull;
    geometry.dilation = 1ull;
    geometry.padding = (recipe->final_kernel - 1ull) / 2ull;
    if (rc == YVEX_OK)
        rc = decoder_convolution(
            run, YVEX_ALIAS_DECODER_POST_WEIGHT, YVEX_ALIAS_DECODER_POST_GAIN,
            YVEX_ALIAS_DECODER_POST_GAIN, 1, 0, 0ull, 0ull, 0ull,
            &geometry, activation, output, err);
    if (rc == YVEX_OK) {
        values = output->bytes / sizeof(float);
        output_address = yvex_cuda_tensor_ptr(output);
        void *parameters[] = {&output_address, &values, &lower, &upper};
        rc = decoder_launch(run, state ? state->clamp_function : NULL,
                            values, parameters, "cuda.alias-decoder.clamp", err);
    }
    if (rc == YVEX_OK)
        rc = yvex_backend_tensor_read(run->backend, output, staged,
                                      output->bytes, err);
    if (rc == YVEX_OK) run->result->d2h_bytes += output->bytes;
    rc = decoder_tensor_close(run, &output, rc, err);
    return decoder_tensor_close(run, &activation, rc, err);
}

int yvex_backend_alias_decoder_execute(
    yvex_backend *backend, const yvex_alias_decoder_request *request,
    yvex_alias_decoder_result *result, yvex_error *err)
{
    alias_decoder_run run = {0};
    const yvex_alias_decoder_recipe *recipe;
    yvex_device_tensor *current = NULL, *next = NULL;
    yvex_convolution_1d_geometry geometry = {0};
    unsigned long long output_length = 0ull, output_values, channels, length, stage;
    float *staged = NULL;
    int rc;
    if (result) memset(result, 0, sizeof(*result));
    if (!backend || !result || yvex_backend_kind_of(backend) != YVEX_BACKEND_KIND_CUDA ||
        !decoder_recipe_valid(request, &output_length) ||
        !yvex_core_u64_mul(request->batch, request->recipe->final_channels,
                           &output_values) ||
        !yvex_core_u64_mul(output_values, output_length, &output_values) ||
        output_values > SIZE_MAX / sizeof(float))
        return decoder_refuse(err, YVEX_ERR_INVALID_ARG,
                              "cuda.alias-decoder.validate",
                              "one complete bounded alias decoder request is required");
    staged = (float *)malloc((size_t)output_values * sizeof(float));
    if (!staged)
        return decoder_refuse(err, YVEX_ERR_NOMEM,
                              "cuda.alias-decoder.allocate",
                              "alias decoder transactional output allocation failed");
    run.backend = backend;
    run.request = request;
    run.result = result;
    recipe = request->recipe;
    rc = decoder_tensor_open(&run, "latent-projection", request->batch,
                             recipe->projection_channels, request->input_length,
                             &current, err);
    geometry.batch = request->batch;
    geometry.input_channels = recipe->input_channels;
    geometry.output_channels = recipe->projection_channels;
    geometry.input_length = request->input_length;
    geometry.kernel_size = recipe->input_kernel;
    geometry.stride = 1ull;
    geometry.dilation = 1ull;
    if (rc == YVEX_OK) {
        rc = decoder_tensor_open(&run, "latent-input", request->batch,
                                 recipe->input_channels, request->input_length,
                                 &next, err);
        if (rc == YVEX_OK)
            rc = yvex_backend_tensor_write(backend, next, request->input,
                                           next->bytes, err);
        if (rc == YVEX_OK) result->h2d_bytes += next->bytes;
        if (rc == YVEX_OK)
            rc = decoder_convolution(
                &run, YVEX_ALIAS_DECODER_INPUT_WEIGHT,
                YVEX_ALIAS_DECODER_INPUT_WEIGHT, YVEX_ALIAS_DECODER_INPUT_BIAS,
                0, 1, 0ull, 0ull, 0ull, &geometry, next, current, err);
        rc = decoder_tensor_close(&run, &next, rc, err);
    }
    channels = recipe->projection_channels;
    length = request->input_length;
    geometry.input_channels = channels;
    geometry.output_channels = recipe->pre_channels;
    geometry.kernel_size = recipe->pre_kernel;
    geometry.padding = (recipe->pre_kernel - 1ull) / 2ull;
    if (rc == YVEX_OK)
        rc = decoder_tensor_open(&run, "decoder-pre", request->batch,
                                 recipe->pre_channels, length, &next, err);
    if (rc == YVEX_OK)
        rc = decoder_convolution(
            &run, YVEX_ALIAS_DECODER_PRE_WEIGHT, YVEX_ALIAS_DECODER_PRE_GAIN,
            YVEX_ALIAS_DECODER_PRE_BIAS, 1, 1, 0ull, 0ull, 0ull,
            &geometry, current, next, err);
    rc = decoder_tensor_close(&run, &current, rc, err);
    if (rc == YVEX_OK) {
        current = next;
        next = NULL;
        channels = recipe->pre_channels;
    }
    for (stage = 0ull; rc == YVEX_OK && stage < recipe->stage_count; ++stage) {
        unsigned long long next_channels = 0ull, next_length = 0ull;
        rc = decoder_cancel(&run, err);
        if (rc == YVEX_OK)
            rc = decoder_stage(&run, stage, request->batch, channels, length,
                               current, &next, &next_channels, &next_length, err);
        if (rc == YVEX_OK) {
            rc = decoder_tensor_close(&run, &current, rc, err);
            current = next;
            next = NULL;
            channels = next_channels;
            length = next_length;
        }
    }
    if (rc == YVEX_OK) rc = decoder_cancel(&run, err);
    if (rc == YVEX_OK)
        rc = decoder_output(&run, current, channels, length, staged, err);
    if (rc == YVEX_OK) {
        unsigned long long index;
        for (index = 0ull; index < output_values; ++index)
            if (!isfinite(staged[index])) {
                rc = decoder_refuse(err, YVEX_ERR_FORMAT,
                                    "cuda.alias-decoder.output",
                                    "alias decoder output contains a non-finite value");
                break;
            }
    }
    if (rc == YVEX_OK) {
        memcpy(request->output, staged, (size_t)output_values * sizeof(float));
        result->output_length = output_length;
        result->output_values = output_values;
        result->peak_device_bytes = run.peak_device_bytes;
        result->complete = 1;
        yvex_error_clear(err);
    }
    rc = decoder_tensor_close(&run, &next, rc, err);
    rc = decoder_tensor_close(&run, &current, rc, err);
    free(staged);
    if (rc != YVEX_OK) memset(result, 0, sizeof(*result));
    return rc;
}
