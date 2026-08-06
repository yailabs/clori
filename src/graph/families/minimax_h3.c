/*
 * Bind MiniMax-H3 component execution recipes to exact physical inputs.
 *
 * Component metadata remains insufficient until the complete file identity and independently
 * derived payload identity agree with this family-owned physical boundary. Numerical execution
 * consumes this admission later; it cannot infer source or model identity from tensor names.
 */
#include <yvex/internal/artifact.h>
#include <yvex/internal/families/minimax_h3.h>

#include "src/graph/private.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    float *data;
    unsigned long long count;
} audio_buffer;

typedef struct {
    yvex_materialization_session *session;
    const yvex_minimax_h3_audio_decode_options *options;
    yvex_minimax_h3_audio_decode_result *result;
    yvex_minimax_h3_audio_execution_failure *failure;
    yvex_error *err;
    unsigned long long live_workspace_bytes;
} audio_execution;

static int audio_execution_refuse(audio_execution *execution,
                                  yvex_minimax_h3_audio_execution_code code,
                                  const char *tensor_name, unsigned long long expected,
                                  unsigned long long actual, yvex_status status,
                                  const char *reason)
{
    if (execution && execution->failure) {
        memset(execution->failure, 0, sizeof(*execution->failure));
        execution->failure->code = code;
        execution->failure->expected = expected;
        execution->failure->actual = actual;
        execution->failure->reason = reason;
        if (tensor_name)
            yvex_core_text_copy(execution->failure->tensor_name,
                                sizeof(execution->failure->tensor_name), tensor_name);
    }
    yvex_error_set(execution ? execution->err : NULL, status,
                   "graph.minimax_h3.audio_vae.execute", reason);
    return status;
}

static int audio_cancel_check(audio_execution *execution)
{
    if (execution->options->cancelled &&
        execution->options->cancelled(execution->options->cancellation_context))
        return audio_execution_refuse(execution, YVEX_MINIMAX_H3_AUDIO_EXECUTION_CANCELLED,
                                      NULL, 0ull, 1ull, YVEX_ERR_CANCELLED,
                                      "Audio VAE execution was cancelled at a layer boundary");
    return YVEX_OK;
}

static int audio_buffer_open(audio_execution *execution, audio_buffer *buffer,
                             unsigned long long count)
{
    unsigned long long bytes;
    unsigned long long next_live;

    memset(buffer, 0, sizeof(*buffer));
    if (!count || !yvex_core_u64_mul(count, sizeof(float), &bytes) ||
        bytes > (unsigned long long)SIZE_MAX ||
        !yvex_core_u64_add(execution->live_workspace_bytes, bytes, &next_live))
        return audio_execution_refuse(execution, YVEX_MINIMAX_H3_AUDIO_EXECUTION_BUDGET,
                                      NULL, 1ull, count, YVEX_ERR_BOUNDS,
                                      "Audio VAE workspace extent overflowed");
    if (next_live > execution->options->max_workspace_bytes)
        return audio_execution_refuse(execution, YVEX_MINIMAX_H3_AUDIO_EXECUTION_BUDGET,
                                      NULL, execution->options->max_workspace_bytes,
                                      next_live, YVEX_ERR_BOUNDS,
                                      "Audio VAE workspace budget was exceeded");
    buffer->data = (float *)malloc((size_t)bytes);
    if (!buffer->data)
        return audio_execution_refuse(execution, YVEX_MINIMAX_H3_AUDIO_EXECUTION_BUDGET,
                                      NULL, bytes, 0ull, YVEX_ERR_NOMEM,
                                      "Audio VAE workspace allocation failed");
    buffer->count = count;
    execution->live_workspace_bytes = next_live;
    if (next_live > execution->result->peak_workspace_bytes)
        execution->result->peak_workspace_bytes = next_live;
    return YVEX_OK;
}

static void audio_buffer_close(audio_execution *execution, audio_buffer *buffer)
{
    unsigned long long bytes = buffer->count * sizeof(float);

    if (bytes <= execution->live_workspace_bytes)
        execution->live_workspace_bytes -= bytes;
    free(buffer->data);
    memset(buffer, 0, sizeof(*buffer));
}

static const yvex_materialized_tensor_binding *
audio_binding_find(const yvex_materialization_session *session, const char *name)
{
    unsigned long long index;

    for (index = 0ull;; ++index) {
        const yvex_materialized_tensor_binding *binding =
            yvex_materialization_session_tensor_at(session, index);
        if (!binding) return NULL;
        if (strcmp(binding->name, name) == 0) return binding;
    }
}

static int audio_tensor_load(audio_execution *execution, const char *name,
                             unsigned int rank, const unsigned long long *dims,
                             audio_buffer *buffer)
{
    const yvex_materialized_tensor_binding *binding =
        audio_binding_find(execution->session, name);
    yvex_materialization_failure materialization_failure;
    unsigned long long count = 1ull;
    unsigned long long expected_bytes = 0ull;
    unsigned int dimension;
    int rc;

    if (!binding)
        return audio_execution_refuse(execution,
                                      YVEX_MINIMAX_H3_AUDIO_EXECUTION_MISSING_TENSOR,
                                      name, 1ull, 0ull, YVEX_ERR_FORMAT,
                                      "Audio VAE execution tensor is missing");
    if (binding->qtype != YVEX_GGUF_QTYPE_F32 || binding->rank != rank)
        return audio_execution_refuse(execution,
                                      YVEX_MINIMAX_H3_AUDIO_EXECUTION_TENSOR_CONTRACT,
                                      name, rank, binding->rank, YVEX_ERR_FORMAT,
                                      "Audio VAE execution tensor rank or dtype differs");
    for (dimension = 0u; dimension < rank; ++dimension) {
        if (binding->dims[dimension] != dims[dimension] ||
            !yvex_core_u64_mul(count, dims[dimension], &count))
            return audio_execution_refuse(execution,
                                          YVEX_MINIMAX_H3_AUDIO_EXECUTION_TENSOR_CONTRACT,
                                          name, dims[dimension], binding->dims[dimension],
                                          YVEX_ERR_FORMAT,
                                          "Audio VAE execution tensor shape differs");
    }
    if (!yvex_core_u64_mul(count, sizeof(float), &expected_bytes) ||
        binding->encoded_bytes != expected_bytes)
        return audio_execution_refuse(execution,
                                      YVEX_MINIMAX_H3_AUDIO_EXECUTION_TENSOR_CONTRACT,
                                      name, expected_bytes, binding->encoded_bytes,
                                      YVEX_ERR_FORMAT,
                                      "Audio VAE execution tensor byte extent differs");
    rc = audio_buffer_open(execution, buffer, count);
    if (rc != YVEX_OK) return rc;
    rc = yvex_materialization_session_read(
        execution->session, binding, 0ull, buffer->data,
        (size_t)binding->encoded_bytes, &materialization_failure, execution->err);
    if (rc != YVEX_OK) {
        audio_buffer_close(execution, buffer);
        if (execution->failure) {
            memset(execution->failure, 0, sizeof(*execution->failure));
            execution->failure->code = YVEX_MINIMAX_H3_AUDIO_EXECUTION_MATERIALIZATION;
            execution->failure->expected = binding->encoded_bytes;
            execution->failure->actual = materialization_failure.actual;
            execution->failure->reason = materialization_failure.reason;
            yvex_core_text_copy(execution->failure->tensor_name,
                                sizeof(execution->failure->tensor_name), name);
        }
        return rc;
    }
    execution->result->tensor_reads++;
    execution->result->payload_bytes_read += binding->encoded_bytes;
    return YVEX_OK;
}

static int audio_name_build(audio_execution *execution, char *output, size_t capacity,
                            const char *prefix, const char *suffix)
{
    int written = snprintf(output, capacity, "%s%s", prefix, suffix);

    if (written < 0 || (size_t)written >= capacity)
        return audio_execution_refuse(execution,
                                      YVEX_MINIMAX_H3_AUDIO_EXECUTION_TENSOR_CONTRACT,
                                      prefix, capacity, written < 0 ? 0ull : (unsigned long long)written,
                                      YVEX_ERR_BOUNDS,
                                      "Audio VAE tensor name exceeded its bounded identity field");
    return YVEX_OK;
}

static int audio_conv_execute(audio_execution *execution, const char *prefix,
                              const yvex_graph_conv1d_geometry *geometry,
                              const float *input, unsigned long long input_count,
                              float *output, unsigned long long output_count,
                              int weight_normalized, int has_bias)
{
    audio_buffer weight = {0};
    audio_buffer gain = {0};
    audio_buffer bias = {0};
    unsigned long long weight_dims[3];
    unsigned long long gain_dims[3];
    unsigned long long bias_dims[1];
    char name[256];
    int rc = audio_cancel_check(execution);

    if (rc != YVEX_OK) return rc;
    weight_dims[0] = geometry->transposed ? geometry->input_channels :
                                              geometry->output_channels;
    weight_dims[1] = geometry->transposed ? geometry->output_channels :
                                              geometry->input_channels;
    weight_dims[2] = geometry->kernel_size;
    rc = audio_name_build(execution, name, sizeof(name), prefix,
                          weight_normalized ? ".weight_v" : ".weight");
    if (rc == YVEX_OK)
        rc = audio_tensor_load(execution, name, 3u, weight_dims, &weight);
    if (weight_normalized && rc == YVEX_OK) {
        gain_dims[0] = geometry->transposed ? geometry->input_channels :
                                             geometry->output_channels;
        gain_dims[1] = 1ull;
        gain_dims[2] = 1ull;
        rc = audio_name_build(execution, name, sizeof(name), prefix, ".weight_g");
        if (rc == YVEX_OK)
            rc = audio_tensor_load(execution, name, 3u, gain_dims, &gain);
    }
    if (has_bias && rc == YVEX_OK) {
        bias_dims[0] = geometry->output_channels;
        rc = audio_name_build(execution, name, sizeof(name), prefix, ".bias");
        if (rc == YVEX_OK)
            rc = audio_tensor_load(execution, name, 1u, bias_dims, &bias);
    }
    if (rc == YVEX_OK)
        rc = yvex_graph_conv1d_f32(
            geometry, input, input_count, weight.data, weight.count,
            bias.data, bias.count, gain.data, gain.count, output, output_count,
            execution->err);
    if (rc != YVEX_OK && execution->failure &&
        execution->failure->code == YVEX_MINIMAX_H3_AUDIO_EXECUTION_NONE) {
        execution->failure->code = YVEX_MINIMAX_H3_AUDIO_EXECUTION_NUMERIC;
        execution->failure->reason = yvex_error_message(execution->err);
        yvex_core_text_copy(execution->failure->tensor_name,
                            sizeof(execution->failure->tensor_name), prefix);
    }
    audio_buffer_close(execution, &bias);
    audio_buffer_close(execution, &gain);
    audio_buffer_close(execution, &weight);
    return rc;
}

static int audio_activation_execute(audio_execution *execution, const char *prefix,
                                    const float *input, unsigned long long batch,
                                    unsigned long long channels, unsigned long long length,
                                    float *output, float *scratch,
                                    unsigned long long scratch_count)
{
    audio_buffer alpha = {0};
    audio_buffer beta = {0};
    audio_buffer up = {0};
    audio_buffer down = {0};
    unsigned long long channel_dims[1] = {channels};
    unsigned long long filter_dims[3] = {1ull, 1ull, 12ull};
    char name[256];
    int rc = audio_cancel_check(execution);

    if (rc != YVEX_OK) return rc;
    rc = audio_name_build(execution, name, sizeof(name), prefix, ".act.alpha");
    if (rc == YVEX_OK)
        rc = audio_tensor_load(execution, name, 1u, channel_dims, &alpha);
    if (rc == YVEX_OK)
        rc = audio_name_build(execution, name, sizeof(name), prefix, ".act.beta");
    if (rc == YVEX_OK)
        rc = audio_tensor_load(execution, name, 1u, channel_dims, &beta);
    if (rc == YVEX_OK)
        rc = audio_name_build(execution, name, sizeof(name), prefix, ".upsample.filter");
    if (rc == YVEX_OK)
        rc = audio_tensor_load(execution, name, 3u, filter_dims, &up);
    if (rc == YVEX_OK)
        rc = audio_name_build(execution, name, sizeof(name), prefix,
                              ".downsample.lowpass.filter");
    if (rc == YVEX_OK)
        rc = audio_tensor_load(execution, name, 3u, filter_dims, &down);
    if (rc == YVEX_OK)
        rc = yvex_graph_alias_snake_f32(
            input, batch, channels, length, alpha.data, beta.data, up.data,
            down.data, output, scratch, scratch_count, execution->err);
    if (rc != YVEX_OK && execution->failure &&
        execution->failure->code == YVEX_MINIMAX_H3_AUDIO_EXECUTION_NONE) {
        execution->failure->code = YVEX_MINIMAX_H3_AUDIO_EXECUTION_NUMERIC;
        execution->failure->reason = yvex_error_message(execution->err);
        yvex_core_text_copy(execution->failure->tensor_name,
                            sizeof(execution->failure->tensor_name), prefix);
    }
    audio_buffer_close(execution, &down);
    audio_buffer_close(execution, &up);
    audio_buffer_close(execution, &beta);
    audio_buffer_close(execution, &alpha);
    return rc;
}

typedef enum {
    AUDIO_NAME_RESBLOCK_ACTIVATION,
    AUDIO_NAME_RESBLOCK_CONV1,
    AUDIO_NAME_RESBLOCK_CONV2,
    AUDIO_NAME_UPSAMPLE
} audio_indexed_name_kind;

static int audio_indexed_name(audio_execution *execution, char *output, size_t capacity,
                              audio_indexed_name_kind kind, unsigned long long first,
                              unsigned long long second)
{
    int written;

    switch (kind) {
    case AUDIO_NAME_RESBLOCK_ACTIVATION:
        written = snprintf(output, capacity, "decoder.resblocks.%llu.activations.%llu",
                           first, second);
        break;
    case AUDIO_NAME_RESBLOCK_CONV1:
        written = snprintf(output, capacity, "decoder.resblocks.%llu.convs1.%llu",
                           first, second);
        break;
    case AUDIO_NAME_RESBLOCK_CONV2:
        written = snprintf(output, capacity, "decoder.resblocks.%llu.convs2.%llu",
                           first, second);
        break;
    case AUDIO_NAME_UPSAMPLE:
        written = snprintf(output, capacity, "decoder.ups.%llu.%llu", first, second);
        break;
    default:
        written = -1;
        break;
    }

    if (written < 0 || (size_t)written >= capacity)
        return audio_execution_refuse(execution,
                                      YVEX_MINIMAX_H3_AUDIO_EXECUTION_TENSOR_CONTRACT,
                                      NULL, capacity,
                                      written < 0 ? 0ull : (unsigned long long)written,
                                      YVEX_ERR_BOUNDS,
                                      "Audio VAE indexed tensor name exceeded its bound");
    return YVEX_OK;
}

static int audio_resblock_execute(audio_execution *execution,
                                  unsigned long long block_index,
                                  const float *input, unsigned long long batch,
                                  unsigned long long channels,
                                  unsigned long long length, float *output)
{
    static const unsigned long long kernels[] = {3ull, 7ull, 11ull};
    static const unsigned long long dilations[] = {1ull, 3ull, 5ull};
    audio_buffer activation = {0};
    audio_buffer convolution = {0};
    audio_buffer scratch = {0};
    unsigned long long values;
    unsigned long long layer;
    char prefix[256];
    int rc;

    if (!yvex_core_u64_mul(batch, channels, &values) ||
        !yvex_core_u64_mul(values, length, &values))
        return audio_execution_refuse(execution,
                                      YVEX_MINIMAX_H3_AUDIO_EXECUTION_BUDGET,
                                      NULL, 1ull, 0ull, YVEX_ERR_BOUNDS,
                                      "Audio VAE residual extent overflowed");
    rc = audio_buffer_open(execution, &activation, values);
    if (rc == YVEX_OK) rc = audio_buffer_open(execution, &convolution, values);
    if (rc == YVEX_OK) rc = audio_buffer_open(execution, &scratch, length * 2ull);
    if (rc != YVEX_OK) goto cleanup;
    memcpy(output, input, (size_t)(values * sizeof(float)));
    for (layer = 0ull; layer < 3ull && rc == YVEX_OK; ++layer) {
        yvex_graph_conv1d_geometry geometry = {
            batch, channels, channels, length, kernels[block_index % 3ull],
            1ull, dilations[layer],
            ((kernels[block_index % 3ull] - 1ull) * dilations[layer]) / 2ull,
            0ull, 0};
        unsigned long long index;

        rc = audio_indexed_name(execution, prefix, sizeof(prefix),
                                AUDIO_NAME_RESBLOCK_ACTIVATION,
                                block_index, layer * 2ull);
        if (rc == YVEX_OK)
            rc = audio_activation_execute(
                execution, prefix, output, batch, channels, length,
                activation.data, scratch.data, scratch.count);
        if (rc == YVEX_OK)
            rc = audio_indexed_name(execution, prefix, sizeof(prefix),
                                    AUDIO_NAME_RESBLOCK_CONV1,
                                    block_index, layer);
        if (rc == YVEX_OK)
            rc = audio_conv_execute(execution, prefix, &geometry,
                                    activation.data, activation.count,
                                    convolution.data, convolution.count, 1, 1);
        geometry.dilation = 1ull;
        geometry.padding = (geometry.kernel_size - 1ull) / 2ull;
        if (rc == YVEX_OK)
            rc = audio_indexed_name(execution, prefix, sizeof(prefix),
                                    AUDIO_NAME_RESBLOCK_ACTIVATION,
                                    block_index, layer * 2ull + 1ull);
        if (rc == YVEX_OK)
            rc = audio_activation_execute(
                execution, prefix, convolution.data, batch, channels, length,
                activation.data, scratch.data, scratch.count);
        if (rc == YVEX_OK)
            rc = audio_indexed_name(execution, prefix, sizeof(prefix),
                                    AUDIO_NAME_RESBLOCK_CONV2,
                                    block_index, layer);
        if (rc == YVEX_OK)
            rc = audio_conv_execute(execution, prefix, &geometry,
                                    activation.data, activation.count,
                                    convolution.data, convolution.count, 1, 1);
        if (rc == YVEX_OK)
            for (index = 0ull; index < values; ++index)
                output[index] += convolution.data[index];
    }
cleanup:
    audio_buffer_close(execution, &scratch);
    audio_buffer_close(execution, &convolution);
    audio_buffer_close(execution, &activation);
    return rc;
}

static int audio_stage_execute(audio_execution *execution, unsigned long long stage,
                               const audio_buffer *input, unsigned long long batch,
                               unsigned long long input_channels,
                               unsigned long long input_length,
                               audio_buffer *output, unsigned long long *output_channels,
                               unsigned long long *output_length)
{
    static const unsigned long long rates[] = {5ull, 5ull, 2ull, 2ull, 2ull, 2ull, 2ull};
    static const unsigned long long kernels[] = {9ull, 9ull, 4ull, 4ull, 4ull, 4ull, 4ull};
    yvex_graph_conv1d_geometry geometry;
    audio_buffer upsampled = {0};
    audio_buffer sum = {0};
    audio_buffer block = {0};
    unsigned long long values;
    unsigned long long block_index;
    char prefix[256];
    int rc;

    memset(output, 0, sizeof(*output));
    if (stage >= sizeof(rates) / sizeof(rates[0]) || input_channels < 2ull)
        return audio_execution_refuse(execution,
                                      YVEX_MINIMAX_H3_AUDIO_EXECUTION_INVALID_ARGUMENT,
                                      NULL, 7ull, stage, YVEX_ERR_INVALID_ARG,
                                      "Audio VAE decoder stage is outside its exact recipe");
    memset(&geometry, 0, sizeof(geometry));
    geometry.batch = batch;
    geometry.input_channels = input_channels;
    geometry.output_channels = input_channels / 2ull;
    geometry.input_length = input_length;
    geometry.kernel_size = kernels[stage];
    geometry.stride = rates[stage];
    geometry.dilation = 1ull;
    geometry.padding = (kernels[stage] - rates[stage]) / 2ull;
    geometry.transposed = 1;
    rc = yvex_graph_conv1d_output_length(&geometry, output_length, execution->err);
    if (rc != YVEX_OK) return rc;
    *output_channels = geometry.output_channels;
    if (!yvex_core_u64_mul(batch, *output_channels, &values) ||
        !yvex_core_u64_mul(values, *output_length, &values))
        return audio_execution_refuse(execution,
                                      YVEX_MINIMAX_H3_AUDIO_EXECUTION_BUDGET,
                                      NULL, 1ull, 0ull, YVEX_ERR_BOUNDS,
                                      "Audio VAE decoder stage extent overflowed");
    rc = audio_buffer_open(execution, &upsampled, values);
    if (rc == YVEX_OK) rc = audio_buffer_open(execution, &sum, values);
    if (rc == YVEX_OK) rc = audio_buffer_open(execution, &block, values);
    if (rc != YVEX_OK) goto cleanup;
    rc = audio_indexed_name(execution, prefix, sizeof(prefix),
                            AUDIO_NAME_UPSAMPLE, stage, 0ull);
    if (rc == YVEX_OK)
        rc = audio_conv_execute(execution, prefix, &geometry, input->data,
                                input->count, upsampled.data, upsampled.count, 1, 1);
    if (rc == YVEX_OK) memset(sum.data, 0, (size_t)(sum.count * sizeof(float)));
    for (block_index = 0ull; block_index < 3ull && rc == YVEX_OK; ++block_index) {
        unsigned long long index;
        rc = audio_resblock_execute(execution, stage * 3ull + block_index,
                                    upsampled.data, batch, *output_channels,
                                    *output_length, block.data);
        if (rc == YVEX_OK)
            for (index = 0ull; index < values; ++index)
                sum.data[index] += block.data[index];
    }
    if (rc == YVEX_OK) {
        unsigned long long index;
        for (index = 0ull; index < values; ++index) sum.data[index] /= 3.0f;
        *output = sum;
        memset(&sum, 0, sizeof(sum));
    }
cleanup:
    audio_buffer_close(execution, &block);
    audio_buffer_close(execution, &sum);
    audio_buffer_close(execution, &upsampled);
    return rc;
}

static int audio_execution_identity(const yvex_materialization_summary *summary,
                                    const yvex_minimax_h3_audio_decode_options *options,
                                    yvex_minimax_h3_audio_decode_result *result)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;

    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.minimax-h3.audio-vae.cpu.v1") ||
        !yvex_sha256_update_text(&hash, summary->artifact_identity) ||
        !yvex_sha256_update_u64_be(&hash, options->batch) ||
        !yvex_sha256_update_u64_be(&hash, options->latent_channels) ||
        !yvex_sha256_update_u64_be(&hash, options->latent_steps))
        return 0;
    for (index = 0ull; index < options->batch * options->latent_channels *
                                     options->latent_steps; ++index) {
        uint32_t bits;
        unsigned char bytes[4];
        memcpy(&bits, &options->latent[index], sizeof(bits));
        bytes[0] = (unsigned char)(bits >> 24u);
        bytes[1] = (unsigned char)(bits >> 16u);
        bytes[2] = (unsigned char)(bits >> 8u);
        bytes[3] = (unsigned char)bits;
        if (!yvex_sha256_update(&hash, bytes, sizeof(bytes))) return 0;
    }
    for (index = 0ull; index < result->output_values; ++index) {
        uint32_t bits;
        unsigned char bytes[4];
        memcpy(&bits, &options->output[index], sizeof(bits));
        bytes[0] = (unsigned char)(bits >> 24u);
        bytes[1] = (unsigned char)(bits >> 16u);
        bytes[2] = (unsigned char)(bits >> 8u);
        bytes[3] = (unsigned char)bits;
        if (!yvex_sha256_update(&hash, bytes, sizeof(bytes))) return 0;
    }
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, result->execution_identity);
    return 1;
}

static int audio_decode_validate(audio_execution *execution)
{
    const yvex_materialization_summary *summary =
        yvex_materialization_session_summary(execution->session);
    unsigned long long latent_values = 0ull;
    unsigned long long output_values = 0ull;
    unsigned long long index;

    if (!execution->options || !execution->result ||
        !execution->options->latent || !execution->options->output ||
        !execution->options->batch || !execution->options->latent_steps ||
        execution->options->latent_channels != 32ull ||
        !execution->options->max_workspace_bytes)
        return audio_execution_refuse(execution,
                                      YVEX_MINIMAX_H3_AUDIO_EXECUTION_INVALID_ARGUMENT,
                                      NULL, 32ull,
                                      execution->options ? execution->options->latent_channels : 0ull,
                                      YVEX_ERR_INVALID_ARG,
                                      "Audio VAE decode requires exact latent geometry and bounded output");
    if (!summary || !summary->committed ||
        strcmp(summary->artifact_identity, YVEX_MINIMAX_H3_AUDIO_ARTIFACT_IDENTITY) != 0)
        return audio_execution_refuse(execution,
                                      YVEX_MINIMAX_H3_AUDIO_EXECUTION_LIFECYCLE,
                                      NULL, 1ull,
                                      summary ? (unsigned long long)summary->committed : 0ull,
                                      YVEX_ERR_STATE,
                                      "Audio VAE decode requires the committed exact component artifact");
    if (!yvex_core_u64_mul(execution->options->batch,
                           execution->options->latent_channels, &latent_values) ||
        !yvex_core_u64_mul(latent_values, execution->options->latent_steps,
                           &latent_values) ||
        !yvex_core_u64_mul(execution->options->batch,
                           execution->options->latent_steps, &output_values) ||
        !yvex_core_u64_mul(output_values, 800ull, &output_values) ||
        execution->options->output_capacity < output_values)
        return audio_execution_refuse(execution,
                                      YVEX_MINIMAX_H3_AUDIO_EXECUTION_BUDGET,
                                      NULL, output_values,
                                      execution->options->output_capacity,
                                      YVEX_ERR_BOUNDS,
                                      "Audio VAE output buffer is smaller than the exact 800x ratio");
    for (index = 0ull; index < latent_values; ++index)
        if (!isfinite(execution->options->latent[index]))
            return audio_execution_refuse(
                execution, YVEX_MINIMAX_H3_AUDIO_EXECUTION_NUMERIC,
                NULL, latent_values, index, YVEX_ERR_FORMAT,
                "Audio VAE latent input contains a non-finite value");
    execution->result->batch = execution->options->batch;
    execution->result->samples_per_channel = execution->options->latent_steps * 800ull;
    execution->result->output_values = output_values;
    yvex_core_text_copy(execution->result->artifact_identity,
                        sizeof(execution->result->artifact_identity),
                        summary->artifact_identity);
    return YVEX_OK;
}

static int audio_vae_decode_cpu(yvex_materialization_session *session,
                                const yvex_minimax_h3_audio_decode_options *options,
                                yvex_minimax_h3_audio_decode_result *result,
                                yvex_minimax_h3_audio_execution_failure *failure,
                                yvex_error *err)
{
    audio_execution execution;
    audio_buffer current = {0};
    audio_buffer next = {0};
    audio_buffer activated = {0};
    audio_buffer scratch = {0};
    yvex_graph_conv1d_geometry geometry;
    unsigned long long channels = 2048ull;
    unsigned long long length;
    unsigned long long stage;
    int rc;

    if (result) memset(result, 0, sizeof(*result));
    if (failure) memset(failure, 0, sizeof(*failure));
    memset(&execution, 0, sizeof(execution));
    execution.session = session;
    execution.options = options;
    execution.result = result;
    execution.failure = failure;
    execution.err = err;
    if (!options || !result)
        return audio_execution_refuse(&execution,
                                      YVEX_MINIMAX_H3_AUDIO_EXECUTION_INVALID_ARGUMENT,
                                      NULL, 2ull, 0ull, YVEX_ERR_INVALID_ARG,
                                      "Audio VAE decode requires options and result");
    rc = audio_decode_validate(&execution);
    if (rc != YVEX_OK) return rc;
    length = options->latent_steps;
    rc = audio_buffer_open(&execution, &current, options->batch * channels * length);
    memset(&geometry, 0, sizeof(geometry));
    geometry.batch = options->batch;
    geometry.input_channels = 32ull;
    geometry.output_channels = channels;
    geometry.input_length = length;
    geometry.kernel_size = 1ull;
    geometry.stride = 1ull;
    geometry.dilation = 1ull;
    if (rc == YVEX_OK)
        rc = audio_conv_execute(&execution, "dec_in_proj", &geometry,
                                options->latent,
                                options->batch * 32ull * length,
                                current.data, current.count, 0, 1);
    geometry.input_channels = channels;
    geometry.output_channels = 1024ull;
    geometry.kernel_size = 7ull;
    geometry.padding = 3ull;
    channels = 1024ull;
    if (rc == YVEX_OK)
        rc = audio_buffer_open(&execution, &next, options->batch * channels * length);
    if (rc == YVEX_OK)
        rc = audio_conv_execute(&execution, "decoder.conv_pre", &geometry,
                                current.data, current.count, next.data, next.count, 1, 1);
    audio_buffer_close(&execution, &current);
    current = next;
    memset(&next, 0, sizeof(next));
    for (stage = 0ull; stage < 7ull && rc == YVEX_OK; ++stage) {
        unsigned long long next_channels = 0ull;
        unsigned long long next_length = 0ull;
        rc = audio_stage_execute(&execution, stage, &current, options->batch,
                                 channels, length, &next, &next_channels, &next_length);
        if (rc == YVEX_OK) {
            audio_buffer_close(&execution, &current);
            current = next;
            memset(&next, 0, sizeof(next));
            channels = next_channels;
            length = next_length;
        }
    }
    if (rc == YVEX_OK)
        rc = audio_buffer_open(&execution, &activated, current.count);
    if (rc == YVEX_OK) rc = audio_buffer_open(&execution, &scratch, length * 2ull);
    if (rc == YVEX_OK)
        rc = audio_activation_execute(
            &execution, "decoder.activation_post", current.data, options->batch,
            channels, length, activated.data, scratch.data, scratch.count);
    memset(&geometry, 0, sizeof(geometry));
    geometry.batch = options->batch;
    geometry.input_channels = channels;
    geometry.output_channels = 1ull;
    geometry.input_length = length;
    geometry.kernel_size = 7ull;
    geometry.stride = 1ull;
    geometry.dilation = 1ull;
    geometry.padding = 3ull;
    if (rc == YVEX_OK)
        rc = audio_conv_execute(&execution, "decoder.conv_post", &geometry,
                                activated.data, activated.count, options->output,
                                result->output_values, 1, 0);
    if (rc == YVEX_OK) {
        unsigned long long index;
        const yvex_materialization_summary *summary =
            yvex_materialization_session_summary(session);
        for (index = 0ull; index < result->output_values; ++index) {
            if (options->output[index] < -1.0f) options->output[index] = -1.0f;
            if (options->output[index] > 1.0f) options->output[index] = 1.0f;
        }
        if (!audio_execution_identity(summary, options, result))
            rc = audio_execution_refuse(&execution,
                                         YVEX_MINIMAX_H3_AUDIO_EXECUTION_NUMERIC,
                                         NULL, 1ull, 0ull, YVEX_ERR_STATE,
                                         "Audio VAE execution identity could not be sealed");
        else
            result->complete = 1;
    }
    audio_buffer_close(&execution, &scratch);
    audio_buffer_close(&execution, &activated);
    audio_buffer_close(&execution, &next);
    audio_buffer_close(&execution, &current);
    if (rc == YVEX_OK) yvex_error_clear(err);
    return rc;
}

typedef struct {
    const char *key;
    const char *value;
} audio_metadata_fact;

static const audio_metadata_fact audio_metadata[] = {
    {"general.architecture", "minimax-h3"},
    {"general.name", "audio_vae"},
    {"yvex.logical.target", YVEX_MINIMAX_H3_TARGET_ID},
    {"yvex.logical.component", "audio_vae"},
    {"yvex.source.snapshot.identity", YVEX_MINIMAX_H3_AUDIO_SOURCE_SNAPSHOT_IDENTITY},
    {"yvex.logical.component.identity", YVEX_MINIMAX_H3_AUDIO_COMPONENT_IDENTITY},
    {"yvex.logical.component_manifest.identity",
     YVEX_MINIMAX_H3_AUDIO_COMPONENT_MANIFEST_IDENTITY},
    {"yvex.logical.architecture.identity", YVEX_MINIMAX_H3_AUDIO_ARCHITECTURE_IDENTITY},
    {"yvex.logical.role_map.identity", YVEX_MINIMAX_H3_AUDIO_ROLE_MAP_IDENTITY},
    {"yvex.logical.unresolved_requirements.identity",
     YVEX_MINIMAX_H3_AUDIO_UNRESOLVED_IDENTITY},
    {"yvex.transformation.identity", YVEX_MINIMAX_H3_AUDIO_TRANSFORM_IDENTITY},
    {"yvex.physical.profile.name", YVEX_MINIMAX_H3_AUDIO_PROFILE_NAME},
    {"yvex.physical.profile.identity", YVEX_MINIMAX_H3_AUDIO_PROFILE_IDENTITY},
    {"yvex.physical.payload_plan.identity", YVEX_MINIMAX_H3_AUDIO_PAYLOAD_PLAN_IDENTITY},
    {"yvex.payload.identity", YVEX_MINIMAX_H3_AUDIO_PAYLOAD_IDENTITY},
    {"yvex.evidence.stage", "component-artifact-planned"},
};

static const yvex_complete_artifact_admission audio_catalog = {
    .artifact_class = YVEX_ARTIFACT_CLASS_COMPONENT_YVEX,
    .metadata_count = 17ull,
    .tensor_count = YVEX_MINIMAX_H3_AUDIO_TENSORS,
    .payload_bytes = YVEX_MINIMAX_H3_AUDIO_PAYLOAD_BYTES,
    .file_bytes = YVEX_MINIMAX_H3_AUDIO_FILE_BYTES,
    .source_snapshot_identity = YVEX_MINIMAX_H3_AUDIO_SOURCE_SNAPSHOT_KEY,
    .mapping_identity = YVEX_MINIMAX_H3_AUDIO_MAPPING_IDENTITY,
    .payload_identity = YVEX_MINIMAX_H3_AUDIO_PAYLOAD_IDENTITY,
    .transform_identity = YVEX_MINIMAX_H3_AUDIO_TRANSFORM_IDENTITY,
    .profile_identity = YVEX_MINIMAX_H3_AUDIO_PROFILE_IDENTITY,
    .profile_name = YVEX_MINIMAX_H3_AUDIO_PROFILE_NAME,
    .quant_execution_identity = YVEX_MINIMAX_H3_AUDIO_QUANT_EXECUTION_IDENTITY,
    .payload_plan_identity = YVEX_MINIMAX_H3_AUDIO_PAYLOAD_PLAN_IDENTITY,
    .payload_byte_identity = YVEX_MINIMAX_H3_AUDIO_PAYLOAD_BYTE_IDENTITY,
    .writer_plan_identity = YVEX_MINIMAX_H3_AUDIO_WRITER_PLAN_IDENTITY,
    .artifact_identity = YVEX_MINIMAX_H3_AUDIO_ARTIFACT_IDENTITY,
    .official_reader_revision = YVEX_GGUF_OFFICIAL_READER_REVISION,
    .logical_target = YVEX_MINIMAX_H3_TARGET_ID,
    .logical_component = "audio_vae",
    .logical_component_identity = YVEX_MINIMAX_H3_AUDIO_COMPONENT_IDENTITY,
    .native_reader_accepted = 1,
    .official_reader_accepted = 1,
    .payload_integrity_accepted = 1,
    .materialization_input_ready = 1,
};

static int audio_refuse(yvex_artifact_admission_failure *failure, const char *field,
                        unsigned long long expected, unsigned long long actual,
                        yvex_status status, yvex_error *err, const char *message)
{
    if (failure) {
        memset(failure, 0, sizeof(*failure));
        failure->code = YVEX_ARTIFACT_ADMISSION_IDENTITY_MISMATCH;
        failure->expected = expected;
        failure->actual = actual;
        yvex_core_text_copy(failure->field, sizeof(failure->field), field);
    }
    yvex_error_set(err, status, "graph.minimax_h3.audio_vae", message);
    return status;
}

static int audio_metadata_validate(const yvex_gguf *gguf,
                                   yvex_artifact_admission_failure *failure,
                                   yvex_error *err)
{
    size_t index;

    if (yvex_gguf_metadata_count(gguf) != audio_catalog.metadata_count)
        return audio_refuse(failure, "metadata-count", audio_catalog.metadata_count,
                            yvex_gguf_metadata_count(gguf), YVEX_ERR_FORMAT, err,
                            "Audio VAE artifact metadata coverage differs from the admitted file");
    for (index = 0u; index < sizeof(audio_metadata) / sizeof(audio_metadata[0]); ++index) {
        const yvex_gguf_value *value = yvex_gguf_metadata_find(gguf, audio_metadata[index].key);
        const char *text = NULL;
        unsigned long long length = 0ull;
        size_t expected = strlen(audio_metadata[index].value);

        if (!value || yvex_gguf_value_as_string(value, &text, &length) != YVEX_OK ||
            length != expected || memcmp(text, audio_metadata[index].value, expected) != 0)
            return audio_refuse(failure, audio_metadata[index].key, expected, length,
                                YVEX_ERR_FORMAT, err,
                                "Audio VAE artifact metadata identity differs from its recipe");
    }
    return YVEX_OK;
}

static int audio_tensors_validate(const yvex_tensor_table *tensors,
                                  yvex_artifact_admission_failure *failure,
                                  yvex_error *err)
{
    unsigned long long elements = 0ull;
    unsigned long long payload = 0ull;
    unsigned long long index;

    if (yvex_tensor_table_count(tensors) != audio_catalog.tensor_count)
        return audio_refuse(failure, "tensor-count", audio_catalog.tensor_count,
                            yvex_tensor_table_count(tensors), YVEX_ERR_FORMAT, err,
                            "Audio VAE tensor coverage differs from the admitted component");
    for (index = 0ull; index < yvex_tensor_table_count(tensors); ++index) {
        const yvex_tensor_info *tensor = yvex_tensor_table_at(tensors, index);
        unsigned long long tensor_elements = 1ull;
        unsigned int dimension;

        if (!tensor || tensor->ggml_type != YVEX_GGUF_QTYPE_F32 || !tensor->rank)
            return audio_refuse(failure, "tensor-qtype", YVEX_GGUF_QTYPE_F32,
                                tensor ? tensor->ggml_type : ~0ull, YVEX_ERR_FORMAT, err,
                                "Audio VAE requires the exact source-faithful F32 inventory");
        for (dimension = 0u; dimension < tensor->rank; ++dimension) {
            if (!tensor->dims[dimension] ||
                !yvex_core_u64_mul(tensor_elements, tensor->dims[dimension], &tensor_elements))
                return audio_refuse(failure, "tensor-elements", 1ull, 0ull,
                                    YVEX_ERR_BOUNDS, err,
                                    "Audio VAE tensor element accounting overflowed");
        }
        if (!yvex_core_u64_add(elements, tensor_elements, &elements) ||
            !yvex_core_u64_add(payload, tensor->storage_bytes, &payload))
            return audio_refuse(failure, "tensor-population", 1ull, 0ull,
                                YVEX_ERR_BOUNDS, err,
                                "Audio VAE aggregate tensor accounting overflowed");
    }
    if (elements != YVEX_MINIMAX_H3_AUDIO_ELEMENTS)
        return audio_refuse(failure, "element-count", YVEX_MINIMAX_H3_AUDIO_ELEMENTS,
                            elements, YVEX_ERR_FORMAT, err,
                            "Audio VAE aggregate element count differs from its recipe");
    if (payload != audio_catalog.payload_bytes)
        return audio_refuse(failure, "payload-bytes", audio_catalog.payload_bytes,
                            payload, YVEX_ERR_FORMAT, err,
                            "Audio VAE aggregate payload extent differs from its recipe");
    return YVEX_OK;
}

static int audio_vae_admit(const yvex_artifact *artifact, const yvex_gguf *gguf,
                           const yvex_tensor_table *tensors,
                           yvex_complete_artifact_admission *out,
                           yvex_artifact_admission_failure *failure, yvex_error *err)
{
    int rc;

    if (!artifact || !gguf || !tensors || !out)
        return audio_refuse(failure, "arguments", 4ull, 0ull, YVEX_ERR_INVALID_ARG,
                            err, "Audio VAE admission requires artifact and structural views");
    rc = audio_metadata_validate(gguf, failure, err);
    if (rc == YVEX_OK) rc = audio_tensors_validate(tensors, failure, err);
    if (rc == YVEX_OK)
        rc = yvex_artifact_admit_component(artifact, &audio_catalog, out, failure, err);
    if (rc == YVEX_OK)
        rc = yvex_artifact_admission_identity_verify(artifact, out, NULL, NULL, failure, err);
    return rc;
}

static int audio_vae_execute_artifact_cpu(
    const yvex_artifact *artifact, const yvex_gguf *gguf,
    const yvex_tensor_table *tensors,
    const yvex_minimax_h3_audio_decode_options *options,
    yvex_minimax_h3_audio_decode_result *result,
    yvex_minimax_h3_audio_execution_failure *failure, yvex_error *err)
{
    yvex_complete_artifact_admission admission;
    yvex_artifact_admission_failure admission_failure;
    yvex_materialization_options materialization_options;
    yvex_materialization_failure materialization_failure;
    yvex_materialization_plan *plan = NULL;
    yvex_materialization_session *session = NULL;
    audio_execution execution = {0};
    int admitted = 0;
    int rc;

    if (result) memset(result, 0, sizeof(*result));
    if (failure) memset(failure, 0, sizeof(*failure));
    execution.options = options;
    execution.result = result;
    execution.failure = failure;
    execution.err = err;
    if (!artifact || !gguf || !tensors || !options || !result)
        return audio_execution_refuse(
            &execution, YVEX_MINIMAX_H3_AUDIO_EXECUTION_INVALID_ARGUMENT,
            NULL, 5ull, 0ull, YVEX_ERR_INVALID_ARG,
            "Audio VAE artifact execution requires structural inputs and output state");
    rc = audio_vae_admit(artifact, gguf, tensors, &admission, &admission_failure, err);
    admitted = rc == YVEX_OK;
    yvex_materialization_options_default(&materialization_options);
    materialization_options.max_chunk_bytes = 64ull * 1024ull * 1024ull;
    if (materialization_options.max_chunk_bytes > options->max_workspace_bytes)
        materialization_options.max_chunk_bytes = options->max_workspace_bytes;
    if (rc == YVEX_OK)
        rc = yvex_materialization_plan_build(
            &plan, &admission, artifact, gguf, tensors, NULL,
            &materialization_options, &materialization_failure, err);
    if (rc == YVEX_OK)
        rc = yvex_materialization_session_open(
            &session, plan, artifact, &materialization_options,
            &materialization_failure, err);
    if (rc == YVEX_OK)
        rc = yvex_materialization_session_commit(session, &materialization_failure, err);
    if (rc == YVEX_OK)
        rc = audio_vae_decode_cpu(session, options, result, failure, err);
    if (rc != YVEX_OK && failure && failure->code == YVEX_MINIMAX_H3_AUDIO_EXECUTION_NONE) {
        failure->code = admitted ? YVEX_MINIMAX_H3_AUDIO_EXECUTION_MATERIALIZATION
                                 : YVEX_MINIMAX_H3_AUDIO_EXECUTION_LIFECYCLE;
        failure->reason = yvex_error_message(err);
    }
    yvex_materialization_session_close(session);
    yvex_materialization_plan_close(plan);
    return rc;
}

const yvex_minimax_h3_graph_api *yvex_graph_register_minimax_h3(void)
{
    static const yvex_minimax_h3_graph_api api = {
        audio_vae_admit,
        audio_vae_decode_cpu,
        audio_vae_execute_artifact_cpu,
    };

    return &api;
}
