#include <math.h>

/* Compute source-compatible weight-normalization factors without rewriting resident weights. */
extern "C" __global__ void yvex_conv_weight_scale_f32(
    const float *weight, const float *gain, float *scale,
    unsigned long long rows, unsigned long long row_width)
{
    unsigned long long row =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    double squared = 0.0;
    unsigned long long index;
    if (!weight || !gain || !scale || row >= rows || !row_width) return;
    for (index = 0ull; index < row_width; ++index) {
        double value = weight[row * row_width + index];
        squared += value * value;
    }
    scale[row] = gain[row] / sqrtf((float)squared);
}

/* One thread owns one output so the source accumulation order remains deterministic. */
extern "C" __global__ void yvex_conv1d_f32(
    const float *input, const float *weight, const float *bias,
    const float *scale, float *output, unsigned long long batch,
    unsigned long long input_channels, unsigned long long output_channels,
    unsigned long long input_length, unsigned long long output_length,
    unsigned long long kernel_size, unsigned long long stride,
    unsigned long long dilation, unsigned long long padding)
{
    unsigned long long task =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long tasks = batch * output_channels * output_length;
    unsigned long long output_position, output_channel, batch_index;
    float sum;
    if (!input || !weight || !output || task >= tasks) return;
    output_position = task % output_length;
    output_channel = (task / output_length) % output_channels;
    batch_index = task / (output_length * output_channels);
    sum = bias ? bias[output_channel] : 0.0f;
    {
        unsigned long long input_channel;
        unsigned long long weight_base =
            output_channel * input_channels * kernel_size;
        float factor = scale ? scale[output_channel] : 1.0f;
        for (input_channel = 0ull; input_channel < input_channels; ++input_channel) {
            unsigned long long kernel;
            for (kernel = 0ull; kernel < kernel_size; ++kernel) {
                unsigned long long projected = output_position * stride + kernel * dilation;
                unsigned long long input_position;
                if (projected < padding) continue;
                input_position = projected - padding;
                if (input_position >= input_length) continue;
                sum += input[(batch_index * input_channels + input_channel) * input_length +
                             input_position] *
                       weight[weight_base + input_channel * kernel_size + kernel] * factor;
            }
        }
    }
    output[task] = sum;
}

/* Solve contributing coordinates directly while retaining source accumulation order. */
extern "C" __global__ void yvex_conv1d_transposed_f32(
    const float *input, const float *weight, const float *bias,
    const float *scale, float *output, unsigned long long batch,
    unsigned long long input_channels, unsigned long long output_channels,
    unsigned long long input_length, unsigned long long output_length,
    unsigned long long kernel_size, unsigned long long stride,
    unsigned long long dilation, unsigned long long padding)
{
    unsigned long long task =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long tasks = batch * output_channels * output_length;
    unsigned long long output_position, output_channel, batch_index;
    unsigned long long target, maximum_kernel_offset, first = 0ull, last;
    float sum;
    if (!input || !weight || !output || task >= tasks || !input_length ||
        !kernel_size || !stride || !dilation)
        return;
    output_position = task % output_length;
    output_channel = (task / output_length) % output_channels;
    batch_index = task / (output_length * output_channels);
    target = output_position + padding;
    maximum_kernel_offset = (kernel_size - 1ull) * dilation;
    if (target > maximum_kernel_offset) {
        unsigned long long lower = target - maximum_kernel_offset;
        first = lower / stride + (lower % stride != 0ull);
    }
    last = target / stride;
    if (last >= input_length) last = input_length - 1ull;
    sum = bias ? bias[output_channel] : 0.0f;
    {
        unsigned long long input_channel;
        for (input_channel = 0ull; input_channel < input_channels; ++input_channel) {
            unsigned long long input_position;
            unsigned long long weight_base =
                input_channel * output_channels * kernel_size;
            float factor = scale ? scale[input_channel] : 1.0f;
            if (first > last) continue;
            for (input_position = first; input_position <= last; ++input_position) {
                unsigned long long delta = target - input_position * stride;
                unsigned long long kernel;
                float value;
                if (delta % dilation != 0ull) continue;
                kernel = delta / dilation;
                if (kernel >= kernel_size) continue;
                value = input[(batch_index * input_channels + input_channel) * input_length +
                              input_position] * factor;
                sum += value * weight[weight_base + output_channel * kernel_size + kernel];
            }
        }
    }
    output[task] = sum;
}

static __device__ unsigned long long reflect_coordinate(
    long long coordinate, unsigned long long extent)
{
    if (extent <= 1ull) return 0ull;
    while (coordinate < 0ll || (unsigned long long)coordinate >= extent) {
        if (coordinate < 0ll) coordinate = -coordinate;
        if ((unsigned long long)coordinate >= extent)
            coordinate = (long long)(extent * 2ull - 2ull) - coordinate;
    }
    return (unsigned long long)coordinate;
}

/* One thread owns one spatial output and retains channel/kernel accumulation order. */
extern "C" __global__ void yvex_conv2d_f32(
    const float *input, const float *weight, const float *bias, float *output,
    unsigned long long batch, unsigned long long input_channels,
    unsigned long long output_channels, unsigned long long input_height,
    unsigned long long input_width, unsigned long long output_height,
    unsigned long long output_width, unsigned long long kernel_height,
    unsigned long long kernel_width, unsigned long long stride_height,
    unsigned long long stride_width, unsigned long long padding_top,
    unsigned long long padding_left, unsigned long long temporal_extent,
    unsigned long long temporal_index, int reflect_padding)
{
    unsigned long long task =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long tasks = batch * output_channels * output_height * output_width;
    unsigned long long x, y, output_channel, batch_index, input_channel, ky, kx;
    float sum;
    if (!input || !weight || !output || task >= tasks) return;
    x = task % output_width;
    y = (task / output_width) % output_height;
    output_channel = (task / (output_width * output_height)) % output_channels;
    batch_index = task / (output_width * output_height * output_channels);
    sum = bias ? bias[output_channel] : 0.0f;
    for (input_channel = 0ull; input_channel < input_channels; ++input_channel) {
        for (ky = 0ull; ky < kernel_height; ++ky) {
            long long source_y = (long long)(y * stride_height + ky) -
                                 (long long)padding_top;
            if (!reflect_padding &&
                (source_y < 0ll || (unsigned long long)source_y >= input_height))
                continue;
            if (reflect_padding)
                source_y = (long long)reflect_coordinate(source_y, input_height);
            for (kx = 0ull; kx < kernel_width; ++kx) {
                long long source_x = (long long)(x * stride_width + kx) -
                                     (long long)padding_left;
                unsigned long long source, kernel;
                if (!reflect_padding &&
                    (source_x < 0ll || (unsigned long long)source_x >= input_width))
                    continue;
                if (reflect_padding)
                    source_x = (long long)reflect_coordinate(source_x, input_width);
                source = ((batch_index * input_channels + input_channel) * input_height +
                          (unsigned long long)source_y) * input_width +
                         (unsigned long long)source_x;
                kernel = (((output_channel * input_channels + input_channel) *
                           temporal_extent + temporal_index) * kernel_height + ky) *
                         kernel_width + kx;
                sum += input[source] * weight[kernel];
            }
        }
    }
    output[task] = sum;
}

/* Image-keyframe normalization is isolated per frame, matching source GroupNorm3D. */
extern "C" __global__ void yvex_group_norm_silu_f32(
    const float *input, const float *weight, const float *bias, float *output,
    unsigned long long batch, unsigned long long channels,
    unsigned long long height, unsigned long long width,
    unsigned long long groups, float epsilon)
{
    unsigned long long task =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long tasks = batch * groups;
    unsigned long long group_channels, channel_start, group_values, index;
    float mean = 0.0f, variance = 0.0f;
    if (!input || !weight || !bias || !output || task >= tasks || !groups ||
        channels % groups) return;
    group_channels = channels / groups;
    channel_start = (task % groups) * group_channels;
    group_values = group_channels * height * width;
    for (index = 0ull; index < group_values; ++index) {
        unsigned long long channel = channel_start + index / (height * width);
        unsigned long long spatial = index % (height * width);
        mean += input[((task / groups) * channels + channel) * height * width + spatial];
    }
    mean /= (float)group_values;
    for (index = 0ull; index < group_values; ++index) {
        unsigned long long channel = channel_start + index / (height * width);
        unsigned long long spatial = index % (height * width);
        float delta = input[((task / groups) * channels + channel) * height * width + spatial] - mean;
        variance += delta * delta;
    }
    variance = rsqrtf(variance / (float)group_values + epsilon);
    for (index = 0ull; index < group_values; ++index) {
        unsigned long long channel = channel_start + index / (height * width);
        unsigned long long spatial = index % (height * width);
        unsigned long long offset =
            ((task / groups) * channels + channel) * height * width + spatial;
        float value = (input[offset] - mean) * variance;
        value = value * weight[channel] + bias[channel];
        output[offset] = value >= 0.0f ? value / (1.0f + expf(-value))
                                      : value * expf(value) / (1.0f + expf(value));
    }
}

/* Upsampling preserves the reflected-edge source policy and padded-position sum order. */
extern "C" __global__ void yvex_alias_snake_up_f32(
    const float *input, const float *alpha_log, const float *beta_log,
    const float *filter, float *scratch, unsigned long long groups,
    unsigned long long channels, unsigned long long length)
{
    unsigned long long task =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long doubled = length * 2ull;
    unsigned long long position, group, raw, first, last, padded;
    float alpha, beta, value = 0.0f;
    if (!input || !alpha_log || !beta_log || !filter || !scratch ||
        task >= groups * doubled) return;
    position = task % doubled;
    group = task / doubled;
    raw = position + 15ull;
    first = raw > 11ull ? (raw - 11ull + 1ull) / 2ull : 0ull;
    last = raw / 2ull;
    if (last >= length + 10ull) last = length + 9ull;
    for (padded = first; padded <= last; ++padded) {
        unsigned long long kernel = raw - padded * 2ull;
        unsigned long long source = padded < 5ull ? 0ull : padded - 5ull;
        if (source >= length) source = length - 1ull;
        value += input[group * length + source] * filter[kernel];
    }
    value *= 2.0f;
    alpha = expf(alpha_log[group % channels]);
    beta = expf(beta_log[group % channels]);
    value += sinf(alpha * value) * sinf(alpha * value) / (beta + 1.0e-9f);
    scratch[task] = value;
}

extern "C" __global__ void yvex_alias_snake_down_f32(
    const float *scratch, const float *filter, float *output,
    unsigned long long groups, unsigned long long length)
{
    unsigned long long task =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long position, group, doubled, kernel;
    float value = 0.0f;
    if (!scratch || !filter || !output || task >= groups * length) return;
    position = task % length;
    group = task / length;
    doubled = length * 2ull;
    for (kernel = 0ull; kernel < 12ull; ++kernel) {
        unsigned long long padded = position * 2ull + kernel;
        unsigned long long source = padded < 5ull ? 0ull : padded - 5ull;
        if (source >= doubled) source = doubled - 1ull;
        value += scratch[group * doubled + source] * filter[kernel];
    }
    output[task] = value;
}

extern "C" __global__ void yvex_vector_update_f32(
    float *destination, const float *source, unsigned long long count,
    float destination_scale, float source_scale)
{
    unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (!destination || !source || index >= count) return;
    destination[index] = destination[index] * destination_scale +
                         source[index] * source_scale;
}

extern "C" __global__ void yvex_clamp_f32(
    float *values, unsigned long long count, float lower, float upper)
{
    unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (!values || index >= count) return;
    if (values[index] < lower) values[index] = lower;
    if (values[index] > upper) values[index] = upper;
}
