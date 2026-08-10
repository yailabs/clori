/* Bind MiniMax-H3 execution recipes only after exact component payload identity is admitted. */
#include <yvex/internal/artifact.h>
#include <yvex/internal/families/minimax_h3.h>
#include <yvex/internal/latent.h>
#include <yvex/internal/runtime.h>
#include <yvex/internal/transformer.h>
#include "src/graph/private.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define VIDEO_COMPONENT_IDENTITY "c45d914061f4a8d71e84d70cf79f286793919bdc040f48e94b4ec83c2ee8a0e7"
#define VIDEO_SOURCE_SNAPSHOT_IDENTITY "897ceaff08708f431132c6643bc8f1041ace8c0444a3ea248bbf727fc7da9943"
#define VIDEO_COMPONENT_MANIFEST_IDENTITY "715f2359aaff048ccca8207976421af5f9f76b08b6f24986b3cc186d2822bc0e"
#define VIDEO_ARCHITECTURE_IDENTITY "47a03bbac2b5346771f70ae39155920f9b1c6e6cec17f2639dd0cbedfa90b517"
#define VIDEO_ROLE_MAP_IDENTITY "61e7a2cfc29e6dd3da966878f5388f1472a406d7e33ba34ef65f44b61f08f013"
#define VIDEO_UNRESOLVED_IDENTITY "935ae0a2371b15131b8920a879462484ebd3f5526ff5a97ef95c4e0af7b7cc1d"
#define VIDEO_TRANSFORM_IDENTITY "438aee784ab722b7c7cb5de1a934fa9ab3067282f30311ee2d595ad128f2d4f8"
#define VIDEO_PROFILE_IDENTITY "2a4211fda0e32dc53e4734a57e4ddc4cd408483b2980eb1439770dabb9bea575"
#define VIDEO_QUANT_EXECUTION_IDENTITY "87f12d8363dbd2a9a5f930a9bcfdbb06533c14db29f2139becc99b2042c76e81"
#define VIDEO_PAYLOAD_PLAN_IDENTITY "baf38268668fd651189dd0e4f90907cb7cc30061275913352dc770ba89a081d8"
#define VIDEO_PAYLOAD_BYTE_IDENTITY "97e4e92a97cb16890346a77f9766b4ad368c22df24715144a60d008a54eef2b7"
#define VIDEO_WRITER_PLAN_IDENTITY "f821e9c691a06f7e9b16261fc5261c160cbac5c7953e0155d6f52fea28ca00d1"
#define VIDEO_ARTIFACT_IDENTITY "29bb1df65227fa05444c4002e18d61934d70d872d8472c4757e93971f9e474cd"
#define VIDEO_PAYLOAD_IDENTITY YVEX_MINIMAX_H3_AUDIO_PAYLOAD_IDENTITY
#define VIDEO_PROFILE_NAME YVEX_MINIMAX_H3_AUDIO_PROFILE_NAME
#define VIDEO_SOURCE_SNAPSHOT_KEY 9907051661387403075ull
#define VIDEO_MAPPING_IDENTITY 16381021892971143870ull
#define VIDEO_TENSORS 560ull
#define VIDEO_ELEMENTS 2603871032ull
#define VIDEO_PAYLOAD_BYTES 10415484128ull
#define VIDEO_FILE_BYTES 10415528096ull
#define TEXT_COMPONENT_IDENTITY YVEX_MINIMAX_H3_TEXT_COMPONENT_IDENTITY
#define TEXT_TRANSFORM_IDENTITY "4e940d589f14194ee827be627afac91ee28ee2a45f1add22753d9ed3dae3962a"
#define TEXT_PROFILE_IDENTITY "5b534130f5114f096db93b96cce26fc6def534c95b6d338b87a668591c20b78f"
#define TEXT_QUANT_EXECUTION_IDENTITY "50fe7545f9fa204dd636fbdf44e660fc92dd52740d2f39b04118a83d72d058ee"
#define TEXT_PAYLOAD_PLAN_IDENTITY "214f0afd6fd2718e8184ce169e55018bc1b93598d34e8930e0514e7fe91328c0"
#define TEXT_PAYLOAD_BYTE_IDENTITY "c95ade3aef89252f46fb190b8d6d80dbbc6c335bef8fab3d2610dc688bcc326f"
#define TEXT_WRITER_PLAN_IDENTITY "ce5d96a027635da6802ca7184c0079206e7c9ba9322412a34064ac1cd7c4dc2c"
#define TEXT_ARTIFACT_IDENTITY "61407a737bf019cef8f0d786394986d419af957bd23a120f6dcc070128abb7ff"
#define TEXT_MAPPING_IDENTITY 17587980532596554443ull
#define TEXT_TENSORS 1058ull
#define TEXT_ELEMENTS 33357390064ull
#define TEXT_PAYLOAD_BYTES 66714780128ull
#define TEXT_FILE_BYTES 66727837152ull
#define TRANSFORMER_COMPONENT_IDENTITY "9745fc5bbf42a0a5d2d42209e50e64f5a58704c7602bce0f71f3225431304318"
#define TRANSFORMER_TRANSFORM_IDENTITY "8f3b16dff00769261df2d5f59c915c114874cb3abfb54ddc11d537875caec58a"
#define TRANSFORMER_PROFILE_IDENTITY "a5441084c92644d91b9285001cf455aa49ac4e3e6c9a779a90f47caab0f29162"
#define TRANSFORMER_QUANT_EXECUTION_IDENTITY "d47a261b45c2411d579b80f7626dee042f687e0454e18dba42fe891a060ec057"
#define TRANSFORMER_PAYLOAD_PLAN_IDENTITY "f14e2bb7963f267b8d2a26e440e344594b557ff3e951ad94302f8a396b5524ae"
#define TRANSFORMER_PAYLOAD_BYTE_IDENTITY "b261084e21a0098eb6947e65a05d559fa9b649d88e88f167120406634c786e85"
#define TRANSFORMER_WRITER_PLAN_IDENTITY "1dcd8cb25b82bf341f880ea42b7b0e2105ae3ea45a864e4ca6a49ce58b90dcde"
#define TRANSFORMER_ARTIFACT_IDENTITY "aa1c84ac801a50f8806b591fb419e60513f0e6bd312b5e3abc5352194a31b992"
#define TRANSFORMER_MAPPING_IDENTITY 17862857563445514422ull
#define TRANSFORMER_TENSORS 535ull
#define TRANSFORMER_ELEMENTS 33122992912ull
#define TRANSFORMER_PAYLOAD_BYTES 66280430144ull
#define TRANSFORMER_FILE_BYTES 66280465664ull
typedef struct { float *data; unsigned long long count; } component_buffer;
typedef struct {
    yvex_materialization_session *session; const yvex_minimax_h3_audio_decode_options *options;
    yvex_minimax_h3_audio_decode_result *result; yvex_minimax_h3_component_execution_failure *failure;
    yvex_error *err;
    unsigned long long live_workspace_bytes;
} audio_execution;
typedef struct {
    yvex_materialization_session *session; const yvex_minimax_h3_video_decode_options *options;
    yvex_minimax_h3_video_decode_result *result; yvex_minimax_h3_component_execution_failure *failure;
    yvex_error *err;
    unsigned long long live_workspace_bytes;
    unsigned long long patches, rows;
} video_execution;
static int component_buffer_open_raw(component_buffer *buffer, unsigned long long count,
                                     unsigned long long maximum, unsigned long long *live,
                                     unsigned long long *peak, yvex_error *err,
                                     const char *stage, const char *label)
{
    unsigned long long bytes, next;
    memset(buffer, 0, sizeof(*buffer));
    if (!count || !yvex_core_u64_mul(count, sizeof(float), &bytes) ||
        bytes > (unsigned long long)SIZE_MAX || !yvex_core_u64_add(*live, bytes, &next)) {
        yvex_error_setf(err, YVEX_ERR_BOUNDS, stage, "%s workspace extent overflowed", label);
        return YVEX_ERR_BOUNDS;
    }
    if (next > maximum) {
        yvex_error_setf(err, YVEX_ERR_BOUNDS, stage, "%s workspace budget was exceeded", label);
        return YVEX_ERR_BOUNDS;
    }
    buffer->data = (float *)malloc((size_t)bytes);
    if (!buffer->data) {
        yvex_error_setf(err, YVEX_ERR_NOMEM, stage, "%s workspace allocation failed", label);
        return YVEX_ERR_NOMEM;
    }
    buffer->count = count;
    *live = next;
    if (next > *peak) *peak = next;
    return YVEX_OK;
}
static void component_buffer_close_raw(component_buffer *buffer, unsigned long long *live)
{
    unsigned long long bytes = buffer->count * sizeof(float);
    if (bytes <= *live) *live -= bytes;
    free(buffer->data);
    memset(buffer, 0, sizeof(*buffer));
}
static int t2va_plan_build(yvex_minimax_h3_t2va_plan *out,
                           unsigned long long text_tokens, unsigned long long width,
                           unsigned long long height, unsigned long long frames,
                           unsigned int inference_steps, yvex_error *err)
{
    static const yvex_runtime_av_plan_policy policy = {
        .schema_version = YVEX_RUNTIME_AV_PLAN_SCHEMA_V1, .maximum_steps = 64u,
        .text_tag = 1u, .audio_tag = 2u, .video_tag = 0u,
        .frame_period = 17ull, .frame_remainder = 5ull,
        .video_latents_per_period = 5ull, .video_latent_remainder = 2ull,
        .spatial_ratio = 16ull, .patch_height = 2ull, .patch_width = 2ull,
        .audio_rate_numerator = 5ull, .audio_rate_denominator = 3ull,
        .audio_channels = 2ull, .video_value_width = 96ull, .audio_value_width = 32ull,
        .temporal_pattern = {1u, 4u, 4u, 4u, 4u}, .temporal_pattern_count = 5u,
        .video_sigma_shift = 12.0f, .audio_sigma_shift = 3.0f,
        .temporal_scale = 5.0 / 3.0, .spatial_scale = 32.0,
        .identity_domain = "yvex.minimax-h3.t2va.res-multistep-layout.v3",
        .target_identity = YVEX_MINIMAX_H3_TARGET_ID,
        .source_revision = YVEX_MINIMAX_H3_REVISION,
    };
    return yvex_runtime_av_plan_build(
        &policy, text_tokens, width, height, frames, inference_steps, out, err);
}
static int audio_execution_refuse(audio_execution *execution,
                                  yvex_minimax_h3_component_execution_code code,
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
        return audio_execution_refuse(execution, YVEX_MINIMAX_H3_COMPONENT_EXECUTION_CANCELLED,
                                      NULL, 0ull, 1ull, YVEX_ERR_CANCELLED,
                                      "Audio VAE execution was cancelled at a layer boundary");
    return YVEX_OK;
}
static int audio_buffer_open(audio_execution *execution, component_buffer *buffer,
                             unsigned long long count)
{
    int rc = component_buffer_open_raw(
        buffer, count, execution->options->max_workspace_bytes,
        &execution->live_workspace_bytes, &execution->result->peak_workspace_bytes,
        execution->err, "graph.minimax_h3.audio_vae.execute", "Audio VAE");
    if (rc == YVEX_OK) return rc;
    return audio_execution_refuse(
        execution, YVEX_MINIMAX_H3_COMPONENT_EXECUTION_BUDGET, NULL,
        execution->options->max_workspace_bytes, count, (yvex_status)rc,
        yvex_error_message(execution->err));
}
static void audio_buffer_close(audio_execution *execution, component_buffer *buffer)
{
    component_buffer_close_raw(buffer, &execution->live_workspace_bytes);
}
static const yvex_materialized_tensor_binding *
component_binding_find(const yvex_materialization_session *session, const char *name)
{
    unsigned long long index;
    for (index = 0ull;; ++index) {
        const yvex_materialized_tensor_binding *binding =
            yvex_materialization_session_tensor_at(session, index);
        if (!binding) return NULL;
        if (strcmp(binding->name, name) == 0) return binding;
    }
}
static int component_tensor_load_raw(
    yvex_materialization_session *session, const char *name,
    unsigned int rank, const unsigned long long *dims, component_buffer *buffer,
    unsigned long long maximum, unsigned long long *live, unsigned long long *peak,
    unsigned long long *reads, unsigned long long *payload,
    yvex_minimax_h3_component_execution_failure *failure,
    yvex_error *err, const char *stage, const char *label)
{
    const yvex_materialized_tensor_binding *binding = component_binding_find(session, name);
    yvex_materialization_failure materialization_failure;
    yvex_minimax_h3_component_execution_failure local_failure = {0};
    unsigned long long count = 1ull, expected_bytes = 0ull;
    unsigned int dimension;
    int rc = YVEX_OK;
    if (!failure) failure = &local_failure;
    if (!binding) {
        failure->code = YVEX_MINIMAX_H3_COMPONENT_EXECUTION_MISSING_TENSOR;
        failure->expected = 1ull;
        goto rejected;
    }
    if (binding->qtype != YVEX_GGUF_QTYPE_F32 || binding->rank != rank) {
        failure->code = YVEX_MINIMAX_H3_COMPONENT_EXECUTION_TENSOR_CONTRACT;
        failure->expected = rank;
        failure->actual = binding->rank;
        goto rejected;
    }
    for (dimension = 0u; dimension < rank; ++dimension)
        if (binding->dims[dimension] != dims[dimension] ||
            !yvex_core_u64_mul(count, dims[dimension], &count)) {
            failure->code = YVEX_MINIMAX_H3_COMPONENT_EXECUTION_TENSOR_CONTRACT;
            failure->expected = dims[dimension];
            failure->actual = binding->dims[dimension];
            goto rejected;
        }
    if (!yvex_core_u64_mul(count, sizeof(float), &expected_bytes) ||
        binding->encoded_bytes != expected_bytes) {
        failure->code = YVEX_MINIMAX_H3_COMPONENT_EXECUTION_TENSOR_CONTRACT;
        failure->expected = expected_bytes;
        failure->actual = binding->encoded_bytes;
        goto rejected;
    }
    rc = component_buffer_open_raw(buffer, count, maximum, live, peak, err, stage, label);
    if (rc != YVEX_OK) {
        failure->code = YVEX_MINIMAX_H3_COMPONENT_EXECUTION_BUDGET;
        goto failed;
    }
    rc = yvex_materialization_session_read(
        session, binding, 0ull, buffer->data, (size_t)binding->encoded_bytes,
        &materialization_failure, err);
    if (rc != YVEX_OK) {
        component_buffer_close_raw(buffer, live);
        failure->code = YVEX_MINIMAX_H3_COMPONENT_EXECUTION_MATERIALIZATION;
        failure->expected = binding->encoded_bytes;
        failure->actual = materialization_failure.actual;
        failure->reason = materialization_failure.reason;
        goto failed;
    }
    (*reads)++;
    *payload += binding->encoded_bytes;
    return YVEX_OK;
rejected:
    rc = YVEX_ERR_FORMAT;
    yvex_error_setf(err, rc, stage, "%s tensor contract rejected %s", label, name);
failed:
    yvex_core_text_copy(failure->tensor_name, sizeof(failure->tensor_name), name);
    if (!failure->reason) failure->reason = yvex_error_message(err);
    return rc;
}
static int component_weight_bind(const yvex_materialization_session *session,
    const yvex_runtime_residency *residency, const char *name,
    yvex_minimax_h3_encoded_weight *weight, yvex_error *err)
{
    const yvex_materialized_tensor_binding *binding = component_binding_find(session, name);
    if (!binding || !binding->row_count) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "minimax-h3.component.binding",
                       "an exact component weight binding is unavailable");
        return YVEX_ERR_FORMAT;
    }
    if (yvex_runtime_residency_binding_view(
            residency, binding, &weight->encoded, &weight->encoded_bytes, err) != YVEX_OK)
        return yvex_error_code(err);
    weight->qtype = binding->qtype; weight->row_count = binding->row_count;
    weight->row_width = binding->row_width;
    weight->row_bytes = binding->encoded_bytes / binding->row_count;
    return YVEX_OK;
}
static int audio_tensor_load(audio_execution *execution, const char *name,
                             unsigned int rank, const unsigned long long *dims,
                             component_buffer *buffer)
{
    return component_tensor_load_raw(
        execution->session, name, rank, dims, buffer,
        execution->options->max_workspace_bytes, &execution->live_workspace_bytes,
        &execution->result->peak_workspace_bytes, &execution->result->tensor_reads,
        &execution->result->payload_bytes_read, execution->failure, execution->err,
        "graph.minimax_h3.audio_vae.execute", "Audio VAE");
}
static int audio_name_build(audio_execution *execution, char *output, size_t capacity,
                            const char *prefix, const char *suffix)
{
    int written = snprintf(output, capacity, "%s%s", prefix, suffix);
    if (written < 0 || (size_t)written >= capacity)
        return audio_execution_refuse(execution,
                                      YVEX_MINIMAX_H3_COMPONENT_EXECUTION_TENSOR_CONTRACT,
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
    component_buffer weight = {0};
    component_buffer gain = {0};
    component_buffer bias = {0};
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
        execution->failure->code == YVEX_MINIMAX_H3_COMPONENT_EXECUTION_NONE) {
        execution->failure->code = YVEX_MINIMAX_H3_COMPONENT_EXECUTION_NUMERIC;
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
    component_buffer alpha = {0};
    component_buffer beta = {0};
    component_buffer up = {0};
    component_buffer down = {0};
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
        execution->failure->code == YVEX_MINIMAX_H3_COMPONENT_EXECUTION_NONE) {
        execution->failure->code = YVEX_MINIMAX_H3_COMPONENT_EXECUTION_NUMERIC;
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
                                      YVEX_MINIMAX_H3_COMPONENT_EXECUTION_TENSOR_CONTRACT,
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
    component_buffer activation = {0};
    component_buffer convolution = {0};
    component_buffer scratch = {0};
    unsigned long long values;
    unsigned long long layer;
    char prefix[256];
    int rc;
    if (!yvex_core_u64_mul(batch, channels, &values) ||
        !yvex_core_u64_mul(values, length, &values))
        return audio_execution_refuse(execution,
                                      YVEX_MINIMAX_H3_COMPONENT_EXECUTION_BUDGET,
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
                               const component_buffer *input, unsigned long long batch,
                               unsigned long long input_channels,
                               unsigned long long input_length,
                               component_buffer *output, unsigned long long *output_channels,
                               unsigned long long *output_length)
{
    static const unsigned long long rates[] = {5ull, 5ull, 2ull, 2ull, 2ull, 2ull, 2ull};
    static const unsigned long long kernels[] = {9ull, 9ull, 4ull, 4ull, 4ull, 4ull, 4ull};
    yvex_graph_conv1d_geometry geometry;
    component_buffer upsampled = {0};
    component_buffer sum = {0};
    component_buffer block = {0};
    unsigned long long values;
    unsigned long long block_index;
    char prefix[256];
    int rc;
    memset(output, 0, sizeof(*output));
    if (stage >= sizeof(rates) / sizeof(rates[0]) || input_channels < 2ull)
        return audio_execution_refuse(execution,
                                      YVEX_MINIMAX_H3_COMPONENT_EXECUTION_INVALID_ARGUMENT,
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
                                      YVEX_MINIMAX_H3_COMPONENT_EXECUTION_BUDGET,
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
    unsigned long long geometry[3] = {
        options->batch, options->latent_channels, options->latent_steps,
    };
    return yvex_graph_f32_execution_identity(
        "yvex.minimax-h3.audio-vae.cpu.v1", summary->artifact_identity,
        geometry, 3ull, options->latent,
        options->batch * options->latent_channels * options->latent_steps,
        options->output, result->output_values, result->execution_identity);
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
                                      YVEX_MINIMAX_H3_COMPONENT_EXECUTION_INVALID_ARGUMENT,
                                      NULL, 32ull,
                                      execution->options ? execution->options->latent_channels : 0ull,
                                      YVEX_ERR_INVALID_ARG,
                                      "Audio VAE decode requires exact latent geometry and bounded output");
    if (!summary || !summary->committed ||
        strcmp(summary->artifact_identity, YVEX_MINIMAX_H3_AUDIO_ARTIFACT_IDENTITY) != 0)
        return audio_execution_refuse(execution,
                                      YVEX_MINIMAX_H3_COMPONENT_EXECUTION_LIFECYCLE,
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
                                      YVEX_MINIMAX_H3_COMPONENT_EXECUTION_BUDGET,
                                      NULL, output_values,
                                      execution->options->output_capacity,
                                      YVEX_ERR_BOUNDS,
                                      "Audio VAE output buffer is smaller than the exact 800x ratio");
    for (index = 0ull; index < latent_values; ++index)
        if (!isfinite(execution->options->latent[index]))
            return audio_execution_refuse(
                execution, YVEX_MINIMAX_H3_COMPONENT_EXECUTION_NUMERIC,
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
                                yvex_minimax_h3_component_execution_failure *failure,
                                yvex_error *err)
{
    audio_execution execution;
    component_buffer current = {0};
    component_buffer next = {0};
    component_buffer activated = {0};
    component_buffer scratch = {0};
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
                                      YVEX_MINIMAX_H3_COMPONENT_EXECUTION_INVALID_ARGUMENT,
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
                                         YVEX_MINIMAX_H3_COMPONENT_EXECUTION_NUMERIC,
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
static int video_execution_refuse(video_execution *execution,
                                  yvex_minimax_h3_component_execution_code code,
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
                   "graph.minimax_h3.video_vae.execute", reason);
    return status;
}
static int video_cancel_check(video_execution *execution)
{
    if (execution->options->cancelled &&
        execution->options->cancelled(execution->options->cancellation_context))
        return video_execution_refuse(
            execution, YVEX_MINIMAX_H3_COMPONENT_EXECUTION_CANCELLED,
            NULL, 0ull, 1ull, YVEX_ERR_CANCELLED,
            "Visual VAE execution was cancelled at a block boundary");
    return YVEX_OK;
}
static int video_buffer_open(video_execution *execution, component_buffer *buffer,
                             unsigned long long count)
{
    int rc = component_buffer_open_raw(
        buffer, count, execution->options->max_workspace_bytes,
        &execution->live_workspace_bytes, &execution->result->peak_workspace_bytes,
        execution->err, "graph.minimax_h3.video_vae.execute", "Visual VAE");
    if (rc == YVEX_OK) return rc;
    return video_execution_refuse(
        execution, YVEX_MINIMAX_H3_COMPONENT_EXECUTION_BUDGET, NULL,
        execution->options->max_workspace_bytes, count, (yvex_status)rc,
        yvex_error_message(execution->err));
}
static void video_buffer_close(video_execution *execution, component_buffer *buffer)
{
    component_buffer_close_raw(buffer, &execution->live_workspace_bytes);
}
static int video_tensor_load(video_execution *execution, const char *name,
                             unsigned int rank, const unsigned long long *dims,
                             component_buffer *buffer)
{
    return component_tensor_load_raw(
        execution->session, name, rank, dims, buffer,
        execution->options->max_workspace_bytes, &execution->live_workspace_bytes,
        &execution->result->peak_workspace_bytes, &execution->result->tensor_reads,
        &execution->result->payload_bytes_read, execution->failure, execution->err,
        "graph.minimax_h3.video_vae.execute", "Visual VAE");
}
static int video_name(video_execution *execution, char output[256],
                      const char *prefix, const char *suffix)
{
    int written = snprintf(output, 256u, "%s%s", prefix, suffix);
    if (written < 0 || written >= 256)
        return video_execution_refuse(
            execution, YVEX_MINIMAX_H3_COMPONENT_EXECUTION_TENSOR_CONTRACT,
            prefix, 255ull, written < 0 ? 0ull : (unsigned long long)written,
            YVEX_ERR_BOUNDS, "Visual VAE tensor name exceeded its bound");
    return YVEX_OK;
}
static int video_linear(video_execution *execution, const char *prefix,
                        unsigned int weight_rank, const float *input,
                        unsigned long long rows, unsigned long long input_width,
                        unsigned long long output_width, float *output)
{
    component_buffer weight = {0}, bias = {0};
    unsigned long long weight_dims[4] = {output_width, input_width, 1ull, 1ull};
    unsigned long long bias_dims[1] = {output_width};
    char name[256];
    int rc = video_name(execution, name, prefix, ".weight");
    if (rc == YVEX_OK)
        rc = video_tensor_load(execution, name, weight_rank, weight_dims, &weight);
    if (rc == YVEX_OK) rc = video_name(execution, name, prefix, ".bias");
    if (rc == YVEX_OK)
        rc = video_tensor_load(execution, name, 1u, bias_dims, &bias);
    if (rc == YVEX_OK)
        rc = yvex_graph_linear_source_f32(
            input, rows * input_width, rows, input_width,
            weight.data, weight.count, bias.data, bias.count, output_width,
            output, rows * output_width, execution->err);
    if (rc != YVEX_OK && execution->failure &&
        execution->failure->code == YVEX_MINIMAX_H3_COMPONENT_EXECUTION_NONE) {
        execution->failure->code = YVEX_MINIMAX_H3_COMPONENT_EXECUTION_NUMERIC;
        execution->failure->reason = yvex_error_message(execution->err);
        yvex_core_text_copy(execution->failure->tensor_name,
                            sizeof(execution->failure->tensor_name), prefix);
    }
    video_buffer_close(execution, &bias);
    video_buffer_close(execution, &weight);
    return rc;
}
static int video_rms_norm(video_execution *execution, const char *prefix,
                          const float *input, unsigned long long rows,
                          unsigned long long width, float *output)
{
    component_buffer weight = {0};
    unsigned long long dims[1] = {width};
    unsigned long long row;
    char name[256];
    int rc = video_name(execution, name, prefix, ".weight");
    if (rc == YVEX_OK) rc = video_tensor_load(execution, name, 1u, dims, &weight);
    if (rc == YVEX_OK) memcpy(output, input, (size_t)(rows * width * sizeof(float)));
    for (row = 0ull; row < rows && rc == YVEX_OK; ++row)
        if (!yvex_attention_rms_norm(output + row * width, width,
                                     weight.data, 1.0e-5))
            rc = video_execution_refuse(
                execution, YVEX_MINIMAX_H3_COMPONENT_EXECUTION_NUMERIC,
                name, width, row, YVEX_ERR_FORMAT,
                "Visual VAE RMSNorm produced a non-finite value");
    video_buffer_close(execution, &weight);
    return rc;
}
static int video_scale_residual(video_execution *execution, const char *name,
                                const float *delta, unsigned long long rows,
                                unsigned long long width, float *hidden)
{
    component_buffer scale = {0};
    unsigned long long dims[1] = {width};
    int rc = video_tensor_load(execution, name, 1u, dims, &scale);
    if (rc == YVEX_OK)
        rc = yvex_graph_scaled_residual_f32(
            hidden, delta, scale.data, rows, width, execution->err);
    if (rc != YVEX_OK)
        rc = video_execution_refuse(
            execution, YVEX_MINIMAX_H3_COMPONENT_EXECUTION_NUMERIC,
            name, rows * width, 0ull, rc,
            "Visual VAE residual update failed its numeric contract");
    video_buffer_close(execution, &scale);
    return rc;
}
static int video_rope_apply(video_execution *execution, float *qkv)
{
    const yvex_minimax_h3_video_decode_options *options = execution->options;
    int rc = yvex_graph_rope_3d_interleaved_qk_f32(
        qkv, execution->rows, options->latent_frames, options->latent_height,
        options->latent_width, 32ull, 64ull, 8ull, 100.0f, execution->err);
    return rc == YVEX_OK ? rc : video_execution_refuse(
        execution, YVEX_MINIMAX_H3_COMPONENT_EXECUTION_NUMERIC,
        NULL, execution->rows * 64ull, 0ull, rc,
        "Visual VAE RoPE failed its numeric contract");
}
static int video_rope_tables(video_execution *execution, float *cosines, float *sines)
{
    unsigned long long token;
    int rc = YVEX_OK;
    memset(cosines, 0, (size_t)(execution->rows * 48ull * sizeof(float)));
    memset(sines, 0, (size_t)(execution->rows * 48ull * sizeof(float)));
    for (token = 0ull; token < execution->rows && rc == YVEX_OK; ++token)
        rc = yvex_graph_rope_3d_row_f32(
            token, execution->options->latent_frames, execution->options->latent_height,
            execution->options->latent_width, 8ull, 100.0f,
            cosines + token * 48ull, sines + token * 48ull, execution->err);
    return rc;
}
static int video_block_name(video_execution *execution, char output[256],
                            unsigned long long block, const char *suffix)
{
    int written = snprintf(output, 256u, "decoder.transformer_blocks.%llu%s",
                           block, suffix);
    if (written < 0 || written >= 256)
        return video_execution_refuse(
            execution, YVEX_MINIMAX_H3_COMPONENT_EXECUTION_TENSOR_CONTRACT,
            suffix, 255ull, written < 0 ? 0ull : (unsigned long long)written,
            YVEX_ERR_BOUNDS, "Visual VAE block tensor name exceeded its bound");
    return YVEX_OK;
}
static int video_dense_weight_name(void *context, unsigned long long block,
                                   unsigned int slot, char output[256], yvex_error *err)
{
    static const char *const suffixes[YVEX_TRANSFORMER_DENSE_DECODER_BLOCK_WEIGHT_COUNT] = {
        ".norm1.weight", ".attn.to_qkv.weight", ".attn.to_qkv.bias",
        ".attn.to_out.weight", ".attn.to_out.bias", ".scale1",
        ".norm2.weight", ".ff.w1.weight", ".ff.w1.bias",
        ".ff.w2.weight", ".ff.w2.bias", ".scale2",
    };
    video_execution *execution = (video_execution *)context;
    if (!execution || slot >= YVEX_TRANSFORMER_DENSE_DECODER_BLOCK_WEIGHT_COUNT)
        return video_execution_refuse(
            execution, YVEX_MINIMAX_H3_COMPONENT_EXECUTION_TENSOR_CONTRACT,
            NULL, YVEX_TRANSFORMER_DENSE_DECODER_BLOCK_WEIGHT_COUNT,
            slot, YVEX_ERR_BOUNDS,
            "Visual VAE decoder weight slot exceeds its exact recipe");
    (void)err;
    return video_block_name(execution, output, block, suffixes[slot]);
}
static int video_block_execute(video_execution *execution,
                               unsigned long long block, component_buffer *hidden,
                               component_buffer *normalized, component_buffer *qkv,
                               component_buffer *projected, component_buffer *fused,
                               component_buffer *gated, component_buffer *attention,
                               component_buffer *scratch)
{
    char name[256];
    int rc = video_cancel_check(execution);
    if (rc == YVEX_OK) rc = video_block_name(execution, name, block, ".norm1");
    if (rc == YVEX_OK)
        rc = video_rms_norm(execution, name, hidden->data, execution->rows, 2048ull,
                            normalized->data);
    if (rc == YVEX_OK)
        rc = video_block_name(execution, name, block, ".attn.to_qkv");
    if (rc == YVEX_OK)
        rc = video_linear(execution, name, 2u, normalized->data,
                          execution->rows, 2048ull, 6144ull, qkv->data);
    if (rc == YVEX_OK)
        rc = yvex_graph_interleaved_qk_norm_f32(
            qkv->data, execution->rows, 32ull, 64ull, 1.0e-5, execution->err);
    if (rc == YVEX_OK) rc = video_rope_apply(execution, qkv->data);
    if (rc == YVEX_OK)
        rc = yvex_graph_full_attention_f32(
            qkv->data, execution->rows, 32ull, 64ull, attention->data,
            scratch->data, scratch->count, execution->err);
    if (rc == YVEX_OK)
        rc = video_block_name(execution, name, block, ".attn.to_out");
    if (rc == YVEX_OK)
        rc = video_linear(execution, name, 2u, attention->data,
                          execution->rows, 2048ull, 2048ull, projected->data);
    if (rc == YVEX_OK) rc = video_block_name(execution, name, block, ".scale1");
    if (rc == YVEX_OK)
        rc = video_scale_residual(execution, name, projected->data,
                                  execution->rows, 2048ull, hidden->data);
    if (rc == YVEX_OK) rc = video_block_name(execution, name, block, ".norm2");
    if (rc == YVEX_OK)
        rc = video_rms_norm(execution, name, hidden->data, execution->rows, 2048ull,
                            normalized->data);
    if (rc == YVEX_OK) rc = video_block_name(execution, name, block, ".ff.w1");
    if (rc == YVEX_OK)
        rc = video_linear(execution, name, 2u, normalized->data,
                          execution->rows, 2048ull, 16384ull, fused->data);
    if (rc == YVEX_OK)
        rc = yvex_graph_silu_gate_f32(fused->data, execution->rows, 8192ull,
                                      gated->data, execution->err);
    if (rc == YVEX_OK) rc = video_block_name(execution, name, block, ".ff.w2");
    if (rc == YVEX_OK)
        rc = video_linear(execution, name, 2u, gated->data,
                          execution->rows, 8192ull, 2048ull, projected->data);
    if (rc == YVEX_OK) rc = video_block_name(execution, name, block, ".scale2");
    if (rc == YVEX_OK)
        rc = video_scale_residual(execution, name, projected->data,
                                  execution->rows, 2048ull, hidden->data);
    if (rc != YVEX_OK && execution->failure &&
        execution->failure->code == YVEX_MINIMAX_H3_COMPONENT_EXECUTION_NONE) {
        execution->failure->code = YVEX_MINIMAX_H3_COMPONENT_EXECUTION_NUMERIC;
        execution->failure->reason = yvex_error_message(execution->err);
    }
    return rc;
}
static int video_final_norm(video_execution *execution, component_buffer *hidden)
{
    component_buffer weight = {0}, bias = {0};
    unsigned long long dims[1] = {2048ull};
    int rc = video_tensor_load(execution, "decoder.norm_out.weight", 1u, dims, &weight);
    if (rc == YVEX_OK)
        rc = video_tensor_load(execution, "decoder.norm_out.bias", 1u, dims, &bias);
    if (rc == YVEX_OK)
        rc = yvex_graph_layer_norm_f32(hidden->data, execution->rows, 2048ull,
                                       weight.data, bias.data, 1.0e-5, execution->err);
    if (rc != YVEX_OK && execution->failure &&
        execution->failure->code == YVEX_MINIMAX_H3_COMPONENT_EXECUTION_NONE)
        rc = video_execution_refuse(
            execution, YVEX_MINIMAX_H3_COMPONENT_EXECUTION_NUMERIC,
            "decoder.norm_out", execution->rows * 2048ull, 0ull, rc,
            "Visual VAE final LayerNorm failed");
    video_buffer_close(execution, &bias);
    video_buffer_close(execution, &weight);
    return rc;
}
static int video_execution_identity(const char *domain,
                                    const yvex_materialization_summary *summary,
                                    const yvex_minimax_h3_video_decode_options *options,
                                    yvex_minimax_h3_video_decode_result *result)
{
    unsigned long long geometry[5] = {
        options->batch, options->latent_channels, options->latent_frames,
        options->latent_height, options->latent_width,
    };
    return yvex_graph_f32_execution_identity(
        domain, summary->artifact_identity, geometry, 5ull, options->latent,
        result->batch * options->latent_channels * options->latent_frames *
            options->latent_height * options->latent_width,
        options->output, result->output_values, result->execution_identity);
}
static int video_decode_validate(video_execution *execution)
{
    const yvex_materialization_summary *summary =
        yvex_materialization_session_summary(execution->session);
    unsigned long long input_values = 0ull, output_values = 0ull, index;
    if (!execution->options || !execution->result || !execution->options->latent ||
        !execution->options->output || execution->options->batch != 1ull ||
        execution->options->latent_channels != 24ull ||
        !execution->options->latent_frames || !execution->options->latent_height ||
        !execution->options->latent_width ||
        !execution->options->max_workspace_bytes)
        return video_execution_refuse(
            execution, YVEX_MINIMAX_H3_COMPONENT_EXECUTION_INVALID_ARGUMENT,
            NULL, 1ull,
            execution->options ? execution->options->output_capacity : 0ull,
            YVEX_ERR_INVALID_ARG,
            "Visual VAE decode requires batch one, 24 channels, and positive latent geometry");
    if (!yvex_core_u64_mul(execution->options->latent_frames,
                           execution->options->latent_height, &execution->patches) ||
        !yvex_core_u64_mul(execution->patches, execution->options->latent_width,
                           &execution->patches) ||
        !yvex_core_u64_add(execution->patches, 5ull, &execution->rows) ||
        execution->options->latent_frames > ULLONG_MAX / 4ull ||
        execution->options->latent_height > ULLONG_MAX / 16ull ||
        execution->options->latent_width > ULLONG_MAX / 16ull ||
        execution->rows > ULLONG_MAX / 16384ull ||
        !yvex_core_u64_mul(execution->patches, 24ull, &input_values) ||
        !yvex_core_u64_mul(execution->patches, 3072ull, &output_values) ||
        output_values > (unsigned long long)SIZE_MAX / sizeof(float) ||
        execution->options->output_capacity < output_values)
        return video_execution_refuse(
            execution, YVEX_MINIMAX_H3_COMPONENT_EXECUTION_INVALID_ARGUMENT,
            NULL, output_values, execution->options->output_capacity,
            YVEX_ERR_BOUNDS,
            "Visual VAE latent geometry or RGB output extent exceeded its bound");
    if (!summary || !summary->committed ||
        strcmp(summary->artifact_identity, VIDEO_ARTIFACT_IDENTITY) != 0)
        return video_execution_refuse(
            execution, YVEX_MINIMAX_H3_COMPONENT_EXECUTION_LIFECYCLE,
            NULL, 1ull, summary ? (unsigned long long)summary->committed : 0ull,
            YVEX_ERR_STATE,
            "Visual VAE decode requires the committed exact component artifact");
    for (index = 0ull; index < input_values; ++index)
        if (!isfinite(execution->options->latent[index]))
            return video_execution_refuse(
                execution, YVEX_MINIMAX_H3_COMPONENT_EXECUTION_NUMERIC,
                NULL, input_values, index, YVEX_ERR_FORMAT,
                "Visual VAE latent input contains a non-finite value");
    execution->result->batch = 1ull;
    execution->result->frames = execution->options->latent_frames * 4ull;
    execution->result->height = execution->options->latent_height * 16ull;
    execution->result->width = execution->options->latent_width * 16ull;
    execution->result->output_values = output_values;
    yvex_core_text_copy(execution->result->artifact_identity,
                        sizeof(execution->result->artifact_identity),
                        summary->artifact_identity);
    return YVEX_OK;
}
static void video_latent_pack(const yvex_minimax_h3_video_decode_options *options,
                              unsigned long long patches, float *packed)
{
    unsigned long long patch, channel;
    for (patch = 0ull; patch < patches; ++patch)
        for (channel = 0ull; channel < 24ull; ++channel)
            packed[patch * 24ull + channel] = options->latent[channel * patches + patch];
}
static int video_prefix_prepare(video_execution *execution, component_buffer *hidden)
{
    component_buffer packed = {0}, post = {0}, registers = {0};
    unsigned long long register_dims[3] = {1ull, 4ull, 2048ull};
    int rc = video_buffer_open(execution, &packed, execution->patches * 24ull);
    if (rc == YVEX_OK) rc = video_buffer_open(execution, &post, execution->patches * 24ull);
    if (rc == YVEX_OK) rc = video_buffer_open(execution, hidden, execution->rows * 2048ull);
    if (rc == YVEX_OK) video_latent_pack(execution->options, execution->patches, packed.data);
    if (rc == YVEX_OK)
        rc = video_linear(execution, "post_quant_conv", 4u, packed.data,
                          execution->patches, 24ull, 24ull, post.data);
    if (rc == YVEX_OK)
        rc = video_linear(execution, "decoder.x_embedder", 2u, post.data,
                          execution->patches, 24ull, 2048ull, hidden->data);
    if (rc == YVEX_OK)
        rc = video_tensor_load(execution, "decoder.register_tokens", 3u,
                               register_dims, &registers);
    if (rc == YVEX_OK) {
        memcpy(hidden->data + execution->patches * 2048ull, registers.data,
               (size_t)(4ull * 2048ull * sizeof(float)));
        memset(hidden->data + (execution->patches + 4ull) * 2048ull,
               0, (size_t)(2048ull * sizeof(float)));
    }
    video_buffer_close(execution, &registers);
    video_buffer_close(execution, &post);
    video_buffer_close(execution, &packed);
    if (rc != YVEX_OK) video_buffer_close(execution, hidden);
    return rc;
}
static void video_output_unpack(const component_buffer *patch_output,
                                const yvex_minimax_h3_video_decode_options *options,
                                yvex_minimax_h3_video_decode_result *result)
{
    unsigned long long patch, value;
    unsigned long long input_plane = options->latent_height * options->latent_width;
    unsigned long long output_plane = result->height * result->width;
    for (patch = 0ull; patch < patch_output->count / 3072ull; ++patch) {
        unsigned long long temporal = patch / input_plane;
        unsigned long long spatial = patch % input_plane;
        unsigned long long height = spatial / options->latent_width;
        unsigned long long width = spatial % options->latent_width;
        for (value = 0ull; value < 3072ull; ++value) {
            unsigned long long channel = value / 1024ull;
            unsigned long long within = value % 1024ull;
            unsigned long long frame = within / 256ull;
            unsigned long long pixel = within % 256ull;
            unsigned long long output_index = channel * result->frames * output_plane +
                (temporal * 4ull + frame) * output_plane +
                (height * 16ull + pixel / 16ull) * result->width +
                width * 16ull + pixel % 16ull;
            options->output[output_index] = patch_output->data[patch * 3072ull + value];
        }
    }
}
static int video_vae_decode_cpu(yvex_materialization_session *session,
                                const yvex_minimax_h3_video_decode_options *options,
                                yvex_minimax_h3_video_decode_result *result,
                                yvex_minimax_h3_component_execution_failure *failure,
                                yvex_error *err)
{
    video_execution execution = {
        .session = session,
        .options = options,
        .result = result,
        .failure = failure,
        .err = err,
    };
    component_buffer hidden = {0}, normalized = {0}, qkv = {0};
    component_buffer projected = {0}, fused = {0}, gated = {0};
    component_buffer attention = {0}, scratch = {0}, patch_output = {0};
    unsigned long long block;
    int rc;
    if (result) memset(result, 0, sizeof(*result));
    if (failure) memset(failure, 0, sizeof(*failure));
    if (!options || !result)
        return video_execution_refuse(
            &execution, YVEX_MINIMAX_H3_COMPONENT_EXECUTION_INVALID_ARGUMENT,
            NULL, 2ull, 0ull, YVEX_ERR_INVALID_ARG,
            "Visual VAE decode requires options and result");
    rc = video_decode_validate(&execution);
    if (rc == YVEX_OK) rc = video_prefix_prepare(&execution, &hidden);
    if (rc == YVEX_OK) rc = video_buffer_open(&execution, &normalized, execution.rows * 2048ull);
    if (rc == YVEX_OK) rc = video_buffer_open(&execution, &qkv, execution.rows * 6144ull);
    if (rc == YVEX_OK) rc = video_buffer_open(&execution, &projected, execution.rows * 2048ull);
    if (rc == YVEX_OK) rc = video_buffer_open(&execution, &fused, execution.rows * 16384ull);
    if (rc == YVEX_OK) rc = video_buffer_open(&execution, &gated, execution.rows * 8192ull);
    if (rc == YVEX_OK) rc = video_buffer_open(&execution, &attention, execution.rows * 2048ull);
    if (rc == YVEX_OK) rc = video_buffer_open(&execution, &scratch, execution.rows);
    for (block = 0ull; block < 36ull && rc == YVEX_OK; ++block)
        rc = video_block_execute(&execution, block, &hidden, &normalized, &qkv,
                                 &projected, &fused, &gated, &attention, &scratch);
    if (rc == YVEX_OK) rc = video_final_norm(&execution, &hidden);
    if (rc == YVEX_OK)
        rc = video_buffer_open(&execution, &patch_output, execution.patches * 3072ull);
    if (rc == YVEX_OK)
        rc = video_linear(&execution, "decoder.proj_out", 2u, hidden.data,
                          execution.patches, 2048ull, 3072ull, patch_output.data);
    if (rc == YVEX_OK) video_output_unpack(&patch_output, options, result);
    if (rc == YVEX_OK) {
        const yvex_materialization_summary *summary =
            yvex_materialization_session_summary(session);
        if (!video_execution_identity(
                result->output_values == 3072ull
                    ? "yvex.minimax-h3.video-vae.cpu.reduced-v1"
                    : "yvex.minimax-h3.video-vae.cpu.geometry-v2",
                summary, options, result))
            rc = video_execution_refuse(
                &execution, YVEX_MINIMAX_H3_COMPONENT_EXECUTION_NUMERIC,
                NULL, 1ull, 0ull, YVEX_ERR_STATE,
                "Visual VAE execution identity could not be sealed");
        else
            result->complete = 1;
    }
    video_buffer_close(&execution, &patch_output);
    video_buffer_close(&execution, &scratch);
    video_buffer_close(&execution, &attention);
    video_buffer_close(&execution, &gated);
    video_buffer_close(&execution, &fused);
    video_buffer_close(&execution, &projected);
    video_buffer_close(&execution, &qkv);
    video_buffer_close(&execution, &normalized);
    video_buffer_close(&execution, &hidden);
    if (rc == YVEX_OK) yvex_error_clear(err);
    return rc;
}
static const yvex_artifact_component_metadata audio_metadata[] = {
    {"general.architecture", "minimax-h3"},
    {"general.name", "audio_vae"},
    {"yvex.logical.target", YVEX_MINIMAX_H3_TARGET_ID},
    {"yvex.logical.component", "audio_vae"},
    {"yvex.source.snapshot.identity", YVEX_MINIMAX_H3_AUDIO_SNAPSHOT_IDENTITY},
    {"yvex.logical.component.identity", YVEX_MINIMAX_H3_AUDIO_COMPONENT_IDENTITY},
    {"yvex.logical.component_manifest.identity",
     YVEX_MINIMAX_H3_AUDIO_MANIFEST_IDENTITY},
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
    .quant_execution_identity = YVEX_MINIMAX_H3_AUDIO_QUANT_IDENTITY,
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
static const yvex_artifact_component_storage audio_storage[] = {
    {YVEX_GGUF_QTYPE_F32, YVEX_MINIMAX_H3_AUDIO_TENSORS},
};
static const yvex_artifact_component_contract audio_contract = {
    &audio_catalog, audio_metadata, audio_storage,
    sizeof(audio_metadata) / sizeof(audio_metadata[0]), 1ull,
    YVEX_MINIMAX_H3_AUDIO_ELEMENTS, 32ull,
};
static const yvex_artifact_component_metadata text_metadata[] = {
    {"general.architecture", "minimax-h3"}, {"general.name", "text_encoder"},
    {"yvex.logical.target", YVEX_MINIMAX_H3_TARGET_ID},
    {"yvex.logical.component", "text_encoder"},
    {"yvex.source.snapshot.identity", VIDEO_SOURCE_SNAPSHOT_IDENTITY},
    {"yvex.logical.component.identity", TEXT_COMPONENT_IDENTITY},
    {"yvex.logical.component_manifest.identity", VIDEO_COMPONENT_MANIFEST_IDENTITY},
    {"yvex.logical.architecture.identity", VIDEO_ARCHITECTURE_IDENTITY},
    {"yvex.logical.role_map.identity", VIDEO_ROLE_MAP_IDENTITY},
    {"yvex.logical.unresolved_requirements.identity", VIDEO_UNRESOLVED_IDENTITY},
    {"yvex.transformation.identity", TEXT_TRANSFORM_IDENTITY},
    {"yvex.physical.profile.name", VIDEO_PROFILE_NAME},
    {"yvex.physical.profile.identity", TEXT_PROFILE_IDENTITY},
    {"yvex.physical.payload_plan.identity", TEXT_PAYLOAD_PLAN_IDENTITY},
    {"yvex.payload.identity", VIDEO_PAYLOAD_IDENTITY},
    {"yvex.evidence.stage", "component-artifact-planned"},
    {"yvex.physical.shape.policy", "reverse-logical-fold-outer-v1"},
    {"tokenizer.ggml.model", "gpt2"}, {"tokenizer.ggml.pre", "qwen2"},
    {"yvex.tokenizer.prompt_policy", "verbatim-no-special-v1"},
    {"yvex.tokenizer.json.sha256", "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7"},
    {"yvex.tokenizer.config.sha256", "a07e942ac874baa13758de8d1fbdb186683cc03416b5589e1b6671c6b3057c68"},
    {"yvex.tokenizer.json.git_oid", "c6cc1014128b19d1fc46b1d30a23e3b1d35db421"},
    {"yvex.tokenizer.config.git_oid", "204d76f78dac6dedc820418c30bf01145de78a21"},
};
static const yvex_complete_artifact_admission text_catalog = {
    .artifact_class = YVEX_ARTIFACT_CLASS_COMPONENT_YVEX,
    .metadata_count = 34ull, .tensor_count = TEXT_TENSORS,
    .payload_bytes = TEXT_PAYLOAD_BYTES, .file_bytes = TEXT_FILE_BYTES,
    .source_snapshot_identity = VIDEO_SOURCE_SNAPSHOT_KEY,
    .mapping_identity = TEXT_MAPPING_IDENTITY,
    .payload_identity = VIDEO_PAYLOAD_IDENTITY,
    .transform_identity = TEXT_TRANSFORM_IDENTITY,
    .profile_identity = TEXT_PROFILE_IDENTITY,
    .profile_name = VIDEO_PROFILE_NAME,
    .quant_execution_identity = TEXT_QUANT_EXECUTION_IDENTITY,
    .payload_plan_identity = TEXT_PAYLOAD_PLAN_IDENTITY,
    .payload_byte_identity = TEXT_PAYLOAD_BYTE_IDENTITY,
    .writer_plan_identity = TEXT_WRITER_PLAN_IDENTITY,
    .artifact_identity = TEXT_ARTIFACT_IDENTITY,
    .official_reader_revision = YVEX_GGUF_OFFICIAL_READER_REVISION,
    .logical_target = YVEX_MINIMAX_H3_TARGET_ID,
    .logical_component = "text_encoder",
    .logical_component_identity = TEXT_COMPONENT_IDENTITY,
    .native_reader_accepted = 1, .official_reader_accepted = 1,
    .payload_integrity_accepted = 1, .materialization_input_ready = 1,
};
static const yvex_artifact_component_storage text_storage[] = {
    {YVEX_GGUF_QTYPE_BF16, TEXT_TENSORS},
};
static const yvex_artifact_component_contract text_contract = {
    &text_catalog, text_metadata, text_storage,
    sizeof(text_metadata) / sizeof(text_metadata[0]), 1ull,
    TEXT_ELEMENTS, 32ull,
};
static const yvex_artifact_component_metadata transformer_metadata[] = {
    {"general.architecture", "minimax-h3"}, {"general.name", "transformer"},
    {"yvex.logical.target", YVEX_MINIMAX_H3_TARGET_ID},
    {"yvex.logical.component", "transformer"},
    {"yvex.source.snapshot.identity", VIDEO_SOURCE_SNAPSHOT_IDENTITY},
    {"yvex.logical.component.identity", TRANSFORMER_COMPONENT_IDENTITY},
    {"yvex.logical.component_manifest.identity", VIDEO_COMPONENT_MANIFEST_IDENTITY},
    {"yvex.logical.architecture.identity", VIDEO_ARCHITECTURE_IDENTITY},
    {"yvex.logical.role_map.identity", VIDEO_ROLE_MAP_IDENTITY},
    {"yvex.logical.unresolved_requirements.identity", VIDEO_UNRESOLVED_IDENTITY},
    {"yvex.transformation.identity", TRANSFORMER_TRANSFORM_IDENTITY},
    {"yvex.physical.profile.name", VIDEO_PROFILE_NAME},
    {"yvex.physical.profile.identity", TRANSFORMER_PROFILE_IDENTITY},
    {"yvex.physical.payload_plan.identity", TRANSFORMER_PAYLOAD_PLAN_IDENTITY},
    {"yvex.payload.identity", VIDEO_PAYLOAD_IDENTITY},
    {"yvex.evidence.stage", "component-artifact-planned"},
};
static const yvex_complete_artifact_admission transformer_catalog = {
    .artifact_class = YVEX_ARTIFACT_CLASS_COMPONENT_YVEX,
    .metadata_count = 17ull, .tensor_count = TRANSFORMER_TENSORS,
    .payload_bytes = TRANSFORMER_PAYLOAD_BYTES, .file_bytes = TRANSFORMER_FILE_BYTES,
    .source_snapshot_identity = VIDEO_SOURCE_SNAPSHOT_KEY,
    .mapping_identity = TRANSFORMER_MAPPING_IDENTITY,
    .payload_identity = VIDEO_PAYLOAD_IDENTITY,
    .transform_identity = TRANSFORMER_TRANSFORM_IDENTITY,
    .profile_identity = TRANSFORMER_PROFILE_IDENTITY,
    .profile_name = VIDEO_PROFILE_NAME,
    .quant_execution_identity = TRANSFORMER_QUANT_EXECUTION_IDENTITY,
    .payload_plan_identity = TRANSFORMER_PAYLOAD_PLAN_IDENTITY,
    .payload_byte_identity = TRANSFORMER_PAYLOAD_BYTE_IDENTITY,
    .writer_plan_identity = TRANSFORMER_WRITER_PLAN_IDENTITY,
    .artifact_identity = TRANSFORMER_ARTIFACT_IDENTITY,
    .official_reader_revision = YVEX_GGUF_OFFICIAL_READER_REVISION,
    .logical_target = YVEX_MINIMAX_H3_TARGET_ID,
    .logical_component = "transformer",
    .logical_component_identity = TRANSFORMER_COMPONENT_IDENTITY,
    .native_reader_accepted = 1, .official_reader_accepted = 1,
    .payload_integrity_accepted = 1, .materialization_input_ready = 1,
};
static const yvex_artifact_component_storage transformer_storage[] = {
    {YVEX_GGUF_QTYPE_F32, 13ull}, {YVEX_GGUF_QTYPE_BF16, 522ull},
};
static const yvex_artifact_component_contract transformer_contract = {
    &transformer_catalog, transformer_metadata, transformer_storage,
    sizeof(transformer_metadata) / sizeof(transformer_metadata[0]), 2ull,
    TRANSFORMER_ELEMENTS, 32ull,
};
static const yvex_artifact_component_metadata video_metadata[] = {
    {"general.architecture", "minimax-h3"},
    {"general.name", "video_vae"},
    {"yvex.logical.target", YVEX_MINIMAX_H3_TARGET_ID},
    {"yvex.logical.component", "video_vae"},
    {"yvex.source.snapshot.identity", VIDEO_SOURCE_SNAPSHOT_IDENTITY},
    {"yvex.logical.component.identity", VIDEO_COMPONENT_IDENTITY},
    {"yvex.logical.component_manifest.identity", VIDEO_COMPONENT_MANIFEST_IDENTITY},
    {"yvex.logical.architecture.identity", VIDEO_ARCHITECTURE_IDENTITY},
    {"yvex.logical.role_map.identity", VIDEO_ROLE_MAP_IDENTITY},
    {"yvex.logical.unresolved_requirements.identity", VIDEO_UNRESOLVED_IDENTITY},
    {"yvex.transformation.identity", VIDEO_TRANSFORM_IDENTITY},
    {"yvex.physical.profile.name", VIDEO_PROFILE_NAME},
    {"yvex.physical.profile.identity", VIDEO_PROFILE_IDENTITY},
    {"yvex.physical.payload_plan.identity", VIDEO_PAYLOAD_PLAN_IDENTITY},
    {"yvex.payload.identity", VIDEO_PAYLOAD_IDENTITY},
    {"yvex.evidence.stage", "component-artifact-planned"},
    {"yvex.physical.shape.policy", "preserve-leading-three-fold-trailing-v1"},
};
static const yvex_complete_artifact_admission video_catalog = {
    .artifact_class = YVEX_ARTIFACT_CLASS_COMPONENT_YVEX,
    .metadata_count = 18ull,
    .tensor_count = VIDEO_TENSORS,
    .payload_bytes = VIDEO_PAYLOAD_BYTES,
    .file_bytes = VIDEO_FILE_BYTES,
    .source_snapshot_identity = VIDEO_SOURCE_SNAPSHOT_KEY,
    .mapping_identity = VIDEO_MAPPING_IDENTITY,
    .payload_identity = VIDEO_PAYLOAD_IDENTITY,
    .transform_identity = VIDEO_TRANSFORM_IDENTITY,
    .profile_identity = VIDEO_PROFILE_IDENTITY,
    .profile_name = VIDEO_PROFILE_NAME,
    .quant_execution_identity = VIDEO_QUANT_EXECUTION_IDENTITY,
    .payload_plan_identity = VIDEO_PAYLOAD_PLAN_IDENTITY,
    .payload_byte_identity = VIDEO_PAYLOAD_BYTE_IDENTITY,
    .writer_plan_identity = VIDEO_WRITER_PLAN_IDENTITY,
    .artifact_identity = VIDEO_ARTIFACT_IDENTITY,
    .official_reader_revision = YVEX_GGUF_OFFICIAL_READER_REVISION,
    .logical_target = YVEX_MINIMAX_H3_TARGET_ID,
    .logical_component = "video_vae",
    .logical_component_identity = VIDEO_COMPONENT_IDENTITY,
    .native_reader_accepted = 1,
    .official_reader_accepted = 1,
    .payload_integrity_accepted = 1,
    .materialization_input_ready = 1,
};
static const yvex_artifact_component_storage video_storage[] = {
    {YVEX_GGUF_QTYPE_F32, VIDEO_TENSORS},
};
static const yvex_artifact_component_contract video_contract = {
    &video_catalog, video_metadata, video_storage,
    sizeof(video_metadata) / sizeof(video_metadata[0]), 1ull,
    VIDEO_ELEMENTS, 32ull,
};
static int component_admit(const char *component, const yvex_artifact *artifact,
                           const yvex_gguf *gguf, const yvex_tensor_table *tensors,
                           yvex_complete_artifact_admission *out,
                           yvex_artifact_admission_failure *failure, yvex_error *err)
{
    const yvex_artifact_component_contract *contract = NULL;
    if (component && strcmp(component, "audio_vae") == 0) contract = &audio_contract;
    if (component && strcmp(component, "video_vae") == 0) contract = &video_contract;
    if (component && strcmp(component, "text_encoder") == 0) contract = &text_contract;
    if (component && strcmp(component, "transformer") == 0) contract = &transformer_contract;
    if (!contract) {
        if (failure) {
            memset(failure, 0, sizeof(*failure));
            failure->code = YVEX_ARTIFACT_ADMISSION_INVALID_ARGUMENT;
            yvex_core_text_copy(failure->field, sizeof(failure->field), "component");
        }
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "graph.minimax_h3.component",
                       "unknown MiniMax-H3 weighted component");
        return YVEX_ERR_INVALID_ARG;
    }
    return yvex_artifact_admit_component(
        artifact, gguf, tensors, contract, out, failure, err);
}
static int component_execute_artifact_cpu(
    const char *component, int video, const yvex_artifact *artifact,
    const yvex_gguf *gguf, const yvex_tensor_table *tensors,
    const void *options, void *result,
    yvex_minimax_h3_component_execution_failure *failure, yvex_error *err)
{
    yvex_complete_artifact_admission admission;
    yvex_artifact_admission_failure admission_failure;
    yvex_runtime_component_session *session = NULL;
    int admitted = 0, rc, cleanup_rc;
    yvex_error cleanup;
    rc = component_admit(component, artifact, gguf, tensors,
                         &admission, &admission_failure, err);
    admitted = rc == YVEX_OK;
    if (rc == YVEX_OK)
        rc = yvex_runtime_component_session_open(
            &session, &admission, artifact, gguf, tensors, YVEX_BACKEND_KIND_CPU,
            admission.payload_bytes, 0ull, err);
    if (rc == YVEX_OK && video)
        rc = video_vae_decode_cpu(
            yvex_runtime_component_session_materialization(session),
            (const yvex_minimax_h3_video_decode_options *)options,
            (yvex_minimax_h3_video_decode_result *)result, failure, err);
    if (rc == YVEX_OK && !video)
        rc = audio_vae_decode_cpu(
            yvex_runtime_component_session_materialization(session),
            (const yvex_minimax_h3_audio_decode_options *)options,
            (yvex_minimax_h3_audio_decode_result *)result, failure, err);
    if (rc != YVEX_OK && failure &&
        failure->code == YVEX_MINIMAX_H3_COMPONENT_EXECUTION_NONE) {
        failure->code = admitted ? YVEX_MINIMAX_H3_COMPONENT_EXECUTION_MATERIALIZATION
                                 : YVEX_MINIMAX_H3_COMPONENT_EXECUTION_LIFECYCLE;
        failure->reason = yvex_error_message(err);
    }
    yvex_error_clear(&cleanup);
    cleanup_rc = yvex_runtime_component_session_close(&session, &cleanup);
    if (cleanup_rc != YVEX_OK) { rc = cleanup_rc; if (err) *err = cleanup; }
    return rc;
}
static int audio_vae_execute_artifact_cpu(
    const yvex_artifact *artifact, const yvex_gguf *gguf, const yvex_tensor_table *tensors,
    const yvex_minimax_h3_audio_decode_options *options,
    yvex_minimax_h3_audio_decode_result *result,
    yvex_minimax_h3_component_execution_failure *failure, yvex_error *err)
{
    if (result) memset(result, 0, sizeof(*result));
    if (failure) memset(failure, 0, sizeof(*failure));
    if (!artifact || !gguf || !tensors || !options || !result) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "graph.minimax_h3.audio_vae.execute",
                       "Audio VAE artifact execution requires structural inputs and output state");
        return YVEX_ERR_INVALID_ARG;
    }
    return component_execute_artifact_cpu(
        "audio_vae", 0, artifact, gguf, tensors, options, result, failure, err);
}
static int video_vae_execute_artifact_cpu(
    const yvex_artifact *artifact, const yvex_gguf *gguf, const yvex_tensor_table *tensors,
    const yvex_minimax_h3_video_decode_options *options,
    yvex_minimax_h3_video_decode_result *result,
    yvex_minimax_h3_component_execution_failure *failure, yvex_error *err)
{
    if (result) memset(result, 0, sizeof(*result));
    if (failure) memset(failure, 0, sizeof(*failure));
    if (!artifact || !gguf || !tensors || !options || !result) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "graph.minimax_h3.video_vae.execute",
                       "Visual VAE artifact execution requires structural inputs and output state");
        return YVEX_ERR_INVALID_ARG;
    }
    return component_execute_artifact_cpu(
        "video_vae", 1, artifact, gguf, tensors, options, result, failure, err);
}
static int video_vae_execute_artifact_cuda(
    const yvex_artifact *artifact, const yvex_gguf *gguf,
    const yvex_tensor_table *tensors,
    const yvex_minimax_h3_video_decode_options *options,
    unsigned long long maximum_device_bytes,
    yvex_minimax_h3_video_decode_result *result,
    yvex_minimax_h3_component_execution_failure *failure, yvex_error *err)
{
    yvex_complete_artifact_admission admission;
    yvex_artifact_admission_failure admission_failure;
    yvex_runtime_component_session *session = NULL;
    yvex_transformer_resident_decoder_request request = {0};
    yvex_transformer_dense_decoder_result decoder = {0};
    video_execution execution = {0};
    component_buffer hidden = {0}, cosines = {0}, sines = {0}, patch_output = {0};
    const yvex_runtime_residency_summary *summary = NULL;
    int admitted = 0, rc, cleanup_rc;
    yvex_error cleanup;
    if (result) memset(result, 0, sizeof(*result));
    if (failure) memset(failure, 0, sizeof(*failure));
    execution.options = options;
    execution.result = result;
    execution.failure = failure;
    execution.err = err;
    if (!artifact || !gguf || !tensors || !options || !result ||
        !maximum_device_bytes)
        return video_execution_refuse(
            &execution, YVEX_MINIMAX_H3_COMPONENT_EXECUTION_INVALID_ARGUMENT,
            NULL, 6ull, 0ull, YVEX_ERR_INVALID_ARG,
            "Visual VAE CUDA execution requires artifact, geometry, and device budget");
    rc = component_admit(
        "video_vae", artifact, gguf, tensors, &admission, &admission_failure, err);
    admitted = rc == YVEX_OK;
    if (rc == YVEX_OK)
        rc = yvex_runtime_component_session_open(
            &session, &admission, artifact, gguf, tensors, YVEX_BACKEND_KIND_CUDA,
            admission.payload_bytes, maximum_device_bytes, err);
    if (rc == YVEX_OK) {
        execution.session = yvex_runtime_component_session_materialization(session);
        summary = yvex_runtime_component_session_summary(session);
        rc = video_decode_validate(&execution);
    }
    if (rc == YVEX_OK) rc = video_prefix_prepare(&execution, &hidden);
    if (rc == YVEX_OK)
        rc = video_buffer_open(&execution, &cosines, execution.rows * 48ull);
    if (rc == YVEX_OK)
        rc = video_buffer_open(&execution, &sines, execution.rows * 48ull);
    if (rc == YVEX_OK)
        rc = video_buffer_open(&execution, &patch_output, execution.patches * 3072ull);
    if (rc == YVEX_OK) {
        rc = video_rope_tables(&execution, cosines.data, sines.data);
    }
    if (rc == YVEX_OK) {
        request.block_weight_name = video_dense_weight_name;
        request.block_weight_name_context = &execution;
        request.final_norm_weight_name = "decoder.norm_out.weight";
        request.final_norm_bias_name = "decoder.norm_out.bias";
        request.output_weight_name = "decoder.proj_out.weight";
        request.output_bias_name = "decoder.proj_out.bias";
        request.execution.hidden = hidden.data;
        request.execution.cosines = cosines.data;
        request.execution.sines = sines.data;
        request.execution.rows = execution.rows;
        request.execution.output_rows = execution.patches;
        request.execution.width = 2048ull;
        request.execution.heads = 32ull;
        request.execution.head_dim = 64ull;
        request.execution.rotary_dim = 48ull;
        request.execution.ffn_width = 8192ull;
        request.execution.block_count = 36ull;
        request.execution.output_width = 3072ull;
        request.execution.output_capacity = patch_output.count;
        request.execution.epsilon = 1.0e-5f;
        request.execution.output = patch_output.data;
        request.execution.cancel_requested = options->cancelled;
        request.execution.cancel_context = options->cancellation_context;
        rc = yvex_runtime_component_dense_decoder_cuda(
            session, &request, &decoder, err);
    }
    if (rc == YVEX_OK) video_output_unpack(&patch_output, options, result);
    if (rc == YVEX_OK &&
        !video_execution_identity("yvex.minimax-h3.video-vae.cuda-f32.v1",
                                  yvex_materialization_session_summary(execution.session),
                                  options, result))
        rc = video_execution_refuse(
            &execution, YVEX_MINIMAX_H3_COMPONENT_EXECUTION_NUMERIC,
            NULL, 1ull, 0ull, YVEX_ERR_STATE,
            "Visual VAE CUDA execution identity could not be sealed");
    if (rc == YVEX_OK) {
        result->kernel_launches = decoder.kernel_launches;
        result->h2d_bytes = decoder.h2d_bytes;
        result->d2h_bytes = decoder.d2h_bytes;
        result->device_bytes = decoder.device_bytes;
        yvex_core_text_copy(result->residency_identity,
                            sizeof(result->residency_identity),
                            summary->residency_identity);
        result->complete = 1;
        yvex_error_clear(err);
    } else if (failure && failure->code == YVEX_MINIMAX_H3_COMPONENT_EXECUTION_NONE) {
        failure->code = admitted ? YVEX_MINIMAX_H3_COMPONENT_EXECUTION_MATERIALIZATION
                                 : YVEX_MINIMAX_H3_COMPONENT_EXECUTION_LIFECYCLE;
        failure->reason = yvex_error_message(err);
    }
    video_buffer_close(&execution, &patch_output);
    video_buffer_close(&execution, &sines);
    video_buffer_close(&execution, &cosines);
    video_buffer_close(&execution, &hidden);
    yvex_error_clear(&cleanup);
    cleanup_rc = yvex_runtime_component_session_close(&session, &cleanup);
    if (cleanup_rc != YVEX_OK) { rc = cleanup_rc; if (err) *err = cleanup; }
    return rc;
}
static const char *const text_layer_weight_suffixes[YVEX_MINIMAX_H3_TEXT_LAYER_WEIGHT_COUNT] = {
    "input_layernorm.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
    "self_attn.v_proj.weight", "self_attn.o_proj.weight", "self_attn.q_norm.weight",
    "self_attn.k_norm.weight", "post_attention_layernorm.weight", "mlp.gate_proj.weight",
    "mlp.up_proj.weight", "mlp.down_proj.weight",
};
static int text_layer_weights_bind(
    const yvex_materialization_session *session, const yvex_runtime_residency *residency,
    yvex_minimax_h3_encoded_weight *weights, unsigned long long layer_count, yvex_error *err)
{
    unsigned long long index, layer, slot;
    char name[160];
    int rc = component_weight_bind(session, residency,
                                   "model.language_model.embed_tokens.weight", weights, err);
    for (layer = 0ull; layer < layer_count; ++layer) {
        for (index = 0ull; rc == YVEX_OK &&
                           index < YVEX_MINIMAX_H3_TEXT_LAYER_WEIGHT_COUNT; ++index) {
            int length = snprintf(name, sizeof(name), "model.language_model.layers.%llu.%s",
                                  layer, text_layer_weight_suffixes[index]);
            slot = 1ull + layer * YVEX_MINIMAX_H3_TEXT_LAYER_WEIGHT_COUNT + index;
            if (length < 0 || (size_t)length >= sizeof(name)) {
                yvex_error_set(err, YVEX_ERR_BOUNDS, "minimax-h3.text-layer.name",
                               "a Qwen layer binding name exceeded its bounded representation");
                return YVEX_ERR_BOUNDS;
            }
            rc = component_weight_bind(session, residency, name, weights + slot, err);
        }
    }
    return rc;
}
static int text_encoder_artifact_cuda(const yvex_artifact *artifact,
    const yvex_gguf *gguf, const yvex_tensor_table *tensors,
    const unsigned int *token_ids, unsigned long long token_count,
    unsigned long long layer_count,
    float *output, unsigned long long output_capacity,
    unsigned long long maximum_host_bytes, unsigned long long maximum_device_bytes,
    yvex_minimax_h3_conditioning_result *result, yvex_error *err)
{
    const yvex_minimax_h3_backend_api *backend = yvex_backend_register_minimax_h3();
    const yvex_materialized_tensor_binding *embedding = NULL;
    yvex_minimax_h3_encoded_weight *weights = NULL;
    yvex_complete_artifact_admission admission;
    yvex_artifact_admission_failure admission_failure;
    yvex_runtime_component_session *component_session = NULL;
    yvex_materialization_session *session = NULL;
    const yvex_runtime_residency_summary *residency_summary = NULL;
    const yvex_runtime_residency *residency = NULL;
    yvex_backend *cuda = NULL;
    const unsigned char *encoded = NULL;
    unsigned long long encoded_bytes = 0ull, output_values = 0ull, output_bytes = 0ull;
    unsigned long long weight_count = 0ull;
    float *staged = NULL;
    yvex_minimax_h3_conditioning_result published = {0};
    int rc, cleanup_rc;
    yvex_error cleanup;
    if (result) memset(result, 0, sizeof(*result));
    if (!artifact || !gguf || !tensors || !backend || !backend->text_embed_cuda ||
        !backend->text_layer_cuda || !token_ids || !token_count || !output || !result ||
        layer_count > YVEX_MINIMAX_H3_TEXT_CONDITIONING_LAYERS ||
        !yvex_core_u64_mul(token_count, 5120ull, &output_values) ||
        output_values > output_capacity ||
        !yvex_core_u64_mul(output_values, sizeof(float), &output_bytes) ||
        output_bytes > SIZE_MAX) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "minimax-h3.text-conditioning",
                       "exact component inputs, bounded output, and CUDA backend are required");
        return YVEX_ERR_INVALID_ARG;
    }
    staged = (float *)malloc((size_t)output_bytes);
    if (!staged) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "minimax-h3.text-conditioning.output",
                       "transactional graph output allocation failed");
        return YVEX_ERR_NOMEM;
    }
    if (layer_count &&
        (!yvex_core_u64_mul(layer_count, YVEX_MINIMAX_H3_TEXT_LAYER_WEIGHT_COUNT,
                            &weight_count) ||
         !yvex_core_u64_add(weight_count, 1ull, &weight_count) ||
         weight_count > SIZE_MAX / sizeof(*weights) ||
         !(weights = (yvex_minimax_h3_encoded_weight *)calloc((size_t)weight_count,
                                                            sizeof(*weights))))) {
        free(staged);
        yvex_error_set(err, YVEX_ERR_NOMEM, "minimax-h3.text-conditioning.weights",
                       "bounded Qwen conditioning weight bindings could not be allocated");
        return YVEX_ERR_NOMEM;
    }
    rc = component_admit(
        "text_encoder", artifact, gguf, tensors, &admission, &admission_failure, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_component_session_open(
            &component_session, &admission, artifact, gguf, tensors, YVEX_BACKEND_KIND_CUDA,
            maximum_host_bytes, maximum_device_bytes, err);
    if (rc == YVEX_OK) {
        session = yvex_runtime_component_session_materialization(component_session);
        residency = yvex_runtime_component_session_residency(component_session);
        residency_summary = yvex_runtime_component_session_summary(component_session);
        cuda = yvex_runtime_component_session_backend(component_session);
    }
    embedding = rc == YVEX_OK
        ? component_binding_find(session, "model.language_model.embed_tokens.weight") : NULL;
    if (rc == YVEX_OK && !embedding) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "minimax-h3.text-conditioning.embedding",
                       "admitted text component lacks its embedding binding");
        rc = YVEX_ERR_FORMAT;
    }
    if (rc == YVEX_OK && !layer_count)
        rc = yvex_runtime_residency_binding_view(
            residency, embedding, &encoded, &encoded_bytes, err);
    if (rc == YVEX_OK && layer_count)
        rc = text_layer_weights_bind(session, residency, weights, layer_count, err);
    if (rc == YVEX_OK && !layer_count)
        rc = backend->text_embed_cuda(
            cuda, encoded, encoded_bytes, embedding->qtype, embedding->row_count,
            embedding->row_width, embedding->encoded_bytes / embedding->row_count,
            residency_summary->residency_identity, residency_summary->encoded_bytes,
            token_ids, token_count, staged, output_values, &published, err);
    if (rc == YVEX_OK && layer_count)
        rc = backend->text_layer_cuda(
            cuda, weights, layer_count, residency_summary->residency_identity,
            residency_summary->encoded_bytes, token_ids, token_count, staged,
            output_values, &published, err);
    yvex_error_clear(&cleanup);
    cleanup_rc = yvex_runtime_component_session_close(&component_session, &cleanup);
    if (cleanup_rc != YVEX_OK) { rc = cleanup_rc; if (err) *err = cleanup; }
    if (rc == YVEX_OK) {
        memcpy(output, staged, (size_t)output_bytes);
        *result = published;
        yvex_error_clear(err);
    }
    free(staged);
    free(weights);
    return rc;
}
static const char *const transformer_external_names[YVEX_MINIMAX_H3_OMNI_EXTERNAL_WEIGHT_COUNT] = {
    "audio_patch_proj.weight", "audio_patch_proj.bias", "video_patch_proj.weight",
    "video_patch_proj.bias", "condition_proj.weight", "condition_proj.bias",
    "time_embedder.proj_in.weight", "time_embedder.proj_in.bias",
    "time_embedder.proj_out.weight", "time_embedder.proj_out.bias",
    "token_refiner.blocks.0.norm1.weight", "token_refiner.blocks.0.attn.qkv_proj.weight",
    "token_refiner.blocks.0.attn.q_norm.weight", "token_refiner.blocks.0.attn.k_norm.weight",
    "token_refiner.blocks.0.attn.out_proj.weight", "token_refiner.blocks.0.norm2.weight",
    "token_refiner.blocks.0.mlp.fc1.weight", "token_refiner.blocks.0.mlp.fc2.weight",
    "token_refiner.blocks.1.norm1.weight", "token_refiner.blocks.1.attn.qkv_proj.weight",
    "token_refiner.blocks.1.attn.q_norm.weight", "token_refiner.blocks.1.attn.k_norm.weight",
    "token_refiner.blocks.1.attn.out_proj.weight", "token_refiner.blocks.1.norm2.weight",
    "token_refiner.blocks.1.mlp.fc1.weight", "token_refiner.blocks.1.mlp.fc2.weight",
    "token_refiner.final_norm.weight", "rope.inv_freq", "final_layer.norm.weight",
    "final_layer.adaln_proj.linear.weight", "final_layer.adaln_proj.linear.bias",
    "final_layer.video_out.weight", "final_layer.video_out.bias",
    "final_layer.audio_out.weight", "final_layer.audio_out.bias",
};
static const char *const transformer_block_suffixes[YVEX_MINIMAX_H3_OMNI_BLOCK_WEIGHT_COUNT] = {
    "norm1.weight", "attn.qkv_proj.weight", "attn.q_norm.weight", "attn.k_norm.weight",
    "attn.out_proj.weight", "norm2.weight", "mlp.fc1.weight", "mlp.fc2.weight",
    "adaln_proj.linear.weight", "adaln_proj.linear.bias",
};
static int transformer_weights_bind(const yvex_materialization_session *session,
    const yvex_runtime_residency *residency, yvex_minimax_h3_encoded_weight *external,
    yvex_minimax_h3_encoded_weight *blocks, unsigned long long block_count, yvex_error *err)
{
    unsigned long long index, block; char name[96]; int rc = YVEX_OK;
    for (index = 0ull; rc == YVEX_OK && index < YVEX_MINIMAX_H3_OMNI_EXTERNAL_WEIGHT_COUNT; ++index)
        rc = component_weight_bind(session, residency, transformer_external_names[index],
                                   external + index, err);
    for (block = 0ull; rc == YVEX_OK && block < block_count; ++block)
        for (index = 0ull; rc == YVEX_OK && index < YVEX_MINIMAX_H3_OMNI_BLOCK_WEIGHT_COUNT; ++index) {
            int length = snprintf(name, sizeof(name), "blocks.%llu.%s", block,
                                  transformer_block_suffixes[index]);
            if (length < 0 || (size_t)length >= sizeof(name)) {
                yvex_error_set(err, YVEX_ERR_BOUNDS, "minimax-h3.transformer.binding",
                               "a Transformer block binding name exceeded its bound");
                return YVEX_ERR_BOUNDS;
            }
            rc = component_weight_bind(session, residency, name,
                blocks + block * YVEX_MINIMAX_H3_OMNI_BLOCK_WEIGHT_COUNT + index, err);
        }
    return rc;
}
static int transformer_component_cuda(yvex_runtime_component_session *session,
    const yvex_minimax_h3_omni_transformer_request *request,
    yvex_minimax_h3_omni_transformer_result *result, yvex_error *err)
{
    const yvex_minimax_h3_backend_api *api = yvex_backend_register_minimax_h3();
    const yvex_runtime_residency_summary *summary;
    yvex_minimax_h3_encoded_weight external[YVEX_MINIMAX_H3_OMNI_EXTERNAL_WEIGHT_COUNT] = {{0}};
    yvex_minimax_h3_encoded_weight *blocks = NULL;
    unsigned long long count = 0ull;
    int rc;
    if (result) memset(result, 0, sizeof(*result));
    if (!session || !request || !result || !api || !api->omni_transformer_cuda ||
        !request->block_count || request->block_count > 50ull ||
        !yvex_core_u64_mul(request->block_count, YVEX_MINIMAX_H3_OMNI_BLOCK_WEIGHT_COUNT, &count) ||
        count > SIZE_MAX / sizeof(*blocks) ||
        !(blocks = calloc((size_t)count, sizeof(*blocks)))) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "minimax-h3.transformer.session",
                       "a resident component and bounded Transformer request are required");
        return YVEX_ERR_INVALID_ARG;
    }
    summary = yvex_runtime_component_session_summary(session);
    rc = transformer_weights_bind(yvex_runtime_component_session_materialization(session),
        yvex_runtime_component_session_residency(session), external, blocks,
        request->block_count, err);
    if (rc == YVEX_OK)
        rc = api->omni_transformer_cuda(yvex_runtime_component_session_backend(session),
            external, blocks, summary->residency_identity, summary->encoded_bytes,
            request, result, err);
    free(blocks);
    return rc;
}
typedef struct {
    const yvex_minimax_h3_t2va_plan *plan;
    const yvex_minimax_h3_t2va_omni_context *context;
    yvex_runtime_latent_evaluator_evidence evidence;
} t2va_omni_execution;
static int t2va_omni_identity(const yvex_minimax_h3_t2va_plan *plan,
    const yvex_minimax_h3_t2va_omni_context *context,
    const yvex_runtime_residency_summary *summary, char output[65], yvex_error *err)
{
    const char *identities[4] = {plan->identity, summary->residency_identity,
                                 context->conditioning_identity,
                                 context->layout_result->layout_identity};
    unsigned long long facts[2] = {context->block_count, summary->encoded_bytes};
    return yvex_runtime_latent_binding_identity(
        "yvex.minimax-h3.t2va.omni-evaluator.v1", identities, 4ull,
        facts, 2ull, output, err);
}
static int t2va_omni_evaluate(void *opaque, const float *video,
    unsigned long long video_values, const float *audio, unsigned long long audio_values,
    float video_timestep, float audio_timestep, float *video_velocity,
    float *audio_velocity, yvex_error *err)
{
    t2va_omni_execution *execution = opaque;
    const yvex_minimax_h3_t2va_plan *plan = execution ? execution->plan : NULL;
    const yvex_minimax_h3_t2va_omni_context *context = execution ? execution->context : NULL;
    yvex_minimax_h3_omni_transformer_request request = {0};
    yvex_minimax_h3_omni_transformer_result result = {0};
    float timesteps[2]; unsigned long long row, expected_video, expected_audio;
    int rc;
    if (!plan || !context || video_timestep > audio_timestep ||
        !yvex_core_u64_mul(plan->video_rows, plan->video_value_width, &expected_video) ||
        !yvex_core_u64_mul(plan->audio_rows, plan->audio_value_width, &expected_audio) ||
        video_values != expected_video || audio_values != expected_audio) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "graph.minimax_h3.t2va.omni",
                       "latent evaluation does not match the admitted FL2VA plan");
        return YVEX_ERR_FORMAT;
    }
    timesteps[0] = video_timestep; timesteps[1] = audio_timestep;
    for (row = 0ull; row < plan->packed_rows; ++row) {
        unsigned int tag = context->layout->token_tags[row];
        if (tag != plan->text_tag && tag != plan->audio_tag && tag != plan->video_tag) {
            yvex_error_set(err, YVEX_ERR_FORMAT, "graph.minimax_h3.t2va.omni",
                           "packed FL2VA row has an unknown modality tag");
            return YVEX_ERR_FORMAT;
        }
        context->timestep_indices[row] =
            tag == plan->audio_tag && audio_timestep != video_timestep ? 1u : 0u;
    }
    request.video = video; request.audio = audio; request.conditioning = context->conditioning;
    request.timesteps = timesteps; request.position_ids = context->layout->position_ids;
    request.video_indices = context->layout->video_indices;
    request.audio_indices = context->layout->audio_indices;
    request.text_indices = context->layout->text_indices;
    request.timestep_indices = context->timestep_indices;
    request.token_tags = context->layout->token_tags;
    request.video_rows = plan->video_rows; request.audio_rows = plan->audio_rows;
    request.text_rows = plan->text_tokens; request.packed_rows = plan->packed_rows;
    request.timestep_count = audio_timestep == video_timestep ? 1ull : 2ull;
    request.block_count = context->block_count; request.video_output = video_velocity;
    request.audio_output = audio_velocity; request.video_output_capacity = video_values;
    request.audio_output_capacity = audio_values;
    rc = transformer_component_cuda(context->transformer_session, &request, &result, err);
    if (rc != YVEX_OK) return rc;
    return yvex_runtime_latent_evaluator_record(
        &execution->evidence, result.residency_identity, result.execution_identity,
        result.kernel_launches, result.h2d_bytes, result.d2h_bytes, result.device_bytes, err);
}
static int t2va_latent_execute(const yvex_minimax_h3_t2va_plan *plan,
    const yvex_minimax_h3_t2va_omni_context *context, unsigned long long seed,
    unsigned long long maximum_workspace_bytes, float *video, unsigned long long video_capacity,
    float *audio, unsigned long long audio_capacity, yvex_runtime_latent_result *latent_result,
    yvex_minimax_h3_t2va_omni_result *omni_result, yvex_error *err)
{
    const yvex_runtime_residency_summary *summary;
    yvex_runtime_latent_request request = {0}; t2va_omni_execution execution = {0};
    unsigned long long conditioning_values;
    int rc;
    if (latent_result) memset(latent_result, 0, sizeof(*latent_result));
    if (omni_result) memset(omni_result, 0, sizeof(*omni_result));
    rc = yvex_runtime_av_layout_matches_plan(
        plan, context ? context->layout : NULL, context ? context->layout_result : NULL, err);
    if (rc != YVEX_OK) return rc;
    summary = yvex_runtime_component_session_summary(context->transformer_session);
    if (!latent_result || !omni_result || !summary || !summary->sealed ||
        !summary->cuda_ready || summary->invalidated ||
        !yvex_sha256_hex_valid(summary->residency_identity) || !context->conditioning ||
        !context->conditioning_identity || !yvex_sha256_hex_valid(context->conditioning_identity) ||
        !yvex_core_u64_mul(plan->text_tokens, 5120ull, &conditioning_values) ||
        context->conditioning_capacity < conditioning_values || !context->timestep_indices ||
        context->timestep_capacity < plan->packed_rows || !context->block_count ||
        context->block_count > 50ull) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "graph.minimax_h3.t2va.latent",
                       "one exact resident Transformer and packed FL2VA layout are required");
        return YVEX_ERR_INVALID_ARG;
    }
    execution.plan = plan; execution.context = context;
    rc = t2va_omni_identity(plan, context, summary, execution.evidence.staged.evaluator_identity, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_latent_evaluator_begin(
            &execution.evidence, "yvex.minimax-h3.t2va.transformer-chain.v1",
            execution.evidence.staged.evaluator_identity, err);
    request.seed = seed; request.maximum_workspace_bytes = maximum_workspace_bytes;
    request.evaluator_identity = execution.evidence.staged.evaluator_identity;
    request.evaluate = t2va_omni_evaluate; request.execution_context = &execution;
    request.cancel_requested = context->cancelled; request.cancel_context = context->cancellation_context;
    if (rc == YVEX_OK)
        rc = yvex_runtime_av_latent_execute(plan, &request, video, video_capacity,
                                             audio, audio_capacity, latent_result, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_latent_evaluator_finish(
            &execution.evidence, plan->model_evaluations, omni_result, err);
    if (rc != YVEX_OK) memset(latent_result, 0, sizeof(*latent_result));
    return rc;
}
const yvex_minimax_h3_graph_api *yvex_graph_register_minimax_h3(void)
{
    static const yvex_minimax_h3_graph_api api = {
        t2va_plan_build, yvex_runtime_av_scheduler_step,
        t2va_latent_execute, yvex_runtime_av_layout_from_plan,
        component_admit, text_encoder_artifact_cuda,
        transformer_component_cuda,
        audio_vae_decode_cpu, audio_vae_execute_artifact_cpu,
        video_vae_decode_cpu, video_vae_execute_artifact_cpu,
        video_vae_execute_artifact_cuda,
    };
    return &api;
}
