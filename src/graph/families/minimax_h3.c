/*
 * Bind MiniMax-H3 component execution recipes to exact physical inputs.
 *
 * Component metadata remains insufficient until the complete file identity and independently
 * derived payload identity agree with this family-owned physical boundary. Numerical execution
 * consumes this admission later; it cannot infer source or model identity from tensor names.
 */
#include <yvex/internal/artifact.h>
#include <yvex/internal/families/minimax_h3.h>
#include <yvex/internal/runtime.h>
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
static int t2va_plan_build(yvex_minimax_h3_t2va_plan *out,
                           unsigned long long text_tokens, unsigned long long width,
                           unsigned long long height, unsigned long long frames,
                           yvex_error *err)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long frame_blocks, spatial_rows;
    unsigned int step;
    if (!out || !text_tokens || width < 32ull || height < 32ull ||
        width % 32ull || height % 32ull || frames < 5ull ||
        (frames - 5ull) % 17ull) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "graph.minimax_h3.t2va.plan",
                       "t2va requires text, 32-aligned geometry, and a 17k+5 frame count");
        return YVEX_ERR_INVALID_ARG;
    }
    memset(out, 0, sizeof(*out));
    out->text_tokens = text_tokens;
    out->frames = frames;
    out->width = width;
    out->height = height;
    frame_blocks = (frames - 5ull) / 17ull;
    if (!yvex_core_u64_mul(frame_blocks, 5ull, &out->video_latent_frames) ||
        !yvex_core_u64_add(out->video_latent_frames, 2ull,
                           &out->video_latent_frames) ||
        !yvex_core_u64_mul(frames, 5ull, &out->audio_latent_steps) ||
        !yvex_core_u64_add(out->audio_latent_steps, 1ull,
                           &out->audio_latent_steps)) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "graph.minimax_h3.t2va.plan",
                       "t2va temporal geometry overflowed");
        return YVEX_ERR_BOUNDS;
    }
    out->audio_latent_steps /= 3ull;
    out->video_latent_height = height / 16ull;
    out->video_latent_width = width / 16ull;
    if (!yvex_core_u64_mul(out->video_latent_height / 2ull,
                           out->video_latent_width / 2ull, &spatial_rows) ||
        !yvex_core_u64_mul(spatial_rows, out->video_latent_frames,
                           &out->video_rows) ||
        !yvex_core_u64_mul(out->audio_latent_steps, 2ull, &out->audio_rows) ||
        !yvex_core_u64_add(text_tokens, out->audio_rows, &out->packed_rows) ||
        !yvex_core_u64_add(out->packed_rows, out->video_rows, &out->packed_rows)) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "graph.minimax_h3.t2va.plan",
                       "t2va packed sequence overflowed");
        return YVEX_ERR_BOUNDS;
    }
    out->sigma_grid_points = 20u;
    out->model_evaluations = out->sigma_grid_points - 1u;
    for (step = 0u; step < out->sigma_grid_points; ++step) {
        float base = (float)(out->sigma_grid_points - 1u - step) /
                     (float)(out->sigma_grid_points - 1u);
        out->video_sigmas[step] = 12.0f * base / (1.0f + 11.0f * base);
        out->audio_sigmas[step] = 3.0f * base / (1.0f + 2.0f * base);
    }
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.minimax-h3.t2va.res-multistep.v2") ||
        !yvex_sha256_update_text(&hash, YVEX_MINIMAX_H3_TARGET_ID) ||
        !yvex_sha256_update_text(&hash, YVEX_MINIMAX_H3_REVISION) ||
        !yvex_sha256_update_u64_be(&hash, text_tokens) ||
        !yvex_sha256_update_u64_be(&hash, width) ||
        !yvex_sha256_update_u64_be(&hash, height) ||
        !yvex_sha256_update_u64_be(&hash, frames) ||
        !yvex_sha256_update_u64_be(&hash, out->sigma_grid_points) ||
        !yvex_sha256_update_u64_be(&hash, out->model_evaluations)) {
        yvex_error_set(err, YVEX_ERR_STATE, "graph.minimax_h3.t2va.plan",
                       "t2va plan identity construction failed");
        return YVEX_ERR_STATE;
    }
    for (step = 0u; step < out->sigma_grid_points; ++step) {
        uint32_t video_bits, audio_bits;
        memcpy(&video_bits, &out->video_sigmas[step], sizeof(video_bits));
        memcpy(&audio_bits, &out->audio_sigmas[step], sizeof(audio_bits));
        if (!yvex_sha256_update_u64_be(&hash, video_bits) ||
            !yvex_sha256_update_u64_be(&hash, audio_bits)) {
            yvex_error_set(err, YVEX_ERR_STATE, "graph.minimax_h3.t2va.plan",
                           "t2va schedule identity construction failed");
            return YVEX_ERR_STATE;
        }
    }
    if (!yvex_sha256_final(&hash, digest)) {
        yvex_error_set(err, YVEX_ERR_STATE, "graph.minimax_h3.t2va.plan",
                       "t2va plan identity finalization failed");
        return YVEX_ERR_STATE;
    }
    yvex_sha256_hex(digest, out->identity);
    out->complete = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}
/* Advance one modality while retaining H3's data-ward velocity sign and F32 blend order. */
static int scheduler_step(float *output, const float *sample, const float *velocity,
                          unsigned long long values, float timestep, float sigma,
                          float sigma_next, yvex_error *err)
{
    unsigned long long index;
    if (!output || !sample || !velocity || !values || !isfinite(timestep) ||
        !isfinite(sigma) || !isfinite(sigma_next) || timestep < 0.0f ||
        timestep >= 1.0f || sigma <= 0.0f || sigma_next < 0.0f ||
        sigma_next >= sigma) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "graph.minimax_h3.scheduler.step",
                       "H3 scheduler step requires finite state and a decreasing sigma interval");
        return YVEX_ERR_INVALID_ARG;
    }
    for (index = 0ull; index < values; ++index)
        if (!isfinite(sample[index]) || !isfinite(velocity[index])) {
            yvex_error_set(err, YVEX_ERR_FORMAT, "graph.minimax_h3.scheduler.step",
                           "H3 scheduler input contains a non-finite value");
            return YVEX_ERR_FORMAT;
        }
    for (index = 0ull; index < values; ++index) {
        float denoised = sample[index] + (1.0f - timestep) * velocity[index];
        float ratio = sigma_next / sigma;
        output[index] = ratio * sample[index] + (1.0f - ratio) * denoised;
    }
    yvex_error_clear(err);
    return YVEX_OK;
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
    unsigned long long bytes;
    unsigned long long next_live;
    memset(buffer, 0, sizeof(*buffer));
    if (!count || !yvex_core_u64_mul(count, sizeof(float), &bytes) ||
        bytes > (unsigned long long)SIZE_MAX ||
        !yvex_core_u64_add(execution->live_workspace_bytes, bytes, &next_live))
        return audio_execution_refuse(execution, YVEX_MINIMAX_H3_COMPONENT_EXECUTION_BUDGET,
                                      NULL, 1ull, count, YVEX_ERR_BOUNDS,
                                      "Audio VAE workspace extent overflowed");
    if (next_live > execution->options->max_workspace_bytes)
        return audio_execution_refuse(execution, YVEX_MINIMAX_H3_COMPONENT_EXECUTION_BUDGET,
                                      NULL, execution->options->max_workspace_bytes,
                                      next_live, YVEX_ERR_BOUNDS,
                                      "Audio VAE workspace budget was exceeded");
    buffer->data = (float *)malloc((size_t)bytes);
    if (!buffer->data)
        return audio_execution_refuse(execution, YVEX_MINIMAX_H3_COMPONENT_EXECUTION_BUDGET,
                                      NULL, bytes, 0ull, YVEX_ERR_NOMEM,
                                      "Audio VAE workspace allocation failed");
    buffer->count = count;
    execution->live_workspace_bytes = next_live;
    if (next_live > execution->result->peak_workspace_bytes)
        execution->result->peak_workspace_bytes = next_live;
    return YVEX_OK;
}
static void audio_buffer_close(audio_execution *execution, component_buffer *buffer)
{
    unsigned long long bytes = buffer->count * sizeof(float);
    if (bytes <= execution->live_workspace_bytes)
        execution->live_workspace_bytes -= bytes;
    free(buffer->data);
    memset(buffer, 0, sizeof(*buffer));
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
static int audio_tensor_load(audio_execution *execution, const char *name,
                             unsigned int rank, const unsigned long long *dims,
                             component_buffer *buffer)
{
    const yvex_materialized_tensor_binding *binding =
        component_binding_find(execution->session, name);
    yvex_materialization_failure materialization_failure;
    unsigned long long count = 1ull;
    unsigned long long expected_bytes = 0ull;
    unsigned int dimension;
    int rc;
    if (!binding)
        return audio_execution_refuse(execution,
                                      YVEX_MINIMAX_H3_COMPONENT_EXECUTION_MISSING_TENSOR,
                                      name, 1ull, 0ull, YVEX_ERR_FORMAT,
                                      "Audio VAE execution tensor is missing");
    if (binding->qtype != YVEX_GGUF_QTYPE_F32 || binding->rank != rank)
        return audio_execution_refuse(execution,
                                      YVEX_MINIMAX_H3_COMPONENT_EXECUTION_TENSOR_CONTRACT,
                                      name, rank, binding->rank, YVEX_ERR_FORMAT,
                                      "Audio VAE execution tensor rank or dtype differs");
    for (dimension = 0u; dimension < rank; ++dimension) {
        if (binding->dims[dimension] != dims[dimension] ||
            !yvex_core_u64_mul(count, dims[dimension], &count))
            return audio_execution_refuse(execution,
                                          YVEX_MINIMAX_H3_COMPONENT_EXECUTION_TENSOR_CONTRACT,
                                          name, dims[dimension], binding->dims[dimension],
                                          YVEX_ERR_FORMAT,
                                          "Audio VAE execution tensor shape differs");
    }
    if (!yvex_core_u64_mul(count, sizeof(float), &expected_bytes) ||
        binding->encoded_bytes != expected_bytes)
        return audio_execution_refuse(execution,
                                      YVEX_MINIMAX_H3_COMPONENT_EXECUTION_TENSOR_CONTRACT,
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
            execution->failure->code = YVEX_MINIMAX_H3_COMPONENT_EXECUTION_MATERIALIZATION;
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
    unsigned long long bytes, next_live;
    memset(buffer, 0, sizeof(*buffer));
    if (!count || !yvex_core_u64_mul(count, sizeof(float), &bytes) ||
        bytes > (unsigned long long)SIZE_MAX ||
        !yvex_core_u64_add(execution->live_workspace_bytes, bytes, &next_live))
        return video_execution_refuse(
            execution, YVEX_MINIMAX_H3_COMPONENT_EXECUTION_BUDGET,
            NULL, 1ull, count, YVEX_ERR_BOUNDS,
            "Visual VAE workspace extent overflowed");
    if (next_live > execution->options->max_workspace_bytes)
        return video_execution_refuse(
            execution, YVEX_MINIMAX_H3_COMPONENT_EXECUTION_BUDGET,
            NULL, execution->options->max_workspace_bytes, next_live,
            YVEX_ERR_BOUNDS, "Visual VAE workspace budget was exceeded");
    buffer->data = (float *)malloc((size_t)bytes);
    if (!buffer->data)
        return video_execution_refuse(
            execution, YVEX_MINIMAX_H3_COMPONENT_EXECUTION_BUDGET,
            NULL, bytes, 0ull, YVEX_ERR_NOMEM,
            "Visual VAE workspace allocation failed");
    buffer->count = count;
    execution->live_workspace_bytes = next_live;
    if (next_live > execution->result->peak_workspace_bytes)
        execution->result->peak_workspace_bytes = next_live;
    return YVEX_OK;
}
static void video_buffer_close(video_execution *execution, component_buffer *buffer)
{
    unsigned long long bytes = buffer->count * sizeof(float);
    if (bytes <= execution->live_workspace_bytes)
        execution->live_workspace_bytes -= bytes;
    free(buffer->data);
    memset(buffer, 0, sizeof(*buffer));
}
static int video_tensor_load(video_execution *execution, const char *name,
                             unsigned int rank, const unsigned long long *dims,
                             component_buffer *buffer)
{
    const yvex_materialized_tensor_binding *binding =
        component_binding_find(execution->session, name);
    yvex_materialization_failure materialization_failure;
    unsigned long long count = 1ull, expected_bytes;
    unsigned int dimension;
    int rc;
    if (!binding)
        return video_execution_refuse(
            execution, YVEX_MINIMAX_H3_COMPONENT_EXECUTION_MISSING_TENSOR,
            name, 1ull, 0ull, YVEX_ERR_FORMAT,
            "Visual VAE execution tensor is missing");
    if (binding->qtype != YVEX_GGUF_QTYPE_F32 || binding->rank != rank)
        return video_execution_refuse(
            execution, YVEX_MINIMAX_H3_COMPONENT_EXECUTION_TENSOR_CONTRACT,
            name, rank, binding->rank, YVEX_ERR_FORMAT,
            "Visual VAE execution tensor rank or dtype differs");
    for (dimension = 0u; dimension < rank; ++dimension) {
        if (binding->dims[dimension] != dims[dimension] ||
            !yvex_core_u64_mul(count, dims[dimension], &count))
            return video_execution_refuse(
                execution, YVEX_MINIMAX_H3_COMPONENT_EXECUTION_TENSOR_CONTRACT,
                name, dims[dimension], binding->dims[dimension], YVEX_ERR_FORMAT,
                "Visual VAE execution tensor shape differs");
    }
    if (!yvex_core_u64_mul(count, sizeof(float), &expected_bytes) ||
        binding->encoded_bytes != expected_bytes)
        return video_execution_refuse(
            execution, YVEX_MINIMAX_H3_COMPONENT_EXECUTION_TENSOR_CONTRACT,
            name, expected_bytes, binding->encoded_bytes, YVEX_ERR_FORMAT,
            "Visual VAE execution tensor byte extent differs");
    rc = video_buffer_open(execution, buffer, count);
    if (rc != YVEX_OK) return rc;
    rc = yvex_materialization_session_read(
        execution->session, binding, 0ull, buffer->data,
        (size_t)binding->encoded_bytes, &materialization_failure, execution->err);
    if (rc != YVEX_OK) {
        video_buffer_close(execution, buffer);
        return video_execution_refuse(
            execution, YVEX_MINIMAX_H3_COMPONENT_EXECUTION_MATERIALIZATION,
            name, binding->encoded_bytes, materialization_failure.actual, rc,
            materialization_failure.reason);
    }
    execution->result->tensor_reads++;
    execution->result->payload_bytes_read += binding->encoded_bytes;
    return YVEX_OK;
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
    unsigned long long index;
    int rc = video_tensor_load(execution, name, 1u, dims, &scale);
    if (rc == YVEX_OK)
        for (index = 0ull; index < rows * width; ++index) {
            float value = hidden[index] + delta[index] * scale.data[index % width];
            if (!isfinite(value)) {
                rc = video_execution_refuse(
                    execution, YVEX_MINIMAX_H3_COMPONENT_EXECUTION_NUMERIC,
                    name, rows * width, index, YVEX_ERR_FORMAT,
                    "Visual VAE residual update produced a non-finite value");
                break;
            }
            hidden[index] = value;
        }
    video_buffer_close(execution, &scale);
    return rc;
}
static int video_qk_normalize(video_execution *execution, float *qkv,
                              unsigned long long rows)
{
    unsigned long long row, head;
    for (row = 0ull; row < rows; ++row)
        for (head = 0ull; head < 32ull; ++head) {
            float *base = qkv + row * 6144ull + head * 192ull;
            if (!yvex_attention_unit_rms_norm(base, 64ull, 1.0e-5) ||
                !yvex_attention_unit_rms_norm(base + 64ull, 64ull, 1.0e-5))
                return video_execution_refuse(
                    execution, YVEX_MINIMAX_H3_COMPONENT_EXECUTION_NUMERIC,
                    NULL, rows * 32ull, row * 32ull + head, YVEX_ERR_FORMAT,
                    "Visual VAE Q/K normalization produced a non-finite value");
        }
    return YVEX_OK;
}
static void video_rope_apply(const video_execution *execution, float *qkv)
{
    const yvex_minimax_h3_video_decode_options *options = execution->options;
    const float tau = 6.28318530717958647692f;
    unsigned long long token, head;
    for (token = 0ull; token < execution->rows; ++token) {
        float coordinates[3] = {0.0f, 0.0f, 0.0f};
        float cosine[24], sine[24];
        unsigned long long coordinate, frequency;
        if (token < execution->patches) {
            unsigned long long plane = options->latent_height * options->latent_width;
            unsigned long long temporal = token / plane;
            unsigned long long spatial = token % plane;
            unsigned long long height = spatial / options->latent_width;
            unsigned long long width = spatial % options->latent_width;
            coordinates[0] = 2.0f * ((float)temporal + 0.5f) /
                                 (float)options->latent_frames - 1.0f;
            coordinates[1] = 2.0f * ((float)height + 0.5f) /
                                 (float)options->latent_height - 1.0f;
            coordinates[2] = 2.0f * ((float)width + 0.5f) /
                                 (float)options->latent_width - 1.0f;
        }
        for (coordinate = 0ull; coordinate < 3ull; ++coordinate)
            for (frequency = 0ull; frequency < 8ull; ++frequency) {
                unsigned long long index = coordinate * 8ull + frequency;
                float angle = tau * coordinates[coordinate] *
                              powf(100.0f, -(float)frequency / 8.0f);
                cosine[index] = cosf(angle);
                sine[index] = sinf(angle);
            }
        for (head = 0ull; head < 32ull; ++head) {
            float *base = qkv + token * 6144ull + head * 192ull;
            unsigned long long kind;
            for (kind = 0ull; kind < 2ull; ++kind)
                for (coordinate = 0ull; coordinate < 3ull; ++coordinate)
                    for (frequency = 0ull; frequency < 8ull; ++frequency) {
                        unsigned long long index = coordinate * 8ull + frequency;
                        float *value = base + kind * 64ull;
                        float first = value[index], second = value[index + 24ull];
                        value[index] = first * cosine[index] - second * sine[index];
                        value[index + 24ull] = second * cosine[index] + first * sine[index];
                    }
        }
    }
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
    if (rc == YVEX_OK) rc = video_qk_normalize(execution, qkv->data, execution->rows);
    if (rc == YVEX_OK) video_rope_apply(execution, qkv->data);
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
static int video_execution_identity(const yvex_materialization_summary *summary,
                                    const yvex_minimax_h3_video_decode_options *options,
                                    yvex_minimax_h3_video_decode_result *result)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(
            &hash, result->output_values == 3072ull
                       ? "yvex.minimax-h3.video-vae.cpu.reduced-v1"
                       : "yvex.minimax-h3.video-vae.cpu.geometry-v2") ||
        !yvex_sha256_update_text(&hash, summary->artifact_identity) ||
        !yvex_sha256_update_u64_be(&hash, options->batch) ||
        !yvex_sha256_update_u64_be(&hash, options->latent_channels) ||
        !yvex_sha256_update_u64_be(&hash, options->latent_frames) ||
        !yvex_sha256_update_u64_be(&hash, options->latent_height) ||
        !yvex_sha256_update_u64_be(&hash, options->latent_width))
        return 0;
    for (index = 0ull; index < result->batch * options->latent_channels *
                                     options->latent_frames * options->latent_height *
                                     options->latent_width; ++index) {
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
    component_buffer packed = {0}, post = {0}, hidden = {0}, normalized = {0}, qkv = {0};
    component_buffer projected = {0}, fused = {0}, gated = {0};
    component_buffer attention = {0}, scratch = {0}, registers = {0}, patch_output = {0};
    unsigned long long register_dims[3] = {1ull, 4ull, 2048ull};
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
    if (rc == YVEX_OK) rc = video_buffer_open(&execution, &packed, execution.patches * 24ull);
    if (rc == YVEX_OK) rc = video_buffer_open(&execution, &post, execution.patches * 24ull);
    if (rc == YVEX_OK) rc = video_buffer_open(&execution, &hidden, execution.rows * 2048ull);
    if (rc == YVEX_OK) rc = video_buffer_open(&execution, &normalized, execution.rows * 2048ull);
    if (rc == YVEX_OK) rc = video_buffer_open(&execution, &qkv, execution.rows * 6144ull);
    if (rc == YVEX_OK) rc = video_buffer_open(&execution, &projected, execution.rows * 2048ull);
    if (rc == YVEX_OK) rc = video_buffer_open(&execution, &fused, execution.rows * 16384ull);
    if (rc == YVEX_OK) rc = video_buffer_open(&execution, &gated, execution.rows * 8192ull);
    if (rc == YVEX_OK) rc = video_buffer_open(&execution, &attention, execution.rows * 2048ull);
    if (rc == YVEX_OK) rc = video_buffer_open(&execution, &scratch, execution.rows);
    if (rc == YVEX_OK) video_latent_pack(options, execution.patches, packed.data);
    if (rc == YVEX_OK)
        rc = video_linear(&execution, "post_quant_conv", 4u, packed.data,
                          execution.patches, 24ull, 24ull, post.data);
    if (rc == YVEX_OK)
        rc = video_linear(&execution, "decoder.x_embedder", 2u, post.data,
                          execution.patches, 24ull, 2048ull, hidden.data);
    if (rc == YVEX_OK)
        rc = video_tensor_load(&execution, "decoder.register_tokens", 3u,
                               register_dims, &registers);
    if (rc == YVEX_OK) {
        memcpy(hidden.data + execution.patches * 2048ull, registers.data,
               (size_t)(4ull * 2048ull * sizeof(float)));
        memset(hidden.data + (execution.patches + 4ull) * 2048ull,
               0, (size_t)(2048ull * sizeof(float)));
    }
    video_buffer_close(&execution, &registers);
    video_buffer_close(&execution, &post);
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
        if (!video_execution_identity(summary, options, result))
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
    video_buffer_close(&execution, &post);
    video_buffer_close(&execution, &packed);
    if (rc == YVEX_OK) yvex_error_clear(err);
    return rc;
}
static const yvex_artifact_component_metadata audio_metadata[] = {
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
static const yvex_artifact_component_storage audio_storage[] = {
    {YVEX_GGUF_QTYPE_F32, YVEX_MINIMAX_H3_AUDIO_TENSORS},
};
static const yvex_artifact_component_contract audio_contract = {
    &audio_catalog, audio_metadata, audio_storage,
    sizeof(audio_metadata) / sizeof(audio_metadata[0]), 1ull,
    YVEX_MINIMAX_H3_AUDIO_ELEMENTS, 32ull,
};
static int component_admit(const char *component, const yvex_artifact *artifact,
                           const yvex_gguf *gguf, const yvex_tensor_table *tensors,
                           yvex_complete_artifact_admission *out,
                           yvex_artifact_admission_failure *failure, yvex_error *err);
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
static int audio_vae_execute_artifact_cpu(
    const yvex_artifact *artifact, const yvex_gguf *gguf,
    const yvex_tensor_table *tensors,
    const yvex_minimax_h3_audio_decode_options *options,
    yvex_minimax_h3_audio_decode_result *result,
    yvex_minimax_h3_component_execution_failure *failure, yvex_error *err)
{
    yvex_complete_artifact_admission admission;
    yvex_artifact_admission_failure admission_failure;
    yvex_materialization_options materialization_options;
    yvex_materialization_failure materialization_failure;
    yvex_materialization_plan *plan = NULL;
    yvex_materialization_session *session = NULL;
    yvex_runtime_residency *residency = NULL;
    yvex_runtime_residency_options residency_options = {0};
    yvex_runtime_residency_failure residency_failure;
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
            &execution, YVEX_MINIMAX_H3_COMPONENT_EXECUTION_INVALID_ARGUMENT,
            NULL, 5ull, 0ull, YVEX_ERR_INVALID_ARG,
            "Audio VAE artifact execution requires structural inputs and output state");
    rc = component_admit(
        "audio_vae", artifact, gguf, tensors, &admission, &admission_failure, err);
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
    residency_options.maximum_host_bytes = admission.payload_bytes;
    if (rc == YVEX_OK)
        rc = yvex_runtime_component_residency_prepare(
            &residency, session, admission.logical_component_identity,
            &residency_options, &residency_failure, err);
    if (rc == YVEX_OK)
        rc = audio_vae_decode_cpu(session, options, result, failure, err);
    if (rc != YVEX_OK && failure && failure->code == YVEX_MINIMAX_H3_COMPONENT_EXECUTION_NONE) {
        failure->code = admitted ? YVEX_MINIMAX_H3_COMPONENT_EXECUTION_MATERIALIZATION
                                 : YVEX_MINIMAX_H3_COMPONENT_EXECUTION_LIFECYCLE;
        failure->reason = yvex_error_message(err);
    }
    {
        yvex_error cleanup;
        int cleanup_rc = yvex_runtime_residency_close(&residency, &cleanup);
        if (cleanup_rc != YVEX_OK) {
            rc = cleanup_rc;
            if (err) *err = cleanup;
        }
    }
    yvex_materialization_session_close(session);
    yvex_materialization_plan_close(plan);
    return rc;
}
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
static int video_vae_execute_artifact_cpu(
    const yvex_artifact *artifact, const yvex_gguf *gguf,
    const yvex_tensor_table *tensors,
    const yvex_minimax_h3_video_decode_options *options,
    yvex_minimax_h3_video_decode_result *result,
    yvex_minimax_h3_component_execution_failure *failure, yvex_error *err)
{
    yvex_complete_artifact_admission admission;
    yvex_artifact_admission_failure admission_failure;
    yvex_materialization_options materialization_options;
    yvex_materialization_failure materialization_failure;
    yvex_materialization_plan *plan = NULL;
    yvex_materialization_session *session = NULL;
    yvex_runtime_residency *residency = NULL;
    yvex_runtime_residency_options residency_options = {0};
    yvex_runtime_residency_failure residency_failure;
    video_execution execution = {0};
    int admitted = 0;
    int rc;
    if (result) memset(result, 0, sizeof(*result));
    if (failure) memset(failure, 0, sizeof(*failure));
    execution.options = options;
    execution.result = result;
    execution.failure = failure;
    execution.err = err;
    if (!artifact || !gguf || !tensors || !options || !result)
        return video_execution_refuse(
            &execution, YVEX_MINIMAX_H3_COMPONENT_EXECUTION_INVALID_ARGUMENT,
            NULL, 5ull, 0ull, YVEX_ERR_INVALID_ARG,
            "Visual VAE artifact execution requires structural inputs and output state");
    rc = component_admit(
        "video_vae", artifact, gguf, tensors, &admission, &admission_failure, err);
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
    residency_options.maximum_host_bytes = admission.payload_bytes;
    if (rc == YVEX_OK)
        rc = yvex_runtime_component_residency_prepare(
            &residency, session, admission.logical_component_identity,
            &residency_options, &residency_failure, err);
    if (rc == YVEX_OK)
        rc = video_vae_decode_cpu(session, options, result, failure, err);
    if (rc != YVEX_OK && failure &&
        failure->code == YVEX_MINIMAX_H3_COMPONENT_EXECUTION_NONE) {
        failure->code = admitted ? YVEX_MINIMAX_H3_COMPONENT_EXECUTION_MATERIALIZATION
                                 : YVEX_MINIMAX_H3_COMPONENT_EXECUTION_LIFECYCLE;
        failure->reason = yvex_error_message(err);
    }
    {
        yvex_error cleanup;
        int cleanup_rc = yvex_runtime_residency_close(&residency, &cleanup);
        if (cleanup_rc != YVEX_OK) {
            rc = cleanup_rc;
            if (err) *err = cleanup;
        }
    }
    yvex_materialization_session_close(session);
    yvex_materialization_plan_close(plan);
    return rc;
}
static const char *const text_layer_weight_suffixes[YVEX_MINIMAX_H3_TEXT_LAYER_WEIGHT_COUNT] = {
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
};
static int text_layer_weights_bind(
    const yvex_materialization_session *session, const yvex_runtime_residency *residency,
    yvex_minimax_h3_encoded_weight *weights, unsigned long long layer_count, yvex_error *err)
{
    unsigned long long index, layer, slot = 0ull;
    char name[160];
    const yvex_materialized_tensor_binding *binding =
        component_binding_find(session, "model.language_model.embed_tokens.weight");
    if (!binding || !binding->row_count ||
        yvex_runtime_residency_binding_view(
            residency, binding, &weights[0].encoded, &weights[0].encoded_bytes, err) != YVEX_OK) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "minimax-h3.text-layer.binding",
                       "the exact Qwen embedding binding is unavailable");
        return YVEX_ERR_FORMAT;
    }
    weights[0].qtype = binding->qtype;
    weights[0].row_count = binding->row_count;
    weights[0].row_width = binding->row_width;
    weights[0].row_bytes = binding->encoded_bytes / binding->row_count;
    for (layer = 0ull; layer < layer_count; ++layer) {
        for (index = 0ull; index < YVEX_MINIMAX_H3_TEXT_LAYER_WEIGHT_COUNT; ++index) {
            int length = snprintf(name, sizeof(name), "model.language_model.layers.%llu.%s",
                                  layer, text_layer_weight_suffixes[index]);
            slot = 1ull + layer * YVEX_MINIMAX_H3_TEXT_LAYER_WEIGHT_COUNT + index;
            if (length < 0 || (size_t)length >= sizeof(name)) {
                yvex_error_set(err, YVEX_ERR_BOUNDS, "minimax-h3.text-layer.name",
                               "a Qwen layer binding name exceeded its bounded representation");
                return YVEX_ERR_BOUNDS;
            }
            binding = component_binding_find(session, name);
            if (!binding || !binding->row_count ||
                yvex_runtime_residency_binding_view(
                    residency, binding, &weights[slot].encoded,
                    &weights[slot].encoded_bytes, err) != YVEX_OK) {
                yvex_error_set(err, YVEX_ERR_FORMAT, "minimax-h3.text-layer.binding",
                               "an exact Qwen BF16 layer binding is unavailable");
                return YVEX_ERR_FORMAT;
            }
            weights[slot].qtype = binding->qtype;
            weights[slot].row_count = binding->row_count;
            weights[slot].row_width = binding->row_width;
            weights[slot].row_bytes = binding->encoded_bytes / binding->row_count;
        }
    }
    return YVEX_OK;
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
    yvex_materialization_options options;
    yvex_materialization_failure failure;
    yvex_materialization_plan *plan = NULL;
    yvex_materialization_session *session = NULL;
    yvex_runtime_residency_options residency_options = {0};
    yvex_runtime_residency_failure residency_failure;
    yvex_runtime_residency_summary residency_summary;
    yvex_runtime_residency *residency = NULL;
    yvex_backend_options backend_options = {0};
    yvex_backend *cuda = NULL;
    const unsigned char *encoded = NULL;
    unsigned long long encoded_bytes = 0ull, output_values = 0ull, output_bytes = 0ull;
    unsigned long long weight_count = 0ull;
    float *staged = NULL;
    yvex_minimax_h3_conditioning_result published = {0};
    int uploaded = 0, rc, cleanup_rc, residency_close_rc;
    yvex_error cleanup, residency_cleanup;
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
    backend_options.kind = YVEX_BACKEND_KIND_CUDA;
    backend_options.memory_limit_bytes = maximum_device_bytes;
    rc = yvex_backend_open(&cuda, &backend_options, err);
    if (rc == YVEX_OK)
        rc = component_admit(
            "text_encoder", artifact, gguf, tensors, &admission, &admission_failure, err);
    yvex_materialization_options_default(&options);
    options.max_chunk_bytes = 64ull * 1024ull * 1024ull;
    if (rc == YVEX_OK)
        rc = yvex_materialization_plan_build(
            &plan, &admission, artifact, gguf, tensors, NULL, &options, &failure, err);
    if (rc == YVEX_OK)
        rc = yvex_materialization_session_open(
            &session, plan, artifact, &options, &failure, err);
    if (rc == YVEX_OK) rc = yvex_materialization_session_commit(session, &failure, err);
    embedding = rc == YVEX_OK
        ? component_binding_find(session, "model.language_model.embed_tokens.weight") : NULL;
    if (rc == YVEX_OK && !embedding) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "minimax-h3.text-conditioning.embedding",
                       "admitted text component lacks its embedding binding");
        rc = YVEX_ERR_FORMAT;
    }
    residency_options.maximum_host_bytes = maximum_host_bytes;
    if (rc == YVEX_OK)
        rc = yvex_runtime_component_residency_prepare(
            &residency, session, admission.logical_component_identity,
            &residency_options, &residency_failure, err);
    if (rc == YVEX_OK && !layer_count)
        rc = yvex_runtime_residency_binding_view(
            residency, embedding, &encoded, &encoded_bytes, err);
    if (rc == YVEX_OK && layer_count)
        rc = text_layer_weights_bind(session, residency, weights, layer_count, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_residency_cuda_session_attach(
            residency, &cuda, maximum_device_bytes, &uploaded, &residency_summary, err);
    if (rc == YVEX_OK && !layer_count)
        rc = backend->text_embed_cuda(
            cuda, encoded, encoded_bytes, embedding->qtype, embedding->row_count,
            embedding->row_width, embedding->encoded_bytes / embedding->row_count,
            residency_summary.residency_identity, residency_summary.encoded_bytes,
            token_ids, token_count, staged, output_values, &published, err);
    if (rc == YVEX_OK && layer_count)
        rc = backend->text_layer_cuda(
            cuda, weights, layer_count, residency_summary.residency_identity,
            residency_summary.encoded_bytes, token_ids, token_count, staged,
            output_values, &published, err);
    yvex_error_clear(&cleanup);
    cleanup_rc = yvex_backend_close_checked(&cuda, &cleanup);
    yvex_error_clear(&residency_cleanup);
    residency_close_rc = yvex_runtime_residency_close(&residency, &residency_cleanup);
    if (cleanup_rc == YVEX_OK && residency_close_rc != YVEX_OK) {
        cleanup_rc = residency_close_rc;
        cleanup = residency_cleanup;
    }
    if (cleanup_rc != YVEX_OK) {
        rc = cleanup_rc;
        if (err) *err = cleanup;
    }
    yvex_materialization_session_close(session);
    yvex_materialization_plan_close(plan);
    if (rc == YVEX_OK) {
        memcpy(output, staged, (size_t)output_bytes);
        *result = published;
        yvex_error_clear(err);
    }
    free(staged);
    free(weights);
    return rc;
}
const yvex_minimax_h3_graph_api *yvex_graph_register_minimax_h3(void)
{
    static const yvex_minimax_h3_graph_api api = {
        t2va_plan_build, scheduler_step, component_admit, text_encoder_artifact_cuda,
        audio_vae_decode_cpu, audio_vae_execute_artifact_cpu,
        video_vae_decode_cpu, video_vae_execute_artifact_cpu,
    };
    return &api;
}
