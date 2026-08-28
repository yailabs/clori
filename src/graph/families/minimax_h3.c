/* Bind MiniMax-H3 execution recipes only after exact component payload identity is admitted. */
#include <yvex/internal/artifact.h>
#include <yvex/internal/compilation.h>
#include <yvex/internal/compiler.h>
#include <yvex/internal/component.h>
#include <yvex/internal/convolution.h>
#include <yvex/internal/family_catalog.h>
#include <yvex/internal/families/minimax_h3.h>
#include <yvex/internal/joint_transformer.h>
#include <yvex/internal/latent.h>
#include <yvex/internal/runtime.h>
#include <yvex/internal/tokenizer.h>
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
typedef yvex_component_f32_buffer component_buffer;
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
        .source_revision = YVEX_SOURCE_MINIMAX_H3_REVISION,
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
static const yvex_alias_decoder_name_templates audio_decoder_names = {
    "dec_in_proj", "decoder.conv_pre", "decoder.ups", "decoder.resblocks",
    "decoder.activation_post", "decoder.conv_post",
};
static const yvex_alias_decoder_recipe audio_decoder_recipe = {
    .input_channels = 32ull, .projection_channels = 2048ull, .input_kernel = 1ull,
    .pre_channels = 1024ull, .pre_kernel = 7ull,
    .stage_count = 7ull, .residual_blocks = 3ull, .residual_layers = 3ull,
    .rates = {5ull, 5ull, 2ull, 2ull, 2ull, 2ull, 2ull},
    .upsample_kernels = {9ull, 9ull, 4ull, 4ull, 4ull, 4ull, 4ull},
    .residual_kernels = {3ull, 7ull, 11ull},
    .residual_dilations = {1ull, 3ull, 5ull},
    .final_channels = 1ull, .final_kernel = 7ull,
};
static int audio_execution_identity(const char *domain,
                                    const yvex_materialization_summary *summary,
                                    const yvex_minimax_h3_audio_decode_options *options,
                                    yvex_minimax_h3_audio_decode_result *result)
{
    unsigned long long geometry[3] = {
        options->batch, options->latent_channels, options->latent_steps,
    };
    return yvex_graph_f32_execution_identity(
        domain, summary->artifact_identity, geometry, 3ull, options->latent,
        options->batch * options->latent_channels * options->latent_steps,
        options->output, result->output_values, result->execution_identity);
}
static int audio_decode_validate(audio_execution *execution)
{
    const yvex_materialization_summary *summary =
        execution && execution->session
            ? yvex_materialization_session_summary(execution->session) : NULL;
    unsigned long long latent_values = 0ull, output_values = 0ull, index;
    if (!execution || !execution->options || !execution->result ||
        !execution->options->latent || !execution->options->output ||
        !execution->options->batch || !execution->options->latent_steps ||
        execution->options->latent_channels != 32ull ||
        !execution->options->max_workspace_bytes)
        return audio_execution_refuse(
            execution, YVEX_MINIMAX_H3_COMPONENT_EXECUTION_INVALID_ARGUMENT, NULL, 32ull,
            execution && execution->options ? execution->options->latent_channels : 0ull,
            YVEX_ERR_INVALID_ARG,
            "Audio VAE decode requires exact latent geometry and bounded output");
    if (!summary || !summary->committed ||
        strcmp(summary->artifact_identity, YVEX_MINIMAX_H3_AUDIO_ARTIFACT_IDENTITY) != 0)
        return audio_execution_refuse(
            execution, YVEX_MINIMAX_H3_COMPONENT_EXECUTION_LIFECYCLE, NULL, 1ull,
            summary ? (unsigned long long)summary->committed : 0ull, YVEX_ERR_STATE,
            "Audio VAE decode requires the committed exact component artifact");
    if (!yvex_core_u64_mul(execution->options->batch,
                           execution->options->latent_channels, &latent_values) ||
        !yvex_core_u64_mul(latent_values, execution->options->latent_steps, &latent_values) ||
        !yvex_core_u64_mul(execution->options->batch,
                           execution->options->latent_steps, &output_values) ||
        !yvex_core_u64_mul(output_values, 800ull, &output_values) ||
        execution->options->output_capacity < output_values)
        return audio_execution_refuse(
            execution, YVEX_MINIMAX_H3_COMPONENT_EXECUTION_BUDGET, NULL, output_values,
            execution->options->output_capacity, YVEX_ERR_BOUNDS,
            "Audio VAE output buffer is smaller than the exact 800x ratio");
    for (index = 0ull; index < latent_values; ++index)
        if (!isfinite(execution->options->latent[index]))
            return audio_execution_refuse(
                execution, YVEX_MINIMAX_H3_COMPONENT_EXECUTION_NUMERIC, NULL,
                latent_values, index, YVEX_ERR_FORMAT,
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
    audio_execution execution = {
        .session = session, .options = options, .result = result,
        .failure = failure, .err = err,
    };
    yvex_alias_decoder_request request = {0};
    yvex_alias_decoder_result decoder = {0};
    int rc;
    if (result) memset(result, 0, sizeof(*result));
    if (failure) memset(failure, 0, sizeof(*failure));
    rc = audio_decode_validate(&execution);
    if (rc != YVEX_OK) return rc;
    request.recipe = &audio_decoder_recipe;
    request.input = options->latent;
    request.batch = options->batch;
    request.input_length = options->latent_steps;
    request.input_count = options->batch * options->latent_channels * options->latent_steps;
    request.output = options->output;
    request.output_capacity = options->output_capacity;
    request.maximum_workspace_bytes = options->max_workspace_bytes;
    request.weight_name = yvex_alias_decoder_template_name;
    request.weight_name_context = (void *)&audio_decoder_names;
    request.cancel_requested = options->cancelled;
    request.cancel_context = options->cancellation_context;
    rc = yvex_runtime_alias_decoder_execute_cpu(session, &request, &decoder, failure, err);
    if (rc == YVEX_OK) {
        const yvex_materialization_summary *summary =
            yvex_materialization_session_summary(session);
        result->tensor_reads = decoder.tensor_reads;
        result->payload_bytes_read = decoder.payload_bytes_read;
        result->peak_workspace_bytes = decoder.peak_host_bytes;
        if (!audio_execution_identity(
                "yvex.minimax-h3.audio-vae.cpu.v1", summary, options, result))
            rc = audio_execution_refuse(
                &execution, YVEX_MINIMAX_H3_COMPONENT_EXECUTION_NUMERIC,
                NULL, 1ull, 0ull, YVEX_ERR_STATE,
                "Audio VAE execution identity could not be sealed");
    }
    if (rc == YVEX_OK) {
        result->complete = 1;
        yvex_error_clear(err);
    }
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
    int rc = yvex_component_buffer_open(
        buffer, count, execution->options->max_workspace_bytes,
        &execution->live_workspace_bytes, &execution->result->peak_workspace_bytes,
        "graph.minimax_h3.video_vae.execute", "Visual VAE", execution->err);
    if (rc == YVEX_OK) return rc;
    return video_execution_refuse(
        execution, YVEX_MINIMAX_H3_COMPONENT_EXECUTION_BUDGET, NULL,
        execution->options->max_workspace_bytes, count, (yvex_status)rc,
        yvex_error_message(execution->err));
}
static void video_buffer_close(video_execution *execution, component_buffer *buffer)
{
    yvex_component_buffer_close(buffer, &execution->live_workspace_bytes);
}
static int video_tensor_load(video_execution *execution, const char *name,
                             unsigned int rank, const unsigned long long *dims,
                             component_buffer *buffer)
{
    yvex_component_load_failure issue = {0};
    int rc = yvex_component_f32_load(
        execution->session, name, rank, dims, buffer,
        execution->options->max_workspace_bytes, &execution->live_workspace_bytes,
        &execution->result->peak_workspace_bytes, &execution->result->tensor_reads,
        &execution->result->payload_bytes_read, &issue,
        "graph.minimax_h3.video_vae.execute", "Visual VAE", execution->err);
    if (rc != YVEX_OK && execution->failure) {
        execution->failure->code = issue.code ? issue.code + 2u
                                               : YVEX_COMPONENT_EXECUTION_TENSOR_CONTRACT;
        execution->failure->expected = issue.expected;
        execution->failure->actual = issue.actual;
        execution->failure->reason = issue.reason ? issue.reason
                                                   : yvex_error_message(execution->err);
        yvex_core_text_copy(execution->failure->tensor_name,
                            sizeof(execution->failure->tensor_name), name);
    }
    return rc;
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
static const yvex_artifact_catalog_contract audio_contract = {
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
static const yvex_artifact_catalog_contract text_contract = {
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
static const yvex_artifact_catalog_contract transformer_contract = {
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
static const yvex_artifact_catalog_contract video_contract = {
    &video_catalog, video_metadata, video_storage,
    sizeof(video_metadata) / sizeof(video_metadata[0]), 1ull,
    VIDEO_ELEMENTS, 32ull,
};
static int component_admit(
    const char *component, const yvex_artifact *artifact, const yvex_gguf *gguf,
    const yvex_tensor_table *tensors, const yvex_artifact_admission_options *options,
    yvex_complete_artifact_admission *out, yvex_artifact_admission_evidence *evidence,
    yvex_artifact_admission_failure *failure, yvex_error *err)
{
    const yvex_artifact_catalog_contract *contract = NULL;
    yvex_artifact_admission_evidence local_evidence;
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
    return yvex_artifact_admit_catalog_with_options(
        artifact, gguf, tensors, contract, options, out, evidence ? evidence : &local_evidence,
        failure, err);
}
static int component_binding_refuse(yvex_component_failure *failure,
                                    yvex_component_failure_code code,
                                    yvex_status status, const char *reason,
                                    yvex_error *err)
{
    if (failure) {
        memset(failure, 0, sizeof(*failure));
        failure->code = code;
        failure->reason = reason;
    }
    yvex_error_set(err, status, "graph.minimax_h3.component.plan", reason);
    return status;
}
static int audio_component_plan(const yvex_component_plan_request *request,
                                yvex_component_plan *out,
                                yvex_component_failure *failure, yvex_error *err)
{
    unsigned long long values;
    if (!request || !out || !request->batch || request->geometry_rank != 1u ||
        !request->geometry[0])
        return component_binding_refuse(
            failure, YVEX_COMPONENT_FAILURE_INVALID_ARGUMENT, YVEX_ERR_INVALID_ARG,
            "Audio VAE requires one latent-step geometry", err);
    if (!yvex_core_u64_mul(request->batch, 32ull, &values) ||
        !yvex_core_u64_mul(values, request->geometry[0], &out->input_values) ||
        !yvex_core_u64_mul(request->batch, request->geometry[0], &values) ||
        !yvex_core_u64_mul(values, 800ull, &out->output_values))
        return component_binding_refuse(
            failure, YVEX_COMPONENT_FAILURE_BUDGET, YVEX_ERR_BOUNDS,
            "Audio VAE geometry overflowed", err);
    out->output_rank = 3u;
    out->output_dims[0] = request->batch;
    out->output_dims[1] = 1ull;
    out->output_dims[2] = request->geometry[0] * 800ull;
    return YVEX_OK;
}
static int video_component_plan(const yvex_component_plan_request *request,
                                yvex_component_plan *out,
                                yvex_component_failure *failure, yvex_error *err)
{
    unsigned long long patches;
    if (!request || !out || request->batch != 1ull || request->geometry_rank != 3u ||
        !request->geometry[0] || !request->geometry[1] || !request->geometry[2])
        return component_binding_refuse(
            failure, YVEX_COMPONENT_FAILURE_INVALID_ARGUMENT, YVEX_ERR_INVALID_ARG,
            "Visual VAE requires batch one and three-dimensional latent geometry", err);
    if (!yvex_core_u64_mul(request->geometry[0], request->geometry[1], &patches) ||
        !yvex_core_u64_mul(patches, request->geometry[2], &patches) ||
        !yvex_core_u64_mul(patches, 24ull, &out->input_values) ||
        !yvex_core_u64_mul(patches, 3072ull, &out->output_values))
        return component_binding_refuse(
            failure, YVEX_COMPONENT_FAILURE_BUDGET, YVEX_ERR_BOUNDS,
            "Visual VAE geometry overflowed", err);
    out->output_rank = 5u;
    out->output_dims[0] = 1ull;
    out->output_dims[1] = 3ull;
    out->output_dims[2] = request->geometry[0] * 4ull;
    out->output_dims[3] = request->geometry[1] * 16ull;
    out->output_dims[4] = request->geometry[2] * 16ull;
    return YVEX_OK;
}
static int audio_component_execute(yvex_materialization_session *session,
                                   const yvex_component_execution_request *request,
                                   yvex_component_execution_result *out,
                                   yvex_component_failure *failure, yvex_error *err)
{
    yvex_minimax_h3_audio_decode_options options = {0};
    yvex_minimax_h3_audio_decode_result result;
    yvex_minimax_h3_component_execution_failure legacy = {0};
    int rc;
    if (!request || !request->plan)
        return component_binding_refuse(
            failure, YVEX_COMPONENT_FAILURE_INVALID_ARGUMENT, YVEX_ERR_INVALID_ARG,
            "Audio VAE execution requires an admitted plan", err);
    options.latent = request->input;
    options.batch = request->plan->batch;
    options.latent_channels = 32ull;
    options.latent_steps = request->plan->geometry[0];
    options.output = request->output;
    options.output_capacity = request->output_capacity;
    options.max_workspace_bytes = request->plan->workspace_bytes;
    options.cancelled = request->cancelled;
    options.cancellation_context = request->cancellation_context;
    rc = audio_vae_decode_cpu(session, &options, &result, &legacy, err);
    if (rc != YVEX_OK) return rc;
    out->batch = result.batch;
    out->output_values = result.output_values;
    out->output_rank = request->plan->output_rank;
    memcpy(out->output_dims, request->plan->output_dims, sizeof(out->output_dims));
    out->tensor_reads = result.tensor_reads;
    out->payload_bytes_read = result.payload_bytes_read;
    out->peak_workspace_bytes = result.peak_workspace_bytes;
    yvex_core_text_copy(out->artifact_identity, sizeof(out->artifact_identity),
                        result.artifact_identity);
    yvex_core_text_copy(out->execution_identity, sizeof(out->execution_identity),
                        result.execution_identity);
    out->complete = result.complete;
    return YVEX_OK;
}
static int video_component_execute(yvex_materialization_session *session,
                                   const yvex_component_execution_request *request,
                                   yvex_component_execution_result *out,
                                   yvex_component_failure *failure, yvex_error *err)
{
    yvex_minimax_h3_video_decode_options options = {0};
    yvex_minimax_h3_video_decode_result result;
    yvex_minimax_h3_component_execution_failure legacy = {0};
    int rc;
    if (!request || !request->plan)
        return component_binding_refuse(
            failure, YVEX_COMPONENT_FAILURE_INVALID_ARGUMENT, YVEX_ERR_INVALID_ARG,
            "Visual VAE execution requires an admitted plan", err);
    options.latent = request->input;
    options.output = request->output;
    options.batch = request->plan->batch;
    options.latent_channels = 24ull;
    options.latent_frames = request->plan->geometry[0];
    options.latent_height = request->plan->geometry[1];
    options.latent_width = request->plan->geometry[2];
    options.output_capacity = request->output_capacity;
    options.max_workspace_bytes = request->plan->workspace_bytes;
    options.cancelled = request->cancelled;
    options.cancellation_context = request->cancellation_context;
    rc = video_vae_decode_cpu(session, &options, &result, &legacy, err);
    if (rc != YVEX_OK) return rc;
    out->batch = result.batch;
    out->output_values = result.output_values;
    out->output_rank = request->plan->output_rank;
    memcpy(out->output_dims, request->plan->output_dims, sizeof(out->output_dims));
    out->tensor_reads = result.tensor_reads;
    out->payload_bytes_read = result.payload_bytes_read;
    out->peak_workspace_bytes = result.peak_workspace_bytes;
    yvex_core_text_copy(out->artifact_identity, sizeof(out->artifact_identity),
                        result.artifact_identity);
    yvex_core_text_copy(out->execution_identity, sizeof(out->execution_identity),
                        result.execution_identity);
    out->complete = result.complete;
    return YVEX_OK;
}
const yvex_component_binding *yvex_component_binding_at(unsigned long long index)
{
    static const yvex_component_binding bindings[] = {
        {YVEX_COMPONENT_BINDING_SCHEMA_V1, 0x6d6d617564696f01ull, 1ull,
         YVEX_MINIMAX_H3_TARGET_ID, "audio-vae", "audio_vae", YVEX_BACKEND_KIND_CPU,
         audio_component_plan, component_admit, audio_component_execute},
        {YVEX_COMPONENT_BINDING_SCHEMA_V1, 0x6d6d766964656f01ull, 1ull,
         YVEX_MINIMAX_H3_TARGET_ID, "video-vae", "video_vae", YVEX_BACKEND_KIND_CPU,
         video_component_plan, component_admit, video_component_execute},
    };
    return index < sizeof(bindings) / sizeof(bindings[0]) ? &bindings[index] : NULL;
}
static int audio_vae_execute_artifact_cuda(
    const yvex_artifact *artifact, const yvex_gguf *gguf, const yvex_tensor_table *tensors,
    const yvex_minimax_h3_audio_decode_options *options, unsigned long long maximum_device_bytes,
    yvex_minimax_h3_audio_decode_result *result,
    yvex_minimax_h3_component_execution_failure *failure, yvex_error *err)
{
    yvex_complete_artifact_admission admission;
    yvex_artifact_admission_failure admission_failure;
    yvex_runtime_component_session *session = NULL;
    const yvex_runtime_residency_summary *summary = NULL;
    yvex_alias_decoder_request request = {0};
    yvex_alias_decoder_result decoder = {0};
    audio_execution execution = {0};
    int admitted = 0, rc, cleanup_rc;
    yvex_error cleanup;
    if (result) memset(result, 0, sizeof(*result));
    if (failure) memset(failure, 0, sizeof(*failure));
    execution.options = options; execution.result = result;
    execution.failure = failure; execution.err = err;
    if (!artifact || !gguf || !tensors || !options || !result || !maximum_device_bytes)
        return audio_execution_refuse(
            &execution, YVEX_MINIMAX_H3_COMPONENT_EXECUTION_INVALID_ARGUMENT,
            NULL, 6ull, 0ull, YVEX_ERR_INVALID_ARG,
            "Audio VAE CUDA execution requires artifact, geometry, and device budget");
    rc = component_admit("audio_vae", artifact, gguf, tensors, NULL, &admission, NULL,
                         &admission_failure, err);
    admitted = rc == YVEX_OK;
    if (rc == YVEX_OK)
        rc = yvex_runtime_component_session_open(
            &session, &admission, artifact, gguf, tensors, YVEX_BACKEND_KIND_CUDA,
            admission.payload_bytes, maximum_device_bytes, err);
    if (rc == YVEX_OK) {
        execution.session = yvex_runtime_component_session_materialization(session);
        summary = yvex_runtime_component_session_summary(session);
        rc = audio_decode_validate(&execution);
    }
    if (rc == YVEX_OK) {
        request = (yvex_alias_decoder_request){
            .recipe = &audio_decoder_recipe, .input = options->latent, .batch = options->batch,
            .input_length = options->latent_steps,
            .input_count = options->batch * options->latent_channels * options->latent_steps,
            .output = options->output, .output_capacity = options->output_capacity,
            .weight_name = yvex_alias_decoder_template_name,
            .weight_name_context = (void *)&audio_decoder_names,
            .cancel_requested = options->cancelled,
            .cancel_context = options->cancellation_context,
        };
        rc = yvex_runtime_component_alias_decoder_cuda(session, &request, &decoder, err);
    }
    if (rc == YVEX_OK && !audio_execution_identity(
            "yvex.minimax-h3.audio-vae.cuda-f32.v1",
            yvex_materialization_session_summary(execution.session), options, result))
        rc = audio_execution_refuse(
            &execution, YVEX_MINIMAX_H3_COMPONENT_EXECUTION_NUMERIC,
            NULL, 1ull, 0ull, YVEX_ERR_STATE,
            "Audio VAE CUDA execution identity could not be sealed");
    if (rc == YVEX_OK) {
        result->kernel_launches = decoder.kernel_launches;
        result->h2d_bytes = decoder.h2d_bytes;
        result->d2h_bytes = decoder.d2h_bytes;
        result->device_bytes = decoder.peak_device_bytes;
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
    yvex_error_clear(&cleanup);
    cleanup_rc = yvex_runtime_component_session_close(&session, &cleanup);
    if (cleanup_rc != YVEX_OK) { rc = cleanup_rc; if (err) *err = cleanup; }
    return rc;
}
static int video_vae_decode_cuda_session(
    yvex_runtime_component_session *session,
    const yvex_minimax_h3_video_decode_options *options,
    yvex_minimax_h3_video_decode_result *result,
    yvex_minimax_h3_component_execution_failure *failure, yvex_error *err)
{
    yvex_transformer_resident_decoder_request request = {0};
    yvex_transformer_dense_decoder_result decoder = {0};
    video_execution execution = {0};
    component_buffer hidden = {0}, cosines = {0}, sines = {0}, patch_output = {0};
    const yvex_runtime_residency_summary *summary = NULL;
    int rc;
    if (result) memset(result, 0, sizeof(*result));
    if (failure) memset(failure, 0, sizeof(*failure));
    execution.options = options; execution.result = result;
    execution.failure = failure; execution.err = err;
    if (!session || !options || !result)
        return video_execution_refuse(
            &execution, YVEX_MINIMAX_H3_COMPONENT_EXECUTION_INVALID_ARGUMENT,
            NULL, 3ull, 0ull, YVEX_ERR_INVALID_ARG,
            "Visual VAE CUDA decode requires one resident component session");
    execution.session = yvex_runtime_component_session_materialization(session);
    summary = yvex_runtime_component_session_summary(session);
    rc = video_decode_validate(&execution);
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
        rc = yvex_runtime_component_dense_decoder_execute(
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
        failure->code = YVEX_MINIMAX_H3_COMPONENT_EXECUTION_MATERIALIZATION;
        failure->reason = yvex_error_message(err);
    }
    video_buffer_close(&execution, &patch_output);
    video_buffer_close(&execution, &sines);
    video_buffer_close(&execution, &cosines);
    video_buffer_close(&execution, &hidden);
    return rc;
}
static const yvex_component_text_recipe text_recipe = {
    .schema_version = YVEX_COMPONENT_TEXT_RECIPE_SCHEMA_V1,
    .semantic_identity = YVEX_MINIMAX_H3_TEXT_COMPONENT_IDENTITY,
    .embedding_identity_domain = "yvex.minimax-h3.text-conditioning.cuda.v1",
    .encoder_identity_domain = "yvex.minimax-h3.qwen-text-stack.cuda.v1",
    .layer_capacity = YVEX_MINIMAX_H3_TEXT_CONDITIONING_LAYERS,
    .hidden_width = 5120ull,
    .ffn_width = 25600ull,
    .query_heads = 64ull,
    .kv_heads = 8ull,
    .head_dimension = 128ull,
    .vocabulary_size = 151936ull,
    .rope_theta = 5000000ull,
    .normalization_epsilon = 1.0e-6f,
};
static const char *const text_layer_weight_suffixes[YVEX_COMPONENT_TEXT_LAYER_WEIGHT_COUNT] = {
    "input_layernorm.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
    "self_attn.v_proj.weight", "self_attn.o_proj.weight", "self_attn.q_norm.weight",
    "self_attn.k_norm.weight", "post_attention_layernorm.weight", "mlp.gate_proj.weight",
    "mlp.up_proj.weight", "mlp.down_proj.weight",
};
static int text_layer_weight_name(
    void *context, unsigned long long layer, unsigned int slot,
    char output[256], yvex_error *err)
{
    int length;
    (void)context;
    if (slot >= YVEX_COMPONENT_TEXT_LAYER_WEIGHT_COUNT) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "minimax-h3.text-layer.name",
                       "text layer weight slot is outside the source recipe");
        return YVEX_ERR_BOUNDS;
    }
    length = snprintf(output, 256u, "model.language_model.layers.%llu.%s",
                      layer, text_layer_weight_suffixes[slot]);
    if (length < 0 || length >= 256) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "minimax-h3.text-layer.name",
                       "a Qwen layer binding name exceeded its bounded representation");
        return YVEX_ERR_BOUNDS;
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
    yvex_complete_artifact_admission admission;
    yvex_artifact_admission_failure admission_failure;
    yvex_component_text_request request = {
        .recipe = &text_recipe,
        .embedding_weight_name = "model.language_model.embed_tokens.weight",
        .layer_weight_name = text_layer_weight_name,
    };
    int rc;
    if (result) memset(result, 0, sizeof(*result));
    if (!artifact || !gguf || !tensors || !result) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "minimax-h3.text-conditioning",
                       "one admitted text component artifact is required");
        return YVEX_ERR_INVALID_ARG;
    }
    rc = component_admit("text_encoder", artifact, gguf, tensors, NULL, &admission, NULL,
                         &admission_failure, err);
    if (rc == YVEX_OK) {
        request.token_ids = token_ids;
        request.token_count = token_count;
        request.layer_count = layer_count;
        request.output = output;
        request.output_capacity = output_capacity;
        request.maximum_host_bytes = maximum_host_bytes;
        request.maximum_device_bytes = maximum_device_bytes;
        rc = yvex_runtime_component_text_artifact_cuda(
            &admission, artifact, gguf, tensors, &request, result, err);
    }
    return rc;
}
static const yvex_transformer_joint_recipe omni_transformer_recipe = {
    .schema_version = YVEX_TRANSFORMER_JOINT_SCHEMA_V4, .identity_domain = "minimax-h3-fl2va-omni-transformer",
    .qkv_layout = YVEX_TRANSFORMER_QKV_LAYOUT_PER_HEAD_THREE,
    .swiglu_layout = YVEX_TRANSFORMER_SWIGLU_LAYOUT_GATE_THEN_UP,
    .hidden_width = 5376ull, .attention_heads = 56ull, .head_dimension = 128ull,
    .attention_width = 7168ull, .ffn_width = 14336ull, .timestep_width = 2688ull,
    .rotary_width = 96ull, .modality_count = 3ull, .modulation_parameters = 6ull,
    .block_count = 50ull, .refiner_block_count = 2ull, .maximum_timesteps = 64ull,
    .maximum_packed_rows = YVEX_MINIMAX_H3_OMNI_MAX_PACKED_ROWS, .video_input_width = 96ull,
    .audio_input_width = 32ull, .condition_input_width = 5120ull,
    .video_output = {
        .operation = YVEX_TRANSFORMER_LINEAR_OPERATION_JOINT_VIDEO_OUTPUT,
        .publication_contract = YVEX_TRANSFORMER_LINEAR_NUMERIC_SOURCE_EXACT, .source_dtype = YVEX_DTYPE_F32,
        .input_width = 5376ull, .output_width = 96ull, .bias = 1},
    .audio_output = {
        .operation = YVEX_TRANSFORMER_LINEAR_OPERATION_JOINT_AUDIO_OUTPUT,
        .publication_contract = YVEX_TRANSFORMER_LINEAR_NUMERIC_SOURCE_EXACT, .source_dtype = YVEX_DTYPE_F32,
        .input_width = 5376ull, .output_width = 32ull, .bias = 1}};
static const char *const transformer_external_names[YVEX_TRANSFORMER_JOINT_EXTERNAL_WEIGHT_COUNT] = {
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
static const char *const transformer_block_suffixes[YVEX_TRANSFORMER_JOINT_BLOCK_WEIGHT_COUNT] = {
    "norm1.weight", "attn.qkv_proj.weight", "attn.q_norm.weight", "attn.k_norm.weight",
    "attn.out_proj.weight", "norm2.weight", "mlp.fc1.weight", "mlp.fc2.weight",
    "adaln_proj.linear.weight", "adaln_proj.linear.bias",
};
static int transformer_block_weight_name(
    void *context, unsigned long long block, unsigned int slot,
    char output[256], yvex_error *err)
{
    int length;
    (void)context;
    if (slot >= YVEX_TRANSFORMER_JOINT_BLOCK_WEIGHT_COUNT) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "minimax-h3.transformer.binding",
                       "joint Transformer weight slot is outside the source recipe");
        return YVEX_ERR_BOUNDS;
    }
    length = snprintf(output, 256u, "blocks.%llu.%s", block,
                      transformer_block_suffixes[slot]);
    if (length < 0 || length >= 256) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "minimax-h3.transformer.binding",
                       "a Transformer block binding name exceeded its bound");
        return YVEX_ERR_BOUNDS;
    }
    return YVEX_OK;
}
static int transformer_component_cuda(yvex_runtime_component_session *session,
    const yvex_minimax_h3_omni_transformer_request *request,
    yvex_minimax_h3_omni_transformer_result *result, yvex_error *err)
{
    if (result) memset(result, 0, sizeof(*result));
    if (!request || request->recipe != &omni_transformer_recipe) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "minimax-h3.transformer.session",
                       "the MiniMax source-authored Transformer recipe is required");
        return YVEX_ERR_INVALID_ARG;
    }
    return yvex_runtime_component_joint_transformer_cuda(
        session, transformer_external_names,
        YVEX_TRANSFORMER_JOINT_EXTERNAL_WEIGHT_COUNT,
        transformer_block_weight_name, NULL, request, result, err);
}
typedef struct {
    const yvex_minimax_h3_t2va_plan *plan;
    const yvex_minimax_h3_t2va_omni_context *context;
    yvex_transformer_linear_physical_plan video_output_physical, audio_output_physical;
    yvex_runtime_latent_evaluator_evidence evidence;
} t2va_omni_execution;
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
    request.recipe = &omni_transformer_recipe;
    request.video_output_physical = execution->video_output_physical;
    request.audio_output_physical = execution->audio_output_physical;
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
        !context->video_output_specialization || !context->audio_output_specialization ||
        !yvex_core_u64_mul(plan->text_tokens, 5120ull, &conditioning_values) ||
        context->conditioning_capacity < conditioning_values || !context->timestep_indices ||
        context->timestep_capacity < plan->packed_rows || !context->block_count ||
        context->block_count > 50ull) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "graph.minimax_h3.t2va.latent",
                       "one exact resident Transformer and packed FL2VA layout are required");
        return YVEX_ERR_INVALID_ARG;
    }
    execution.plan = plan; execution.context = context;
    execution.video_output_physical = *context->video_output_specialization;
    execution.audio_output_physical = *context->audio_output_specialization;
    rc = yvex_runtime_latent_binding_identity(
        "yvex.minimax-h3.t2va.omni-evaluator.v1", (const char *[4]){plan->identity, summary->residency_identity,
            context->conditioning_identity, context->layout_result->layout_identity}, 4ull,
        (unsigned long long[2]){context->block_count, summary->encoded_bytes}, 2ull,
        execution.evidence.staged.evaluator_identity, err);
    if (rc == YVEX_OK) rc = yvex_runtime_latent_evaluator_begin(
            &execution.evidence, "yvex.minimax-h3.t2va.transformer-chain.v1",
            execution.evidence.staged.evaluator_identity, err);
    request.seed = seed; request.maximum_workspace_bytes = maximum_workspace_bytes;
    request.evaluator_identity = execution.evidence.staged.evaluator_identity;
    request.evaluate = t2va_omni_evaluate; request.execution_context = &execution;
    request.cancel_requested = context->cancelled; request.cancel_context = context->cancellation_context;
    request.observe = context->observe; request.observer_context = context->observer_context;
    if (rc == YVEX_OK) rc = yvex_runtime_av_latent_execute(plan, &request, video, video_capacity,
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
        &omni_transformer_recipe, t2va_plan_build, yvex_runtime_av_scheduler_step,
        t2va_latent_execute, yvex_runtime_av_layout_from_plan,
        component_admit, text_encoder_artifact_cuda,
        transformer_component_cuda,
        audio_vae_decode_cpu, audio_vae_execute_artifact_cuda,
        video_vae_decode_cpu, video_vae_decode_cuda_session,
    };
    return &api;
}
static int component_variant_id(const char *name, yvex_minimax_h3_component_id *out)
{
    static const yvex_minimax_h3_component_id weighted[] = {
        YVEX_MINIMAX_H3_COMPONENT_TEXT_ENCODER,
        YVEX_MINIMAX_H3_COMPONENT_TRANSFORMER,
        YVEX_MINIMAX_H3_COMPONENT_VIDEO_VAE,
        YVEX_MINIMAX_H3_COMPONENT_AUDIO_VAE};
    const yvex_minimax_h3_api *family = yvex_model_register_minimax_h3();
    size_t index;
    if (!name || !out) return 0;
    for (index = 0u; index < sizeof(weighted) / sizeof(weighted[0]); ++index) {
        if (strcmp(name, family->component_name(weighted[index])) == 0) {
            *out = weighted[index];
            return 1;
        }
    }
    return 0;
}
static void component_variant_close(void *owner)
{
    yvex_minimax_h3_handoff *handoff = owner;
    yvex_model_minimax_h3_handoff_api()->close(&handoff);
}
static int component_variant_open(yvex_component_variant_source *out,
                                  const yvex_component_variant_source_request *request,
                                  yvex_error *err)
{
    const yvex_minimax_h3_handoff_api *handoff_api = yvex_model_minimax_h3_handoff_api();
    const yvex_minimax_h3_api *family = yvex_model_register_minimax_h3();
    yvex_minimax_h3_handoff_options options = {0};
    yvex_minimax_h3_handoff_failure failure = {0};
    yvex_minimax_h3_handoff *handoff = NULL;
    yvex_minimax_h3_component_id component_id;
    const yvex_minimax_h3_handoff_summary *source;
    const yvex_minimax_h3_target *target;
    const yvex_minimax_h3_summary *summary;
    const yvex_minimax_h3_component *component;
    int rc;
    if (out) memset(out, 0, sizeof(*out));
    if (!out || !request || !component_variant_id(request->component_id, &component_id)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "minimax-h3.component-variant",
                       "component must be text_encoder, transformer, video_vae, or audio_vae");
        return YVEX_ERR_INVALID_ARG;
    }
    options.source_root = request->source_path;
    options.component = component_id;
    options.transformer_q8 = request->candidate_q8;
    yvex_source_payload_budget_default(&options.budget);
    options.budget.maximum_open_handles = 4u;
    options.budget.maximum_streams = 1u;
    options.budget.maximum_inflight_host_bytes = options.budget.chunk_bytes;
    options.chunk_bytes = options.budget.chunk_bytes;
    options.page_bytes = options.budget.page_bytes;
    rc = handoff_api->open(&handoff, &options, &failure, err);
    target = handoff ? handoff_api->target(handoff) : NULL;
    source = handoff ? handoff_api->summary(handoff) : NULL;
    summary = target ? family->summary(target) : NULL;
    component = target ? family->component_at(target, component_id) : NULL;
    if (rc == YVEX_OK && (!source || !summary || !component)) {
        yvex_error_set(err, YVEX_ERR_STATE, "minimax-h3.component-variant",
                       "weighted component disappeared after source admission");
        rc = YVEX_ERR_STATE;
    }
    if (rc != YVEX_OK) {
        handoff_api->close(&handoff);
        return rc;
    }
    out->owner = handoff;
    out->close = component_variant_close;
    out->transform_ir = handoff_api->transform_ir(handoff);
    out->transform_binding = handoff_api->binding(handoff);
#define COPY(field, value) yvex_core_text_copy(out->field, sizeof(out->field), (value))
    COPY(architecture, "minimax-h3");
    COPY(target_id, YVEX_MINIMAX_H3_TARGET_ID);
    COPY(component_id, component->canonical_id);
    COPY(source_snapshot_identity, summary->source_snapshot_identity);
    COPY(component_identity, component->identity);
    COPY(component_manifest_identity, summary->component_manifest_identity);
    COPY(architecture_identity, summary->architecture_identity);
    COPY(role_map_identity, summary->role_map_identity);
#undef COPY
    out->source_snapshot_key = summary->source_snapshot_key;
    out->summary.schema_version = YVEX_PHYSICAL_VARIANT_SESSION_SCHEMA_V1;
    out->summary.kind = YVEX_PHYSICAL_VARIANT_COMPONENT;
    out->summary.shards = source->shards;
    out->summary.tensors = source->tensors;
    out->summary.elements = source->elements;
    out->summary.payload_execution_bytes_read = source->payload_execution_bytes_read;
    out->summary.source_verified = source->complete;
#define COPY_SUMMARY(field, value) \
    yvex_core_text_copy(out->summary.field, sizeof(out->summary.field), (value))
    COPY_SUMMARY(target_id, YVEX_MINIMAX_H3_TARGET_ID);
    COPY_SUMMARY(family, "minimax-h3");
    COPY_SUMMARY(component_id, component->canonical_id);
    COPY_SUMMARY(source_revision, YVEX_SOURCE_MINIMAX_H3_REVISION);
    COPY_SUMMARY(source_snapshot_identity, source->source_snapshot_identity);
    COPY_SUMMARY(component_identity, source->component_identity);
    COPY_SUMMARY(transform_identity, source->transform_identity);
#undef COPY_SUMMARY
    return YVEX_OK;
}
const yvex_component_variant_adapter *yvex_graph_minimax_h3_component_adapter(void)
{
    static const yvex_media_execution_recipe media = {
        .schema_version = YVEX_MEDIA_EXECUTION_RECIPE_SCHEMA_V1,
        .conditioning_layers = YVEX_MINIMAX_H3_TEXT_CONDITIONING_LAYERS, .transformer_blocks = 50ull,
        .maximum_prompt_tokens = YVEX_MINIMAX_H3_TEXT_MAX_TOKENS,
        .maximum_packed_rows = YVEX_MINIMAX_H3_OMNI_MAX_PACKED_ROWS, .component_backend = YVEX_BACKEND_KIND_CUDA,
        .output_semantic_domain = omni_transformer_recipe.identity_domain,
        .video_output_requirement = &omni_transformer_recipe.video_output,
        .audio_output_requirement = &omni_transformer_recipe.audio_output,
        .plan_build = t2va_plan_build, .layout_build = yvex_runtime_av_layout_from_plan,
        .component_admit = component_admit, .condition = text_encoder_artifact_cuda,
        .latent = t2va_latent_execute, .video_decode = video_vae_decode_cuda_session,
        .audio_decode = audio_vae_execute_artifact_cuda};
    static const yvex_component_variant_adapter adapter = {
        .schema_version = YVEX_PHYSICAL_VARIANT_SESSION_SCHEMA_V1,
        .target_id = YVEX_MINIMAX_H3_TARGET_ID,
        .family = "minimax-h3",
        .source_revision = YVEX_SOURCE_MINIMAX_H3_REVISION,
        .profile_name = "minimax-h3-source-faithful-v1",
        .candidate_profile_name = YVEX_MINIMAX_H3_TRANSFORMER_Q8_PROFILE_NAME,
        .candidate_component_id = "transformer",
        .candidate_q8_semantic_role_mask = YVEX_MINIMAX_H3_TRANSFORMER_Q8_ROLE_MASK,
        .source_open = component_variant_open,
        .physical_variant = yvex_graph_physical_variant_api_get,
        .media_target_profile = yvex_model_minimax_h3_media_target_profile,
        .media_execution = &media};
    return &adapter;
}
typedef struct {
    yvex_semantic_model_ir *semantic_model;
    yvex_transform_ir *transform_ir;
} minimax_source_owner;
static void minimax_source_release(void *pointer)
{
    minimax_source_owner *owner = pointer;
    if (!owner) return;
    yvex_semantic_model_ir_close(&owner->semantic_model);
    yvex_transform_ir_release(&owner->transform_ir);
    free(owner);
}
static int minimax_source_compile(yvex_family_source_products *out,
                                  const yvex_compilation_runtime_binding_request *request,
                                  yvex_error *err)
{
    const yvex_minimax_h3_api *family = yvex_model_register_minimax_h3();
    yvex_minimax_h3_open_options options = {0};
    yvex_minimax_h3_failure failure = {0};
    yvex_minimax_h3_target *target = NULL;
    minimax_source_owner *owner = NULL;
    yvex_semantic_component components[YVEX_MINIMAX_H3_COMPONENT_COUNT] = {0};
    yvex_semantic_phase_edge edges[YVEX_MINIMAX_H3_PHASE_EDGES] = {0};
    yvex_semantic_composite_request composite = {0};
    yvex_semantic_model_ir_request semantic = {0};
    const yvex_minimax_h3_summary *summary;
    unsigned long long index;
    char derivation[YVEX_SHA256_HEX_BYTES] = {0};
    int rc;
    if (out) memset(out, 0, sizeof(*out));
    if (!out || !request || !request->source_path || !request->source_path[0]) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "minimax-h3.source-compiler",
                       "exact MiniMax source path is required");
        return YVEX_ERR_INVALID_ARG;
    }
    options.source_root = request->source_path;
    rc = family->open(&target, &options, &failure, err);
    summary = rc == YVEX_OK ? family->summary(target) : NULL;
    if (rc != YVEX_OK || !summary) goto cleanup;
    owner = calloc(1u, sizeof(*owner));
    if (!owner) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "minimax-h3.source-compiler",
                       "source compiler ownership allocation failed");
        rc = YVEX_ERR_NOMEM;
        goto cleanup;
    }
    rc = yvex_model_minimax_h3_transform_api()->build(
        &owner->transform_ir, derivation, target, err);
    for (index = 0ull; rc == YVEX_OK && index < summary->component_count; ++index) {
        const yvex_minimax_h3_component *source = family->component_at(target, index);
        if (!source) {
            rc = YVEX_ERR_STATE;
            yvex_error_set(err, rc, "minimax-h3.source-compiler",
                           "semantic component disappeared after source admission");
            break;
        }
        yvex_core_text_copy(components[index].canonical_id,
                            sizeof(components[index].canonical_id), source->canonical_id);
        yvex_core_text_copy(components[index].identity,
                            sizeof(components[index].identity), source->identity);
        components[index].shards = source->shard_count;
        components[index].tensors = source->tensor_count;
        components[index].phase = (unsigned int)source->phase;
        components[index].weighted = source->weighted;
        components[index].release_after_phase = source->release_after_phase;
    }
    for (index = 0ull; rc == YVEX_OK && index < summary->phase_edge_count; ++index) {
        const yvex_minimax_h3_phase_edge *source = family->phase_edge_at(index);
        if (!source) {
            rc = YVEX_ERR_STATE;
            yvex_error_set(err, rc, "minimax-h3.source-compiler",
                           "semantic phase edge disappeared after source admission");
            break;
        }
        edges[index].source_phase = (unsigned int)source->source_phase;
        edges[index].destination_phase = (unsigned int)source->destination_phase;
        edges[index].data_classes = source->data_classes;
        edges[index].lifetime = (unsigned int)source->lifetime;
    }
    composite = (yvex_semantic_composite_request){
        .repository = YVEX_SOURCE_MINIMAX_H3_REPOSITORY,
        .revision = YVEX_SOURCE_MINIMAX_H3_REVISION,
        .subtree = YVEX_MINIMAX_H3_SUBTREE,
        .source_snapshot_identity = summary->source_snapshot_identity,
        .component_manifest_identity = summary->component_manifest_identity,
        .phase_dag_identity = summary->phase_dag_identity,
        .architecture_identity = summary->architecture_identity,
        .role_map_identity = summary->role_map_identity,
        .unresolved_requirements_identity = summary->unresolved_requirements_identity,
        .weighted_components = summary->weighted_component_count,
        .shards = summary->shard_count,
        .tensors = summary->tensor_count,
        .elements = summary->element_count,
        .payload_bytes = summary->payload_bytes,
        .components = components,
        .component_count = summary->component_count,
        .phase_edges = edges,
        .phase_edge_count = summary->phase_edge_count};
    semantic = (yvex_semantic_model_ir_request){
        .schema_version = YVEX_SEMANTIC_MODEL_IR_SCHEMA_V1,
        .family_adapter_id = 0x4d494e494d4158ull,
        .family_adapter_version = 1ull,
        .target_id = YVEX_MINIMAX_H3_TARGET_ID,
        .source_model_identity = summary->source_snapshot_identity,
        .logical_model_identity = summary->target_identity,
        .semantic_payload_identity = summary->architecture_identity,
        .composite = &composite};
    if (rc == YVEX_OK)
        rc = yvex_semantic_model_ir_seal(&owner->semantic_model, &semantic, err);
    if (rc == YVEX_OK) {
        out->owner = owner;
        out->release = minimax_source_release;
        out->semantic_model = owner->semantic_model;
        out->transform_ir = owner->transform_ir;
        yvex_core_text_copy(out->derivation_identity,
                            sizeof(out->derivation_identity), derivation);
        owner = NULL;
    }
cleanup:
    family->close(&target);
    minimax_source_release(owner);
    return rc;
}
static int minimax_tokenizer_policy(
    yvex_tokenizer_family_policy *out, yvex_error *err)
{
    static const yvex_tokenizer_direct_policy policy = {
        .family_adapter_id = 0x4d4d4833ull,
        .family_adapter_version = 1ull,
        .tokenizer_kind = YVEX_TOKENIZER_KIND_GGML_GPT2,
        .model_policy = YVEX_TOKENIZER_MODEL_BPE_BYTELEVEL,
        .prompt_policy = YVEX_TOKENIZER_PROMPT_MINIMAX_H3_FL2VA,
        .vocabulary_size = 151669ull,
        .base_vocabulary_size = 151643ull,
        .merge_count = 151387ull,
        .added_token_count = 26ull,
        .special_token_count = 14ull,
        .eos_token_id = 151645u,
        .pad_token_id = 151643u,
        .eos_present = 1,
        .pad_present = 1,
        .architecture = "minimax-h3",
        .tokenizer_model = "gpt2",
        .tokenizer_pre = "qwen2",
        .tokenizer_json_identity =
            "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7",
        .tokenizer_config_identity =
            "a07e942ac874baa13758de8d1fbdb186683cc03416b5589e1b6671c6b3057c68",
        .prompt_name = "verbatim-no-special-v1"};
    return yvex_tokenizer_family_policy_compile_direct(out, &policy, err) == YVEX_OK;
}
const yvex_family_source_adapter *yvex_graph_minimax_h3_source_adapter(void)
{
    static const yvex_family_source_adapter adapter = {
        .schema_version = YVEX_FAMILY_SOURCE_ADAPTER_SCHEMA_V1, .target_id = YVEX_MINIMAX_H3_TARGET_ID,
        .family = "minimax-h3", .tokenizer_architecture = "minimax-h3",
        .tokenizer_policy = minimax_tokenizer_policy, .compile = minimax_source_compile};
    return &adapter;
}
