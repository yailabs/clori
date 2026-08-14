/*
 * Stage one identity-bound audio-video request across independently resident components.
 *
 * Family callbacks retain numeric policy. This owner supplies the common transactional
 * lifecycle: tokenizer, conditioning, latent state, late VAE residency, synchronization,
 * and publication are advanced in order and no partial decoded output becomes visible.
 */
#include <yvex/internal/media.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <yvex/gguf.h>
#include <yvex/internal/family_catalog.h>
#include <yvex/model.h>
#include <yvex/tokenizer.h>

typedef struct {
    yvex_artifact *artifact;
    yvex_gguf *gguf;
    yvex_tensor_table *tensors;
} component_view;

typedef struct {
    const yvex_runtime_av_generation_request *request;
    yvex_runtime_av_plan plan;
    yvex_runtime_av_layout_output layout;
    yvex_runtime_av_layout_result layout_result;
    yvex_runtime_latent_result latent_result;
    yvex_runtime_latent_evaluator_result evaluator_result;
    yvex_runtime_av_unpack_result unpack_result;
    yvex_runtime_av_video_reconstruction_result video_result;
    yvex_runtime_av_audio_decode_result audio_result;
    yvex_media_avi_result media_result;
    yvex_runtime_av_conditioning_result conditioning_result;
    float *conditioning, *video_rows, *audio_rows, *video_latent, *audio_latent;
    float *rgb, *pcm;
    unsigned int *timestep_indices;
    unsigned long long conditioning_values, video_row_values, audio_row_values;
    unsigned long long video_latent_values, audio_latent_values, rgb_values, pcm_values;
    unsigned long long layout_position_values;
    unsigned long long host_live, host_peak;
    char prompt_identity[YVEX_SHA256_HEX_CAP];
    char execution_identity[YVEX_SHA256_HEX_CAP];
} generation_state;

typedef struct {
    const yvex_runtime_av_generation_request *request;
    yvex_runtime_component_session *session;
} video_decode_context;

static int generation_fail(
    yvex_error *err, yvex_status status, const char *where, const char *message)
{
    yvex_error_set(err, status, where, message);
    return status;
}

static int generation_cancelled(
    const yvex_runtime_av_generation_request *request, yvex_error *err)
{
    if (request->cancel_requested && request->cancel_requested(request->cancel_context))
        return generation_fail(err, YVEX_ERR_CANCELLED, "runtime.av-generation",
                               "audio-video generation was cancelled");
    return YVEX_OK;
}

static int component_view_open(const char *path, component_view *view, yvex_error *err)
{
    yvex_artifact_options options = {0};
    int rc;
    memset(view, 0, sizeof(*view));
    options.path = path;
    options.readonly = 1;
    rc = yvex_artifact_open(&view->artifact, &options, err);
    if (rc == YVEX_OK) rc = yvex_gguf_open(&view->gguf, view->artifact, err);
    if (rc == YVEX_OK) rc = yvex_tensor_table_from_gguf(&view->tensors, view->gguf, err);
    return rc;
}

static int fixture_prompt_identity(const yvex_tokens *tokens,
                                   char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;
    yvex_sha256_init(&hash);
    if (!tokens || !tokens->ids || !tokens->len ||
        !yvex_sha256_update_text(&hash, "yvex.runtime.av-generation.fixture-token-ids.v1") ||
        !yvex_sha256_update_u64_be(&hash, tokens->len))
        return 0;
    for (index = 0ull; index < tokens->len; ++index)
        if (!yvex_sha256_update_u64_be(&hash, tokens->ids[index])) return 0;
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

static void component_view_close(component_view *view)
{
    if (!view) return;
    yvex_tensor_table_close(view->tensors);
    yvex_gguf_close(view->gguf);
    yvex_artifact_close(view->artifact);
    memset(view, 0, sizeof(*view));
}

static int host_allocate(
    generation_state *state, unsigned long long count, size_t element_bytes,
    void **out, const char *label, yvex_error *err)
{
    unsigned long long bytes, next;
    if (out) *out = NULL;
    if (!state || !out || !count || !element_bytes ||
        !yvex_core_u64_mul(count, element_bytes, &bytes) || bytes > SIZE_MAX ||
        !yvex_core_u64_add(state->host_live, bytes, &next) ||
        next > state->request->maximum_workspace_bytes ||
        next > state->request->maximum_host_bytes) {
        yvex_error_setf(err, YVEX_ERR_BOUNDS, "runtime.av-generation.workspace",
                        "%s exceeds the pipeline workspace budget", label);
        return YVEX_ERR_BOUNDS;
    }
    *out = calloc(1u, (size_t)bytes);
    if (!*out) {
        yvex_error_setf(err, YVEX_ERR_NOMEM, "runtime.av-generation.workspace",
                        "%s allocation failed", label);
        return YVEX_ERR_NOMEM;
    }
    state->host_live = next;
    if (next > state->host_peak) state->host_peak = next;
    return YVEX_OK;
}

static void host_release(
    generation_state *state, void **owned, unsigned long long count, size_t element_bytes)
{
    unsigned long long bytes = 0ull;
    if (!state || !owned || !*owned) return;
    if (yvex_core_u64_mul(count, element_bytes, &bytes) && bytes <= state->host_live)
        state->host_live -= bytes;
    free(*owned);
    *owned = NULL;
}

static void generation_state_close(generation_state *state)
{
    if (!state) return;
    host_release(state, (void **)&state->pcm, state->pcm_values, sizeof(float));
    host_release(state, (void **)&state->rgb, state->rgb_values, sizeof(float));
    host_release(state, (void **)&state->audio_latent, state->audio_latent_values, sizeof(float));
    host_release(state, (void **)&state->video_latent, state->video_latent_values, sizeof(float));
    host_release(state, (void **)&state->audio_rows, state->audio_row_values, sizeof(float));
    host_release(state, (void **)&state->video_rows, state->video_row_values, sizeof(float));
    host_release(state, (void **)&state->conditioning,
                 state->conditioning_values, sizeof(float));
    host_release(state, (void **)&state->timestep_indices, state->plan.packed_rows,
                 sizeof(unsigned int));
    host_release(state, (void **)&state->layout.text_indices, state->plan.text_tokens,
                 sizeof(unsigned int));
    host_release(state, (void **)&state->layout.audio_indices, state->plan.audio_rows,
                 sizeof(unsigned int));
    host_release(state, (void **)&state->layout.video_indices, state->plan.video_rows,
                 sizeof(unsigned int));
    host_release(state, (void **)&state->layout.token_tags, state->plan.packed_rows,
                 sizeof(unsigned int));
    host_release(state, (void **)&state->layout.position_ids, state->layout_position_values,
                 sizeof(float));
}

static int request_validate(
    const yvex_runtime_av_generation_request *request, yvex_error *err)
{
    if (!request || request->schema_version != YVEX_RUNTIME_AV_GENERATION_SCHEMA_V1 ||
        !request->target || !request->prompt || !request->prompt[0] || !request->output_path ||
        !request->text_artifact_path || !request->transformer_artifact_path ||
        !request->video_artifact_path || !request->audio_artifact_path ||
        !request->source_identity || !yvex_sha256_hex_valid(request->source_identity) ||
        !request->frames || !request->width || !request->height ||
        !request->fps_numerator || !request->fps_denominator || !request->audio_sample_rate ||
        !request->inference_steps || !request->conditioning_layers ||
        !request->transformer_blocks || !request->maximum_prompt_tokens ||
        !request->maximum_packed_rows ||
        !request->maximum_host_bytes || !request->maximum_device_bytes ||
        !request->maximum_workspace_bytes || !request->maximum_file_bytes ||
        (request->component_backend != YVEX_BACKEND_KIND_CPU &&
         request->component_backend != YVEX_BACKEND_KIND_CUDA) ||
        !request->video_temporal_ratio || !request->video_clip_length ||
        !request->video_spatial_ratio || !request->video_tile_size ||
        !request->video_mean || !request->video_std || !request->audio_mean ||
        !request->audio_std || !request->pixel_mean || !request->pixel_std ||
        !request->video_channels || !request->audio_channels || !request->pixel_channels ||
        !request->audio_output_channels || request->audio_output_channels > 2ull ||
        !request->audio_samples_per_step || !request->plan_build || !request->layout_build ||
        !request->component_admit || !request->condition || !request->latent ||
        !request->video_decode || !request->audio_decode)
        return generation_fail(err, YVEX_ERR_INVALID_ARG, "runtime.av-generation",
                               "one exact family adapter and bounded AV request are required");
    return YVEX_OK;
}

static int conditioning_execute(generation_state *state, yvex_error *err)
{
    const yvex_runtime_av_generation_request *request = state->request;
    yvex_model_context context = {0};
    yvex_tokenizer_encode_options options = {
        0, 0, 1, state->request->maximum_prompt_tokens
    };
    yvex_tokenizer_encode_result encoded = {0};
    yvex_tokens fixture = {0};
    const yvex_tokens *tokens = NULL;
    int rc = generation_cancelled(request, err);
    if (rc == YVEX_OK)
        rc = yvex_model_context_open(request->text_artifact_path, &context, err);
    if (rc == YVEX_OK) {
        rc = yvex_family_tokenizer_open(&context.tokenizer, context.gguf, err);
        if (rc == YVEX_ERR_UNSUPPORTED) {
            yvex_error_clear(err);
            rc = yvex_tokenizer_from_gguf(
                &context.tokenizer, context.gguf, context.model, err);
        }
    }
    if (rc == YVEX_OK && yvex_tokenizer_plan_summary_get(context.tokenizer)) {
        rc = yvex_tokenizer_encode(
            context.tokenizer, (const unsigned char *)request->prompt,
            (unsigned long long)strlen(request->prompt), &options, &encoded, err);
        tokens = &encoded.tokens;
    } else if (rc == YVEX_OK) {
        rc = yvex_tokenize_text(context.tokenizer, request->prompt, &fixture, err);
        tokens = &fixture;
    }
    if (rc == YVEX_OK &&
        (!tokens || !tokens->len ||
         !yvex_core_u64_mul(tokens->len, 5120ull, &state->conditioning_values)))
        rc = generation_fail(err, YVEX_ERR_FORMAT, "runtime.av-generation.conditioning",
                             "prompt tokenization did not produce exact conditioning geometry");
    if (rc == YVEX_OK)
        rc = host_allocate(state, state->conditioning_values, sizeof(float),
                           (void **)&state->conditioning, "conditioning", err);
    if (rc == YVEX_OK)
        rc = request->condition(
            context.artifact, context.gguf, context.table, tokens->ids, tokens->len,
            request->conditioning_layers, state->conditioning,
            state->conditioning_values, request->maximum_host_bytes,
            request->maximum_device_bytes, &state->conditioning_result, err);
    if (rc == YVEX_OK &&
         (!state->conditioning_result.complete ||
         state->conditioning_result.token_count != tokens->len ||
         state->conditioning_result.hidden_width != 5120ull ||
         !yvex_sha256_hex_valid(state->conditioning_result.execution_identity)))
        rc = generation_fail(err, YVEX_ERR_STATE, "runtime.av-generation.conditioning",
                             "family conditioning returned incomplete execution evidence");
    if (rc == YVEX_OK && encoded.completed)
        yvex_core_text_copy(state->prompt_identity, sizeof(state->prompt_identity),
                            encoded.encoding_identity);
    else if (rc == YVEX_OK && !fixture_prompt_identity(&fixture, state->prompt_identity))
        rc = generation_fail(err, YVEX_ERR_STATE, "runtime.av-generation.conditioning",
                             "fixture prompt identity could not be sealed");
    yvex_tokenizer_encode_result_clear(&encoded);
    yvex_tokens_clear(&fixture);
    yvex_model_context_close(&context);
    return rc;
}

static int plan_and_layout_build(generation_state *state, yvex_error *err)
{
    const yvex_runtime_av_generation_request *request = state->request;
    int rc = request->plan_build(
        &state->plan, state->conditioning_result.token_count, request->width,
        request->height, request->frames, request->inference_steps, err);
    if (rc == YVEX_OK && state->plan.packed_rows > request->maximum_packed_rows)
        rc = generation_fail(err, YVEX_ERR_BOUNDS, "runtime.av-generation.plan",
                             "packed AV plan exceeds the admitted execution capacity");
    if (rc == YVEX_OK &&
        !yvex_core_u64_mul(state->plan.packed_rows, 3ull, &state->layout_position_values))
        rc = generation_fail(err, YVEX_ERR_BOUNDS, "runtime.av-generation.layout",
                             "layout position extent overflowed");
    if (rc == YVEX_OK)
        rc = host_allocate(state, state->layout_position_values, sizeof(float),
                           (void **)&state->layout.position_ids, "layout positions", err);
    if (rc == YVEX_OK)
        rc = host_allocate(state, state->plan.packed_rows, sizeof(unsigned int),
                           (void **)&state->layout.token_tags, "layout tags", err);
    if (rc == YVEX_OK)
        rc = host_allocate(state, state->plan.video_rows, sizeof(unsigned int),
                           (void **)&state->layout.video_indices, "video indices", err);
    if (rc == YVEX_OK)
        rc = host_allocate(state, state->plan.audio_rows, sizeof(unsigned int),
                           (void **)&state->layout.audio_indices, "audio indices", err);
    if (rc == YVEX_OK)
        rc = host_allocate(state, state->plan.text_tokens, sizeof(unsigned int),
                           (void **)&state->layout.text_indices, "text indices", err);
    if (rc == YVEX_OK)
        rc = host_allocate(state, state->plan.packed_rows, sizeof(unsigned int),
                           (void **)&state->timestep_indices, "timestep indices", err);
    state->layout.position_capacity = state->plan.packed_rows * 3ull;
    state->layout.tag_capacity = state->plan.packed_rows;
    state->layout.video_capacity = state->plan.video_rows;
    state->layout.audio_capacity = state->plan.audio_rows;
    state->layout.text_capacity = state->plan.text_tokens;
    if (rc == YVEX_OK)
        rc = request->layout_build(&state->plan, &state->layout, &state->layout_result, err);
    return rc;
}

static int latent_execute(generation_state *state, yvex_error *err)
{
    const yvex_runtime_av_generation_request *request = state->request;
    component_view view = {0};
    yvex_complete_artifact_admission admission = {0};
    yvex_artifact_admission_failure failure = {0};
    yvex_runtime_component_session *session = NULL;
    yvex_runtime_av_latent_context context = {0};
    yvex_error cleanup;
    int rc, cleanup_rc;
    if (!yvex_core_u64_mul(state->plan.video_rows, state->plan.video_value_width,
                           &state->video_row_values) ||
        !yvex_core_u64_mul(state->plan.audio_rows, state->plan.audio_value_width,
                           &state->audio_row_values))
        return generation_fail(err, YVEX_ERR_BOUNDS, "runtime.av-generation.latent",
                               "paired latent extent overflowed");
    rc = host_allocate(state, state->video_row_values, sizeof(float),
                       (void **)&state->video_rows, "video latent rows", err);
    if (rc == YVEX_OK)
        rc = host_allocate(state, state->audio_row_values, sizeof(float),
                           (void **)&state->audio_rows, "audio latent rows", err);
    if (rc == YVEX_OK) rc = component_view_open(request->transformer_artifact_path, &view, err);
    if (rc == YVEX_OK)
        rc = request->component_admit(
            "transformer", view.artifact, view.gguf, view.tensors, &admission, &failure, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_component_session_open(
            &session, &admission, view.artifact, view.gguf, view.tensors,
            request->component_backend, request->maximum_host_bytes,
            request->maximum_device_bytes, err);
    context.transformer_session = session;
    context.conditioning = state->conditioning;
    context.conditioning_capacity = state->conditioning_values;
    context.conditioning_identity = state->conditioning_result.execution_identity;
    context.layout = &state->layout;
    context.layout_result = &state->layout_result;
    context.timestep_indices = state->timestep_indices;
    context.timestep_capacity = state->plan.packed_rows;
    context.block_count = request->transformer_blocks;
    context.cancelled = request->cancel_requested;
    context.cancellation_context = request->cancel_context;
    if (rc == YVEX_OK)
        rc = request->latent(
            &state->plan, &context, request->seed,
            request->maximum_workspace_bytes, state->video_rows, state->video_row_values,
            state->audio_rows, state->audio_row_values, &state->latent_result,
            &state->evaluator_result, err);
    yvex_error_clear(&cleanup);
    cleanup_rc = yvex_runtime_component_session_close(&session, &cleanup);
    if (cleanup_rc != YVEX_OK) { rc = cleanup_rc; if (err) *err = cleanup; }
    component_view_close(&view);
    if (rc == YVEX_OK &&
        (!state->latent_result.completed || !state->evaluator_result.complete ||
         !yvex_sha256_hex_valid(state->latent_result.execution_identity)))
        rc = generation_fail(err, YVEX_ERR_STATE, "runtime.av-generation.latent",
                             "paired latent execution returned incomplete evidence");
    return rc;
}

static int latent_unpack(generation_state *state, yvex_error *err)
{
    const yvex_runtime_av_generation_request *request = state->request;
    yvex_runtime_av_unpack_request unpack = {0};
    yvex_runtime_av_unpack_output output = {0};
    int rc;
    if (!yvex_core_u64_mul(request->video_channels, state->plan.video_latent_frames,
                           &state->video_latent_values) ||
        !yvex_core_u64_mul(state->video_latent_values, state->plan.video_latent_height,
                           &state->video_latent_values) ||
        !yvex_core_u64_mul(state->video_latent_values, state->plan.video_latent_width,
                           &state->video_latent_values) ||
        !yvex_core_u64_mul(request->audio_output_channels, request->audio_channels,
                           &state->audio_latent_values) ||
        !yvex_core_u64_mul(state->audio_latent_values, state->plan.audio_latent_steps,
                           &state->audio_latent_values))
        return generation_fail(err, YVEX_ERR_BOUNDS, "runtime.av-generation.unpack",
                               "component latent extent overflowed");
    rc = host_allocate(state, state->video_latent_values, sizeof(float),
                       (void **)&state->video_latent, "Visual VAE latent", err);
    if (rc == YVEX_OK)
        rc = host_allocate(state, state->audio_latent_values, sizeof(float),
                           (void **)&state->audio_latent, "Audio VAE latent", err);
    unpack.schema_version = YVEX_RUNTIME_AV_UNPACK_SCHEMA_V1;
    unpack.plan = &state->plan;
    unpack.video_rows = state->video_rows;
    unpack.audio_rows = state->audio_rows;
    unpack.video_row_capacity = state->video_row_values;
    unpack.audio_row_capacity = state->audio_row_values;
    unpack.video_channel_mean = request->video_mean;
    unpack.video_channel_std = request->video_std;
    unpack.audio_channel_mean = request->audio_mean;
    unpack.audio_channel_std = request->audio_std;
    unpack.video_channel_count = request->video_channels;
    unpack.audio_channel_count = request->audio_channels;
    unpack.maximum_workspace_bytes = request->maximum_workspace_bytes;
    unpack.latent_execution_identity = state->latent_result.execution_identity;
    output.video = state->video_latent;
    output.audio = state->audio_latent;
    output.video_capacity = state->video_latent_values;
    output.audio_capacity = state->audio_latent_values;
    if (rc == YVEX_OK) rc = yvex_runtime_av_unpack(&unpack, &output, &state->unpack_result, err);
    return rc;
}

static int video_decode(
    void *opaque, const yvex_runtime_av_video_decode_window *window,
    yvex_runtime_av_video_decode_evidence *evidence, yvex_error *err)
{
    video_decode_context *context = opaque;
    yvex_runtime_av_video_decode_options options = {0};
    yvex_runtime_av_video_decode_result result = {0};
    yvex_component_execution_failure failure = {0};
    int rc;
    options.latent = window->latent;
    options.output = window->output;
    options.batch = 1ull;
    options.latent_channels = window->latent_channels;
    options.latent_frames = window->latent_frames;
    options.latent_height = window->latent_height;
    options.latent_width = window->latent_width;
    options.output_capacity = window->output_capacity;
    options.max_workspace_bytes = context->request->maximum_workspace_bytes;
    options.cancelled = context->request->cancel_requested;
    options.cancellation_context = context->request->cancel_context;
    rc = context->request->video_decode(
        context->session, &options, &result, &failure, err);
    if (rc == YVEX_OK) {
        evidence->output_values = result.output_values;
        evidence->kernel_launches = result.kernel_launches;
        evidence->h2d_bytes = result.h2d_bytes;
        evidence->d2h_bytes = result.d2h_bytes;
        evidence->device_bytes = result.device_bytes;
        yvex_core_text_copy(evidence->execution_identity,
                            sizeof(evidence->execution_identity),
                            result.execution_identity);
        evidence->complete = result.complete;
    }
    return rc;
}

static int video_execute(generation_state *state, yvex_error *err)
{
    const yvex_runtime_av_generation_request *request = state->request;
    yvex_runtime_av_video_reconstruction_request plan_request = {0};
    yvex_runtime_av_video_reconstruction_plan plan;
    yvex_runtime_av_video_reconstruction_execution execution = {0};
    component_view view = {0};
    yvex_complete_artifact_admission admission = {0};
    yvex_artifact_admission_failure failure = {0};
    video_decode_context context = {request, NULL};
    yvex_error cleanup;
    int rc, cleanup_rc;
    if (!yvex_core_u64_mul(3ull, request->frames, &state->rgb_values) ||
        !yvex_core_u64_mul(state->rgb_values, request->height, &state->rgb_values) ||
        !yvex_core_u64_mul(state->rgb_values, request->width, &state->rgb_values))
        return generation_fail(err, YVEX_ERR_BOUNDS, "runtime.av-generation.video",
                               "decoded RGB extent overflowed");
    rc = host_allocate(state, state->rgb_values, sizeof(float),
                       (void **)&state->rgb, "decoded RGB", err);
    plan_request.schema_version = YVEX_RUNTIME_AV_VIDEO_RECONSTRUCTION_SCHEMA_V1;
    plan_request.frames = request->frames;
    plan_request.width = request->width;
    plan_request.height = request->height;
    plan_request.latent_frames = state->plan.video_latent_frames;
    plan_request.latent_height = state->plan.video_latent_height;
    plan_request.latent_width = state->plan.video_latent_width;
    plan_request.temporal_ratio = request->video_temporal_ratio;
    plan_request.clip_length = request->video_clip_length;
    plan_request.token_drop = request->video_token_drop;
    plan_request.spatial_ratio = request->video_spatial_ratio;
    plan_request.tile_size = request->video_tile_size;
    plan_request.minimum_tile_overlap = request->video_minimum_tile_overlap;
    plan_request.source_identity = request->source_identity;
    if (rc == YVEX_OK)
        rc = yvex_runtime_av_video_reconstruction_plan_build(&plan_request, &plan, err);
    if (rc == YVEX_OK) rc = component_view_open(request->video_artifact_path, &view, err);
    if (rc == YVEX_OK)
        rc = request->component_admit(
            "video_vae", view.artifact, view.gguf, view.tensors, &admission, &failure, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_component_session_open(
            &context.session, &admission, view.artifact, view.gguf, view.tensors,
            request->component_backend, request->maximum_host_bytes,
            request->maximum_device_bytes, err);
    execution.schema_version = YVEX_RUNTIME_AV_VIDEO_RECONSTRUCTION_SCHEMA_V1;
    execution.plan = &plan;
    execution.latent = state->video_latent;
    execution.latent_channels = request->video_channels;
    execution.latent_capacity = state->video_latent_values;
    execution.maximum_workspace_bytes = request->maximum_workspace_bytes;
    execution.output_channel_mean = request->pixel_mean;
    execution.output_channel_std = request->pixel_std;
    execution.output_channel_count = request->pixel_channels;
    execution.decode = video_decode;
    execution.decode_context = &context;
    execution.cancel_requested = request->cancel_requested;
    execution.cancel_context = request->cancel_context;
    if (rc == YVEX_OK)
        rc = yvex_runtime_av_video_reconstruct(
            &execution, state->rgb, state->rgb_values, &state->video_result, err);
    yvex_error_clear(&cleanup);
    cleanup_rc = yvex_runtime_component_session_close(&context.session, &cleanup);
    if (cleanup_rc != YVEX_OK) { rc = cleanup_rc; if (err) *err = cleanup; }
    component_view_close(&view);
    return rc;
}

static int audio_execute(generation_state *state, yvex_error *err)
{
    const yvex_runtime_av_generation_request *request = state->request;
    component_view view = {0};
    yvex_runtime_av_audio_decode_options options = {0};
    yvex_component_execution_failure failure = {0};
    unsigned long long expected_samples;
    int rc;
    if (!yvex_core_u64_mul(request->audio_output_channels,
                           state->plan.audio_latent_steps, &state->pcm_values) ||
        !yvex_core_u64_mul(state->pcm_values, request->audio_samples_per_step,
                           &state->pcm_values))
        return generation_fail(err, YVEX_ERR_BOUNDS, "runtime.av-generation.audio",
                               "decoded PCM extent overflowed");
    rc = host_allocate(state, state->pcm_values, sizeof(float),
                       (void **)&state->pcm, "decoded PCM", err);
    if (rc == YVEX_OK) rc = component_view_open(request->audio_artifact_path, &view, err);
    options.latent = state->audio_latent;
    options.batch = request->audio_output_channels;
    options.latent_channels = request->audio_channels;
    options.latent_steps = state->plan.audio_latent_steps;
    options.output = state->pcm;
    options.output_capacity = state->pcm_values;
    options.max_workspace_bytes = request->maximum_workspace_bytes;
    options.cancelled = request->cancel_requested;
    options.cancellation_context = request->cancel_context;
    if (rc == YVEX_OK)
        rc = request->audio_decode(
            view.artifact, view.gguf, view.tensors, &options,
            request->maximum_device_bytes, &state->audio_result, &failure, err);
    component_view_close(&view);
    if (rc == YVEX_OK &&
        !yvex_core_u64_mul(state->plan.audio_latent_steps,
                           request->audio_samples_per_step, &expected_samples))
        rc = generation_fail(err, YVEX_ERR_BOUNDS, "runtime.av-generation.audio",
                             "decoded PCM evidence extent overflowed");
    if (rc == YVEX_OK &&
        (!state->audio_result.complete ||
         state->audio_result.samples_per_channel != expected_samples))
        rc = generation_fail(err, YVEX_ERR_STATE, "runtime.av-generation.audio",
                             "Audio VAE returned incomplete decoded evidence");
    return rc;
}

static int phase_identity(generation_state *state, yvex_error *err)
{
    const char *identities[] = {
        state->prompt_identity, state->conditioning_result.execution_identity,
        state->plan.identity, state->layout_result.layout_identity,
        state->latent_result.execution_identity, state->unpack_result.input_identity,
        state->video_result.execution_identity, state->audio_result.execution_identity,
    };
    const unsigned long long facts[] = {
        state->request->frames, state->request->width, state->request->height,
        state->request->inference_steps, state->request->transformer_blocks,
        state->request->seed,
    };
    return yvex_runtime_latent_binding_identity(
        "yvex.runtime.av-generation.staged.v1", identities,
        sizeof(identities) / sizeof(identities[0]), facts,
        sizeof(facts) / sizeof(facts[0]), state->execution_identity, err);
}

static int media_publish(generation_state *state, yvex_error *err)
{
    const yvex_runtime_av_generation_request *request = state->request;
    yvex_media_avi_request media = {0};
    int rc = phase_identity(state, err);
    media.schema_version = YVEX_MEDIA_AVI_SCHEMA_V1;
    media.path = request->output_path;
    media.video = state->rgb;
    media.audio = state->pcm;
    media.video_channels = 3ull;
    media.frames = request->frames;
    media.width = request->width;
    media.height = request->height;
    media.fps_numerator = request->fps_numerator;
    media.fps_denominator = request->fps_denominator;
    media.audio_channels = request->audio_output_channels;
    media.audio_samples = state->audio_result.samples_per_channel;
    media.audio_sample_rate = request->audio_sample_rate;
    media.maximum_file_bytes = request->maximum_file_bytes;
    media.video_identity = state->video_result.execution_identity;
    media.audio_identity = state->audio_result.execution_identity;
    media.execution_identity = state->execution_identity;
    media.cancel_requested = request->cancel_requested;
    media.cancel_context = request->cancel_context;
    if (rc == YVEX_OK) rc = yvex_media_avi_publish(&media, &state->media_result, err);
    return rc;
}

static int result_publish(
    const generation_state *state, yvex_runtime_av_generation_result *result,
    yvex_error *err)
{
    unsigned long long launches;
    result->schema_version = YVEX_RUNTIME_AV_GENERATION_SCHEMA_V1;
    result->prompt_tokens = state->conditioning_result.token_count;
    result->frames = state->request->frames;
    result->width = state->request->width;
    result->height = state->request->height;
    result->audio_samples = state->media_result.audio_samples_used;
    result->model_evaluations = state->evaluator_result.model_evaluations;
    if (!yvex_core_u64_add(state->conditioning_result.kernel_launches,
                           state->evaluator_result.kernel_launches, &launches) ||
        !yvex_core_u64_add(launches, state->video_result.kernel_launches, &launches) ||
        !yvex_core_u64_add(launches, state->audio_result.kernel_launches, &launches))
        return generation_fail(err, YVEX_ERR_BOUNDS, "runtime.av-generation.result",
                               "kernel launch evidence overflowed");
    result->kernel_launches = launches;
    result->peak_device_bytes = state->evaluator_result.peak_device_bytes;
    if (state->video_result.peak_device_bytes > result->peak_device_bytes)
        result->peak_device_bytes = state->video_result.peak_device_bytes;
    if (state->audio_result.device_bytes > result->peak_device_bytes)
        result->peak_device_bytes = state->audio_result.device_bytes;
    result->peak_workspace_bytes = state->host_peak;
    if (state->conditioning_result.peak_workspace_bytes > result->peak_workspace_bytes)
        result->peak_workspace_bytes = state->conditioning_result.peak_workspace_bytes;
    if (state->latent_result.peak_workspace_bytes > result->peak_workspace_bytes)
        result->peak_workspace_bytes = state->latent_result.peak_workspace_bytes;
    if (state->video_result.peak_workspace_bytes > result->peak_workspace_bytes)
        result->peak_workspace_bytes = state->video_result.peak_workspace_bytes;
    if (state->audio_result.peak_workspace_bytes > result->peak_workspace_bytes)
        result->peak_workspace_bytes = state->audio_result.peak_workspace_bytes;
    result->file_bytes = state->media_result.file_bytes;
    yvex_core_text_copy(result->prompt_identity, sizeof(result->prompt_identity),
                        state->prompt_identity);
    yvex_core_text_copy(result->conditioning_identity, sizeof(result->conditioning_identity),
                        state->conditioning_result.execution_identity);
    yvex_core_text_copy(result->plan_identity, sizeof(result->plan_identity),
                        state->plan.identity);
    yvex_core_text_copy(result->layout_identity, sizeof(result->layout_identity),
                        state->layout_result.layout_identity);
    yvex_core_text_copy(result->latent_identity, sizeof(result->latent_identity),
                        state->latent_result.execution_identity);
    yvex_core_text_copy(result->vae_input_identity, sizeof(result->vae_input_identity),
                        state->unpack_result.input_identity);
    yvex_core_text_copy(result->video_identity, sizeof(result->video_identity),
                        state->video_result.execution_identity);
    yvex_core_text_copy(result->audio_identity, sizeof(result->audio_identity),
                        state->audio_result.execution_identity);
    yvex_core_text_copy(result->execution_identity, sizeof(result->execution_identity),
                        state->execution_identity);
    yvex_core_text_copy(result->file_identity, sizeof(result->file_identity),
                        state->media_result.file_identity);
    yvex_core_text_copy(result->publication_identity, sizeof(result->publication_identity),
                        state->media_result.publication_identity);
    result->complete = 1;
    return YVEX_OK;
}

int yvex_runtime_av_generate(
    const yvex_runtime_av_generation_request *request,
    yvex_runtime_av_generation_result *result, yvex_error *err)
{
    generation_state state = {0};
    int rc;
    if (result) memset(result, 0, sizeof(*result));
    rc = request_validate(request, err);
    if (rc != YVEX_OK || !result) {
        if (rc == YVEX_OK)
            rc = generation_fail(err, YVEX_ERR_INVALID_ARG, "runtime.av-generation",
                                 "generation result is required");
        return rc;
    }
    state.request = request;
    rc = conditioning_execute(&state, err);
    if (rc == YVEX_OK) rc = plan_and_layout_build(&state, err);
    if (rc == YVEX_OK) rc = generation_cancelled(request, err);
    if (rc == YVEX_OK) rc = latent_execute(&state, err);
    if (rc == YVEX_OK) rc = latent_unpack(&state, err);
    host_release(&state, (void **)&state.video_rows, state.video_row_values, sizeof(float));
    host_release(&state, (void **)&state.audio_rows, state.audio_row_values, sizeof(float));
    host_release(&state, (void **)&state.conditioning,
                 state.conditioning_values, sizeof(float));
    if (rc == YVEX_OK) rc = generation_cancelled(request, err);
    if (rc == YVEX_OK) rc = video_execute(&state, err);
    host_release(&state, (void **)&state.video_latent,
                 state.video_latent_values, sizeof(float));
    if (rc == YVEX_OK) rc = generation_cancelled(request, err);
    if (rc == YVEX_OK) rc = audio_execute(&state, err);
    host_release(&state, (void **)&state.audio_latent,
                 state.audio_latent_values, sizeof(float));
    if (rc == YVEX_OK) rc = media_publish(&state, err);
    if (rc == YVEX_OK) {
        rc = result_publish(&state, result, err);
    }
    if (rc == YVEX_OK) {
        yvex_error_clear(err);
    }
    generation_state_close(&state);
    return rc;
}
