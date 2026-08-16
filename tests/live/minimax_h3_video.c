/* Verify and execute the exact reduced MiniMax-H3 Visual VAE component artifact. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <yvex/artifact.h>
#include <yvex/gguf.h>
#include <yvex/internal/artifact.h>
#include <yvex/internal/component.h>
#include <yvex/internal/families/minimax_h3.h>
#include <yvex/internal/latent.h>
#include <yvex/internal/runtime.h>

typedef struct {
    yvex_runtime_component_session *session;
    yvex_component_execution_failure failure;
} reconstruction_context;

static int reconstruction_decode(
    void *opaque, const yvex_runtime_av_video_decode_window *window,
    yvex_runtime_av_video_decode_evidence *evidence, yvex_error *err)
{
    reconstruction_context *context = opaque;
    yvex_runtime_av_video_decode_options options = {0};
    yvex_runtime_av_video_decode_result result;
    int rc;
    options.latent = window->latent; options.output = window->output;
    options.batch = 1ull; options.latent_channels = window->latent_channels;
    options.latent_frames = window->latent_frames;
    options.latent_height = window->latent_height; options.latent_width = window->latent_width;
    options.output_capacity = window->output_capacity;
    options.max_workspace_bytes = 256ull * 1024ull * 1024ull;
    rc = yvex_graph_register_minimax_h3()->video_vae_decode_cuda(
        context->session, &options, &result, &context->failure, err);
    if (rc == YVEX_OK) {
        memset(evidence, 0, sizeof(*evidence));
        evidence->output_values = result.output_values;
        evidence->kernel_launches = result.kernel_launches;
        evidence->h2d_bytes = result.h2d_bytes; evidence->d2h_bytes = result.d2h_bytes;
        evidence->device_bytes = result.device_bytes;
        memcpy(evidence->execution_identity, result.execution_identity,
               sizeof(evidence->execution_identity));
        evidence->complete = result.complete;
    }
    return rc;
}

static int file_read_exact(const char *path, void *output, size_t bytes)
{
    FILE *file = fopen(path, "rb");
    int accepted = file && fread(output, bytes, 1u, file) == 1u && fgetc(file) == EOF;

    if (file) fclose(file);
    return accepted;
}

static int compare_reference(const char *path, const float *output, size_t count)
{
    float *reference = (float *)malloc(count * sizeof(*reference));
    float maximum = 0.0f;
    size_t index;

    if (!reference || !file_read_exact(path, reference, count * sizeof(*reference))) {
        fprintf(stderr, "video_vae_reference_read=refused\n");
        free(reference);
        return 1;
    }
    for (index = 0u; index < count; ++index) {
        float absolute = fabsf(output[index] - reference[index]);
        if (!isfinite(absolute)) {
            fprintf(stderr, "video_vae_conformance=refused non-finite-value\n");
            free(reference);
            return 1;
        }
        if (absolute > maximum) maximum = absolute;
    }
    free(reference);
    printf("oracle_max_absolute_error=%.9g\n", maximum);
    if (maximum > 1.0e-5f) {
        fprintf(stderr, "video_vae_conformance=refused tolerance=1e-5\n");
        return 1;
    }
    printf("video_vae_conformance=accepted tolerance=1e-5\n");
    return 0;
}

static int compare_reconstruction_reference(const char *path, const float *output, size_t count)
{
    float *reference = (float *)malloc(count * sizeof(*reference));
    long double reference_squared = 0.0L, difference_squared = 0.0L;
    float maximum = 0.0f;
    size_t index;

    if (!reference || !file_read_exact(path, reference, count * sizeof(*reference))) {
        fprintf(stderr, "video_reconstruction_reference_read=refused\n");
        free(reference);
        return 1;
    }
    for (index = 0u; index < count; ++index) {
        float difference = output[index] - reference[index];
        float absolute = fabsf(difference);

        if (!isfinite(absolute)) {
            fprintf(stderr, "video_reconstruction_conformance=refused non-finite-value\n");
            free(reference);
            return 1;
        }
        if (absolute > maximum) maximum = absolute;
        reference_squared += (long double)reference[index] * reference[index];
        difference_squared += (long double)difference * difference;
    }
    free(reference);
    if (reference_squared <= 0.0L) {
        fprintf(stderr, "video_reconstruction_conformance=refused zero-reference-norm\n");
        return 1;
    }
    printf("reconstruction_oracle_max_absolute_error=%.9g\n", maximum);
    printf("reconstruction_oracle_relative_l2=%.12Lg\n",
           sqrtl(difference_squared / reference_squared));
    if (maximum > 1.0e-4f || sqrtl(difference_squared / reference_squared) > 1.0e-4L) {
        fprintf(stderr, "video_reconstruction_conformance=refused tolerance=1e-4\n");
        return 1;
    }
    printf("video_reconstruction_conformance=accepted tolerance=1e-4\n");
    return 0;
}

static int output_write(const char *path, const float *output, size_t count)
{
    FILE *file = fopen(path, "wb");
    size_t written;
    int close_rc;

    if (!file) return 0;
    written = fwrite(output, sizeof(float), count, file);
    close_rc = fclose(file);
    return written == count && close_rc == 0;
}

static int parse_dimension(const char *text, unsigned long long *out)
{
    char *end = NULL;
    unsigned long long value = strtoull(text, &end, 10);

    if (!text[0] || !end || *end || !value || value > 1024ull) return 0;
    *out = value;
    return 1;
}

int main(int argc, char **argv)
{
    yvex_artifact_options options = {0};
    yvex_artifact_admission_failure admission_failure = {0};
    yvex_complete_artifact_admission admission;
    yvex_artifact *artifact = NULL;
    yvex_tensor_table *tensors = NULL;
    yvex_gguf *gguf = NULL;
    yvex_error err;
    const char *latent_path = NULL, *output_path = NULL, *reference_path = NULL;
    unsigned long long frames = 1ull, height = 1ull, width = 1ull;
    int expect_refused = 0, decode = 0, cuda = 0, reconstruct = 0;
    int rc;

    if ((argc == 5 || argc == 6) &&
        (strcmp(argv[1], "--decode") == 0 || strcmp(argv[1], "--decode-cuda") == 0)) {
        decode = 1;
        cuda = strcmp(argv[1], "--decode-cuda") == 0;
        options.path = argv[2];
        latent_path = argv[3];
        output_path = argv[4];
        if (argc == 6) reference_path = argv[5];
    } else if ((argc == 8 || argc == 9) &&
               (strcmp(argv[1], "--decode-geometry") == 0 ||
                strcmp(argv[1], "--decode-geometry-cuda") == 0)) {
        decode = 1;
        cuda = strcmp(argv[1], "--decode-geometry-cuda") == 0;
        options.path = argv[2];
        latent_path = argv[3];
        output_path = argv[4];
        if (!parse_dimension(argv[5], &frames) ||
            !parse_dimension(argv[6], &height) ||
            !parse_dimension(argv[7], &width) ||
            frames > 4096ull / height || frames * height > 4096ull / width) {
            fprintf(stderr, "video_vae_geometry=refused\n");
            return 2;
        }
        if (argc == 9) reference_path = argv[8];
    } else if ((argc == 5 || argc == 6) && strcmp(argv[1], "--reconstruct-cuda") == 0) {
        reconstruct = 1; cuda = 1; options.path = argv[2];
        latent_path = argv[3]; output_path = argv[4];
        if (argc == 6) reference_path = argv[5];
    } else if (argc == 3 && strcmp(argv[1], "--expect-refused") == 0) {
        expect_refused = 1;
        options.path = argv[2];
    } else if (argc == 2) {
        options.path = argv[1];
    } else {
        fprintf(stderr,
                "usage: minimax_h3_video [--expect-refused] VIDEO_VAE_GGUF\n"
                "       minimax_h3_video --decode VIDEO_VAE_GGUF LATENT_F32 OUTPUT_F32 "
                "[REFERENCE_F32]\n"
                "       minimax_h3_video --decode-geometry VIDEO_VAE_GGUF LATENT_F32 "
                "OUTPUT_F32 T H W [REFERENCE_F32]\n"
                "       minimax_h3_video --reconstruct-cuda VIDEO_VAE_GGUF LATENT_F32 "
                "OUTPUT_F32 [REFERENCE_F32]\n"
                "       replace --decode with --decode-cuda for resident CUDA execution\n");
        return 2;
    }
    options.readonly = 1;
    rc = yvex_artifact_open(&artifact, &options, &err);
    if (rc == YVEX_OK) rc = yvex_gguf_open(&gguf, artifact, &err);
    if (rc == YVEX_OK) rc = yvex_tensor_table_from_gguf(&tensors, gguf, &err);
    if (reconstruct && rc == YVEX_OK) {
        static const char source_identity[] =
            "897ceaff08708f431132c6643bc8f1041ace8c0444a3ea248bbf727fc7da9943";
        yvex_runtime_av_video_reconstruction_request plan_request = {
            .schema_version = YVEX_RUNTIME_AV_VIDEO_RECONSTRUCTION_SCHEMA_V1,
            .frames = 124ull, .width = 32ull, .height = 32ull,
            .latent_frames = 37ull, .latent_width = 2ull, .latent_height = 2ull,
            .temporal_ratio = 4ull, .clip_length = 17ull, .token_drop = 3ull,
            .spatial_ratio = 16ull, .tile_size = 256ull, .minimum_tile_overlap = 64ull,
            .source_identity = source_identity,
        };
        yvex_runtime_av_video_reconstruction_plan plan;
        yvex_runtime_av_video_reconstruction_execution execution = {0};
        yvex_runtime_av_video_reconstruction_result result;
        reconstruction_context context = {0};
        const yvex_minimax_h3_latent_normalization *normalization =
            yvex_model_register_minimax_h3()->latent_normalization();
        yvex_complete_artifact_admission component;
        yvex_artifact_admission_failure failure;
        unsigned long long latent_values = 24ull * 37ull * 2ull * 2ull;
        unsigned long long output_values = 3ull * 124ull * 32ull * 32ull;
        float *latent = malloc((size_t)(latent_values * sizeof(float)));
        float *output = malloc((size_t)(output_values * sizeof(float)));
        if (!latent || !output ||
            !file_read_exact(latent_path, latent, (size_t)(latent_values * sizeof(float)))) {
            fprintf(stderr, "video_reconstruction_latent_read=refused\n");
            rc = YVEX_ERR_FORMAT;
        }
        if (rc == YVEX_OK)
            rc = yvex_runtime_av_video_reconstruction_plan_build(&plan_request, &plan, &err);
        if (rc == YVEX_OK)
            rc = yvex_graph_register_minimax_h3()->component_admit(
                "video_vae", artifact, gguf, tensors, &component, &failure, &err);
        if (rc == YVEX_OK)
            rc = yvex_runtime_component_session_open(
                &context.session, &component, artifact, gguf, tensors, YVEX_BACKEND_KIND_CUDA,
                component.payload_bytes, 16ull * 1024ull * 1024ull * 1024ull, &err);
        execution.schema_version = YVEX_RUNTIME_AV_VIDEO_RECONSTRUCTION_SCHEMA_V1;
        execution.plan = &plan; execution.latent = latent;
        execution.latent_channels = 24ull; execution.latent_capacity = latent_values;
        execution.maximum_workspace_bytes = 64ull * 1024ull * 1024ull;
        execution.output_channel_mean = normalization->pixel_mean;
        execution.output_channel_std = normalization->pixel_std;
        execution.output_channel_count = normalization->pixel_channels;
        execution.decode = reconstruction_decode; execution.decode_context = &context;
        if (rc == YVEX_OK)
            rc = yvex_runtime_av_video_reconstruct(
                &execution, output, output_values, &result, &err);
        if (rc == YVEX_OK && !output_write(output_path, output, (size_t)output_values))
            rc = YVEX_ERR_IO;
        if (rc == YVEX_OK && reference_path &&
            compare_reconstruction_reference(reference_path, output,
                                             (size_t)output_values) != 0)
            rc = YVEX_ERR_FORMAT;
        if (rc == YVEX_OK)
            printf("video_reconstruction=accepted shape=1x3x124x32x32 calls=%llu "
                   "kernels=%llu workspace=%llu device=%llu identity=%s\n",
                   result.decode_calls, result.kernel_launches, result.peak_workspace_bytes,
                   result.peak_device_bytes, result.execution_identity);
        else
            fprintf(stderr, "video_reconstruction=refused code=%d where=%s message=%s\n",
                    context.failure.code, yvex_error_where(&err), yvex_error_message(&err));
        {
            yvex_error cleanup;
            yvex_error_clear(&cleanup);
            if (yvex_runtime_component_session_close(&context.session, &cleanup) != YVEX_OK &&
                rc == YVEX_OK) { rc = yvex_error_code(&cleanup); err = cleanup; }
        }
        free(output); free(latent);
    } else if (decode && rc == YVEX_OK) {
        yvex_runtime_av_video_decode_options decode_options;
        yvex_runtime_av_video_decode_result result;
        yvex_component_execution_failure execution_failure;
        yvex_runtime_component_session *session = NULL;
        unsigned long long patches = frames * height * width;
        size_t latent_values = (size_t)(patches * 24ull);
        size_t output_values = (size_t)(patches * 3072ull);
        float *latent = (float *)malloc(latent_values * sizeof(*latent));
        float *output = (float *)malloc(output_values * sizeof(*output));

        memset(&decode_options, 0, sizeof(decode_options));
        memset(&execution_failure, 0, sizeof(execution_failure));
        if (!latent || !output ||
            !file_read_exact(latent_path, latent, latent_values * sizeof(*latent))) {
            fprintf(stderr, "video_vae_latent_read=refused\n");
            rc = YVEX_ERR_FORMAT;
        }
        decode_options.latent = latent;
        decode_options.output = output;
        decode_options.output_capacity = output_values;
        decode_options.batch = 1ull;
        decode_options.latent_channels = 24ull;
        decode_options.latent_frames = frames;
        decode_options.latent_height = height;
        decode_options.latent_width = width;
        decode_options.max_workspace_bytes = 256ull * 1024ull * 1024ull;
        if (rc == YVEX_OK)
            rc = yvex_graph_register_minimax_h3()->component_admit(
                "video_vae", artifact, gguf, tensors, &admission, &admission_failure, &err);
        if (rc == YVEX_OK)
            rc = yvex_runtime_component_session_open(
                &session, &admission, artifact, gguf, tensors,
                cuda ? YVEX_BACKEND_KIND_CUDA : YVEX_BACKEND_KIND_CPU,
                admission.payload_bytes,
                cuda ? 16ull * 1024ull * 1024ull * 1024ull : 0ull, &err);
        if (rc == YVEX_OK && !cuda)
            rc = yvex_graph_register_minimax_h3()->video_vae_decode_cpu(
                yvex_runtime_component_session_materialization(session), &decode_options,
                &result, &execution_failure, &err);
        if (rc == YVEX_OK && cuda)
            rc = yvex_graph_register_minimax_h3()->video_vae_decode_cuda(
                session, &decode_options, &result, &execution_failure, &err);
        if (rc != YVEX_OK) {
            fprintf(stderr,
                    "video_vae_decode=refused code=%d tensor=%s expected=%llu actual=%llu "
                    "where=%s message=%s\n",
                    execution_failure.code, execution_failure.tensor_name,
                    execution_failure.expected, execution_failure.actual,
                    yvex_error_where(&err), yvex_error_message(&err));
        } else if (!output_write(output_path, output, output_values)) {
            fprintf(stderr, "video_vae_output_write=refused\n");
            rc = YVEX_ERR_IO;
        } else {
            printf("video_vae_decode=accepted\n");
            printf("output_shape=1x3x%llux%llux%llu\n",
                   result.frames, result.height, result.width);
            printf("tensor_reads=%llu\n", result.tensor_reads);
            printf("payload_bytes_read=%llu\n", result.payload_bytes_read);
            printf("peak_workspace_bytes=%llu\n", result.peak_workspace_bytes);
            printf("kernel_launches=%llu\n", result.kernel_launches);
            printf("h2d_bytes=%llu\n", result.h2d_bytes);
            printf("d2h_bytes=%llu\n", result.d2h_bytes);
            printf("device_bytes=%llu\n", result.device_bytes);
            printf("residency_identity=%s\n", result.residency_identity);
            printf("execution_identity=%s\n", result.execution_identity);
            if (reference_path && compare_reference(reference_path, output, output_values) != 0)
                rc = YVEX_ERR_FORMAT;
        }
        {
            yvex_error cleanup;
            yvex_error_clear(&cleanup);
            if (yvex_runtime_component_session_close(&session, &cleanup) != YVEX_OK &&
                rc == YVEX_OK) { rc = yvex_error_code(&cleanup); err = cleanup; }
        }
        free(output);
        free(latent);
    } else if (rc == YVEX_OK) {
        rc = yvex_graph_register_minimax_h3()->component_admit(
            "video_vae", artifact, gguf, tensors, &admission, &admission_failure, &err);
        if (expect_refused) {
            if (rc == YVEX_OK) {
                fprintf(stderr, "video_vae_corruption=accepted\n");
                rc = YVEX_ERR_STATE;
            } else {
                printf("video_vae_corruption=refused field=%s\n", admission_failure.field);
                rc = YVEX_OK;
            }
        } else if (rc == YVEX_OK) {
            printf("video_vae_admission=accepted\n");
            printf("artifact_identity=%s\n", admission.artifact_identity);
            printf("admission_identity=%s\n", admission.admission_identity);
            printf("component_identity=%s\n", admission.logical_component_identity);
            printf("tensors=%llu\n", admission.tensor_count);
            printf("payload_bytes=%llu\n", admission.payload_bytes);
        }
    }
    if (rc != YVEX_OK && !decode && !expect_refused)
        fprintf(stderr, "video_vae=refused field=%s where=%s message=%s\n",
                admission_failure.field, yvex_error_where(&err), yvex_error_message(&err));
    yvex_tensor_table_close(tensors);
    yvex_gguf_close(gguf);
    yvex_artifact_close(artifact);
    return rc == YVEX_OK ? 0 : 1;
}
