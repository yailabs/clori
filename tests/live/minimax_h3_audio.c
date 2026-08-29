/* Verify and characterize the exact source-faithful Audio VAE artifact. */
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <yvex/artifact.h>
#include <yvex/gguf.h>
#include <yvex/internal/artifact.h>
#include <yvex/internal/component.h>
#include <yvex/internal/families/minimax_h3.h>
#include <yvex/internal/runtime.h>

typedef enum {
    MODE_ADMIT = 0,
    MODE_EXPECT_REFUSED,
    MODE_DECODE_CPU,
    MODE_DECODE_CUDA,
    MODE_UNPACK_AUDIO,
} audio_mode;

typedef struct {
    audio_mode mode;
    const char *artifact_path, *component, *input_path, *output_path, *reference_path;
    unsigned long long batch, latent_steps;
} audio_arguments;

static int fail(const char *phase, const yvex_artifact_admission_failure *failure,
                const yvex_error *err)
{
    fprintf(stderr,
            "%s=refused code=%s field=%s expected=%llu actual=%llu where=%s message=%s\n",
            phase, yvex_artifact_admission_code_name(failure->code), failure->field,
            failure->expected, failure->actual, yvex_error_where(err), yvex_error_message(err));
    return 1;
}

static int parse_extent(const char *text, unsigned long long *value)
{
    char *end = NULL;
    unsigned long long parsed;
    errno = 0;
    parsed = strtoull(text, &end, 10);
    if (errno || !text[0] || !end || *end || !parsed) return 0;
    *value = parsed;
    return 1;
}

static int tensor_extent(unsigned long long batch, unsigned long long channels,
                         unsigned long long length, size_t *values)
{
    if (!batch || !channels || !length || batch > SIZE_MAX / channels ||
        batch * channels > SIZE_MAX / length)
        return 0;
    *values = (size_t)(batch * channels * length);
    return 1;
}

static int file_read_exact(const char *path, float *values, size_t count)
{
    FILE *file = fopen(path, "rb");
    int ok = file && fread(values, sizeof(*values), count, file) == count && fgetc(file) == EOF;
    if (file) fclose(file);
    return ok;
}

static int file_write_exact(const char *path, const float *values, size_t count)
{
    FILE *file = fopen(path, "wb");
    int ok = file && fwrite(values, sizeof(*values), count, file) == count;
    if (file && fclose(file) != 0) ok = 0;
    return ok;
}

static int compare_reference(const char *path, const float *output, size_t count)
{
    float *reference = (float *)malloc(count * sizeof(*reference));
    float max_absolute = 0.0f;
    size_t index;
    int rc = 0;

    if (!reference || !file_read_exact(path, reference, count)) {
        fprintf(stderr, "audio_vae_reference_read=refused\n");
        free(reference);
        return 1;
    }
    for (index = 0u; index < count; ++index) {
        float absolute = fabsf(output[index] - reference[index]);
        if (absolute > max_absolute) max_absolute = absolute;
    }
    printf("oracle_max_absolute_error=%.9g\n", max_absolute);
    if (max_absolute > 1.0e-5f) {
        fprintf(stderr, "audio_vae_conformance=refused tolerance=1e-5\n");
        rc = 1;
    } else {
        printf("audio_vae_conformance=accepted tolerance=1e-5\n");
    }
    free(reference);
    return rc;
}

static double elapsed_seconds(const struct timespec *begin, const struct timespec *end)
{
    return (double)(end->tv_sec - begin->tv_sec) +
           (double)(end->tv_nsec - begin->tv_nsec) / 1000000000.0;
}

static int unpack_audio(const audio_arguments *arguments)
{
    const yvex_minimax_h3_latent_normalization *normalization =
        yvex_model_register_minimax_h3()->latent_normalization();
    size_t values, source, destination;
    unsigned long long batch, channel, step;
    float *packed = NULL, *decoded = NULL;
    int rc = 1;

    if (!normalization || normalization->audio_channels != 32ull ||
        !tensor_extent(arguments->batch, 32ull, arguments->latent_steps, &values)) {
        fprintf(stderr, "audio_latent_unpack=refused geometry\n");
        return 1;
    }
    packed = (float *)malloc(values * sizeof(*packed));
    decoded = (float *)malloc(values * sizeof(*decoded));
    if (!packed || !decoded || !file_read_exact(arguments->input_path, packed, values)) {
        fprintf(stderr, "audio_latent_unpack=refused input\n");
        goto done;
    }
    for (batch = 0ull; batch < arguments->batch; ++batch)
        for (step = 0ull; step < arguments->latent_steps; ++step)
            for (channel = 0ull; channel < 32ull; ++channel) {
                source = (size_t)((batch * arguments->latent_steps + step) * 32ull + channel);
                destination =
                    (size_t)((batch * 32ull + channel) * arguments->latent_steps + step);
                decoded[destination] = packed[source] * normalization->audio_std[channel] +
                                       normalization->audio_mean[channel];
            }
    if (!file_write_exact(arguments->output_path, decoded, values)) {
        fprintf(stderr, "audio_latent_unpack=refused output\n");
        goto done;
    }
    printf("audio_latent_unpack=accepted\n");
    printf("shape=%llux32x%llu\n", arguments->batch, arguments->latent_steps);
    printf("values=%zu\n", values);
    rc = 0;
done:
    free(decoded);
    free(packed);
    return rc;
}

static int decode_cpu(const audio_arguments *arguments, const yvex_artifact *artifact,
                      const yvex_gguf *gguf, const yvex_tensor_table *tensors,
                      yvex_error *err)
{
    yvex_component_plan_request plan_request = {
        .target_id = YVEX_MINIMAX_H3_TARGET_ID,
        .component_id = "audio-vae",
        .backend = YVEX_BACKEND_KIND_CPU,
        .batch = 1ull,
        .geometry_rank = 1u,
        .geometry = {1ull},
        .maximum_host_bytes = 256ull * 1024ull * 1024ull,
    };
    yvex_component_plan component_plan;
    yvex_component_execution_request request = {0};
    yvex_component_execution_result result;
    yvex_component_failure failure = {0};
    float latent[32], output[800];
    int rc;

    if (!file_read_exact(arguments->input_path, latent, 32u)) {
        fprintf(stderr, "latent_read=refused\n");
        return YVEX_ERR_FORMAT;
    }
    rc = yvex_runtime_component_api_get()->plan_build(
        &plan_request, &component_plan, &failure, err);
    request.plan = &component_plan;
    request.input = latent;
    request.output = output;
    request.output_capacity = 800ull;
    if (rc == YVEX_OK)
        rc = yvex_runtime_component_api_get()->execute(
            artifact, gguf, tensors, &request, &result, &failure, err);
    if (rc != YVEX_OK) {
        fprintf(stderr,
                "audio_vae_decode=refused code=%d tensor=%s expected=%llu actual=%llu "
                "where=%s message=%s\n",
                failure.code, failure.tensor_name, failure.expected, failure.actual,
                yvex_error_where(err), yvex_error_message(err));
        return rc;
    }
    if (!file_write_exact(arguments->output_path, output, 800u))
        return YVEX_ERR_IO;
    printf("audio_vae_decode=accepted backend=cpu\n");
    printf("samples=%llu\n", result.output_dims[2]);
    printf("tensor_reads=%llu\n", result.tensor_reads);
    printf("payload_bytes_read=%llu\n", result.payload_bytes_read);
    printf("peak_workspace_bytes=%llu\n", result.peak_workspace_bytes);
    printf("execution_identity=%s\n", result.execution_identity);
    if (arguments->reference_path && compare_reference(arguments->reference_path, output, 800u))
        return YVEX_ERR_FORMAT;
    return YVEX_OK;
}

static int decode_cuda(const audio_arguments *arguments, const yvex_artifact *artifact,
                       const yvex_gguf *gguf, const yvex_tensor_table *tensors,
                       yvex_error *err)
{
    yvex_minimax_h3_audio_decode_options options = {0};
    yvex_minimax_h3_audio_decode_result result;
    yvex_minimax_h3_component_execution_failure failure = {0};
    yvex_artifact_admission_failure admission_failure = {0};
    yvex_complete_artifact_admission admission;
    yvex_runtime_component_session *session = NULL;
    yvex_component_execution component = {0};
    struct timespec begin, end;
    yvex_error cleanup;
    size_t latent_values, output_values;
    float *latent = NULL, *output = NULL;
    int rc = YVEX_ERR_BOUNDS;

    if (!tensor_extent(arguments->batch, 32ull, arguments->latent_steps, &latent_values) ||
        !tensor_extent(arguments->batch, 800ull, arguments->latent_steps, &output_values)) {
        fprintf(stderr, "audio_vae_cuda=refused geometry\n");
        return rc;
    }
    latent = (float *)malloc(latent_values * sizeof(*latent));
    output = (float *)malloc(output_values * sizeof(*output));
    if (!latent || !output || !file_read_exact(arguments->input_path, latent, latent_values)) {
        fprintf(stderr, "audio_vae_cuda=refused input\n");
        rc = YVEX_ERR_FORMAT;
        goto done;
    }
    options.latent = latent;
    options.batch = arguments->batch;
    options.latent_channels = 32ull;
    options.latent_steps = arguments->latent_steps;
    options.output = output;
    options.output_capacity = output_values;
    options.max_workspace_bytes = 16ull * 1024ull * 1024ull * 1024ull;
    (void)clock_gettime(CLOCK_MONOTONIC, &begin);
    rc = yvex_graph_register_minimax_h3()->component_admit(
        "audio_vae", artifact, gguf, tensors, NULL, &admission, NULL,
        &admission_failure, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_component_session_open(
            &session, &admission, artifact, gguf, tensors, YVEX_BACKEND_KIND_CUDA,
            admission.payload_bytes, 16ull * 1024ull * 1024ull * 1024ull, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_component_session_borrow(session, &component, err);
    if (rc == YVEX_OK)
        rc = yvex_graph_register_minimax_h3()->audio_vae_decode_backend(
            &component, &options, &result, &failure, err);
    yvex_error_clear(&cleanup);
    if (yvex_runtime_component_session_close(&session, &cleanup) != YVEX_OK &&
        rc == YVEX_OK) {
        rc = yvex_error_code(&cleanup);
        if (err) *err = cleanup;
    }
    (void)clock_gettime(CLOCK_MONOTONIC, &end);
    printf("decode_wall_seconds=%.6f\n", elapsed_seconds(&begin, &end));
    if (rc != YVEX_OK) {
        fprintf(stderr,
                "audio_vae_cuda=refused code=%d tensor=%s expected=%llu actual=%llu "
                "where=%s message=%s\n",
                failure.code, failure.tensor_name, failure.expected, failure.actual,
                yvex_error_where(err), yvex_error_message(err));
        goto done;
    }
    if (!file_write_exact(arguments->output_path, output, output_values)) {
        rc = YVEX_ERR_IO;
        goto done;
    }
    printf("audio_vae_decode=accepted backend=cuda\n");
    printf("shape=%llux1x%llu\n", result.batch, result.samples_per_channel);
    printf("output_values=%llu\n", result.output_values);
    printf("kernel_launches=%llu\n", result.kernel_launches);
    printf("h2d_bytes=%llu\n", result.h2d_bytes);
    printf("d2h_bytes=%llu\n", result.d2h_bytes);
    printf("peak_device_bytes=%llu\n", result.device_bytes);
    printf("artifact_identity=%s\n", result.artifact_identity);
    printf("execution_identity=%s\n", result.execution_identity);
    printf("residency_identity=%s\n", result.residency_identity);
    if (arguments->reference_path &&
        compare_reference(arguments->reference_path, output, output_values))
        rc = YVEX_ERR_FORMAT;
done:
    free(output);
    free(latent);
    return rc;
}

static int arguments_parse(int argc, char **argv, audio_arguments *arguments)
{
    memset(arguments, 0, sizeof(*arguments));
    arguments->component = "audio_vae";
    if ((argc == 5 || argc == 6) && strcmp(argv[1], "--decode") == 0) {
        arguments->mode = MODE_DECODE_CPU;
        arguments->artifact_path = argv[2]; arguments->input_path = argv[3];
        arguments->output_path = argv[4];
        if (argc == 6) arguments->reference_path = argv[5];
    } else if ((argc == 7 || argc == 8) && strcmp(argv[1], "--decode-cuda") == 0 &&
               parse_extent(argv[5], &arguments->batch) &&
               parse_extent(argv[6], &arguments->latent_steps)) {
        arguments->mode = MODE_DECODE_CUDA;
        arguments->artifact_path = argv[2]; arguments->input_path = argv[3];
        arguments->output_path = argv[4];
        if (argc == 8) arguments->reference_path = argv[7];
    } else if (argc == 6 && strcmp(argv[1], "--unpack-audio") == 0 &&
               parse_extent(argv[4], &arguments->batch) &&
               parse_extent(argv[5], &arguments->latent_steps)) {
        arguments->mode = MODE_UNPACK_AUDIO;
        arguments->input_path = argv[2]; arguments->output_path = argv[3];
    } else if (argc == 4 && strcmp(argv[1], "--admit-component") == 0) {
        arguments->component = argv[2]; arguments->artifact_path = argv[3];
    } else if (argc == 3 && strcmp(argv[1], "--expect-refused") == 0) {
        arguments->mode = MODE_EXPECT_REFUSED; arguments->artifact_path = argv[2];
    } else if (argc == 2) {
        arguments->artifact_path = argv[1];
    } else {
        fprintf(stderr,
                "usage: minimax_h3_audio [--expect-refused] AUDIO_VAE_GGUF\n"
                "       minimax_h3_audio --admit-component COMPONENT GGUF\n"
                "       minimax_h3_audio --decode AUDIO LATENT OUTPUT [REFERENCE]\n"
                "       minimax_h3_audio --decode-cuda AUDIO LATENT OUTPUT BATCH STEPS "
                "[REFERENCE]\n"
                "       minimax_h3_audio --unpack-audio PACKED OUTPUT BATCH STEPS\n");
        return 0;
    }
    return 1;
}

int main(int argc, char **argv)
{
    audio_arguments arguments;
    yvex_artifact_options options = {0};
    yvex_artifact_admission_failure failure = {0};
    yvex_complete_artifact_admission admission;
    yvex_artifact *artifact = NULL;
    yvex_tensor_table *tensors = NULL;
    yvex_gguf *gguf = NULL;
    yvex_error err;
    int rc;

    if (!arguments_parse(argc, argv, &arguments)) return 2;
    if (arguments.mode == MODE_UNPACK_AUDIO) return unpack_audio(&arguments);
    options.path = arguments.artifact_path;
    options.readonly = 1;
    rc = yvex_artifact_open(&artifact, &options, &err);
    if (rc == YVEX_OK) rc = yvex_gguf_open(&gguf, artifact, &err);
    if (rc == YVEX_OK) rc = yvex_tensor_table_from_gguf(&tensors, gguf, &err);
    if (rc == YVEX_OK && arguments.mode != MODE_DECODE_CUDA)
        rc = yvex_graph_register_minimax_h3()->component_admit(
            arguments.component, artifact, gguf, tensors, NULL, &admission, NULL,
            &failure, &err);
    if (arguments.mode == MODE_EXPECT_REFUSED) {
        if (rc == YVEX_OK) {
            fprintf(stderr, "audio_vae_corruption=accepted\n");
            rc = YVEX_ERR_STATE;
        } else {
            printf("audio_vae_corruption=refused field=%s\n", failure.field);
            rc = YVEX_OK;
        }
    } else if (rc != YVEX_OK) {
        (void)fail("component_admission", &failure, &err);
    } else if (arguments.mode == MODE_DECODE_CPU) {
        rc = decode_cpu(&arguments, artifact, gguf, tensors, &err);
    } else if (arguments.mode == MODE_DECODE_CUDA) {
        rc = decode_cuda(&arguments, artifact, gguf, tensors, &err);
    } else {
        printf("component_admission=accepted component=%s\n", arguments.component);
        printf("artifact_identity=%s\n", admission.artifact_identity);
        printf("admission_identity=%s\n", admission.admission_identity);
        printf("payload_byte_identity=%s\n", admission.payload_byte_identity);
        printf("component_identity=%s\n", admission.logical_component_identity);
        printf("tensors=%llu\n", admission.tensor_count);
        printf("payload_bytes=%llu\n", admission.payload_bytes);
        printf("artifact_bytes_hashed=%llu\n", admission.artifact_bytes_hashed);
        printf("artifact_identity_verified=%d\n", admission.artifact_identity_verified);
    }
    yvex_tensor_table_close(tensors);
    yvex_gguf_close(gguf);
    yvex_artifact_close(artifact);
    return rc == YVEX_OK ? 0 : 1;
}
