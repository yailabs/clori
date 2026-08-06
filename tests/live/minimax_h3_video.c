/* Verify and execute the exact reduced MiniMax-H3 Visual VAE component artifact. */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <yvex/artifact.h>
#include <yvex/gguf.h>
#include <yvex/internal/artifact.h>
#include <yvex/internal/families/minimax_h3.h>

static int file_read_exact(const char *path, void *output, size_t bytes)
{
    FILE *file = fopen(path, "rb");
    int accepted = file && fread(output, bytes, 1u, file) == 1u && fgetc(file) == EOF;

    if (file) fclose(file);
    return accepted;
}

static int compare_reference(const char *path, const float output[3072])
{
    float reference[3072];
    float maximum = 0.0f;
    unsigned long long index;

    if (!file_read_exact(path, reference, sizeof(reference))) {
        fprintf(stderr, "video_vae_reference_read=refused\n");
        return 1;
    }
    for (index = 0ull; index < 3072ull; ++index) {
        float absolute = fabsf(output[index] - reference[index]);
        if (absolute > maximum) maximum = absolute;
    }
    printf("oracle_max_absolute_error=%.9g\n", maximum);
    if (maximum > 1.0e-5f) {
        fprintf(stderr, "video_vae_conformance=refused tolerance=1e-5\n");
        return 1;
    }
    printf("video_vae_conformance=accepted tolerance=1e-5\n");
    return 0;
}

static int output_write(const char *path, const float output[3072])
{
    FILE *file = fopen(path, "wb");
    size_t written;
    int close_rc;

    if (!file) return 0;
    written = fwrite(output, sizeof(float), 3072u, file);
    close_rc = fclose(file);
    return written == 3072u && close_rc == 0;
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
    int expect_refused = 0, decode = 0;
    int rc;

    if ((argc == 5 || argc == 6) && strcmp(argv[1], "--decode") == 0) {
        decode = 1;
        options.path = argv[2];
        latent_path = argv[3];
        output_path = argv[4];
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
                "[REFERENCE_F32]\n");
        return 2;
    }
    options.readonly = 1;
    rc = yvex_artifact_open(&artifact, &options, &err);
    if (rc == YVEX_OK) rc = yvex_gguf_open(&gguf, artifact, &err);
    if (rc == YVEX_OK) rc = yvex_tensor_table_from_gguf(&tensors, gguf, &err);
    if (decode && rc == YVEX_OK) {
        yvex_minimax_h3_video_decode_options decode_options;
        yvex_minimax_h3_video_decode_result result;
        yvex_minimax_h3_component_execution_failure execution_failure;
        float latent[24];
        float output[3072];

        memset(&decode_options, 0, sizeof(decode_options));
        memset(&execution_failure, 0, sizeof(execution_failure));
        if (!file_read_exact(latent_path, latent, sizeof(latent))) {
            fprintf(stderr, "video_vae_latent_read=refused\n");
            rc = YVEX_ERR_FORMAT;
        }
        decode_options.latent = latent;
        decode_options.output = output;
        decode_options.output_capacity = 3072ull;
        decode_options.batch = 1ull;
        decode_options.latent_channels = 24ull;
        decode_options.latent_frames = 1ull;
        decode_options.latent_height = 1ull;
        decode_options.latent_width = 1ull;
        decode_options.max_workspace_bytes = 256ull * 1024ull * 1024ull;
        if (rc == YVEX_OK)
            rc = yvex_graph_register_minimax_h3()->video_vae_execute_artifact_cpu(
                artifact, gguf, tensors, &decode_options, &result,
                &execution_failure, &err);
        if (rc != YVEX_OK) {
            fprintf(stderr,
                    "video_vae_decode=refused code=%d tensor=%s expected=%llu actual=%llu "
                    "where=%s message=%s\n",
                    execution_failure.code, execution_failure.tensor_name,
                    execution_failure.expected, execution_failure.actual,
                    yvex_error_where(&err), yvex_error_message(&err));
        } else if (!output_write(output_path, output)) {
            fprintf(stderr, "video_vae_output_write=refused\n");
            rc = YVEX_ERR_IO;
        } else {
            printf("video_vae_decode=accepted\n");
            printf("output_shape=1x3x%llux%llux%llu\n",
                   result.frames, result.height, result.width);
            printf("tensor_reads=%llu\n", result.tensor_reads);
            printf("payload_bytes_read=%llu\n", result.payload_bytes_read);
            printf("peak_workspace_bytes=%llu\n", result.peak_workspace_bytes);
            printf("execution_identity=%s\n", result.execution_identity);
            if (reference_path && compare_reference(reference_path, output) != 0)
                rc = YVEX_ERR_FORMAT;
        }
    } else if (rc == YVEX_OK) {
        rc = yvex_graph_register_minimax_h3()->video_vae_admit(
            artifact, gguf, tensors, &admission, &admission_failure, &err);
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
