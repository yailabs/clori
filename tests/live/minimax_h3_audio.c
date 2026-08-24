/* Verify the exact source-faithful Audio VAE artifact without executing tensor payloads. */
#include <stdio.h>
#include <math.h>
#include <string.h>

#include <yvex/artifact.h>
#include <yvex/gguf.h>
#include <yvex/internal/artifact.h>
#include <yvex/internal/families/minimax_h3.h>
#include <yvex/internal/runtime.h>

static int fail(const char *phase, const yvex_artifact_admission_failure *failure,
                const yvex_error *err)
{
    fprintf(stderr,
            "%s=refused code=%s field=%s expected=%llu actual=%llu where=%s message=%s\n",
            phase, yvex_artifact_admission_code_name(failure->code), failure->field,
            failure->expected, failure->actual, yvex_error_where(err), yvex_error_message(err));
    return 1;
}

static int compare_reference(const char *path, const float output[800])
{
    float reference[800];
    float max_absolute = 0.0f;
    FILE *file = fopen(path, "rb");
    unsigned long long index;

    if (!file || fread(reference, sizeof(reference), 1u, file) != 1u ||
        fgetc(file) != EOF) {
        if (file) fclose(file);
        fprintf(stderr, "audio_vae_reference_read=refused\n");
        return 1;
    }
    fclose(file);
    for (index = 0ull; index < 800ull; ++index) {
        float absolute = fabsf(output[index] - reference[index]);
        if (absolute > max_absolute) max_absolute = absolute;
    }
    printf("oracle_max_absolute_error=%.9g\n", max_absolute);
    if (max_absolute > 1.0e-5f) {
        fprintf(stderr, "audio_vae_conformance=refused tolerance=1e-5\n");
        return 1;
    }
    printf("audio_vae_conformance=accepted tolerance=1e-5\n");
    return 0;
}

int main(int argc, char **argv)
{
    yvex_artifact_options options = {0};
    yvex_artifact_admission_failure failure = {0};
    yvex_complete_artifact_admission admission;
    yvex_artifact *artifact = NULL;
    yvex_tensor_table *tensors = NULL;
    yvex_gguf *gguf = NULL;
    yvex_error err;
    int expect_refused = 0;
    int decode = 0;
    const char *latent_path = NULL;
    const char *output_path = NULL;
    const char *reference_path = NULL;
    const char *component = "audio_vae";
    int rc;

    if ((argc == 5 || argc == 6) && strcmp(argv[1], "--decode") == 0) {
        decode = 1;
        options.path = argv[2];
        latent_path = argv[3];
        output_path = argv[4];
        if (argc == 6) reference_path = argv[5];
    } else if (argc == 4 && strcmp(argv[1], "--admit-component") == 0) {
        component = argv[2];
        options.path = argv[3];
    } else if (argc == 3 && strcmp(argv[1], "--expect-refused") == 0) {
        expect_refused = 1;
        options.path = argv[2];
    } else if (argc == 2) {
        options.path = argv[1];
    } else {
        fprintf(stderr,
                "usage: minimax_h3_audio [--expect-refused] AUDIO_VAE_GGUF\n"
                "       minimax_h3_audio --admit-component COMPONENT GGUF\n"
                "       minimax_h3_audio --decode AUDIO_VAE_GGUF LATENT_F32 OUTPUT_F32 "
                "[REFERENCE_F32]\n");
        return 2;
    }
    options.readonly = 1;
    rc = yvex_artifact_open(&artifact, &options, &err);
    if (rc == YVEX_OK) rc = yvex_gguf_open(&gguf, artifact, &err);
    if (rc == YVEX_OK) rc = yvex_tensor_table_from_gguf(&tensors, gguf, &err);
    if (rc == YVEX_OK)
        rc = yvex_graph_register_minimax_h3()->component_admit(
            component, artifact, gguf, tensors, NULL, &admission, NULL, &failure, &err);
    if (expect_refused) {
        if (rc == YVEX_OK) {
            fprintf(stderr, "audio_vae_corruption=accepted\n");
            rc = YVEX_ERR_STATE;
        } else {
            printf("audio_vae_corruption=refused field=%s\n", failure.field);
            rc = YVEX_OK;
        }
    } else if (rc != YVEX_OK) {
        (void)fail("component_admission", &failure, &err);
    } else if (decode) {
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
        yvex_component_execution_request execution_request = {0};
        yvex_component_execution_result result;
        yvex_component_failure execution_failure;
        float latent[32];
        float output[800];
        FILE *file = fopen(latent_path, "rb");

        memset(&execution_failure, 0, sizeof(execution_failure));

        if (!file || fread(latent, sizeof(latent), 1u, file) != 1u ||
            fgetc(file) != EOF) {
            fprintf(stderr, "latent_read=refused\n");
            if (file) fclose(file);
            rc = YVEX_ERR_FORMAT;
        } else {
            fclose(file);
            rc = yvex_runtime_component_api_get()->plan_build(
                &plan_request, &component_plan, &execution_failure, &err);
            execution_request.plan = &component_plan;
            execution_request.input = latent;
            execution_request.output = output;
            execution_request.output_capacity = 800ull;
            if (rc == YVEX_OK)
                rc = yvex_runtime_component_api_get()->execute(
                    artifact, gguf, tensors, &execution_request, &result,
                    &execution_failure, &err);
            if (rc != YVEX_OK) {
                fprintf(stderr,
                        "audio_vae_decode=refused code=%d tensor=%s expected=%llu actual=%llu "
                        "where=%s message=%s\n",
                        execution_failure.code, execution_failure.tensor_name,
                        execution_failure.expected, execution_failure.actual,
                        yvex_error_where(&err), yvex_error_message(&err));
            } else {
                file = fopen(output_path, "wb");
                if (!file) {
                    fprintf(stderr, "audio_vae_output_write=refused\n");
                    rc = YVEX_ERR_IO;
                } else {
                    size_t written = fwrite(output, sizeof(output), 1u, file);
                    int close_rc = fclose(file);
                    file = NULL;
                    if (written != 1u || close_rc != 0) {
                        fprintf(stderr, "audio_vae_output_write=refused\n");
                        rc = YVEX_ERR_IO;
                    } else {
                        printf("audio_vae_decode=accepted\n");
                        printf("samples=%llu\n", result.output_dims[2]);
                        printf("tensor_reads=%llu\n", result.tensor_reads);
                        printf("payload_bytes_read=%llu\n", result.payload_bytes_read);
                        printf("peak_workspace_bytes=%llu\n", result.peak_workspace_bytes);
                        printf("execution_identity=%s\n", result.execution_identity);
                        if (reference_path && compare_reference(reference_path, output) != 0)
                            rc = YVEX_ERR_FORMAT;
                    }
                }
            }
        }
    } else {
        printf("component_admission=accepted component=%s\n", component);
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
