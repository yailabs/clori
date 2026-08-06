/* Verify the exact source-faithful Audio VAE artifact without executing tensor payloads. */
#include <stdio.h>
#include <string.h>

#include <yvex/artifact.h>
#include <yvex/gguf.h>
#include <yvex/internal/artifact.h>
#include <yvex/internal/families/minimax_h3.h>

static int fail(const char *phase, const yvex_artifact_admission_failure *failure,
                const yvex_error *err)
{
    fprintf(stderr,
            "%s=refused code=%s field=%s expected=%llu actual=%llu where=%s message=%s\n",
            phase, yvex_artifact_admission_code_name(failure->code), failure->field,
            failure->expected, failure->actual, yvex_error_where(err), yvex_error_message(err));
    return 1;
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
    int rc;

    if (argc == 3 && strcmp(argv[1], "--expect-refused") == 0) {
        expect_refused = 1;
        options.path = argv[2];
    } else if (argc == 2) {
        options.path = argv[1];
    } else {
        fprintf(stderr, "usage: minimax_h3_audio [--expect-refused] AUDIO_VAE_GGUF\n");
        return 2;
    }
    options.readonly = 1;
    rc = yvex_artifact_open(&artifact, &options, &err);
    if (rc == YVEX_OK) rc = yvex_gguf_open(&gguf, artifact, &err);
    if (rc == YVEX_OK) rc = yvex_tensor_table_from_gguf(&tensors, gguf, &err);
    if (rc == YVEX_OK)
        rc = yvex_graph_register_minimax_h3()->audio_vae_admit(
            artifact, gguf, tensors, &admission, &failure, &err);
    if (expect_refused) {
        if (rc == YVEX_OK) {
            fprintf(stderr, "audio_vae_corruption=accepted\n");
            rc = YVEX_ERR_STATE;
        } else {
            printf("audio_vae_corruption=refused field=%s\n", failure.field);
            rc = YVEX_OK;
        }
    } else if (rc != YVEX_OK) {
        (void)fail("audio_vae_admission", &failure, &err);
    } else {
        printf("audio_vae_admission=accepted\n");
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
