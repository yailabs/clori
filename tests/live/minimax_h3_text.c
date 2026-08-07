/* Exercise the exact Text Encoder embedding through staged GB10 residency. */
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <yvex/artifact.h>
#include <yvex/gguf.h>
#include <yvex/internal/families/minimax_h3.h>

enum { TEXT_HIDDEN = 5120u };

static int reference_compare(const char *path, const float output[TEXT_HIDDEN])
{
    float reference[TEXT_HIDDEN];
    float maximum = 0.0f;
    FILE *file = fopen(path, "rb");
    unsigned long long index;

    if (!file || fread(reference, sizeof(reference), 1u, file) != 1u || fgetc(file) != EOF) {
        if (file) fclose(file);
        fprintf(stderr, "text_reference_read=refused\n");
        return 0;
    }
    fclose(file);
    for (index = 0ull; index < TEXT_HIDDEN; ++index) {
        float absolute = fabsf(reference[index] - output[index]);
        if (absolute > maximum) maximum = absolute;
    }
    printf("oracle_max_absolute_error=%.9g\n", maximum);
    return maximum == 0.0f;
}

static int output_write(const char *path, const float output[TEXT_HIDDEN])
{
    FILE *file = fopen(path, "wb");
    size_t written;
    int close_rc;

    if (!file) return 0;
    written = fwrite(output, sizeof(float), TEXT_HIDDEN, file);
    close_rc = fclose(file);
    return written == TEXT_HIDDEN && close_rc == 0;
}

int main(int argc, char **argv)
{
    yvex_artifact_options options = {0};
    yvex_artifact *artifact = NULL;
    yvex_tensor_table *tensors = NULL;
    yvex_gguf *gguf = NULL;
    yvex_minimax_h3_conditioning_result result;
    float output[TEXT_HIDDEN];
    char *end = NULL;
    unsigned long token_value;
    unsigned int token;
    yvex_error err;
    int rc;

    if (argc != 5) {
        fprintf(stderr, "usage: minimax_h3_text TEXT_GGUF TOKEN OUTPUT_F32 REFERENCE_F32\n");
        return 2;
    }
    errno = 0;
    token_value = strtoul(argv[2], &end, 10);
    if (errno || !end || *end || token_value > 151935ul) {
        fprintf(stderr, "text_token=refused\n");
        return 2;
    }
    token = (unsigned int)token_value;
    options.path = argv[1];
    options.readonly = 1;
    rc = yvex_artifact_open(&artifact, &options, &err);
    if (rc == YVEX_OK) rc = yvex_gguf_open(&gguf, artifact, &err);
    if (rc == YVEX_OK) rc = yvex_tensor_table_from_gguf(&tensors, gguf, &err);
    if (rc == YVEX_OK)
        rc = yvex_graph_register_minimax_h3()->text_encoder_embed_artifact_cuda(
            artifact, gguf, tensors, &token, 1ull, output, TEXT_HIDDEN,
            70ull * 1024ull * 1024ull * 1024ull, 256ull * 1024ull * 1024ull,
            &result, &err);
    if (rc == YVEX_OK && !output_write(argv[3], output)) {
        yvex_error_set(&err, YVEX_ERR_IO, "minimax-h3.text.output",
                       "conditioning output could not be written completely");
        rc = YVEX_ERR_IO;
    }
    if (rc == YVEX_OK && !reference_compare(argv[4], output)) {
        yvex_error_set(&err, YVEX_ERR_FORMAT, "minimax-h3.text.oracle",
                       "YVEX conditioning differs from the independent BF16 oracle");
        rc = YVEX_ERR_FORMAT;
    }
    if (rc == YVEX_OK) {
        printf("text_conditioning=accepted\n");
        printf("tokens=%llu hidden=%llu resident_bytes=%llu\n",
               result.token_count, result.hidden_width, result.resident_bytes);
        printf("kernel_launches=%llu h2d_bytes=%llu d2h_bytes=%llu device_bytes=%llu\n",
               result.kernel_launches, result.h2d_bytes, result.d2h_bytes, result.device_bytes);
        printf("residency_identity=%s\nexecution_identity=%s\n",
               result.residency_identity, result.execution_identity);
    } else {
        fprintf(stderr, "text_conditioning=refused where=%s message=%s\n",
                yvex_error_where(&err), yvex_error_message(&err));
    }
    yvex_tensor_table_close(tensors);
    yvex_gguf_close(gguf);
    yvex_artifact_close(artifact);
    return rc == YVEX_OK ? 0 : 1;
}
