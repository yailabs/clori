/*
 * Publish explicit model-target sidecars through atomic bounded writes.
 *
 * Sidecar writer APIs write explicit local files only and never process operator streams. Sidecar
 * writer availability does not create artifact emission capability, runtime support, generation
 * support, benchmark evidence, or release readiness.
 */
#include <yvex/internal/model_target.h>
#include <yvex/internal/core.h>
#include <yvex/internal/io.h>

#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int sidecar_open_tmp(const char *path, char *tmp, size_t tmp_cap, FILE **out)
{
    yvex_error err;
    int n;

    if (!path || !tmp || tmp_cap == 0 || !out) return 0;
    *out = NULL;
    yvex_error_clear(&err);
    if (yvex_core_mkdir_parent(path, "model_target.sidecar", &err) != YVEX_OK)
        return 0;
    n = snprintf(tmp, tmp_cap, "%s.tmp", path);
    if (n < 0 || (size_t)n >= tmp_cap) return 0;
    *out = fopen(tmp, "wb");
    return *out != NULL;
}

static int sidecar_close_tmp(FILE *fp, const char *tmp, const char *path)
{
    if (!fp || !tmp || !path) return 0;
    if (fclose(fp) != 0) {
        remove(tmp);
        return 0;
    }
    if (rename(tmp, path) != 0) {
        remove(tmp);
        return 0;
    }
    return 1;
}

/* Serialize one typed sidecar variant through the shared atomic file boundary. */
int yvex_model_target_write_sidecar(yvex_model_target_sidecar_kind kind, const char *path,
                                    const char *target_id, const char *family,
                                    const char *status, const char *coverage)
{
    char tmp[1024];
    FILE *fp;

    if (!path || !path[0]) return 1;
    if (kind < YVEX_MODEL_TARGET_SIDECAR_TENSOR_MAP ||
        kind > YVEX_MODEL_TARGET_SIDECAR_TOKENIZER ||
        !sidecar_open_tmp(path, tmp, sizeof(tmp), &fp)) return 0;
    fprintf(fp, "{\n");
    if (kind == YVEX_MODEL_TARGET_SIDECAR_TOKENIZER)
        fprintf(fp, "  \"schema_version\": \"yvex.source.tokenizer_map.v1\",\n");
    else
        fprintf(fp, "  \"schema\": \"yvex.source.%s.v1\",\n",
                kind == YVEX_MODEL_TARGET_SIDECAR_TENSOR_MAP ? "tensor_map"
                                                            : "output_head_map");
    fprintf(fp, "  \"row\": \"%s\",\n",
            kind == YVEX_MODEL_TARGET_SIDECAR_TOKENIZER
                ? "V010.MAP.7" : "MODELS.SOURCE.MAP.HANDOFF.0");
    fprintf(fp, "  \"target_id\": ");
    yvex_file_json_write_string(fp, target_id);
    fprintf(fp, ",\n  \"family\": ");
    yvex_file_json_write_string(fp, family);
    if (kind == YVEX_MODEL_TARGET_SIDECAR_TENSOR_MAP) {
        fprintf(fp, ",\n  \"tensor_map_status\": ");
        yvex_file_json_write_string(fp, status);
        fprintf(fp, ",\n  \"required_role_coverage_status\": ");
        yvex_file_json_write_string(fp, coverage);
    } else if (kind == YVEX_MODEL_TARGET_SIDECAR_OUTPUT_HEAD) {
        fprintf(fp, ",\n  \"output_head_map_status\": ");
        yvex_file_json_write_string(fp, status);
        fprintf(fp, ",\n  \"output_head_status\": ");
        yvex_file_json_write_string(fp, strcmp(status, "output-head-missing") == 0
                                            ? "missing" : "present");
    } else {
        fprintf(fp, ",\n  \"tokenizer_map_status\": ");
        yvex_file_json_write_string(fp, status);
        fprintf(fp, ",\n  \"status\": ");
        yvex_file_json_write_string(fp, status);
    }
    fprintf(fp, "\n}\n");
    return sidecar_close_tmp(fp, tmp, path);
}
