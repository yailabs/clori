/*
 * Serialize canonical model registry state through atomic file publication.
 *
 * Writer output targets caller-provided local file paths only and never operator streams. Registry
 * JSON writing is not artifact emission, model verification, runtime support, generation
 * readiness, benchmark evidence, or release readiness.
 */
#include <yvex/internal/model_artifact.h>
#include <yvex/internal/core.h>

#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static void write_escaped(FILE *fp, const char *s)
{
    if (!s) s = "";
    fputc('"', fp);
    while (*s) {
        unsigned char ch = (unsigned char)*s++;
        if (ch == '"' || ch == '\\') {
            fputc('\\', fp);
            fputc((int)ch, fp);
        } else if (ch == '\n') {
            fputs("\\n", fp);
        } else if (ch == '\r') {
            fputs("\\r", fp);
        } else if (ch == '\t') {
            fputs("\\t", fp);
        } else {
            fputc((int)ch, fp);
        }
    }
    fputc('"', fp);
}

static void write_field(FILE *fp, const char *indent, const char *key, const char *value, int comma)
{
    fputs(indent, fp);
    fprintf(fp, "\"%s\": ", key);
    write_escaped(fp, value);
    fprintf(fp, "%s\n", comma ? "," : "");
}

static void write_u64_field(FILE *fp,
                            const char *indent,
                            const char *key,
                            unsigned long long value,
                            int comma)
{
    fputs(indent, fp);
    fprintf(fp, "\"%s\": %llu%s\n", key, value, comma ? "," : "");
}

int yvex_model_registry_write_json_file(const yvex_model_registry *registry,
                                        const char *path,
                                        yvex_error *err)
{
    char tmp[4096];
    FILE *fp;
    unsigned long long i;
    int n;
    int rc;

    if (!registry || !path || !path[0]) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "model_registry_json", "registry and path are required");
        return YVEX_ERR_INVALID_ARG;
    }
    rc = yvex_core_mkdir_parent(path, "model_registry_json", err);
    if (rc != YVEX_OK) return rc;
    n = snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    if (n < 0 || (size_t)n >= sizeof(tmp)) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "model_registry_json", "temporary path too long");
        return YVEX_ERR_BOUNDS;
    }
    fp = fopen(tmp, "wb");
    if (!fp) {
        yvex_error_setf(err, YVEX_ERR_IO, "model_registry_json", "cannot write registry: %s", tmp);
        return YVEX_ERR_IO;
    }
    fprintf(fp, "{\n");
    write_field(fp, "  ", "schema", "yvex.models.local.v3", 1);
    fprintf(fp, "  \"models\": [\n");
    for (i = 0; i < registry->count; ++i) {
        const yvex_model_registry_owned_entry *e = &registry->entries[i];
        fprintf(fp, "    {\n");
        write_field(fp, "      ", "alias", e->alias, 1);
        write_field(fp, "      ", "family", e->family, 1);
        write_field(fp, "      ", "model", e->model, 1);
        write_field(fp, "      ", "scope", e->scope, 1);
        write_field(fp, "      ", "artifact_class", e->artifact_class, 1);
        write_field(fp, "      ", "qprofile", e->qprofile, 1);
        write_field(fp, "      ", "calibration", e->calibration, 1);
        write_field(fp, "      ", "producer", e->producer, 1);
        write_field(fp, "      ", "schema_version", e->schema_version, 1);
        write_field(fp, "      ", "path", e->path, 1);
        write_field(fp, "      ", "sha256", e->sha256, 1);
        write_u64_field(fp, "      ", "file_size", e->file_size, 1);
        write_field(fp, "      ", "format", e->format, 1);
        write_field(fp, "      ", "architecture", e->architecture, 1);
        write_u64_field(fp, "      ", "tensor_count", e->tensor_count, 1);
        write_u64_field(fp, "      ", "known_tensor_bytes", e->known_tensor_bytes, 1);
        write_field(fp, "      ", "primary_tensor_name", e->primary_tensor_name, 1);
        write_field(fp, "      ", "primary_tensor_role", e->primary_tensor_role, 1);
        write_field(fp, "      ", "primary_tensor_dtype", e->primary_tensor_dtype, 1);
        write_u64_field(fp, "      ", "primary_tensor_rank", e->primary_tensor_rank, 1);
        write_field(fp, "      ", "primary_tensor_dims", e->primary_tensor_dims, 1);
        write_u64_field(fp, "      ", "primary_tensor_bytes", e->primary_tensor_bytes, 1);
        write_field(fp, "      ", "support_level", e->support_level, 1);
        fprintf(fp, "      \"selected_embedding_ready\": %s,\n",
                e->selected_embedding_ready ? "true" : "false");
        write_u64_field(fp, "      ", "selected_embedding_hidden_size",
                        e->selected_embedding_hidden_size, 1);
        write_u64_field(fp, "      ", "selected_embedding_vocab_size",
                        e->selected_embedding_vocab_size, 1);
        write_u64_field(fp, "      ", "selected_embedding_output_count",
                        e->selected_embedding_output_count, 1);
        write_u64_field(fp, "      ", "selected_embedding_slice_bytes",
                        e->selected_embedding_slice_bytes, 1);
        fprintf(fp, "      \"execution_ready\": %s,\n", e->execution_ready ? "true" : "false");
        write_field(fp, "      ", "runtime_binding", e->runtime_binding, 1);
        write_field(fp, "      ", "runtime_target", e->runtime_target, 1);
        write_field(fp, "      ", "runtime_backend", e->runtime_backend, 1);
        write_u64_field(fp, "      ", "runtime_context", e->runtime_context, 0);
        fprintf(fp, "    }%s\n", (i + 1u < registry->count) ? "," : "");
    }
    fprintf(fp, "  ]\n");
    fprintf(fp, "}\n");
    if (fflush(fp) != 0 || fclose(fp) != 0) {
        remove(tmp);
        yvex_error_setf(err, YVEX_ERR_IO, "model_registry_json", "cannot close registry: %s", tmp);
        return YVEX_ERR_IO;
    }
    if (rename(tmp, path) != 0) {
        remove(tmp);
        yvex_error_setf(err, YVEX_ERR_IO, "model_registry_json", "cannot replace registry: %s", path);
        return YVEX_ERR_IO;
    }
    return YVEX_OK;
}
