/*
 * Resolve model references and immutable artifact metadata snapshots.
 *
 * References preserve public model_ref API behavior and do not imply support. Resolving a model
 * reference is not artifact verification, runtime support, generation readiness, benchmark
 * evidence, or release readiness.
 */
#include <yvex/registry.h>

#include <stdio.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <yvex/artifact.h>
#include <yvex/internal/core.h>
#include <yvex/model.h>

static void metadata_dims_text(const unsigned long long *dims,
                               unsigned int rank,
                               char *out,
                               size_t out_cap)
{
    unsigned int i;
    size_t used = 0u;

    if (!out || out_cap == 0u) return;
    out[0] = '\0';
    if (used + 1u < out_cap) out[used++] = '[';
    for (i = 0u; i < rank && used < out_cap; ++i) {
        int n = snprintf(out + used, out_cap - used, "%s%llu",
                         i == 0u ? "" : ",", dims[i]);
        if (n < 0 || (size_t)n >= out_cap - used) {
            out[out_cap - 1u] = '\0';
            return;
        }
        used += (size_t)n;
    }
    if (used + 1u < out_cap) out[used++] = ']';
    out[used] = '\0';
}

static const char *metadata_support(const yvex_model_registry_entry *entry)
{
    if (entry && entry->primary_tensor_name && entry->primary_tensor_name[0]) {
        return "selected-tensor-materialized";
    }
    if (entry && entry->format && entry->format[0]) return "descriptor-only";
    return "";
}

void yvex_model_ref_registry_entry_view(const yvex_model_ref *ref,
                                        yvex_model_registry_entry *entry)
{
    if (!entry) return;
    memset(entry, 0, sizeof(*entry));
    entry->schema_version = YVEX_MODEL_REGISTRY_ENTRY_SCHEMA_CURRENT;
    if (!ref) return;
    entry->alias = ref->alias;
    entry->path = ref->path;
    entry->sha256 = ref->sha256;
    entry->file_size = ref->registered_file_size;
    entry->format = ref->format;
    entry->architecture = ref->architecture;
    entry->tensor_count = ref->tensor_count;
    entry->known_tensor_bytes = ref->known_tensor_bytes;
    entry->primary_tensor_name = ref->primary_tensor_name;
    entry->primary_tensor_role = ref->primary_tensor_role;
    entry->primary_tensor_dtype = ref->primary_tensor_dtype;
    entry->primary_tensor_rank = ref->primary_tensor_rank;
    entry->primary_tensor_dims = ref->primary_tensor_dims;
    entry->primary_tensor_bytes = ref->primary_tensor_bytes;
    entry->support_level = ref->support_level;
    entry->selected_embedding_ready = ref->selected_embedding_ready;
    entry->selected_embedding_hidden_size = ref->selected_embedding_hidden_size;
    entry->selected_embedding_vocab_size = ref->selected_embedding_vocab_size;
    entry->selected_embedding_output_count = ref->selected_embedding_output_count;
    entry->selected_embedding_slice_bytes = ref->selected_embedding_slice_bytes;
    entry->execution_ready = ref->execution_ready;
}

int yvex_model_metadata_snapshot_read(yvex_model_metadata_snapshot *snapshot,
                                      const char *path_or_alias,
                                      yvex_error *err)
{
    yvex_model_context context;
    const yvex_tensor_info *primary = NULL;
    const yvex_tensor_info *embedding = NULL;
    yvex_selected_embedding_shape selected_shape;
    unsigned long long known_bytes = 0ull;
    unsigned long long i;
    int rc;

    if (!snapshot || !path_or_alias || !path_or_alias[0]) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "model_metadata",
                       "metadata snapshot and path are required");
        return YVEX_ERR_INVALID_ARG;
    }
    memset(snapshot, 0, sizeof(*snapshot));
    memset(&context, 0, sizeof(context));
    rc = yvex_model_context_open(path_or_alias, &context, err);
    if (rc != YVEX_OK) return rc;

    snprintf(snapshot->format, sizeof(snapshot->format), "gguf");
    snprintf(snapshot->architecture, sizeof(snapshot->architecture), "%s",
             yvex_arch_name(yvex_model_arch(context.model)));
    for (i = 0ull; i < yvex_tensor_table_count(context.table); ++i) {
        const yvex_tensor_info *tensor = yvex_tensor_table_at(context.table, i);
        if (!tensor) continue;
        if (ULLONG_MAX - known_bytes < tensor->storage_bytes) {
            yvex_model_context_close(&context);
            yvex_error_set(err, YVEX_ERR_BOUNDS, "model_metadata",
                           "known tensor byte count overflow");
            return YVEX_ERR_BOUNDS;
        }
        known_bytes += tensor->storage_bytes;
        if (!primary && strcmp(tensor->name, "token_embd.weight") == 0) {
            primary = tensor;
            embedding = tensor;
        }
    }
    if (!primary && yvex_tensor_table_count(context.table) > 0ull) {
        primary = yvex_tensor_table_at(context.table, 0ull);
    }
    if (primary) {
        snprintf(snapshot->primary_tensor_name,
                 sizeof(snapshot->primary_tensor_name), "%s",
                 primary->name ? primary->name : "");
        snprintf(snapshot->primary_tensor_role,
                 sizeof(snapshot->primary_tensor_role), "%s",
                 yvex_tensor_role_name(primary->role));
        snprintf(snapshot->primary_tensor_dtype,
                 sizeof(snapshot->primary_tensor_dtype), "%s",
                 yvex_dtype_name(primary->dtype));
        metadata_dims_text(primary->dims, primary->rank,
                           snapshot->primary_tensor_dims,
                           sizeof(snapshot->primary_tensor_dims));
        snapshot->entry.primary_tensor_rank = primary->rank;
        snapshot->entry.primary_tensor_bytes = primary->storage_bytes;
    }
    if (embedding) {
        yvex_error shape_error;
        yvex_error_clear(&shape_error);
        memset(&selected_shape, 0, sizeof(selected_shape));
        if (yvex_selected_embedding_shape_validate(embedding, 0u,
                                                    &selected_shape,
                                                    &shape_error) == YVEX_OK) {
            snapshot->entry.selected_embedding_ready = 1;
            snapshot->entry.selected_embedding_hidden_size = selected_shape.hidden_size;
            snapshot->entry.selected_embedding_vocab_size = selected_shape.vocab_size;
            snapshot->entry.selected_embedding_output_count = selected_shape.output_count;
            snapshot->entry.selected_embedding_slice_bytes = selected_shape.slice_bytes;
        }
    }
    snapshot->entry.path = path_or_alias;
    snapshot->entry.format = snapshot->format;
    snapshot->entry.architecture = snapshot->architecture;
    snapshot->entry.tensor_count = yvex_tensor_table_count(context.table);
    snapshot->entry.known_tensor_bytes = known_bytes;
    snapshot->entry.primary_tensor_name = snapshot->primary_tensor_name;
    snapshot->entry.primary_tensor_role = snapshot->primary_tensor_role;
    snapshot->entry.primary_tensor_dtype = snapshot->primary_tensor_dtype;
    snapshot->entry.primary_tensor_dims = snapshot->primary_tensor_dims;
    snprintf(snapshot->support_level, sizeof(snapshot->support_level), "%s",
             metadata_support(&snapshot->entry));
    snapshot->entry.support_level = snapshot->support_level;
    yvex_model_context_close(&context);
    yvex_error_clear(err);
    return YVEX_OK;
}

void yvex_model_ref_clear(yvex_model_ref *ref)
{
    if (!ref) return;
    free((char *)ref->input);
    free((char *)ref->path);
    free((char *)ref->alias);
    free((char *)ref->family);
    free((char *)ref->sha256);
    free((char *)ref->support_level);
    free((char *)ref->format);
    free((char *)ref->architecture);
    free((char *)ref->primary_tensor_name);
    free((char *)ref->primary_tensor_role);
    free((char *)ref->primary_tensor_dtype);
    free((char *)ref->primary_tensor_dims);
    memset(ref, 0, sizeof(*ref));
}

static int set_path_ref(yvex_model_ref *out, const char *input, yvex_error *err)
{
    memset(out, 0, sizeof(*out));
    out->input = yvex_core_strdup(input);
    out->path = yvex_core_strdup(input);
    out->alias = yvex_core_strdup("");
    out->family = yvex_core_strdup("");
    out->sha256 = yvex_core_strdup("");
    out->support_level = yvex_core_strdup("");
    out->format = yvex_core_strdup("");
    out->architecture = yvex_core_strdup("");
    out->primary_tensor_name = yvex_core_strdup("");
    out->primary_tensor_role = yvex_core_strdup("");
    out->primary_tensor_dtype = yvex_core_strdup("");
    out->primary_tensor_dims = yvex_core_strdup("");
    out->status = YVEX_MODEL_REF_STATUS_RESOLVED;
    out->kind = YVEX_MODEL_REF_PATH;
    out->execution_ready = 0;
    if (!out->input || !out->path || !out->alias || !out->family ||
        !out->sha256 || !out->support_level || !out->format ||
        !out->architecture || !out->primary_tensor_name ||
        !out->primary_tensor_role || !out->primary_tensor_dtype ||
        !out->primary_tensor_dims) {
        yvex_model_ref_clear(out);
        yvex_error_set(err, YVEX_ERR_NOMEM, "model_ref", "path reference allocation failed");
        return YVEX_ERR_NOMEM;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

static int is_path_like_reference(const char *input)
{
    size_t len;

    if (!input || !input[0]) return 0;
    if (strchr(input, '/') || strchr(input, '\\')) return 1;
    len = strlen(input);
    if (len >= 5u && strcmp(input + len - 5u, ".gguf") == 0) return 1;
    return 0;
}

static int ref_copy_entry(yvex_model_ref *out,
                                   const char *input,
                                   const yvex_model_registry_entry *entry,
                                   yvex_error *err)
{
    memset(out, 0, sizeof(*out));
    if (!input || !entry ||
        entry->schema_version != YVEX_MODEL_REGISTRY_ENTRY_SCHEMA_CURRENT ||
        !entry->alias || !entry->path) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "model_ref", "input and registry entry are required");
        return YVEX_ERR_INVALID_ARG;
    }
    out->input = yvex_core_strdup(input);
    out->path = yvex_core_strdup(entry->path);
    out->alias = yvex_core_strdup(entry->alias);
    out->family = yvex_core_strdup(entry->family);
    out->sha256 = yvex_core_strdup(entry->sha256);
    out->registered_file_size = entry->file_size;
    out->support_level = yvex_core_strdup(entry->support_level);
    out->format = yvex_core_strdup(entry->format);
    out->architecture = yvex_core_strdup(entry->architecture);
    out->tensor_count = entry->tensor_count;
    out->known_tensor_bytes = entry->known_tensor_bytes;
    out->primary_tensor_name = yvex_core_strdup(entry->primary_tensor_name);
    out->primary_tensor_role = yvex_core_strdup(entry->primary_tensor_role);
    out->primary_tensor_dtype = yvex_core_strdup(entry->primary_tensor_dtype);
    out->primary_tensor_rank = entry->primary_tensor_rank;
    out->primary_tensor_dims = yvex_core_strdup(entry->primary_tensor_dims);
    out->primary_tensor_bytes = entry->primary_tensor_bytes;
    out->selected_embedding_ready = entry->selected_embedding_ready;
    out->selected_embedding_hidden_size = entry->selected_embedding_hidden_size;
    out->selected_embedding_vocab_size = entry->selected_embedding_vocab_size;
    out->selected_embedding_output_count = entry->selected_embedding_output_count;
    out->selected_embedding_slice_bytes = entry->selected_embedding_slice_bytes;
    out->status = YVEX_MODEL_REF_STATUS_RESOLVED;
    out->kind = YVEX_MODEL_REF_ALIAS;
    out->execution_ready = entry->execution_ready;
    if (!out->input || !out->path || !out->alias || !out->family ||
        !out->sha256 || !out->support_level || !out->format ||
        !out->architecture || !out->primary_tensor_name ||
        !out->primary_tensor_role || !out->primary_tensor_dtype ||
        !out->primary_tensor_dims) {
        yvex_model_ref_clear(out);
        yvex_error_set(err, YVEX_ERR_NOMEM, "model_ref", "alias reference allocation failed");
        return YVEX_ERR_NOMEM;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

static void append_available_aliases(char *buf,
                                     size_t cap,
                                     const yvex_model_registry *registry)
{
    unsigned long long i;
    size_t used;

    if (!buf || cap == 0 || !registry) return;
    used = strlen(buf);
    for (i = 0; i < yvex_model_registry_count(registry); ++i) {
        const yvex_model_registry_entry *entry = yvex_model_registry_at(registry, i);
        int n;
        if (!entry || !entry->alias || !entry->alias[0]) continue;
        if (used + 4u >= cap) break;
        n = snprintf(buf + used, cap - used, "\n  %s", entry->alias);
        if (n < 0 || (size_t)n >= cap - used) {
            buf[cap - 1u] = '\0';
            break;
        }
        used += (size_t)n;
    }
}

int yvex_model_ref_resolve(yvex_model_ref *out,
                           const char *input,
                           const yvex_model_ref_options *options,
                           yvex_error *err)
{
    yvex_model_registry *registry = NULL;
    yvex_model_registry_options registry_options;
    const yvex_model_registry_entry *entry;
    char message[1024];
    int rc;

    if (!out || !input || !input[0]) {
        if (out) {
            memset(out, 0, sizeof(*out));
            out->status = YVEX_MODEL_REF_STATUS_INVALID;
        }
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "model_ref", "model reference is required");
        return YVEX_ERR_INVALID_ARG;
    }

    memset(out, 0, sizeof(*out));
    if (access(input, F_OK) == 0) {
        return set_path_ref(out, input, err);
    }

    if (is_path_like_reference(input)) {
        return set_path_ref(out, input, err);
    }

    if (options && !options->allow_registry) {
        out->status = YVEX_MODEL_REF_STATUS_NOT_FOUND;
        out->kind = YVEX_MODEL_REF_UNKNOWN;
        yvex_error_setf(err, YVEX_ERR_INVALID_ARG, "model_ref",
                        "model path does not exist: %s", input);
        return YVEX_ERR_INVALID_ARG;
    }

    memset(&registry_options, 0, sizeof(registry_options));
    registry_options.registry_path = options ? options->registry_path : NULL;
    registry_options.create_if_missing = 0;
    rc = yvex_model_registry_open(&registry, &registry_options, err);
    if (rc != YVEX_OK) {
        const char *env_registry = getenv("YVEX_MODELS_REGISTRY");

        out->status = YVEX_MODEL_REF_STATUS_REGISTRY_UNAVAILABLE;
        out->kind = YVEX_MODEL_REF_UNKNOWN;
        if (env_registry && env_registry[0]) {
            yvex_error_setf(err, YVEX_ERR_IO, "model_ref",
                            "model registry unavailable for reference: %s; "
                            "YVEX_MODELS_REGISTRY=%s; hint: register the alias in that "
                            "registry, unset YVEX_MODELS_REGISTRY, or pass an existing path",
                            input, env_registry);
        } else {
            yvex_error_setf(err, YVEX_ERR_IO, "model_ref",
                            "model registry unavailable for reference: %s; hint: run "
                            "'./yvex models list' or pass an existing path",
                            input);
        }
        return YVEX_ERR_IO;
    }

    entry = yvex_model_registry_find(registry, input);
    if (!entry) {
        snprintf(message, sizeof(message),
                 "model reference not found: %s; hint: run './yvex models list'; available models:",
                 input);
        append_available_aliases(message, sizeof(message), registry);
        out->status = YVEX_MODEL_REF_STATUS_NOT_FOUND;
        out->kind = YVEX_MODEL_REF_UNKNOWN;
        yvex_model_registry_close(registry);
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "model_ref", message);
        return YVEX_ERR_INVALID_ARG;
    }

    if (access(entry->path, F_OK) != 0) {
        out->status = YVEX_MODEL_REF_STATUS_ALIAS_PATH_MISSING;
        out->kind = YVEX_MODEL_REF_ALIAS;
        out->alias = yvex_core_strdup(entry->alias);
        out->path = yvex_core_strdup(entry->path);
        yvex_error_setf(err, YVEX_ERR_IO, "model_ref",
                        "model alias exists but path is missing: alias=%s path=%s; hint: "
                        "update or remove the registry entry with "
                        "'./yvex models remove %s'",
                        entry->alias, entry->path, entry->alias);
        yvex_model_registry_close(registry);
        return YVEX_ERR_IO;
    }

    rc = ref_copy_entry(out, input, entry, err);
    yvex_model_registry_close(registry);
    return rc;
}
