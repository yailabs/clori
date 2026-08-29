/*
 * Own canonical model registry storage, metadata comparison, and scanning.
 *
 * Registry operations preserve public model_registry API behavior and never serialize operator
 * output. Registry facts are not artifact emission, source verification, runtime support,
 * generation readiness, benchmark evidence, or release readiness.
 */
#include <yvex/registry.h>
#include <yvex/internal/core.h>
#include <yvex/internal/model_artifact.h>

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <yvex/artifact.h>

static int registry_parse_json(const char *path,
                                               yvex_model_registry *registry,
                                               yvex_error *err);

typedef struct {
    size_t owned_offset;
    size_t view_offset;
    const char *json_key;
} registry_string_field;

static const registry_string_field registry_string_fields[] = {
    {offsetof(yvex_model_registry_owned_entry, alias),
     offsetof(yvex_model_registry_entry, alias), "alias"},
    {offsetof(yvex_model_registry_owned_entry, family),
     offsetof(yvex_model_registry_entry, family), "family"},
    {offsetof(yvex_model_registry_owned_entry, model),
     offsetof(yvex_model_registry_entry, model), "model"},
    {offsetof(yvex_model_registry_owned_entry, scope),
     offsetof(yvex_model_registry_entry, scope), "scope"},
    {offsetof(yvex_model_registry_owned_entry, artifact_class),
     offsetof(yvex_model_registry_entry, artifact_class), "artifact_class"},
    {offsetof(yvex_model_registry_owned_entry, qprofile),
     offsetof(yvex_model_registry_entry, qprofile), "qprofile"},
    {offsetof(yvex_model_registry_owned_entry, calibration),
     offsetof(yvex_model_registry_entry, calibration), "calibration"},
    {offsetof(yvex_model_registry_owned_entry, producer),
     offsetof(yvex_model_registry_entry, producer), "producer"},
    {offsetof(yvex_model_registry_owned_entry, artifact_schema),
     offsetof(yvex_model_registry_entry, artifact_schema), "schema_version"},
    {offsetof(yvex_model_registry_owned_entry, path),
     offsetof(yvex_model_registry_entry, path), "path"},
    {offsetof(yvex_model_registry_owned_entry, sha256),
     offsetof(yvex_model_registry_entry, sha256), "sha256"},
    {offsetof(yvex_model_registry_owned_entry, format),
     offsetof(yvex_model_registry_entry, format), "format"},
    {offsetof(yvex_model_registry_owned_entry, architecture),
     offsetof(yvex_model_registry_entry, architecture), "architecture"},
    {offsetof(yvex_model_registry_owned_entry, primary_tensor_name),
     offsetof(yvex_model_registry_entry, primary_tensor_name), "primary_tensor_name"},
    {offsetof(yvex_model_registry_owned_entry, primary_tensor_role),
     offsetof(yvex_model_registry_entry, primary_tensor_role), "primary_tensor_role"},
    {offsetof(yvex_model_registry_owned_entry, primary_tensor_dtype),
     offsetof(yvex_model_registry_entry, primary_tensor_dtype), "primary_tensor_dtype"},
    {offsetof(yvex_model_registry_owned_entry, primary_tensor_dims),
     offsetof(yvex_model_registry_entry, primary_tensor_dims), "primary_tensor_dims"},
    {offsetof(yvex_model_registry_owned_entry, support_level),
     offsetof(yvex_model_registry_entry, support_level), "support_level"},
    {offsetof(yvex_model_registry_owned_entry, runtime_profile),
     offsetof(yvex_model_registry_entry, runtime_profile), "runtime_profile"},
    {offsetof(yvex_model_registry_owned_entry, runtime_installation),
     offsetof(yvex_model_registry_entry, runtime_installation), "runtime_installation"},
    {offsetof(yvex_model_registry_owned_entry, runtime_binding),
     offsetof(yvex_model_registry_entry, runtime_binding), "runtime_binding"},
    {offsetof(yvex_model_registry_owned_entry, runtime_target),
     offsetof(yvex_model_registry_entry, runtime_target), "runtime_target"},
    {offsetof(yvex_model_registry_owned_entry, runtime_backend),
     offsetof(yvex_model_registry_entry, runtime_backend), "runtime_backend"},
    {offsetof(yvex_model_registry_owned_entry, runtime_engine_kind),
     offsetof(yvex_model_registry_entry, runtime_engine_kind), "runtime_engine_kind"},
    {offsetof(yvex_model_registry_owned_entry, runtime_execution_strategy),
     offsetof(yvex_model_registry_entry, runtime_execution_strategy),
     "runtime_execution_strategy"}
};

static size_t registry_string_field_count(void)
{
    return sizeof(registry_string_fields) / sizeof(registry_string_fields[0]);
}

static char **owned_string_field(yvex_model_registry_owned_entry *entry,
                                 size_t offset)
{
    return (char **)(void *)((unsigned char *)entry + offset);
}

static const char **view_string_field(yvex_model_registry_entry *entry,
                                      size_t offset)
{
    return (const char **)(void *)((unsigned char *)entry + offset);
}

static const char *view_string_value(const yvex_model_registry_entry *entry,
                                     size_t offset)
{
    return *(const char *const *)(const void *)
        ((const unsigned char *)entry + offset);
}

static void registry_owned_entry_clear(yvex_model_registry_owned_entry *entry)
{
    size_t field;

    if (!entry) return;
    for (field = 0u; field < registry_string_field_count(); ++field)
        free(*owned_string_field(entry, registry_string_fields[field].owned_offset));
    memset(entry, 0, sizeof(*entry));
}

static void registry_entry_view(const yvex_model_registry_owned_entry *owned,
                                yvex_model_registry_entry *view)
{
    size_t field;

    memset(view, 0, sizeof(*view));
    if (!owned) return;
    view->schema_version = YVEX_MODEL_REGISTRY_ENTRY_SCHEMA_CURRENT;
    for (field = 0u; field < registry_string_field_count(); ++field) {
        const registry_string_field *spec = &registry_string_fields[field];
        *view_string_field(view, spec->view_offset) =
            *owned_string_field((yvex_model_registry_owned_entry *)(void *)owned,
                                spec->owned_offset);
    }
    view->file_size = owned->file_size;
    view->format = owned->format;
    view->architecture = owned->architecture;
    view->tensor_count = owned->tensor_count;
    view->known_tensor_bytes = owned->known_tensor_bytes;
    view->primary_tensor_name = owned->primary_tensor_name;
    view->primary_tensor_role = owned->primary_tensor_role;
    view->primary_tensor_dtype = owned->primary_tensor_dtype;
    view->primary_tensor_rank = owned->primary_tensor_rank;
    view->primary_tensor_dims = owned->primary_tensor_dims;
    view->primary_tensor_bytes = owned->primary_tensor_bytes;
    view->support_level = owned->support_level;
    view->selected_embedding_ready = owned->selected_embedding_ready;
    view->selected_embedding_hidden_size = owned->selected_embedding_hidden_size;
    view->selected_embedding_vocab_size = owned->selected_embedding_vocab_size;
    view->selected_embedding_output_count = owned->selected_embedding_output_count;
    view->selected_embedding_slice_bytes = owned->selected_embedding_slice_bytes;
    view->execution_ready = owned->execution_ready;
    view->runtime_context = owned->runtime_context;
}

static int registry_copy_entry(yvex_model_registry_owned_entry *dst,
                               const yvex_model_registry_entry *src,
                               yvex_error *err)
{
    size_t field;

    memset(dst, 0, sizeof(*dst));
    if (!src || src->schema_version != YVEX_MODEL_REGISTRY_ENTRY_SCHEMA_CURRENT) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "model_registry",
                       "current registry entry schema is required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (!src->alias || !src->path) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "model_registry",
                       "entry alias and path are required");
        return YVEX_ERR_INVALID_ARG;
    }
    for (field = 0u; field < registry_string_field_count(); ++field) {
        const registry_string_field *spec = &registry_string_fields[field];
        *owned_string_field(dst, spec->owned_offset) =
            yvex_core_strdup(view_string_value(src, spec->view_offset)
                                 ? view_string_value(src, spec->view_offset)
                                 : "");
    }
    dst->file_size = src->file_size;
    dst->tensor_count = src->tensor_count;
    dst->known_tensor_bytes = src->known_tensor_bytes;
    dst->primary_tensor_rank = src->primary_tensor_rank;
    dst->primary_tensor_bytes = src->primary_tensor_bytes;
    dst->selected_embedding_ready = src->selected_embedding_ready;
    dst->selected_embedding_hidden_size = src->selected_embedding_hidden_size;
    dst->selected_embedding_vocab_size = src->selected_embedding_vocab_size;
    dst->selected_embedding_output_count = src->selected_embedding_output_count;
    dst->selected_embedding_slice_bytes = src->selected_embedding_slice_bytes;
    dst->execution_ready = src->execution_ready;
    dst->runtime_context = src->runtime_context;
    for (field = 0u; field < registry_string_field_count(); ++field)
        if (!*owned_string_field(dst, registry_string_fields[field].owned_offset)) {
            registry_owned_entry_clear(dst);
            yvex_error_set(err, YVEX_ERR_NOMEM, "model_registry",
                           "entry allocation failed");
            return YVEX_ERR_NOMEM;
        }
    return YVEX_OK;
}

static const char *metadata_value_or_empty(const char *s)
{
    return s ? s : "";
}

static void metadata_set_status(char *dst, size_t cap, const char *status)
{
    if (!dst || cap == 0u) return;
    snprintf(dst, cap, "%s", status ? status : "");
}

static void metadata_add_issue(yvex_model_metadata_drift_report *out,
                               const char *code,
                               const char *registered_value,
                               const char *current_value)
{
    yvex_model_metadata_issue *issue;

    if (!out || out->issue_count >= YVEX_MODEL_METADATA_MAX_ISSUES) {
        return;
    }
    issue = &out->issues[out->issue_count++];
    snprintf(issue->code, sizeof(issue->code), "%s", code ? code : "");
    snprintf(issue->registered_value, sizeof(issue->registered_value), "%s",
             registered_value ? registered_value : "");
    snprintf(issue->current_value, sizeof(issue->current_value), "%s",
             current_value ? current_value : "");
}

static int metadata_string_missing(const char *s)
{
    return !s || !s[0];
}

static int registry_artifact_support_valid(const char *level)
{
    static const char *const levels[] = {
        "none",
        "descriptor-only",
        "selected-tensor-materialized",
        "full-weights-materialized",
        "partial-graph-executable",
        "prefill-ready",
        "decode-ready",
        "generation-ready"
    };
    size_t i;

    if (!level || !level[0]) return 1;
    for (i = 0u; i < sizeof(levels) / sizeof(levels[0]); ++i) {
        if (strcmp(level, levels[i]) == 0) return 1;
    }
    return 0;
}

static int metadata_registered_summary_missing(const yvex_model_registry_entry *entry)
{
    if (!entry) return 1;
    if (metadata_string_missing(entry->support_level)) return 1;
    if (metadata_string_missing(entry->format)) return 1;
    if (metadata_string_missing(entry->architecture)) return 1;
    if (entry->tensor_count == 0ull) return 1;
    if (metadata_string_missing(entry->primary_tensor_name)) return 1;
    if (metadata_string_missing(entry->primary_tensor_role)) return 1;
    if (metadata_string_missing(entry->primary_tensor_dtype)) return 1;
    if (entry->primary_tensor_rank == 0u) return 1;
    if (metadata_string_missing(entry->primary_tensor_dims)) return 1;
    return 0;
}

static void metadata_u64_to_text(unsigned long long value,
                                 char out[YVEX_MODEL_METADATA_VALUE_CAP])
{
    snprintf(out, YVEX_MODEL_METADATA_VALUE_CAP, "%llu", value);
}

static void metadata_bool_to_text(int value,
                                  char out[YVEX_MODEL_METADATA_VALUE_CAP])
{
    snprintf(out, YVEX_MODEL_METADATA_VALUE_CAP, "%s", value ? "true" : "false");
}

static void metadata_compare_string_field(yvex_model_metadata_drift_report *out,
                                          const char *code,
                                          const char *registered_value,
                                          const char *current_value)
{
    registered_value = metadata_value_or_empty(registered_value);
    current_value = metadata_value_or_empty(current_value);
    if (strcmp(registered_value, current_value) != 0) {
        metadata_add_issue(out, code, registered_value, current_value);
    }
}

static void metadata_compare_u64_field(yvex_model_metadata_drift_report *out,
                                       const char *code,
                                       unsigned long long registered_value,
                                       unsigned long long current_value)
{
    char registered_text[YVEX_MODEL_METADATA_VALUE_CAP];
    char current_text[YVEX_MODEL_METADATA_VALUE_CAP];

    if (registered_value == current_value) return;
    metadata_u64_to_text(registered_value, registered_text);
    metadata_u64_to_text(current_value, current_text);
    metadata_add_issue(out, code, registered_text, current_text);
}

static void metadata_compare_bool_field(yvex_model_metadata_drift_report *out,
                                        const char *code,
                                        int registered_value,
                                        int current_value)
{
    char registered_text[YVEX_MODEL_METADATA_VALUE_CAP];
    char current_text[YVEX_MODEL_METADATA_VALUE_CAP];

    if (!!registered_value == !!current_value) return;
    metadata_bool_to_text(registered_value, registered_text);
    metadata_bool_to_text(current_value, current_text);
    metadata_add_issue(out, code, registered_text, current_text);
}

int yvex_model_registry_compare_metadata(
    const yvex_model_registry_entry *registered,
    const yvex_model_registry_entry *current,
    yvex_model_metadata_drift_report *out,
    yvex_error *err)
{
    unsigned int before_selected_issues;

    if (!registered || !current || !out) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "model_registry_metadata",
                       "registered, current, and report are required");
        return YVEX_ERR_INVALID_ARG;
    }

    memset(out, 0, sizeof(*out));
    metadata_set_status(out->metadata_status, sizeof(out->metadata_status), "pass");
    metadata_set_status(out->readiness_status, sizeof(out->readiness_status), "pass");

    if (metadata_registered_summary_missing(registered)) {
        metadata_set_status(out->metadata_status, sizeof(out->metadata_status), "missing");
        if (strcmp(metadata_value_or_empty(registered->support_level),
                   "selected-tensor-materialized") == 0) {
            metadata_set_status(out->readiness_status, sizeof(out->readiness_status), "missing");
        }
        metadata_add_issue(out, "registered-metadata-missing", "missing", "available");
        return YVEX_OK;
    }

    metadata_compare_string_field(out, "support-level-mismatch",
                                  registered->support_level, current->support_level);
    metadata_compare_string_field(out, "format-mismatch",
                                  registered->format, current->format);
    metadata_compare_string_field(out, "architecture-mismatch",
                                  registered->architecture, current->architecture);
    metadata_compare_u64_field(out, "tensor-count-mismatch",
                               registered->tensor_count, current->tensor_count);
    metadata_compare_u64_field(out, "known-tensor-bytes-mismatch",
                               registered->known_tensor_bytes, current->known_tensor_bytes);
    metadata_compare_string_field(out, "primary-tensor-name-mismatch",
                                  registered->primary_tensor_name, current->primary_tensor_name);
    metadata_compare_string_field(out, "primary-tensor-role-mismatch",
                                  registered->primary_tensor_role, current->primary_tensor_role);
    metadata_compare_string_field(out, "primary-tensor-dtype-mismatch",
                                  registered->primary_tensor_dtype, current->primary_tensor_dtype);
    metadata_compare_u64_field(out, "primary-tensor-rank-mismatch",
                               registered->primary_tensor_rank, current->primary_tensor_rank);
    metadata_compare_string_field(out, "primary-tensor-dims-mismatch",
                                  registered->primary_tensor_dims, current->primary_tensor_dims);
    metadata_compare_u64_field(out, "primary-tensor-bytes-mismatch",
                               registered->primary_tensor_bytes, current->primary_tensor_bytes);

    before_selected_issues = out->issue_count;
    metadata_compare_bool_field(out, "selected-embedding-readiness-mismatch",
                                registered->selected_embedding_ready,
                                current->selected_embedding_ready);
    metadata_compare_u64_field(out, "selected-embedding-hidden-size-mismatch",
                               registered->selected_embedding_hidden_size,
                               current->selected_embedding_hidden_size);
    metadata_compare_u64_field(out, "selected-embedding-vocab-size-mismatch",
                               registered->selected_embedding_vocab_size,
                               current->selected_embedding_vocab_size);
    metadata_compare_u64_field(out, "selected-embedding-output-count-mismatch",
                               registered->selected_embedding_output_count,
                               current->selected_embedding_output_count);
    metadata_compare_u64_field(out, "selected-embedding-slice-bytes-mismatch",
                               registered->selected_embedding_slice_bytes,
                               current->selected_embedding_slice_bytes);
    if (out->issue_count > before_selected_issues) {
        metadata_set_status(out->readiness_status, sizeof(out->readiness_status), "fail");
    }

    if (out->issue_count > 0u) {
        metadata_set_status(out->metadata_status, sizeof(out->metadata_status), "fail");
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

static int registry_reserve(yvex_model_registry *registry,
                            unsigned long long need,
                            yvex_error *err)
{
    yvex_model_registry_owned_entry *next;
    unsigned long long cap;

    if (need <= registry->cap) return YVEX_OK;
    cap = registry->cap ? registry->cap * 2u : 4u;
    while (cap < need) cap *= 2u;
    next = (yvex_model_registry_owned_entry *)realloc(registry->entries, (size_t)cap * sizeof(*next));
    if (!next) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "model_registry", "registry allocation failed");
        return YVEX_ERR_NOMEM;
    }
    memset(next + registry->cap, 0, (size_t)(cap - registry->cap) * sizeof(*next));
    registry->entries = next;
    registry->cap = cap;
    return YVEX_OK;
}

static int is_ambiguous_token(const char *alias)
{
    return strcmp(alias, "latest") == 0 ||
           strcmp(alias, "final") == 0 ||
           strcmp(alias, "new") == 0 ||
           strcmp(alias, "test") == 0 ||
           strcmp(alias, "tmp") == 0 ||
           strcmp(alias, "debug") == 0 ||
           strstr(alias, "-latest") || strstr(alias, "latest-") ||
           strstr(alias, "-final") || strstr(alias, "final-") ||
           strstr(alias, "-new") || strstr(alias, "new-") ||
           strstr(alias, "-test") || strstr(alias, "test-") ||
           strstr(alias, "-tmp") || strstr(alias, "tmp-") ||
           strstr(alias, "-debug") || strstr(alias, "debug-");
}

int yvex_model_alias_validate(const char *alias, yvex_error *err)
{
    const char *p;
    int hyphens = 0;

    if (!alias || !alias[0]) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "model_alias", "alias is required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (alias[0] == '-' || alias[strlen(alias) - 1u] == '-') {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "model_alias", "alias must not start or end with hyphen");
        return YVEX_ERR_INVALID_ARG;
    }
    if (strstr(alias, "--")) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "model_alias", "alias must not contain empty segments");
        return YVEX_ERR_INVALID_ARG;
    }
    if (strchr(alias, '/') || strstr(alias, "..")) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "model_alias", "alias must not be path-like");
        return YVEX_ERR_INVALID_ARG;
    }
    for (p = alias; *p; ++p) {
        if (*p == '-') {
            hyphens++;
            continue;
        }
        if (!((*p >= 'a' && *p <= 'z') || (*p >= '0' && *p <= '9'))) {
            yvex_error_set(err, YVEX_ERR_INVALID_ARG, "model_alias", "alias uses invalid characters");
            return YVEX_ERR_INVALID_ARG;
        }
    }
    if (hyphens < 3) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "model_alias",
                       "alias must include family, model, scope, and artifact class; "
                       "example: deepseek4-v4-flash-dspark-selected-embed");
        return YVEX_ERR_INVALID_ARG;
    }
    if (is_ambiguous_token(alias)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "model_alias", "alias contains ambiguous vocabulary");
        return YVEX_ERR_INVALID_ARG;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_model_registry_default_path(char *out,
                                     unsigned long long out_size,
                                     yvex_error *err)
{
    const char *env = getenv("YVEX_MODELS_REGISTRY");
    yvex_paths paths;
    int n;

    if (!out || out_size == 0) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "model_registry_path", "output buffer is required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (env && env[0]) {
        n = snprintf(out, (size_t)out_size, "%s", env);
    } else {
        int rc = yvex_paths_default(&paths, err);
        if (rc != YVEX_OK) return rc;
        n = snprintf(out, (size_t)out_size, "%s/models.local.json", paths.data_dir);
    }
    if (n < 0 || (unsigned long long)n >= out_size) {
        out[0] = '\0';
        yvex_error_set(err, YVEX_ERR_BOUNDS, "model_registry_path", "registry path buffer too small");
        return YVEX_ERR_BOUNDS;
    }
    return YVEX_OK;
}

/*
 * Validate the persisted facts needed to construct one future server invocation.
 *
 * This deliberately stops before artifact or binding admission: registry inspection is inert,
 * while the server remains the only owner allowed to authenticate and open the runtime model.
 */
int yvex_model_registry_startup_validate(const yvex_model_registry_entry *entry,
                                         yvex_error *err)
{
    struct stat installation;
    int composite, media, text, not_applicable, target_only, speculative;

    if (!entry ||
        entry->schema_version != YVEX_MODEL_REGISTRY_ENTRY_SCHEMA_CURRENT) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "model_registry_startup",
                       "current registry entry schema is required");
        return YVEX_ERR_INVALID_ARG;
    }
    composite = entry->runtime_profile &&
                strcmp(entry->runtime_profile, "composite") == 0;
    media = entry->runtime_engine_kind &&
            strcmp(entry->runtime_engine_kind, "media") == 0;
    text = entry->runtime_engine_kind &&
           strcmp(entry->runtime_engine_kind, "text") == 0;
    not_applicable = entry->runtime_execution_strategy &&
                     strcmp(entry->runtime_execution_strategy, "not-applicable") == 0;
    target_only = entry->runtime_execution_strategy &&
                  strcmp(entry->runtime_execution_strategy, "target-only") == 0;
    speculative = entry->runtime_execution_strategy &&
                  strcmp(entry->runtime_execution_strategy, "speculative") == 0;
    if ((entry->runtime_profile && entry->runtime_profile[0] &&
         strcmp(entry->runtime_profile, "single-artifact") != 0 && !composite) ||
        !entry->runtime_target ||
        !entry->runtime_target[0] || !entry->runtime_backend ||
        (strcmp(entry->runtime_backend, "cpu") != 0 &&
         strcmp(entry->runtime_backend, "cuda") != 0) ||
        (!text && !media) ||
        (!target_only && !speculative && !not_applicable) ||
        (composite &&
         ((!entry->runtime_installation || entry->runtime_installation[0] != '/') ||
          (entry->runtime_binding && entry->runtime_binding[0]) ||
          !media || !not_applicable ||
          strcmp(entry->runtime_backend, "cuda") != 0 ||
          entry->runtime_context != 0ull)) ||
        (!composite &&
         ((!entry->path || entry->path[0] != '/') ||
          (!entry->runtime_binding || entry->runtime_binding[0] != '/') ||
          (entry->runtime_installation && entry->runtime_installation[0]) ||
          !text || not_applicable ||
          entry->runtime_context == 0ull))) {
        yvex_error_set(err, YVEX_ERR_STATE, "model_registry_startup",
                       "model has no complete startup profile");
        return YVEX_ERR_STATE;
    }
    if (composite &&
        (lstat(entry->runtime_installation, &installation) != 0 ||
         !S_ISDIR(installation.st_mode) || S_ISLNK(installation.st_mode) ||
         access(entry->runtime_installation, R_OK | X_OK) != 0)) {
        yvex_error_setf(err, YVEX_ERR_IO, "model_registry_startup",
                        "registered composite installation is not readable: %s",
                        entry->runtime_installation);
        return YVEX_ERR_IO;
    }
    if (!composite && access(entry->path, R_OK) != 0) {
        yvex_error_setf(err, YVEX_ERR_IO, "model_registry_startup",
                        "registered artifact is not readable: %s", entry->path);
        return YVEX_ERR_IO;
    }
    if (!composite && access(entry->runtime_binding, R_OK) != 0) {
        yvex_error_setf(err, YVEX_ERR_IO, "model_registry_startup",
                        "registered runtime binding is not readable: %s",
                        entry->runtime_binding);
        return YVEX_ERR_IO;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_model_registry_open(yvex_model_registry **out,
                             const yvex_model_registry_options *options,
                             yvex_error *err)
{
    yvex_model_registry *registry;
    char path[4096];
    const char *registry_path = NULL;
    int rc;

    if (!out) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "model_registry_open", "out is required");
        return YVEX_ERR_INVALID_ARG;
    }
    *out = NULL;
    registry = (yvex_model_registry *)calloc(1u, sizeof(*registry));
    if (!registry) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "model_registry_open", "registry allocation failed");
        return YVEX_ERR_NOMEM;
    }
    if (options && options->registry_path && options->registry_path[0]) {
        registry_path = options->registry_path;
    } else {
        rc = yvex_model_registry_default_path(path, sizeof(path), err);
        if (rc != YVEX_OK) {
            free(registry);
            return rc;
        }
        registry_path = path;
    }
    if (access(registry_path, F_OK) == 0) {
        rc = registry_parse_json(registry_path, registry, err);
        if (rc != YVEX_OK) {
            yvex_model_registry_close(registry);
            return rc;
        }
    } else if (!(options && options->create_if_missing)) {
        yvex_model_registry_close(registry);
        yvex_error_setf(err, YVEX_ERR_IO, "model_registry_open", "registry does not exist: %s", registry_path);
        return YVEX_ERR_IO;
    }
    *out = registry;
    yvex_error_clear(err);
    return YVEX_OK;
}

void yvex_model_registry_close(yvex_model_registry *registry)
{
    unsigned long long i;

    if (!registry) return;
    for (i = 0; i < registry->count; ++i) {
        registry_owned_entry_clear(&registry->entries[i]);
    }
    free(registry->entries);
    free(registry);
}

unsigned long long yvex_model_registry_count(const yvex_model_registry *registry)
{
    return registry ? registry->count : 0u;
}

const yvex_model_registry_entry *yvex_model_registry_at(const yvex_model_registry *registry,
                                                        unsigned long long index)
{
    static yvex_model_registry_entry view;

    if (!registry || index >= registry->count) return NULL;
    registry_entry_view(&registry->entries[index], &view);
    return &view;
}

const yvex_model_registry_entry *yvex_model_registry_find(const yvex_model_registry *registry,
                                                          const char *alias)
{
    unsigned long long i;
    static yvex_model_registry_entry view;

    if (!registry || !alias) return NULL;
    for (i = 0; i < registry->count; ++i) {
        if (strcmp(registry->entries[i].alias, alias) == 0) {
            registry_entry_view(&registry->entries[i], &view);
            return &view;
        }
    }
    return NULL;
}

int yvex_model_registry_add(yvex_model_registry *registry,
                            const yvex_model_registry_entry *entry,
                            yvex_error *err)
{
    yvex_model_registry_owned_entry copy;
    int rc;

    if (!registry || !entry ||
        entry->schema_version != YVEX_MODEL_REGISTRY_ENTRY_SCHEMA_CURRENT) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "model_registry_add",
                       "registry and current entry schema are required");
        return YVEX_ERR_INVALID_ARG;
    }
    rc = yvex_model_alias_validate(entry->alias, err);
    if (rc != YVEX_OK) return rc;
    if (yvex_model_registry_find(registry, entry->alias)) {
        yvex_error_setf(err, YVEX_ERR_STATE, "model_registry_add", "duplicate alias: %s", entry->alias);
        return YVEX_ERR_STATE;
    }
    if (!registry_artifact_support_valid(entry->support_level)) {
        yvex_error_setf(err, YVEX_ERR_INVALID_ARG, "model_registry_add",
                        "support level is not an artifact capability: %s",
                        entry->support_level);
        return YVEX_ERR_INVALID_ARG;
    }
    if (access(entry->path, F_OK) != 0) {
        yvex_error_setf(err, YVEX_ERR_IO, "model_registry_add", "model path does not exist: %s", entry->path);
        return YVEX_ERR_IO;
    }
    if ((entry->runtime_profile && entry->runtime_profile[0]) ||
        (entry->runtime_installation && entry->runtime_installation[0]) ||
        (entry->runtime_binding && entry->runtime_binding[0]) ||
        (entry->runtime_target && entry->runtime_target[0]) ||
        (entry->runtime_backend && entry->runtime_backend[0]) ||
        (entry->runtime_engine_kind && entry->runtime_engine_kind[0]) ||
        (entry->runtime_execution_strategy &&
         entry->runtime_execution_strategy[0]) ||
        entry->runtime_context != 0ull) {
        rc = yvex_model_registry_startup_validate(entry, err);
        if (rc != YVEX_OK) return rc;
    }
    rc = registry_reserve(registry, registry->count + 1u, err);
    if (rc != YVEX_OK) return rc;
    rc = registry_copy_entry(&copy, entry, err);
    if (rc != YVEX_OK) return rc;
    registry->entries[registry->count++] = copy;
    return YVEX_OK;
}

int yvex_model_registry_remove(yvex_model_registry *registry,
                               const char *alias,
                               yvex_error *err)
{
    unsigned long long i;

    if (!registry || !alias) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "model_registry_remove", "registry and alias are required");
        return YVEX_ERR_INVALID_ARG;
    }
    for (i = 0; i < registry->count; ++i) {
        if (strcmp(registry->entries[i].alias, alias) == 0) {
            registry_owned_entry_clear(&registry->entries[i]);
            if (i + 1u < registry->count) {
                memmove(&registry->entries[i], &registry->entries[i + 1u],
                        (size_t)(registry->count - i - 1u) * sizeof(registry->entries[0]));
            }
            registry->count--;
            memset(&registry->entries[registry->count], 0, sizeof(registry->entries[0]));
            return YVEX_OK;
        }
    }
    yvex_error_setf(err, YVEX_ERR_STATE, "model_registry_remove", "alias not found: %s", alias);
    return YVEX_ERR_STATE;
}

/*
 * Serialize canonical model registry state through atomic file publication.
 *
 * Writer output targets caller-provided local file paths only and never operator streams. Registry
 * JSON writing is not artifact emission, model verification, runtime support, generation
 * readiness, benchmark evidence, or release readiness.
 */
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

static int registry_write_json_file(const yvex_model_registry *registry,
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
    write_field(fp, "  ", "schema", YVEX_MODEL_REGISTRY_SCHEMA_CURRENT, 1);
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
        write_field(fp, "      ", "schema_version", e->artifact_schema, 1);
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
        write_field(fp, "      ", "runtime_profile", e->runtime_profile, 1);
        write_field(fp, "      ", "runtime_installation", e->runtime_installation, 1);
        write_field(fp, "      ", "runtime_binding", e->runtime_binding, 1);
        write_field(fp, "      ", "runtime_target", e->runtime_target, 1);
        write_field(fp, "      ", "runtime_backend", e->runtime_backend, 1);
        write_field(fp, "      ", "runtime_engine_kind", e->runtime_engine_kind, 1);
        write_field(fp, "      ", "runtime_execution_strategy",
                    e->runtime_execution_strategy, 1);
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

int yvex_model_registry_save(const yvex_model_registry *registry,
                             const char *path,
                             yvex_error *err)
{
    char default_path[4096];

    if (!registry) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "model_registry_save", "registry is required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (!path || !path[0]) {
        int rc = yvex_model_registry_default_path(default_path, sizeof(default_path), err);
        if (rc != YVEX_OK) return rc;
        path = default_path;
    }
    return registry_write_json_file(registry, path, err);
}

static int stem_alias_append(char *output, size_t capacity, size_t *used,
                             const char *component)
{
    size_t separator = *used != 0u;
    size_t extent = strlen(component);
    if (*used >= capacity || separator >= capacity - *used ||
        extent >= capacity - *used - separator)
        return 0;
    if (separator) output[(*used)++] = '-';
    memcpy(output + *used, component, extent);
    *used += extent;
    output[*used] = '\0';
    return 1;
}

/*
 * Split one canonical artifact stem into identity-bearing registry components.
 *
 * Stem plus bounded outputs for every required artifact-name component. Writes complete components
 * and the derived alias on success.
 */
static int split_canonical_stem(const char *stem,
                                char *family, size_t family_cap,
                                char *model, size_t model_cap,
                                char *scope, size_t scope_cap,
                                char *artifact_class, size_t class_cap,
                                char *qprofile, size_t qprofile_cap,
                                char *calibration, size_t calibration_cap,
                                char *producer, size_t producer_cap,
                                char *schema, size_t schema_cap,
                                char *alias, size_t alias_cap)
{
    char buf[1024];
    char *parts[64];
    int count = 0;
    char *tok;
    int tail;
    int i;
    size_t pos = 0;

    if (strlen(stem) >= sizeof(buf)) return 0;
    strcpy(buf, stem);
    tok = strtok(buf, "-");
    while (tok && count < 64) {
        parts[count++] = tok;
        tok = strtok(NULL, "-");
    }
    if (count < 8) return 0;
    tail = count - 4;
    snprintf(qprofile, qprofile_cap, "%s", parts[tail]);
    snprintf(calibration, calibration_cap, "%s", parts[tail + 1]);
    snprintf(producer, producer_cap, "%s", parts[tail + 2]);
    snprintf(schema, schema_cap, "%s", parts[tail + 3]);
    snprintf(family, family_cap, "%s", parts[0]);
    snprintf(scope, scope_cap, "%s", parts[tail - 2]);
    snprintf(artifact_class, class_cap, "%s", parts[tail - 1]);
    model[0] = '\0';
    for (i = 1; i < tail - 2; ++i) {
        int n = snprintf(model + pos, model_cap > pos ? model_cap - pos : 0,
                         "%s%s", pos ? "-" : "", parts[i]);
        if (n < 0 || (size_t)n >= (model_cap > pos ? model_cap - pos : 0)) return 0;
        pos += (size_t)n;
    }
    pos = 0u;
    alias[0] = '\0';
    if (!stem_alias_append(alias, alias_cap, &pos, family) ||
        !stem_alias_append(alias, alias_cap, &pos, model) ||
        !stem_alias_append(alias, alias_cap, &pos, scope) ||
        !stem_alias_append(alias, alias_cap, &pos, artifact_class))
        return 0;
    return strcmp(producer, "yvex") == 0 && strcmp(schema, "v1") == 0 &&
           family[0] && model[0] && scope[0] && artifact_class[0] &&
           qprofile[0] && calibration[0];
}

int yvex_model_registry_entry_derive_from_path(yvex_model_registry_entry *entry,
                                               const char *path,
                                               yvex_error *err)
{
    static char alias[256];
    static char family[128];
    static char model[128];
    static char scope[64];
    static char artifact_class[128];
    static char qprofile[64];
    static char calibration[128];
    static char producer[64];
    static char schema[64];
    static char path_copy[4096];
    char stem[1024];
    const char *base;
    size_t len;

    if (!entry || !path || !path[0]) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "model_registry_derive", "entry and path are required");
        return YVEX_ERR_INVALID_ARG;
    }
    base = strrchr(path, '/');
    base = base ? base + 1 : path;
    len = strlen(base);
    if (len <= 5u || strcmp(base + len - 5u, ".gguf") != 0 || len >= sizeof(stem)) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "model_registry_derive", "filename is not a canonical GGUF artifact name");
        return YVEX_ERR_FORMAT;
    }
    memcpy(stem, base, len - 5u);
    stem[len - 5u] = '\0';
    if (!split_canonical_stem(stem, family, sizeof(family), model, sizeof(model),
                              scope, sizeof(scope), artifact_class, sizeof(artifact_class),
                              qprofile, sizeof(qprofile), calibration, sizeof(calibration),
                              producer, sizeof(producer), schema, sizeof(schema),
                              alias, sizeof(alias))) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "model_registry_derive",
                       "filename does not match YVEX artifact naming grammar");
        return YVEX_ERR_FORMAT;
    }
    if (yvex_model_alias_validate(alias, err) != YVEX_OK) return yvex_error_code(err);
    snprintf(path_copy, sizeof(path_copy), "%s", path);
    memset(entry, 0, sizeof(*entry));
    entry->schema_version = YVEX_MODEL_REGISTRY_ENTRY_SCHEMA_CURRENT;
    entry->alias = alias;
    entry->family = family;
    entry->model = model;
    entry->scope = scope;
    entry->artifact_class = artifact_class;
    entry->qprofile = qprofile;
    entry->calibration = calibration;
    entry->producer = producer;
    entry->artifact_schema = schema;
    entry->path = path_copy;
    entry->sha256 = "";
    entry->file_size = 0ull;
    entry->format = "";
    entry->architecture = "";
    entry->tensor_count = 0ull;
    entry->known_tensor_bytes = 0ull;
    entry->primary_tensor_name = "";
    entry->primary_tensor_role = "";
    entry->primary_tensor_dtype = "";
    entry->primary_tensor_rank = 0u;
    entry->primary_tensor_dims = "";
    entry->primary_tensor_bytes = 0ull;
    entry->support_level = "";
    entry->selected_embedding_ready = 0;
    entry->selected_embedding_hidden_size = 0ull;
    entry->selected_embedding_vocab_size = 0ull;
    entry->selected_embedding_output_count = 0ull;
    entry->selected_embedding_slice_bytes = 0ull;
    entry->execution_ready = 0;
    entry->runtime_profile = "";
    entry->runtime_installation = "";
    entry->runtime_binding = "";
    entry->runtime_target = "";
    entry->runtime_backend = "";
    entry->runtime_engine_kind = "";
    entry->runtime_execution_strategy = "";
    entry->runtime_context = 0ull;
    return YVEX_OK;
}

static int read_file(const char *path, char **out, yvex_error *err)
{
    FILE *fp;
    long size;
    char *buf;

    fp = fopen(path, "rb");
    if (!fp) {
        yvex_error_setf(err, YVEX_ERR_IO, "model_registry_json", "cannot open registry: %s", path);
        return YVEX_ERR_IO;
    }
    if (fseek(fp, 0, SEEK_END) != 0 || (size = ftell(fp)) < 0 || fseek(fp, 0, SEEK_SET) != 0) {
        fclose(fp);
        yvex_error_setf(err, YVEX_ERR_IO, "model_registry_json", "cannot size registry: %s", path);
        return YVEX_ERR_IO;
    }
    buf = (char *)malloc((size_t)size + 1u);
    if (!buf) {
        fclose(fp);
        yvex_error_set(err, YVEX_ERR_NOMEM, "model_registry_json", "read allocation failed");
        return YVEX_ERR_NOMEM;
    }
    if (fread(buf, 1, (size_t)size, fp) != (size_t)size) {
        free(buf);
        fclose(fp);
        yvex_error_setf(err, YVEX_ERR_IO, "model_registry_json", "cannot read registry: %s", path);
        return YVEX_ERR_IO;
    }
    fclose(fp);
    buf[size] = '\0';
    *out = buf;
    return YVEX_OK;
}

static char *extract_string_in(const char *start, const char *end, const char *key)
{
    char needle[128];
    const char *p;
    const char *colon;
    const char *s;
    char *out;
    size_t n = 0;

    snprintf(needle, sizeof(needle), "\"%s\"", key);
    p = strstr(start, needle);
    if (!p || (end && p >= end)) return yvex_core_strdup("");
    colon = strchr(p, ':');
    if (!colon || (end && colon >= end)) return NULL;
    s = colon + 1;
    while ((!end || s < end) && (*s == ' ' || *s == '\n' || *s == '\r' || *s == '\t')) s++;
    if (end && s >= end) return NULL;
    if (*s != '"') return NULL;
    s++;
    out = (char *)malloc((size_t)(end ? end - s : (long)strlen(s)) + 1u);
    if (!out) return NULL;
    while (*s && (!end || s < end)) {
        char ch = *s++;
        if (ch == '"') {
            out[n] = '\0';
            return out;
        }
        if (ch == '\\' && *s && (!end || s < end)) {
            ch = *s++;
            if (ch == 'n') out[n++] = '\n';
            else if (ch == 'r') out[n++] = '\r';
            else if (ch == 't') out[n++] = '\t';
            else out[n++] = ch;
        } else {
            out[n++] = ch;
        }
    }
    free(out);
    return NULL;
}

static int extract_bool_in(const char *start, const char *end, const char *key)
{
    char needle[128];
    const char *p;
    const char *colon;
    const char *s;

    snprintf(needle, sizeof(needle), "\"%s\"", key);
    p = strstr(start, needle);
    if (!p || (end && p >= end)) return 0;
    colon = strchr(p, ':');
    if (!colon || (end && colon >= end)) return 0;
    s = colon + 1;
    while ((!end || s < end) && (*s == ' ' || *s == '\n' || *s == '\r' || *s == '\t')) s++;
    return strncmp(s, "true", 4) == 0 ? 1 : 0;
}

static unsigned long long extract_ull_in(const char *start, const char *end, const char *key)
{
    char needle[128];
    const char *p;
    const char *colon;
    const char *s;
    unsigned long long value = 0ull;

    snprintf(needle, sizeof(needle), "\"%s\"", key);
    p = strstr(start, needle);
    if (!p || (end && p >= end)) return 0ull;
    colon = strchr(p, ':');
    if (!colon || (end && colon >= end)) return 0ull;
    s = colon + 1;
    while ((!end || s < end) && (*s == ' ' || *s == '\n' || *s == '\r' || *s == '\t')) s++;
    while ((!end || s < end) && *s >= '0' && *s <= '9') {
        unsigned int digit = (unsigned int)(*s - '0');
        if (value > (ULLONG_MAX - (unsigned long long)digit) / 10ull) {
            return 0ull;
        }
        value = value * 10ull + (unsigned long long)digit;
        s++;
    }
    return value;
}

static void free_entry_view_strings(yvex_model_registry_entry *view)
{
    size_t field;

    if (!view) return;
    for (field = 0u; field < registry_string_field_count(); ++field)
        free((char *)view_string_value(view, registry_string_fields[field].view_offset));
    memset(view, 0, sizeof(*view));
}

static int legacy_startup_axes(const char *mode, const char *target,
                               const char **engine_kind,
                               const char **execution_strategy)
{
    const char *kind = "";
    const char *strategy = "";

    if (mode && mode[0]) {
        if (strcmp(mode, "media") == 0) {
            kind = "media";
            strategy = "not-applicable";
        } else if (strcmp(mode, "dspark") == 0) {
            kind = "text";
            strategy = "speculative";
        } else if (strcmp(mode, "target-only") == 0) {
            kind = "text";
            strategy = "target-only";
        } else {
            return 0;
        }
    } else if (target && target[0]) {
        kind = "text";
        strategy = "target-only";
    }
    *engine_kind = yvex_core_strdup(kind);
    *execution_strategy = yvex_core_strdup(strategy);
    return *engine_kind && *execution_strategy;
}

static int parse_entry_strings(const char *start,
                               const char *end,
                               yvex_model_registry_entry *view,
                               int import_legacy_startup_axes)
{
    size_t field;

    for (field = 0u; field < registry_string_field_count(); ++field) {
        const registry_string_field *spec = &registry_string_fields[field];
        char *value = extract_string_in(start, end, spec->json_key);
        if (import_legacy_startup_axes &&
            (strcmp(spec->json_key, "runtime_engine_kind") == 0 ||
             strcmp(spec->json_key, "runtime_execution_strategy") == 0)) {
            free(value);
            continue;
        }
        if (!value) return 0;
        *view_string_field(view, spec->view_offset) = value;
    }
    if (import_legacy_startup_axes) {
        char *mode = extract_string_in(start, end, "runtime_mode");
        int ok = legacy_startup_axes(
            mode, view->runtime_target,
            &view->runtime_engine_kind, &view->runtime_execution_strategy);
        free(mode);
        if (!ok) return 0;
    }
    return 1;
}

static int append_owned_registry_entry(yvex_model_registry *registry,
                                       yvex_model_registry_owned_entry *owned,
                                       yvex_error *err)
{
    int rc;

    if (!registry || !owned) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "model_registry_json",
                       "registry and owned entry are required");
        return YVEX_ERR_INVALID_ARG;
    }
    rc = registry_reserve(registry, registry->count + 1u, err);
    if (rc != YVEX_OK) {
        return rc;
    }
    registry->entries[registry->count++] = *owned;
    memset(owned, 0, sizeof(*owned));
    return YVEX_OK;
}

static const char *find_matching_object_end(const char *start)
{
    int depth = 0;
    int in_string = 0;
    int escape = 0;
    const char *p;

    for (p = start; *p; ++p) {
        if (in_string) {
            if (escape) escape = 0;
            else if (*p == '\\') escape = 1;
            else if (*p == '"') in_string = 0;
            continue;
        }
        if (*p == '"') in_string = 1;
        else if (*p == '{') depth++;
        else if (*p == '}') {
            depth--;
            if (depth == 0) return p + 1;
        }
    }
    return NULL;
}

static int registry_parse_json(const char *path,
                                        yvex_model_registry *registry,
                                        yvex_error *err)
{
    char *json = NULL;
    char *schema = NULL;
    const char *models;
    const char *p;
    int import_legacy_startup_axes;
    int rc;

    if (!path || !registry) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "model_registry_json", "path and registry are required");
        return YVEX_ERR_INVALID_ARG;
    }
    rc = read_file(path, &json, err);
    if (rc != YVEX_OK) return rc;
    models = strstr(json, "\"models\"");
    if (!models) {
        free(json);
        yvex_error_set(err, YVEX_ERR_FORMAT, "model_registry_json", "models array missing");
        return YVEX_ERR_FORMAT;
    }
    schema = extract_string_in(json, models, "schema");
    import_legacy_startup_axes = schema &&
        (strcmp(schema, "yvex.models.local.v1") == 0 ||
         strcmp(schema, "yvex.models.local.v2") == 0 ||
         strcmp(schema, "yvex.models.local.v3") == 0 ||
         strcmp(schema, "yvex.models.local.v4") == 0 ||
         strcmp(schema, YVEX_MODEL_REGISTRY_SCHEMA_V5) == 0);
    if (!schema || (!import_legacy_startup_axes &&
                    strcmp(schema, YVEX_MODEL_REGISTRY_SCHEMA_CURRENT) != 0)) {
        free(schema);
        free(json);
        yvex_error_set(err, YVEX_ERR_FORMAT, "model_registry_json", "registry schema missing or unsupported");
        return YVEX_ERR_FORMAT;
    }
    free(schema);
    p = strchr(models, '[');
    if (!p) {
        free(json);
        yvex_error_set(err, YVEX_ERR_FORMAT, "model_registry_json", "models array malformed");
        return YVEX_ERR_FORMAT;
    }
    p++;
    while (*p) {
        const char *obj_start;
        const char *obj_end;
        yvex_model_registry_entry view;
        yvex_model_registry_owned_entry owned;
        memset(&view, 0, sizeof(view));
        view.schema_version = YVEX_MODEL_REGISTRY_ENTRY_SCHEMA_CURRENT;
        while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t' || *p == ',') p++;
        if (*p == ']') break;
        if (*p != '{') {
            free(json);
            yvex_error_set(err, YVEX_ERR_FORMAT, "model_registry_json", "model entry malformed");
            return YVEX_ERR_FORMAT;
        }
        obj_start = p;
        obj_end = find_matching_object_end(obj_start);
        if (!obj_end) {
            free(json);
            yvex_error_set(err, YVEX_ERR_FORMAT, "model_registry_json", "model entry unterminated");
            return YVEX_ERR_FORMAT;
        }
        if (!parse_entry_strings(obj_start, obj_end, &view,
                                 import_legacy_startup_axes)) {
            free_entry_view_strings(&view);
            free(json);
            yvex_error_set(err, YVEX_ERR_FORMAT, "model_registry_json",
                           "model entry has malformed string field");
            return YVEX_ERR_FORMAT;
        }
        view.file_size = extract_ull_in(obj_start, obj_end, "file_size");
        view.tensor_count = extract_ull_in(obj_start, obj_end, "tensor_count");
        view.known_tensor_bytes = extract_ull_in(obj_start, obj_end, "known_tensor_bytes");
        view.primary_tensor_rank = (unsigned int)extract_ull_in(obj_start, obj_end, "primary_tensor_rank");
        view.primary_tensor_bytes = extract_ull_in(obj_start, obj_end, "primary_tensor_bytes");
        view.selected_embedding_ready = extract_bool_in(obj_start, obj_end, "selected_embedding_ready");
        view.selected_embedding_hidden_size = extract_ull_in(obj_start, obj_end, "selected_embedding_hidden_size");
        view.selected_embedding_vocab_size = extract_ull_in(obj_start, obj_end, "selected_embedding_vocab_size");
        view.selected_embedding_output_count = extract_ull_in(obj_start, obj_end, "selected_embedding_output_count");
        view.selected_embedding_slice_bytes = extract_ull_in(obj_start, obj_end, "selected_embedding_slice_bytes");
        view.execution_ready = extract_bool_in(obj_start, obj_end, "execution_ready");
        view.runtime_context = extract_ull_in(obj_start, obj_end, "runtime_context");
        rc = yvex_model_alias_validate(view.alias, err);
        if (rc != YVEX_OK) {
            free_entry_view_strings(&view);
            free(json);
            return rc;
        }
        rc = registry_copy_entry(&owned, &view, err);
        free_entry_view_strings(&view);
        if (rc != YVEX_OK) {
            free(json);
            return rc;
        }
        rc = append_owned_registry_entry(registry, &owned, err);
        if (rc != YVEX_OK) {
            registry_owned_entry_clear(&owned);
            free(json);
            return rc;
        }
        p = obj_end;
    }
    free(json);
    return YVEX_OK;
}

static int append_scan_entry(yvex_model_registry_entry **entries,
                             unsigned long long *count,
                             unsigned long long *cap,
                             const yvex_model_registry_entry *entry,
                             yvex_error *err)
{
    yvex_model_registry_owned_entry owned;
    yvex_model_registry_entry view;
    yvex_model_registry_entry *next;
    int rc;

    if (*count == *cap) {
        unsigned long long new_cap = *cap ? *cap * 2u : 8u;
        next = (yvex_model_registry_entry *)realloc(*entries, (size_t)new_cap * sizeof(*next));
        if (!next) {
            yvex_error_set(err, YVEX_ERR_NOMEM, "model_registry_scan", "scan allocation failed");
            return YVEX_ERR_NOMEM;
        }
        memset(next + *cap, 0, (size_t)(new_cap - *cap) * sizeof(*next));
        *entries = next;
        *cap = new_cap;
    }
    rc = registry_copy_entry(&owned, entry, err);
    if (rc != YVEX_OK) return rc;
    registry_entry_view(&owned, &view);
    (*entries)[*count] = view;
    memset(&owned, 0, sizeof(owned));
    (*count)++;
    return YVEX_OK;
}

static int scan_dir(const char *dir,
                    yvex_model_registry_entry **entries,
                    unsigned long long *count,
                    unsigned long long *cap,
                    yvex_error *err)
{
    DIR *dp;
    struct dirent *de;

    dp = opendir(dir);
    if (!dp) {
        yvex_error_setf(err, YVEX_ERR_IO, "model_registry_scan", "cannot open scan root: %s", dir);
        return YVEX_ERR_IO;
    }
    while ((de = readdir(dp)) != NULL) {
        char path[4096];
        struct stat st;
        size_t len;
        yvex_model_registry_entry entry;
        int n;

        if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0) continue;
        n = snprintf(path, sizeof(path), "%s/%s", dir, de->d_name);
        if (n < 0 || (size_t)n >= sizeof(path)) continue;
        if (stat(path, &st) != 0) continue;
        if (S_ISDIR(st.st_mode)) {
            int rc = scan_dir(path, entries, count, cap, err);
            if (rc != YVEX_OK) {
                closedir(dp);
                return rc;
            }
            continue;
        }
        if (!S_ISREG(st.st_mode)) continue;
        len = strlen(path);
        if (len <= 5u || strcmp(path + len - 5u, ".gguf") != 0) continue;
        if (yvex_model_registry_entry_derive_from_path(&entry, path, err) == YVEX_OK) {
            int rc = append_scan_entry(entries, count, cap, &entry, err);
            if (rc != YVEX_OK) {
                closedir(dp);
                return rc;
            }
        }
    }
    closedir(dp);
    return YVEX_OK;
}

int yvex_model_registry_scan_root(const char *root,
                                  yvex_model_registry_entry **entries_out,
                                  unsigned long long *count_out,
                                  yvex_error *err)
{
    yvex_model_registry_entry *entries = NULL;
    unsigned long long count = 0;
    unsigned long long cap = 0;
    int rc;

    if (!root || !entries_out || !count_out) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "model_registry_scan", "root and outputs are required");
        return YVEX_ERR_INVALID_ARG;
    }
    *entries_out = NULL;
    *count_out = 0;
    rc = scan_dir(root, &entries, &count, &cap, err);
    if (rc != YVEX_OK) {
        yvex_model_registry_scan_free(entries, count);
        return rc;
    }
    *entries_out = entries;
    *count_out = count;
    return YVEX_OK;
}

void yvex_model_registry_scan_free(yvex_model_registry_entry *entries,
                                   unsigned long long count)
{
    unsigned long long i;

    if (!entries) return;
    for (i = 0; i < count; ++i) {
        free_entry_view_strings(&entries[i]);
    }
    free(entries);
}
