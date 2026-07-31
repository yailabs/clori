/*
 * Provide small CLI-only helper implementations shared by model-artifacts command family surfaces.
 *
 * Shared rendering stays policy-free, table-driven, and below the repository unit limit. Shared
 * CLI helpers do not imply runtime support or artifact emission.
 */
#include "src/cli/model_artifacts/private.h"
#include "src/cli/render/private.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <yvex/artifact.h>
#include <yvex/internal/core.h>

typedef struct {
    const char *section;
    const char *reason;
} fullmodel_static_blocker;

typedef enum {
    PLAN_COLLECTION_EMBEDDING,
    PLAN_COLLECTION_NORMALIZATION,
    PLAN_COLLECTION_ATTENTION,
    PLAN_COLLECTION_MLP,
    PLAN_COLLECTION_MOE,
    PLAN_COLLECTION_OUTPUT,
    PLAN_COLLECTION_TOKENIZER,
    PLAN_COLLECTION_KV,
    PLAN_COLLECTION_SCRATCH,
    PLAN_COLLECTION_UNKNOWN
} plan_collection_rule;

typedef struct {
    const char *name;
    size_t count_offset;
    size_t bytes_offset;
    plan_collection_rule rule;
    int required_for_generation;
    const char *phase;
    const char *runtime_consumer;
    const char *missing_blocker;
} plan_collection_spec;

static const fullmodel_static_blocker materialization_static_blockers[] = {
    {"runtime-consumer", "full collection runtime consumers are not implemented"},
    {"generation-boundary", "real transformer prefill not implemented"},
    {"generation-boundary", "real attention-backed KV not implemented"},
    {"generation-boundary", "real DeepSeek decode not implemented"},
    {"generation-boundary", "real output-head logits not implemented"},
    {"generation-boundary", "real vocabulary sampling not implemented"},
    {"runtime-family", "runtime family adapter not implemented"},
};

static const plan_collection_spec plan_collections[] = {
    {"embedding", offsetof(yvex_fullmodel_collections, embedding),
     offsetof(yvex_fullmodel_collections, embedding_bytes), PLAN_COLLECTION_EMBEDDING, 1,
     "backend-residency", "planned", "embedding collection missing"},
    {"normalization", offsetof(yvex_fullmodel_collections, normalization),
     offsetof(yvex_fullmodel_collections, normalization_bytes), PLAN_COLLECTION_NORMALIZATION, 1,
     "backend-residency", "planned", "normalization collection missing"},
    {"attention", offsetof(yvex_fullmodel_collections, attention),
     offsetof(yvex_fullmodel_collections, attention_bytes), PLAN_COLLECTION_ATTENTION, 1,
     "backend-residency", "planned", "attention collection missing"},
    {"mlp", offsetof(yvex_fullmodel_collections, mlp),
     offsetof(yvex_fullmodel_collections, mlp_bytes), PLAN_COLLECTION_MLP, 1, "backend-residency",
     "planned", "MLP collection missing"},
    {"moe", offsetof(yvex_fullmodel_collections, moe),
     offsetof(yvex_fullmodel_collections, moe_bytes), PLAN_COLLECTION_MOE, 1, "backend-residency",
     "planned", "MoE collection missing or not identified"},
    {"output", offsetof(yvex_fullmodel_collections, output),
     offsetof(yvex_fullmodel_collections, output_bytes), PLAN_COLLECTION_OUTPUT, 1,
     "backend-residency", "planned", "output collection missing"},
    {"tokenizer-runtime-input", offsetof(yvex_fullmodel_collections, tokenizer),
     offsetof(yvex_fullmodel_collections, tokenizer_bytes), PLAN_COLLECTION_TOKENIZER, 1,
     "preflight", "planned", "tokenizer/full runtime metadata incomplete"},
    {"kv-cache-runtime", (size_t)-1, (size_t)-1, PLAN_COLLECTION_KV, 1, "kv-residency",
     "unsupported", "real attention-backed KV not implemented"},
    {"scratch-runtime", (size_t)-1, (size_t)-1, PLAN_COLLECTION_SCRATCH, 1, "scratch-residency",
     "planned", "scratch sizing remains planned"},
    {"unknown", offsetof(yvex_fullmodel_collections, unknown),
     offsetof(yvex_fullmodel_collections, unknown_bytes), PLAN_COLLECTION_UNKNOWN, 0,
     "collection-grouping", "unsupported", "unknown tensor role"},
};

static const char *const literal_pair_1[] = {"  }", "}"};

static const char *const literal_pair_2[] = {"  ]", "}"};

static const char *const literal_pair_3[] = {"  },", "  \"tensors\": ["};

static const char *const literal_pair_4[] = {"cleanup: preserved-partial-source",
                                             "lock_cleanup: not-attempted"};

static const char *const literal_pair_5[] = {"backend_memory_total_bytes: unknown",
                                             "backend_memory_available_bytes: unknown"};

static const char *const literal_pair_6[] = {
    "plan_cleanup_required: true",
    "plan_cleanup_phases: "
    "release-backend-buffers,release-host-staging,release-scratch,clear-partial-"
    "residency,preserve-failure-report"};

static const char *const literal_pair_7[] = {"plan_collection_count: 10", "plan_phase_count: 11"};

static const char *const literal_pair_8[] = {"plan_kind: full-model-materialization",
                                             "plan_source: tensor-inventory"};

static const char *const literal_pair_9[] = {"fullmodel: materialization-plan",
                                             "status: fullmodel-materialization-plan"};

static const char *const literal_lines_0[] = {
    "materialization_attempted: false",   "full_materialization_proof: false",
    "full_model_execution: unsupported",  "generation_ready: false",
    "generation: unsupported-full-model", "benchmark_status: not-measured"};

static const char *const literal_lines_1[] = {
    "cleanup_plan_required: true",
    "cleanup_plan_phases: "
    "release-backend-buffers,release-host-staging,release-scratch,clear-partial-"
    "residency,preserve-failure-report",
    "cleanup_idempotent_required: true",
    "cleanup_failure_policy: preserve-failure-report",
    "next_required_row: FULLMODEL.2",
    "proof_ready_for_fullmodel_2: false"};

static const char *const literal_lines_2[] = {
    "action: planned", "manifest: skipped", "native_inventory: skipped",
    "boundary: no payload downloaded, runtime unsupported"};

static const char *const literal_lines_3[] = {"upstream_identity_verified: false",
                                              "remote_lookup_performed: false",
                                              "payload_hash_verified: false",
                                              "payload_loaded: false",
                                              "gguf_created: false",
                                              "materialized: false",
                                              "runtime_ready: false",
                                              "generation: unsupported",
                                              "eval: unsupported",
                                              "benchmark_status: not-measured"};

#define DOWNLOAD_FIELD(key_, kind_, member_, fallback_)                                            \
    { key_, kind_, offsetof(yvex_model_download_report, member_), fallback_ }

static const yvex_render_field_spec download_audit_identity_fields[] = {
    DOWNLOAD_FIELD("status", YVEX_RENDER_FIELD_TEXT_ARRAY, status, "unknown"),
    DOWNLOAD_FIELD("target_id", YVEX_RENDER_FIELD_TEXT_ARRAY, target_id, "unknown"),
    DOWNLOAD_FIELD("family", YVEX_RENDER_FIELD_TEXT_ARRAY, family, "unknown"),
    DOWNLOAD_FIELD("provider", YVEX_RENDER_FIELD_TEXT_ARRAY, provider, "unknown"),
    DOWNLOAD_FIELD("repo_id", YVEX_RENDER_FIELD_TEXT_ARRAY, repo_id, "unknown"),
    DOWNLOAD_FIELD("revision", YVEX_RENDER_FIELD_TEXT_ARRAY, revision, "unknown"),
    DOWNLOAD_FIELD("local_source_dir", YVEX_RENDER_FIELD_TEXT_ARRAY, local_source_dir, "unknown"),
    DOWNLOAD_FIELD("models_root", YVEX_RENDER_FIELD_TEXT_ARRAY, models_root, "unknown"),
    DOWNLOAD_FIELD("models_root_source", YVEX_RENDER_FIELD_TEXT_ARRAY, models_root_source,
                   "unknown"),
    DOWNLOAD_FIELD("provider_cli", YVEX_RENDER_FIELD_TEXT_ARRAY, provider_cli_path, "missing"),
    DOWNLOAD_FIELD("provider_cli_source", YVEX_RENDER_FIELD_TEXT_ARRAY, provider_cli_source,
                   "none"),
    DOWNLOAD_FIELD("provider_cli_status", YVEX_RENDER_FIELD_TEXT_ARRAY, provider_cli_status,
                   "unknown"),
    DOWNLOAD_FIELD("account_hint", YVEX_RENDER_FIELD_TEXT_ARRAY, account_hint, "unknown"),
    DOWNLOAD_FIELD("credential_source", YVEX_RENDER_FIELD_TEXT_ARRAY, credential_source, "unknown"),
    DOWNLOAD_FIELD("accounts_state_path", YVEX_RENDER_FIELD_TEXT_ARRAY, accounts_state_path,
                   "unknown"),
};

static const yvex_render_field_spec download_normal_identity_fields[] = {
    DOWNLOAD_FIELD("family", YVEX_RENDER_FIELD_TEXT_ARRAY, family, ""),
    DOWNLOAD_FIELD("provider", YVEX_RENDER_FIELD_TEXT_ARRAY, provider, ""),
    DOWNLOAD_FIELD("repo", YVEX_RENDER_FIELD_TEXT_ARRAY, repo_id, ""),
    DOWNLOAD_FIELD("source", YVEX_RENDER_FIELD_TEXT_ARRAY, local_source_dir, ""),
};

static const yvex_render_field_spec download_audit_process_fields[] = {
    DOWNLOAD_FIELD("hf_cli", YVEX_RENDER_FIELD_TEXT_ARRAY, hf_cli_path, "missing"),
    DOWNLOAD_FIELD("hf_cli_source", YVEX_RENDER_FIELD_TEXT_ARRAY, hf_cli_source, "none"),
    DOWNLOAD_FIELD("hf_exit_code", YVEX_RENDER_FIELD_I32, hf_exit_code, NULL),
    DOWNLOAD_FIELD("provider_exit_code", YVEX_RENDER_FIELD_I32, provider_exit_code, NULL),
    DOWNLOAD_FIELD("auth_state", YVEX_RENDER_FIELD_TEXT_ARRAY, auth_state, "unknown"),
    DOWNLOAD_FIELD("token_env_name", YVEX_RENDER_FIELD_TEXT_ARRAY, token_env_name, "unknown"),
};

static const yvex_render_field_spec download_audit_telemetry_fields[] = {
    DOWNLOAD_FIELD("tick_count", YVEX_RENDER_FIELD_U64, tick_count, NULL),
    DOWNLOAD_FIELD("stdout_streamed", YVEX_RENDER_FIELD_BOOL, stdout_streamed, NULL),
    DOWNLOAD_FIELD("stderr_streamed", YVEX_RENDER_FIELD_BOOL, stderr_streamed, NULL),
    DOWNLOAD_FIELD("stdout_bytes", YVEX_RENDER_FIELD_U64, stdout_bytes, NULL),
    DOWNLOAD_FIELD("stderr_bytes", YVEX_RENDER_FIELD_U64, stderr_bytes, NULL),
};

static const yvex_render_field_spec download_audit_lifecycle_fields[] = {
    DOWNLOAD_FIELD("signal_forwarded", YVEX_RENDER_FIELD_BOOL, signal_forwarded, NULL),
    DOWNLOAD_FIELD("child_signal_forwarded", YVEX_RENDER_FIELD_BOOL, signal_forwarded, NULL),
    DOWNLOAD_FIELD("child_terminated", YVEX_RENDER_FIELD_BOOL, child_terminated, NULL),
    DOWNLOAD_FIELD("child_killed_after_timeout", YVEX_RENDER_FIELD_BOOL, child_killed_after_timeout,
                   NULL),
    DOWNLOAD_FIELD("child_exit_status", YVEX_RENDER_FIELD_TEXT_ARRAY, child_exit_status, "unknown"),
    DOWNLOAD_FIELD("orphan_check_performed", YVEX_RENDER_FIELD_BOOL, orphan_check_performed, NULL),
    DOWNLOAD_FIELD("orphan_check_status", YVEX_RENDER_FIELD_TEXT_ARRAY, orphan_check_status,
                   "unknown"),
    DOWNLOAD_FIELD("partial_source_preserved", YVEX_RENDER_FIELD_BOOL, partial_source_preserved,
                   NULL),
};

static const yvex_render_field_spec download_audit_source_fields[] = {
    DOWNLOAD_FIELD("source_file_count", YVEX_RENDER_FIELD_U64, source_scan.file_count, NULL),
    DOWNLOAD_FIELD("file_count", YVEX_RENDER_FIELD_U64, source_scan.file_count, NULL),
    DOWNLOAD_FIELD("safetensors_count", YVEX_RENDER_FIELD_U64, source_scan.safetensors_count, NULL),
    DOWNLOAD_FIELD("config_present", YVEX_RENDER_FIELD_BOOL, source_scan.config_present, NULL),
    DOWNLOAD_FIELD("tokenizer_present", YVEX_RENDER_FIELD_BOOL, source_scan.tokenizer_present,
                   NULL),
    DOWNLOAD_FIELD("total_regular_file_bytes", YVEX_RENDER_FIELD_U64,
                   source_scan.total_regular_file_bytes, NULL),
    DOWNLOAD_FIELD("largest_file_name", YVEX_RENDER_FIELD_TEXT_ARRAY, source_scan.largest_file_name,
                   "none"),
    DOWNLOAD_FIELD("largest_file_bytes", YVEX_RENDER_FIELD_U64, source_scan.largest_file_bytes,
                   NULL),
    DOWNLOAD_FIELD("manifest_path", YVEX_RENDER_FIELD_TEXT_ARRAY, manifest_path, "unknown"),
    DOWNLOAD_FIELD("native_inventory_path", YVEX_RENDER_FIELD_TEXT_ARRAY, native_inventory_path,
                   "unknown"),
    DOWNLOAD_FIELD("download_report_path", YVEX_RENDER_FIELD_TEXT_ARRAY, download_report_path,
                   "unknown"),
    DOWNLOAD_FIELD("registry_path", YVEX_RENDER_FIELD_TEXT_ARRAY, registry_path, "unknown"),
    DOWNLOAD_FIELD("receipt_path", YVEX_RENDER_FIELD_TEXT_ARRAY, receipt_path, "unknown"),
    DOWNLOAD_FIELD("stdout_log", YVEX_RENDER_FIELD_TEXT_ARRAY, stdout_log_path, "unknown"),
    DOWNLOAD_FIELD("stderr_log", YVEX_RENDER_FIELD_TEXT_ARRAY, stderr_log_path, "unknown"),
    DOWNLOAD_FIELD("created_at", YVEX_RENDER_FIELD_TEXT_ARRAY, created_at, "unknown"),
};

#undef DOWNLOAD_FIELD

int models_registry_open(yvex_model_registry **registry, const char *registry_path,
                         int create_if_missing, yvex_error *err) {
    yvex_model_registry_options options;

    memset(&options, 0, sizeof(options));
    options.registry_path = registry_path;
    options.create_if_missing = create_if_missing;
    return yvex_model_registry_open(registry, &options, err);
}

void print_metadata_drift_cli(const yvex_model_metadata_drift_report *report) {
    unsigned int i;

    if (!report)
        return;
    yvex_cli_out_writef(stdout, "metadata_status: %s\n",
                        report->metadata_status[0] ? report->metadata_status : "unknown");
    yvex_cli_out_writef(stdout, "readiness_status: %s\n",
                        report->readiness_status[0] ? report->readiness_status : "unknown");
    for (i = 0u; i < report->issue_count; ++i) {
        yvex_cli_out_writef(stdout, "metadata_issue_%u_code: %s\n", i, report->issues[i].code);
        yvex_cli_out_writef(stdout, "metadata_issue_%u_registered: %s\n", i,
                            report->issues[i].registered_value);
        yvex_cli_out_writef(stdout, "metadata_issue_%u_current: %s\n", i,
                            report->issues[i].current_value);
    }
}

int path_exists(const char *path) {
    return path && path[0] && access(path, F_OK) == 0;
}

int is_path_like_reference(const char *input) {
    size_t len;

    if (!input || !input[0])
        return 0;
    if (strchr(input, '/') || strchr(input, '\\'))
        return 1;
    len = strlen(input);
    return len >= 5u && strcmp(input + len - 5u, ".gguf") == 0;
}

void model_artifact_append_role(char *out, size_t out_cap, const char *role)
{
    size_t used;

    if (!out || out_cap == 0u || !role || !role[0]) return;
    used = strlen(out);
    if (used > 0u) {
        if (used + 1u >= out_cap) return;
        out[used++] = ',';
        out[used] = '\0';
    }
    snprintf(out + used, used < out_cap ? out_cap - used : 0u, "%s", role);
}

static void write_escaped(FILE *fp, const char *s) {
    if (!s)
        s = "";
    yvex_cli_out_char(fp, '"');
    while (*s) {
        unsigned char ch = (unsigned char)*s++;
        if (ch == '"' || ch == '\\') {
            yvex_cli_out_char(fp, '\\');
            yvex_cli_out_char(fp, (int)ch);
        } else if (ch == '\n') {
            yvex_cli_out_fputs("\\n", fp);
        } else if (ch == '\r') {
            yvex_cli_out_fputs("\\r", fp);
        } else if (ch == '\t') {
            yvex_cli_out_fputs("\\t", fp);
        } else {
            yvex_cli_out_char(fp, (int)ch);
        }
    }
    yvex_cli_out_char(fp, '"');
}

static void write_field(FILE *fp, const char *indent, const char *key, const char *value,
                        int comma) {
    yvex_cli_out_fputs(indent, fp);
    yvex_cli_out_writef(fp, "\"%s\": ", key);
    write_escaped(fp, value);
    yvex_cli_out_writef(fp, "%s\n", comma ? "," : "");
}

static void write_u64_field(FILE *fp, const char *indent, const char *key, unsigned long long value,
                            int comma) {
    yvex_cli_out_fputs(indent, fp);
    yvex_cli_out_writef(fp, "\"%s\": %llu%s\n", key, value, comma ? "," : "");
}

static void write_bool_field(FILE *fp, const char *indent, const char *key, int value, int comma) {
    yvex_cli_out_fputs(indent, fp);
    yvex_cli_out_writef(fp, "\"%s\": %s%s\n", key, value ? "true" : "false", comma ? "," : "");
}

int parse_models_output_mode(const char *value, yvex_models_output_mode *mode) {
    if (!value || !mode)
        return 0;
    if (strcmp(value, "normal") == 0)
        *mode = YVEX_MODELS_OUTPUT_NORMAL;
    else if (strcmp(value, "table") == 0)
        *mode = YVEX_MODELS_OUTPUT_TABLE;
    else if (strcmp(value, "audit") == 0)
        *mode = YVEX_MODELS_OUTPUT_AUDIT;
    else
        return 0;
    return 1;
}

int parse_models_bound_option(const char *command, int arg_count, char **args, int *index,
                              void *options, const yvex_models_option_spec *specs,
                              size_t spec_count, int *handled) {
    size_t i;

    if (handled)
        *handled = 0;
    if (!command || !args || !index || !options || !specs || !handled)
        return 2;
    for (i = 0; i < spec_count; ++i) {
        const yvex_models_option_spec *spec = &specs[i];
        unsigned char *field;
        const char *value;

        if (strcmp(args[*index], spec->flag) != 0)
            continue;
        *handled = 1;
        field = (unsigned char *)options + spec->offset;
        if (spec->kind == YVEX_MODELS_OPTION_FLAG) {
            *(int *)field = 1;
            return 0;
        }
        if (*index + 1 >= arg_count) {
            yvex_cli_out_writef(stderr, "yvex: %s %s requires a value\n", command, spec->flag);
            return 2;
        }
        value = args[++(*index)];
        if (!cli_arg_value_valid(value)) {
            yvex_cli_out_writef(stderr, "yvex: %s %s value is empty or invalid\n", command,
                                spec->flag);
            return 2;
        }
        if (spec->kind == YVEX_MODELS_OPTION_TEXT) {
            *(const char **)field = value;
            return 0;
        }
        if (!parse_models_output_mode(value, (yvex_models_output_mode *)field)) {
            yvex_cli_out_writef(stderr, "yvex: %s unsupported output mode: %s\n", command, value);
            return 2;
        }
        return 0;
    }
    return 0;
}

void print_model_registry_entry_cli(const yvex_model_registry_entry *entry) {
    if (!entry)
        return;
    yvex_cli_out_writef(stdout, "%-46s %-10s %-22s %5llu %13llu  %s\n",
                        entry->alias ? entry->alias : "", entry->family ? entry->family : "",
                        entry->artifact_class ? entry->artifact_class : "", entry->tensor_count,
                        entry->known_tensor_bytes, entry->selected_embedding_ready ? "yes" : "no");
}

void print_model_registry_entry_audit(const yvex_model_registry_entry *entry) {
    if (!entry)
        return;
    yvex_cli_out_writef(stdout, "model: %s\n", entry->alias ? entry->alias : "");
    yvex_cli_out_writef(stdout, "path: %s\n", entry->path ? entry->path : "");
    yvex_cli_out_writef(stdout, "family: %s\n", entry->family ? entry->family : "");
    yvex_cli_out_writef(stdout, "model_name: %s\n", entry->model ? entry->model : "");
    yvex_cli_out_writef(stdout, "scope: %s\n", entry->scope ? entry->scope : "");
    yvex_cli_out_writef(stdout, "artifact_class: %s\n",
                        entry->artifact_class ? entry->artifact_class : "");
    yvex_cli_out_writef(stdout, "qprofile: %s\n", entry->qprofile ? entry->qprofile : "");
    yvex_cli_out_writef(stdout, "calibration: %s\n", entry->calibration ? entry->calibration : "");
    yvex_cli_out_writef(stdout, "registered_file_size: %llu\n", entry->file_size);
    yvex_cli_out_writef(stdout, "registered_sha256: %s\n",
                        entry->sha256 && entry->sha256[0] ? entry->sha256 : "absent");
    yvex_cli_out_writef(stdout, "registered_format: %s\n", entry->format ? entry->format : "");
    yvex_cli_out_writef(stdout, "registered_architecture: %s\n",
                        entry->architecture ? entry->architecture : "");
    yvex_cli_out_writef(stdout, "registered_tensor_count: %llu\n", entry->tensor_count);
    yvex_cli_out_writef(stdout, "registered_known_tensor_bytes: %llu\n", entry->known_tensor_bytes);
    yvex_cli_out_writef(stdout, "registered_selected_embedding_ready: %s\n",
                        entry->selected_embedding_ready ? "true" : "false");
}

void print_model_registry_scan_entry_cli(const yvex_model_registry_entry *entry) {
    if (!entry)
        return;
    yvex_cli_out_writef(stdout, "candidate: %s\n", entry->alias ? entry->alias : "");
    yvex_cli_out_writef(stdout, "path: %s\n", entry->path ? entry->path : "");
}

void dims_to_text(const unsigned long long *dims, unsigned int rank, char *out, size_t out_cap) {
    size_t used = 0u;
    unsigned int i;

    if (!out || out_cap == 0u)
        return;
    out[0] = '\0';
    for (i = 0; i < rank && used < out_cap; ++i) {
        int n = snprintf(out + used, out_cap - used, "%s%llu", i == 0u ? "" : "x",
                         dims ? dims[i] : 0ull);
        if (n < 0)
            return;
        if ((size_t)n >= out_cap - used) {
            out[out_cap - 1u] = '\0';
            return;
        }
        used += (size_t)n;
    }
}

int populate_registry_identity(yvex_model_registry_entry *entry, char *sha256, char *format,
                               char *architecture, char *primary_name, char *primary_role,
                               char *primary_dtype, char *primary_dims, yvex_error *err) {
    yvex_artifact_file_identity identity;
    yvex_model_metadata_snapshot snapshot;
    yvex_error_clear(err);
    memset(&identity, 0, sizeof(identity));
    memset(&snapshot, 0, sizeof(snapshot));
    if (yvex_artifact_identity_read(entry->path, &identity, err) != YVEX_OK)
        return yvex_error_code(err);
    if (yvex_model_metadata_snapshot_read(&snapshot, entry->path, err) != YVEX_OK)
        return yvex_error_code(err);
    snprintf(sha256, YVEX_SHA256_HEX_CAP, "%s", identity.sha256);
    snprintf(format, 16u, "%s", snapshot.format);
    snprintf(architecture, 64u, "%s", snapshot.architecture);
    snprintf(primary_name, 128u, "%s", snapshot.primary_tensor_name);
    snprintf(primary_role, 64u, "%s", snapshot.primary_tensor_role);
    snprintf(primary_dtype, 32u, "%s", snapshot.primary_tensor_dtype);
    snprintf(primary_dims, 128u, "%s", snapshot.primary_tensor_dims);
    entry->sha256 = sha256;
    entry->file_size = identity.file_size;
    entry->format = format;
    entry->architecture = architecture;
    entry->tensor_count = snapshot.entry.tensor_count;
    entry->known_tensor_bytes = snapshot.entry.known_tensor_bytes;
    entry->primary_tensor_name = primary_name;
    entry->primary_tensor_role = primary_role;
    entry->primary_tensor_dtype = primary_dtype;
    entry->primary_tensor_rank = snapshot.entry.primary_tensor_rank;
    entry->primary_tensor_dims = primary_dims;
    entry->primary_tensor_bytes = snapshot.entry.primary_tensor_bytes;
    entry->selected_embedding_ready = snapshot.entry.selected_embedding_ready;
    entry->selected_embedding_hidden_size = snapshot.entry.selected_embedding_hidden_size;
    entry->selected_embedding_vocab_size = snapshot.entry.selected_embedding_vocab_size;
    entry->selected_embedding_output_count = snapshot.entry.selected_embedding_output_count;
    entry->selected_embedding_slice_bytes = snapshot.entry.selected_embedding_slice_bytes;
    return YVEX_OK;
}

void model_stage_print(const char *stage, const char *status) {
    yvex_cli_out_writef(stdout, "stage: %s %s\n", stage ? stage : "", status ? status : "");
}

const char *model_requested_family(const char *family) {
    return family && family[0] ? family : "auto";
}

/*
 * Render one indexed model phase through the shared CLI phase grammar.
 *
 * Borrowed prefix, phase name, status, and fallback text.
 */
void model_phase_print(const char *prefix, unsigned int index, const char *name, const char *status,
                       const char *fallback) {
    yvex_cli_out_writef(stdout, "%s.%u.name: %s\n", prefix, index, name ? name : "");
    yvex_cli_out_writef(stdout, "%s.%u.status: %s\n", prefix, index, status ? status : fallback);
}

void model_print_runtime_generation(const char *runtime_execution) {
    yvex_cli_out_writef(stdout, "runtime_execution: %s\n",
                        runtime_execution ? runtime_execution : "not-performed");
    yvex_cli_out_writef(stdout, "generation: unsupported\n");
}

int cli_arg_value_valid(const char *value) {
    return value && value[0] && !strchr(value, '\n') && !strchr(value, '\r');
}

int parse_models_value_option(const char *command, const char *flag, int arg_count, char **args,
                              int *index, const char **value) {
    if (*index + 1 >= arg_count) {
        yvex_cli_out_writef(stderr, "yvex: %s %s requires a value\n", command, flag);
        return 2;
    }
    *value = args[++(*index)];
    if (!cli_arg_value_valid(*value)) {
        yvex_cli_out_writef(stderr, "yvex: %s %s value is empty or invalid\n", command, flag);
        return 2;
    }
    return 0;
}

int model_backend_kind_from_name(const char *backend_name, yvex_backend_kind *kind) {
    if (!kind)
        return 0;
    if (!backend_name || strcmp(backend_name, "cpu") == 0) {
        *kind = YVEX_BACKEND_KIND_CPU;
        return 1;
    }
    if (strcmp(backend_name, "cuda") == 0) {
        *kind = YVEX_BACKEND_KIND_CUDA;
        return 1;
    }
    return 0;
}

int expand_operator_path(const char *input, char *out, size_t out_cap, yvex_error *err,
                         const char *where) {
    const char *home;
    int n;

    if (!input || !out || out_cap == 0u) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, where, "path value is required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (!cli_arg_value_valid(input)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, where,
                       "path value is empty or contains a newline");
        return YVEX_ERR_INVALID_ARG;
    }
    if (input[0] == '~' && input[1] == '/') {
        home = getenv("HOME");
        if (!home || !home[0]) {
            yvex_error_set(err, YVEX_ERR_INVALID_ARG, where, "HOME is required to expand ~/ paths");
            return YVEX_ERR_INVALID_ARG;
        }
        n = snprintf(out, out_cap, "%s/%s", home, input + 2);
    } else {
        n = snprintf(out, out_cap, "%s", input);
    }
    if (n < 0 || (size_t)n >= out_cap) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, where, "path is too long");
        return YVEX_ERR_BOUNDS;
    }
    return YVEX_OK;
}

int path_join2(char *out, size_t out_cap, const char *dir, const char *file, yvex_error *err,
               const char *where) {
    int n = snprintf(out, out_cap, "%s/%s", dir, file);
    if (n < 0 || (size_t)n >= out_cap) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, where, "resolved path is too long");
        return YVEX_ERR_BOUNDS;
    }
    return YVEX_OK;
}

int path_parent_dir(const char *path, char *out, size_t out_cap) {
    const char *slash;
    size_t len;

    if (!path || !out || out_cap == 0u)
        return 0;
    slash = strrchr(path, '/');
    if (!slash) {
        snprintf(out, out_cap, ".");
        return 1;
    }
    len = (size_t)(slash - path);
    if (len == 0u)
        len = 1u;
    if (len >= out_cap)
        return 0;
    memcpy(out, path, len);
    out[len] = '\0';
    return 1;
}

void fullmodel_print_largest(const fullmodel_largest_tensor *top, unsigned int top_count) {
    unsigned int i;

    for (i = 0; i < top_count; ++i) {
        char dims[128];
        const yvex_tensor_info *tensor = top[i].tensor;
        if (!tensor)
            continue;
        dims_to_text(tensor->dims, tensor->rank, dims, sizeof(dims));
        yvex_cli_out_writef(stdout,
                            "largest_tensor_%u: name=%s dtype=%s role=%s dims=%s bytes=%llu\n", i,
                            tensor->name ? tensor->name : "", yvex_dtype_name(tensor->dtype),
                            yvex_tensor_role_name(tensor->role), dims, tensor->storage_bytes);
    }
}

static int fullmodel_residency_is_future_unsupported(const char *residency) {
    return residency &&
           (strcmp(residency, "ssd-streamed") == 0 || strcmp(residency, "managed-memory") == 0 ||
            strcmp(residency, "distributed") == 0);
}

static const char *fullmodel_placement_for_residency(const char *backend, const char *residency,
                                                     int present) {
    if (!present)
        return "not-planned";
    if (fullmodel_residency_is_future_unsupported(residency))
        return "unsupported";
    if (strcmp(residency, "host-staged") == 0)
        return "host-staged";
    if (strcmp(residency, "ssd-staged") == 0)
        return "ssd-staged";
    if (strcmp(residency, "hybrid") == 0)
        return "hybrid";
    return backend && strcmp(backend, "cuda") == 0 ? "cuda-resident" : "cpu-resident";
}

static const char *fullmodel_required_bool(int value) {
    return value ? "true" : "false";
}

static int fullmodel_collection_missing(const yvex_fullmodel_collections *collections,
                                        plan_collection_rule rule) {
    if (!collections)
        return 1;
    switch (rule) {
    case PLAN_COLLECTION_EMBEDDING:
        return !collections->has_token_embedding;
    case PLAN_COLLECTION_NORMALIZATION:
        return !fullmodel_has_normalization_collection(collections);
    case PLAN_COLLECTION_ATTENTION:
        return !fullmodel_has_attention_collection(collections);
    case PLAN_COLLECTION_MLP:
        return !fullmodel_has_mlp_collection(collections);
    case PLAN_COLLECTION_MOE:
        return !collections->has_moe_router && !collections->has_moe_expert;
    case PLAN_COLLECTION_OUTPUT:
        return !collections->has_output_head;
    case PLAN_COLLECTION_TOKENIZER:
        return !collections->has_tokenizer_metadata;
    case PLAN_COLLECTION_KV:
    case PLAN_COLLECTION_SCRATCH:
    case PLAN_COLLECTION_UNKNOWN:
        return 0;
    }
    return 1;
}

static unsigned int
fullmodel_plan_missing_collection_blockers(const yvex_fullmodel_collections *collections) {
    unsigned int count = 0u;
    size_t i;

    if (!collections)
        return 0u;
    for (i = 0u; i < sizeof(plan_collections) / sizeof(plan_collections[0]); ++i) {
        count += (unsigned int)fullmodel_collection_missing(collections, plan_collections[i].rule);
    }
    return count;
}

static unsigned int fullmodel_plan_blocker_count(const yvex_fullmodel_collections *collections,
                                                 int selected_target, const char *residency,
                                                 const yvex_fullmodel_backend_fit *fit) {
    unsigned int count = 7u; /* runtime-consumer plus six generation-boundary blockers. */

    count += fullmodel_plan_missing_collection_blockers(collections);
    if (selected_target)
        count++;
    if (fullmodel_residency_is_future_unsupported(residency))
        count++;
    if (fit && !fit->available)
        count++;
    if (fit && strcmp(fit->fit_status ? fit->fit_status : "unknown", "does-not-fit") == 0)
        count++;
    return count;
}

static const char *fullmodel_plan_status(const yvex_fullmodel_collections *collections,
                                         int selected_target, const char *residency,
                                         const yvex_fullmodel_backend_fit *fit) {
    if (fullmodel_residency_is_future_unsupported(residency))
        return "unsupported";
    if (fit && !fit->available)
        return "partial";
    if (selected_target)
        return "partial";
    if (fullmodel_plan_missing_collection_blockers(collections) > 0u)
        return "partial";
    return "ready";
}

void fullmodel_print_phase(unsigned int index, const char *name, const char *status,
                           unsigned long long tensor_count, unsigned long long tensor_bytes,
                           const char *residency, int required, int blocked, const char *blocker) {
    yvex_cli_out_writef(stdout, "phase.%u.name: %s\n", index, name ? name : "");
    yvex_cli_out_writef(stdout, "phase.%u.status: %s\n", index, status ? status : "planned");
    yvex_cli_out_writef(stdout, "phase.%u.tensor_count: %llu\n", index, tensor_count);
    yvex_cli_out_writef(stdout, "phase.%u.tensor_bytes: %llu\n", index, tensor_bytes);
    yvex_cli_out_writef(stdout, "phase.%u.residency: %s\n", index,
                        residency ? residency : "not-applicable");
    yvex_cli_out_writef(stdout, "phase.%u.required: %s\n", index, required ? "true" : "false");
    yvex_cli_out_writef(stdout, "phase.%u.blocked: %s\n", index, blocked ? "true" : "false");
    yvex_cli_out_writef(stdout, "phase.%u.blocker: %s\n", index,
                        blocker && blocker[0] ? blocker : "none");
}

static void fullmodel_print_collection_plan(const char *name, const char *status,
                                            unsigned long long tensor_count,
                                            unsigned long long tensor_bytes,
                                            int required_for_generation, int present,
                                            const char *placement, const char *phase,
                                            const char *runtime_consumer, const char *blocker) {
    yvex_cli_out_writef(stdout, "collection.%s.status: %s\n", name, status ? status : "planned");
    yvex_cli_out_writef(stdout, "collection.%s.tensor_count: %llu\n", name, tensor_count);
    yvex_cli_out_writef(stdout, "collection.%s.tensor_bytes: %llu\n", name, tensor_bytes);
    yvex_cli_out_writef(stdout, "collection.%s.required_for_generation: %s\n", name,
                        fullmodel_required_bool(required_for_generation));
    yvex_cli_out_writef(stdout, "collection.%s.present: %s\n", name,
                        fullmodel_required_bool(present));
    yvex_cli_out_writef(stdout, "collection.%s.placement: %s\n", name,
                        placement ? placement : "unknown");
    yvex_cli_out_writef(stdout, "collection.%s.materialization_phase: %s\n", name,
                        phase ? phase : "collection-grouping");
    yvex_cli_out_writef(stdout, "collection.%s.runtime_consumer: %s\n", name,
                        runtime_consumer ? runtime_consumer : "planned");
    yvex_cli_out_writef(stdout, "collection.%s.blocker: %s\n", name,
                        blocker && blocker[0] ? blocker : "none");
}

unsigned int fullmodel_print_blocker(unsigned int index, const char *category, const char *severity,
                                     const char *message, int blocks_full_materialization,
                                     int blocks_generation) {
    yvex_cli_out_writef(stdout, "blocker.%u.category: %s\n", index,
                        category ? category : "runtime-consumer");
    yvex_cli_out_writef(stdout, "blocker.%u.severity: %s\n", index,
                        severity ? severity : "warning");
    yvex_cli_out_writef(stdout, "blocker.%u.message: %s\n", index, message ? message : "");
    yvex_cli_out_writef(stdout, "blocker.%u.blocks_full_materialization: %s\n", index,
                        blocks_full_materialization ? "true" : "false");
    yvex_cli_out_writef(stdout, "blocker.%u.blocks_generation: %s\n", index,
                        blocks_generation ? "true" : "false");
    return index + 1u;
}

static void
fullmodel_print_missing_collection_blockers(unsigned int *index,
                                            const yvex_fullmodel_collections *collections) {
    size_t i;

    if (!index || !collections)
        return;
    for (i = 0u; i < sizeof(plan_collections) / sizeof(plan_collections[0]); ++i) {
        const plan_collection_spec *spec = &plan_collections[i];
        const char *category =
            spec->rule == PLAN_COLLECTION_TOKENIZER ? "tokenizer" : "role-coverage";
        const char *severity =
            spec->rule == PLAN_COLLECTION_NORMALIZATION || spec->rule == PLAN_COLLECTION_TOKENIZER
                ? "error"
            : spec->rule == PLAN_COLLECTION_MOE ? "warning"
                                                : "fatal";

        if (fullmodel_collection_missing(collections, spec->rule)) {
            *index =
                fullmodel_print_blocker(*index, category, severity, spec->missing_blocker, 1, 1);
        }
    }
}

static void fullmodel_print_materialization_phases(
    const yvex_fullmodel_backend_fit *fit, const char *backend, const char *residency,
    const char *role_coverage, unsigned long long tensor_count,
    unsigned long long total_tensor_bytes, unsigned long long artifact_bytes, int selected_target,
    int future_residency, int backend_blocked) {
    fullmodel_print_phase(0u, "preflight", future_residency ? "unsupported" : "ready", tensor_count,
                          total_tensor_bytes, residency, 1, future_residency,
                          future_residency ? "residency mode is not implemented" : "none");
    fullmodel_print_phase(1u, "artifact-identity", "ready", 0ull, artifact_bytes, "host", 1, 0,
                          "none");
    fullmodel_print_phase(2u, "tensor-directory", "ready", tensor_count, total_tensor_bytes, "host",
                          1, 0, "none");
    fullmodel_print_phase(3u, "tensor-range-validation", "ready", tensor_count, total_tensor_bytes,
                          "host", 1, 0, "none");
    fullmodel_print_phase(
        4u, "collection-grouping",
        selected_target || strcmp(role_coverage ? role_coverage : "partial", "observed") != 0
            ? "partial"
            : "ready",
        tensor_count, total_tensor_bytes, "host", 1, selected_target,
        selected_target ? "selected artifacts cannot satisfy full materialization" : "none");
    fullmodel_print_phase(5u, "backend-capability", !fit->available ? "blocked" : "ready", 0ull,
                          0ull, backend, 1, !fit->available,
                          fit->available ? "none" : fit->fit_reason);
    fullmodel_print_phase(6u, "host-residency", "planned", tensor_count, total_tensor_bytes,
                          "host-staged", 1, 0, "none");
    fullmodel_print_phase(7u, "backend-residency", backend_blocked ? "blocked" : "planned",
                          tensor_count, total_tensor_bytes,
                          fullmodel_placement_for_residency(backend, residency, 1), 1,
                          backend_blocked, backend_blocked ? fit->fit_reason : "none");
    fullmodel_print_phase(8u, "kv-residency", "unsupported", 0ull, 0ull, "not-planned", 1, 1,
                          "real attention-backed KV not implemented");
    fullmodel_print_phase(9u, "scratch-residency", "planned", 0ull, 0ull, "host-staged", 1, 0,
                          "scratch sizing remains planned");
    fullmodel_print_phase(10u, "cleanup", "planned", 0ull, 0ull, "not-applicable", 1, 0, "none");
}

static void fullmodel_print_planned_collection(const plan_collection_spec *spec,
                                               const yvex_fullmodel_collections *collections,
                                               const char *backend, const char *residency) {
    unsigned long long count = cli_collection_value(collections, spec->count_offset);
    unsigned long long bytes = cli_collection_value(collections, spec->bytes_offset);
    int present = count > 0ull;
    int ready = present;
    const char *status;
    const char *placement;
    const char *blocker;

    if (spec->rule == PLAN_COLLECTION_ATTENTION) {
        ready = fullmodel_has_attention_collection(collections);
    } else if (spec->rule == PLAN_COLLECTION_MLP) {
        ready = fullmodel_has_mlp_collection(collections);
    }
    status = ready ? "planned" : "blocked";
    placement = fullmodel_placement_for_residency(backend, residency, present);
    blocker = ready ? "none" : spec->missing_blocker;
    if (spec->rule == PLAN_COLLECTION_TOKENIZER) {
        placement = present ? "host-staged" : "not-planned";
    } else if (spec->rule == PLAN_COLLECTION_KV) {
        status = "unsupported";
        placement = "not-planned";
        blocker = spec->missing_blocker;
    } else if (spec->rule == PLAN_COLLECTION_SCRATCH) {
        status = "planned";
        placement = "host-staged";
        blocker = spec->missing_blocker;
    } else if (spec->rule == PLAN_COLLECTION_UNKNOWN) {
        status = present ? "partial" : "not-applicable";
        placement = present ? "unknown" : "not-planned";
        blocker = present ? spec->missing_blocker : "none";
    }
    fullmodel_print_collection_plan(spec->name, status, count, bytes, spec->required_for_generation,
                                    present, placement, spec->phase, spec->runtime_consumer,
                                    blocker);
}

static void fullmodel_print_collection_plans(const yvex_fullmodel_collections *collections,
                                             const char *backend, const char *residency) {
    size_t i;

    for (i = 0; i < sizeof(plan_collections) / sizeof(plan_collections[0]); ++i) {
        fullmodel_print_planned_collection(&plan_collections[i], collections, backend, residency);
    }
}

void fullmodel_print_materialization_plan(
    const yvex_cli_fullmodel_options *options, const yvex_model_ref *ref, const char *target_id,
    const char *target_class, unsigned long long artifact_bytes, yvex_arch arch,
    unsigned long long tensor_count, unsigned long long total_tensor_bytes,
    const yvex_fullmodel_collections *collections, const char *dtype_summary,
    const char *role_coverage, const char *missing_roles, int selected_target) {
    yvex_fullmodel_backend_fit fit;
    size_t static_blocker_index;
    const char *plan_status;
    int materialization_plan_ready;
    unsigned int blocker_count;
    unsigned int blocker_index = 0u;
    const char *backend = options && options->backend ? options->backend : "cpu";
    const char *residency = options && options->residency ? options->residency : "resident";
    int backend_blocked;
    int future_residency;

    fullmodel_probe_backend_fit(backend, total_tensor_bytes, &fit);
    plan_status = fullmodel_plan_status(collections, selected_target, residency, &fit);
    future_residency = fullmodel_residency_is_future_unsupported(residency);
    materialization_plan_ready = !selected_target && !future_residency && tensor_count > 0ull;
    blocker_count = fullmodel_plan_blocker_count(collections, selected_target, residency, &fit);
    backend_blocked = !fit.available || strcmp(fit.fit_status, "does-not-fit") == 0;

    yvex_cli_out_lines(stdout, literal_pair_9, sizeof(literal_pair_9) / sizeof(literal_pair_9[0]));
    yvex_cli_out_writef(stdout, "model: %s\n", options && options->model ? options->model : "");
    yvex_cli_out_writef(stdout, "model_resolved_path: %s\n", ref && ref->path ? ref->path : "");
    yvex_cli_out_writef(stdout, "target_id: %s\n", target_id ? target_id : "path");
    yvex_cli_out_writef(stdout, "target_class: %s\n",
                        target_class ? target_class : "candidate-GGUF-path");
    yvex_cli_out_writef(stdout, "artifact_exists: true\n");
    yvex_cli_out_writef(stdout, "artifact_bytes: %llu\n", artifact_bytes);
    yvex_cli_out_writef(stdout, "artifact_identity_status: %s\n",
                        fullmodel_identity_status(ref, artifact_bytes));
    yvex_cli_out_writef(stdout, "tensor_inventory_status: pass\n");
    yvex_cli_out_writef(stdout, "tensor_count: %llu\n", tensor_count);
    yvex_cli_out_writef(stdout, "total_tensor_bytes: %llu\n", total_tensor_bytes);
    yvex_cli_out_writef(stdout, "architecture: %s\n", yvex_arch_name(arch));
    yvex_cli_out_writef(stdout, "family: %s\n", fullmodel_family_from_arch(arch));
    yvex_cli_out_writef(stdout, "qtype_summary: %s\n", dtype_summary ? dtype_summary : "none");
    yvex_cli_out_writef(stdout, "dtype_summary: %s\n", dtype_summary ? dtype_summary : "none");
    yvex_cli_out_writef(stdout, "required_role_coverage: %s\n",
                        role_coverage ? role_coverage : "partial");
    yvex_cli_out_writef(stdout, "missing_required_roles: %s\n",
                        missing_roles ? missing_roles : "unknown");
    yvex_cli_out_writef(stdout, "backend: %s\n", backend);
    yvex_cli_out_writef(stdout, "residency: %s\n", residency);
    yvex_cli_out_writef(stdout, "plan_status: %s\n", plan_status);
    yvex_cli_out_writef(stdout, "materialization_plan_ready: %s\n",
                        materialization_plan_ready ? "true" : "false");
    yvex_cli_out_lines(stdout, literal_lines_0,
                       sizeof(literal_lines_0) / sizeof(literal_lines_0[0]));
    yvex_cli_out_writef(stdout, "plan_id: fullmodel-materialization:%s:%s:%s\n",
                        target_id ? target_id : "path", backend, residency);
    yvex_cli_out_lines(stdout, literal_pair_8, sizeof(literal_pair_8) / sizeof(literal_pair_8[0]));
    yvex_cli_out_writef(stdout, "plan_backend: %s\n", backend);
    yvex_cli_out_writef(stdout, "plan_residency: %s\n", residency);
    yvex_cli_out_writef(stdout, "plan_tensor_count: %llu\n", tensor_count);
    yvex_cli_out_writef(stdout, "plan_tensor_bytes: %llu\n", total_tensor_bytes);
    yvex_cli_out_lines(stdout, literal_pair_7, sizeof(literal_pair_7) / sizeof(literal_pair_7[0]));
    yvex_cli_out_writef(stdout, "plan_blocker_count: %u\n", blocker_count);
    yvex_cli_out_lines(stdout, literal_pair_6, sizeof(literal_pair_6) / sizeof(literal_pair_6[0]));

    yvex_cli_out_writef(stdout, "backend_available: %s\n", fit.available ? "true" : "false");
    yvex_cli_out_writef(stdout, "backend_memory_known: %s\n", fit.memory_known ? "true" : "false");
    if (fit.memory_known) {
        yvex_cli_out_writef(stdout, "backend_memory_total_bytes: %llu\n", fit.total_bytes);
        yvex_cli_out_writef(stdout, "backend_memory_available_bytes: %llu\n", fit.available_bytes);
    } else {
        yvex_cli_out_lines(stdout, literal_pair_5,
                           sizeof(literal_pair_5) / sizeof(literal_pair_5[0]));
    }
    yvex_cli_out_writef(stdout, "backend_required_bytes: %llu\n", fit.required_bytes);
    yvex_cli_out_writef(stdout, "backend_fit_status: %s\n", fit.fit_status);
    yvex_cli_out_writef(stdout, "backend_fit_reason: %s\n", fit.fit_reason);
    yvex_cli_out_writef(stdout, "backend_allocation_attempted: false\n");

    fullmodel_print_materialization_phases(&fit, backend, residency, role_coverage, tensor_count,
                                           total_tensor_bytes, artifact_bytes, selected_target,
                                           future_residency, backend_blocked);

    fullmodel_print_collection_plans(collections, backend, residency);

    if (selected_target) {
        blocker_index =
            fullmodel_print_blocker(blocker_index, "artifact", "fatal",
                                    "selected artifacts cannot satisfy full materialization", 1, 1);
    }
    if (future_residency) {
        blocker_index =
            fullmodel_print_blocker(blocker_index, "backend-capability", "error",
                                    "requested residency mode is planned but unsupported", 1, 1);
    }
    if (!fit.available) {
        blocker_index = fullmodel_print_blocker(blocker_index, "backend-capability", "error",
                                                fit.fit_reason, 1, 0);
    } else if (strcmp(fit.fit_status ? fit.fit_status : "unknown", "does-not-fit") == 0) {
        blocker_index =
            fullmodel_print_blocker(blocker_index, "backend-memory", "error", fit.fit_reason, 1, 0);
    }
    fullmodel_print_missing_collection_blockers(&blocker_index, collections);
    for (static_blocker_index = 0u;
         static_blocker_index <
         sizeof(materialization_static_blockers) / sizeof(materialization_static_blockers[0]);
         ++static_blocker_index) {
        const fullmodel_static_blocker *blocker =
            &materialization_static_blockers[static_blocker_index];
        blocker_index = fullmodel_print_blocker(blocker_index, blocker->section, "fatal",
                                                blocker->reason, 0, 1);
    }

    yvex_cli_out_lines(stdout, literal_lines_1,
                       sizeof(literal_lines_1) / sizeof(literal_lines_1[0]));
    yvex_cli_out_writef(stdout, "fullmodel_2_blockers: %s\n",
                        selected_target ? "full tensor set missing; full materialization executor "
                                          "not implemented; cleanup proof not implemented"
                                        : "full materialization executor not implemented; cleanup "
                                          "proof not implemented; runtime "
                                          "descriptor not implemented");
}

static void model_download_print_audit_patterns(const yvex_cli_models_download_options *options) {
    unsigned int i;

    for (i = 0; i < model_download_effective_include_count(options); ++i) {
        yvex_cli_out_writef(stdout, "include.%u: %s\n", i,
                            model_download_effective_include_at(options, i));
    }
    for (i = 0; i < model_download_effective_exclude_count(options); ++i) {
        yvex_cli_out_writef(stdout, "exclude.%u: %s\n", i,
                            model_download_effective_exclude_at(options, i));
    }
}

static void model_download_print_normal(const yvex_cli_models_download_options *options,
                                        const yvex_model_download_report *report) {
    char bytes_text[32];
    char largest_text[32];
    char largest_name[64];

    if (strcmp(report->status, "model-download-dry-run") == 0) {
        yvex_cli_out_writef(stdout, "model-download: dry-run target=%s\n", report->target_id);
        render_object_fields(stdout, report, download_normal_identity_fields,
                             sizeof(download_normal_identity_fields) /
                                 sizeof(download_normal_identity_fields[0]));
        yvex_cli_out_writef(stdout, "account_provider: %s\n", report->stage_account_provider);
        yvex_cli_out_lines(stdout, literal_lines_2,
                           sizeof(literal_lines_2) / sizeof(literal_lines_2[0]));
        yvex_cli_out_writef(stdout, "status: %s\n", report->status);
        return;
    }
    if (strcmp(report->status, "model-download-blocked") == 0 ||
        strcmp(report->status, "model-download-resume-blocked") == 0) {
        yvex_cli_out_writef(stdout, "%s target=%s\n",
                            strcmp(report->status, "model-download-resume-blocked") == 0
                                ? "model-download-resume: blocked"
                                : "model-download: blocked",
                            report->target_id);
        render_object_fields(stdout, report, download_normal_identity_fields, 3u);
        yvex_cli_out_writef(stdout, "stage: account-provider %s\n", report->stage_account_provider);
        yvex_cli_out_writef(stdout, "auth_state: %s\n", report->auth_state);
        yvex_cli_out_writef(stdout, "top_blocker: %s\n",
                            report->top_blocker[0] ? report->top_blocker : "unknown");
        yvex_cli_out_writef(stdout, "next: %s\n",
                            report->error[0] ? report->error : "resolve provider account state");
        yvex_cli_out_writef(stdout, "boundary: no payload downloaded, runtime unsupported\n");
        yvex_cli_out_writef(stdout, "status: %s\n", report->status);
        return;
    }
    if (strcmp(report->status, "model-download-pass") == 0 ||
        strcmp(report->status, "model-download-resume-pass") == 0) {
        yvex_cli_out_writef(stdout, "%s target=%s\n",
                            strcmp(report->status, "model-download-resume-pass") == 0
                                ? "model-download-resume: pass"
                                : "model-download: pass",
                            report->target_id);
        render_object_fields(stdout, report, download_normal_identity_fields,
                             sizeof(download_normal_identity_fields) /
                                 sizeof(download_normal_identity_fields[0]));
        yvex_cli_out_writef(stdout, "account_provider: %s\n", report->stage_account_provider);
        model_download_format_bytes(bytes_text, sizeof(bytes_text),
                                    report->source_scan.total_regular_file_bytes);
        yvex_cli_out_writef(stdout, "files: %llu partial=%llu safetensors=%llu bytes=%s\n",
                            report->source_scan.file_count, report->source_scan.partial_file_count,
                            report->source_scan.safetensors_count, bytes_text);
        yvex_cli_out_writef(stdout, "manifest: %s\n",
                            report->source_manifest_written ? report->manifest_path : "skipped");
        yvex_cli_out_writef(stdout, "native_inventory: %s\n",
                            report->native_inventory_written ? report->native_inventory_path
                                                             : "skipped");
        yvex_cli_out_writef(stdout, "boundary: source tensors only, runtime unsupported\n");
        yvex_cli_out_writef(stdout, "status: %s\n", report->status);
        return;
    }
    if (strcmp(report->status, "model-download-interrupted") == 0) {
        yvex_cli_out_writef(stdout, "model-download: interrupted target=%s\n", report->target_id);
        yvex_cli_out_writef(stdout, "family: %s\n", report->family);
        yvex_cli_out_writef(stdout, "repo: %s\n", report->repo_id);
        yvex_cli_out_writef(stdout, "source: %s\n", report->local_source_dir);
        yvex_cli_out_writef(stdout, "stage: download interrupted\n");
        yvex_cli_out_writef(stdout, "signal: %s\n",
                            model_download_signal_name(report->interrupt_signal));
        yvex_cli_out_writef(stdout, "child_signal_forwarded: %s\n",
                            report->signal_forwarded ? "true" : "false");
        yvex_cli_out_writef(stdout, "child_exit_status: %s\n", report->child_exit_status);
        model_download_format_bytes(bytes_text, sizeof(bytes_text),
                                    report->source_scan.total_regular_file_bytes);
        model_download_format_bytes(largest_text, sizeof(largest_text),
                                    report->source_scan.largest_file_bytes);
        model_download_short_file_name(largest_name, sizeof(largest_name),
                                       report->source_scan.largest_file_name);
        yvex_cli_out_writef(stdout, "files: %llu partial=%llu safetensors=%llu bytes=%s\n",
                            report->source_scan.file_count, report->source_scan.partial_file_count,
                            report->source_scan.safetensors_count, bytes_text);
        yvex_cli_out_writef(stdout, "largest: %s (%s)\n", largest_name, largest_text);
        yvex_cli_out_lines(stdout, literal_pair_4,
                           sizeof(literal_pair_4) / sizeof(literal_pair_4[0]));
        yvex_cli_out_writef(stdout, "partial_source_preserved: %s\n",
                            report->partial_source_preserved ? "true" : "false");
        yvex_cli_out_writef(stdout, "lock_files_deleted: %s\n",
                            report->lock_files_deleted ? "true" : "false");
        yvex_cli_out_writef(stdout, "stdout_log: %s\n", report->stdout_log_path);
        yvex_cli_out_writef(stdout, "stderr_log: %s\n", report->stderr_log_path);
        yvex_cli_out_writef(stdout,
                            "boundary: partial source files may exist; runtime unsupported\n");
        yvex_cli_out_writef(stdout, "status: %s\n", report->status);
        return;
    }

    yvex_cli_out_writef(stdout, "model-download: fail target=%s\n", report->target_id);
    render_object_fields(stdout, report, download_normal_identity_fields,
                         sizeof(download_normal_identity_fields) /
                             sizeof(download_normal_identity_fields[0]));
    yvex_cli_out_writef(stdout, "hf_exit_code: %d\n", report->hf_exit_code);
    yvex_cli_out_writef(stdout, "stderr_log: %s\n", report->stderr_log_path);
    if (report->error[0])
        yvex_cli_out_writef(stdout, "reason: %s\n", report->error);
    yvex_cli_out_writef(stdout, "boundary: no runtime, no generation, no benchmark\n");
    yvex_cli_out_writef(stdout, "status: %s\n", report->status);
    (void)options;
}

static void model_download_print_table(const yvex_model_download_report *report) {
    char bytes_text[32];

    model_download_format_bytes(bytes_text, sizeof(bytes_text),
                                report->source_scan.total_regular_file_bytes);
    yvex_cli_out_writef(stdout, "TARGET       PROVIDER     FAMILY  ACCOUNT  STATUS                 "
                                "      FILES  PARTIAL  SAFETENSORS  BYTES\n");
    yvex_cli_out_writef(stdout, "%-12s %-11s  %-6s  %-7s  %-27s  %5llu  %7llu  %11llu  %s\n",
                        report->target_id, report->provider, report->family,
                        report->stage_account_provider, report->status,
                        report->source_scan.file_count, report->source_scan.partial_file_count,
                        report->source_scan.safetensors_count, bytes_text);
    yvex_cli_out_writef(stdout, "status: %s\n", report->status);
}

static void model_download_print_audit(const yvex_cli_models_download_options *options,
                                       const yvex_model_download_report *report) {
    yvex_cli_out_writef(stdout, "models: download\n");
    render_object_fields(stdout, report, download_audit_identity_fields,
                         sizeof(download_audit_identity_fields) /
                             sizeof(download_audit_identity_fields[0]));
    yvex_cli_out_writef(stdout, "auth_mode: %s\n",
                        model_download_auth_mode_name(options->auth_mode));
    render_object_fields(stdout, report, download_audit_process_fields,
                         sizeof(download_audit_process_fields) /
                             sizeof(download_audit_process_fields[0]));
    yvex_cli_out_writef(stdout, "token_value_redacted: %s\n",
                        strcmp(report->auth_state, "env-token-present") == 0 ? "true" : "false");
    model_download_print_audit_patterns(options);
    yvex_cli_out_writef(stdout, "progress_mode: %s\n",
                        model_download_progress_mode_name(options->progress_mode));
    yvex_cli_out_writef(stdout, "tick_seconds: %llu\n", options->tick_seconds);
    render_object_fields(stdout, report, download_audit_telemetry_fields,
                         sizeof(download_audit_telemetry_fields) /
                             sizeof(download_audit_telemetry_fields[0]));
    model_stage_print("resolve-target", report->stage_resolve_target);
    model_stage_print("resolve-paths", report->stage_resolve_paths);
    model_stage_print("prepare-dirs", report->stage_prepare_dirs);
    model_stage_print("account-provider", report->stage_account_provider);
    model_stage_print("provider-cli", report->stage_provider_cli);
    model_stage_print("hf-cli", report->stage_hf_cli);
    model_stage_print("download", report->stage_download);
    model_stage_print("progress-stream", report->stage_progress_stream);
    model_stage_print("progress-ticks", report->stage_progress_ticks);
    model_stage_print("source-scan", report->stage_source_scan);
    model_stage_print("source-manifest", report->stage_source_manifest);
    model_stage_print("native-inventory", report->stage_native_inventory);
    model_stage_print("sidecar", report->stage_sidecar);
    yvex_cli_out_writef(stdout, "provider_pid: %lld\n", (long long)report->provider_pid);
    yvex_cli_out_writef(stdout, "provider_process_group: %lld\n",
                        (long long)report->provider_process_group);
    yvex_cli_out_writef(stdout, "interrupted: %s\n", report->interrupted ? "true" : "false");
    yvex_cli_out_writef(stdout, "interrupt_signal: %s\n",
                        model_download_signal_name(report->interrupt_signal));
    yvex_cli_out_writef(stdout, "signal: %s\n",
                        model_download_signal_name(report->interrupt_signal));
    render_object_fields(stdout, report, download_audit_lifecycle_fields,
                         sizeof(download_audit_lifecycle_fields) /
                             sizeof(download_audit_lifecycle_fields[0]));
    yvex_cli_out_writef(stdout, "lock_cleanup: not-attempted\n");
    yvex_cli_out_writef(stdout, "lock_files_deleted: %s\n",
                        report->lock_files_deleted ? "true" : "false");
    render_object_fields(stdout, report, download_audit_source_fields,
                         sizeof(download_audit_source_fields) /
                             sizeof(download_audit_source_fields[0]));
    yvex_cli_out_writef(stdout, "yvex_version: %s\n", yvex_version_string());
    yvex_cli_out_lines(stdout, literal_lines_3,
                       sizeof(literal_lines_3) / sizeof(literal_lines_3[0]));
    if (report->top_blocker[0])
        yvex_cli_out_writef(stdout, "top_blocker: %s\n", report->top_blocker);
    if (report->error[0])
        yvex_cli_out_writef(stdout, "reason: %s\n", report->error);
}

static void model_download_print(const yvex_cli_models_download_options *options,
                                 const yvex_model_download_report *report) {
    if (options && options->output_mode == YVEX_MODELS_OUTPUT_AUDIT) {
        model_download_print_audit(options, report);
    } else if (options && options->output_mode == YVEX_MODELS_OUTPUT_TABLE) {
        model_download_print_table(report);
    } else {
        model_download_print_normal(options, report);
    }
}

int model_download_finish(const yvex_cli_models_download_options *options,
                          yvex_model_download_report *report) {
    model_download_print(options, report);
    if (strcmp(report->status, "model-download-pass") == 0 ||
        strcmp(report->status, "model-download-resume-pass") == 0 ||
        strcmp(report->status, "model-download-dry-run") == 0) {
        return 0;
    }
    if (strcmp(report->status, "model-download-blocked") == 0 ||
        strcmp(report->status, "model-download-resume-blocked") == 0) {
        return exit_for_status(YVEX_ERR_UNSUPPORTED);
    }
    return 1;
}

static void model_download_write_pattern_array(FILE *fp, const char *name,
                                               const yvex_cli_models_download_options *options,
                                               int includes, int comma) {
    unsigned int count = includes ? model_download_effective_include_count(options)
                                  : model_download_effective_exclude_count(options);
    unsigned int i;

    yvex_cli_out_writef(fp, "  \"%s\": [", name);
    for (i = 0; i < count; ++i) {
        if (i > 0)
            yvex_cli_out_fputs(", ", fp);
        write_escaped(fp, includes ? model_download_effective_include_at(options, i)
                                   : model_download_effective_exclude_at(options, i));
    }
    yvex_cli_out_writef(fp, "]%s\n", comma ? "," : "");
}

int model_download_write_native_inventory_json(const char *path, const char *source_dir,
                                               const yvex_native_weight_table *table,
                                               yvex_error *err) {
    yvex_native_weight_summary summary;
    FILE *fp;
    unsigned long long count;
    unsigned long long i;

    if (!path || !source_dir || !table) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "models_download_native_inventory",
                       "path, source directory, and table are required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (yvex_native_weight_table_summary(table, &summary, err) != YVEX_OK) {
        return yvex_error_code(err);
    }
    fp = fopen(path, "wb");
    if (!fp) {
        yvex_error_setf(err, YVEX_ERR_IO, "models_download_native_inventory",
                        "cannot open native inventory: %s", path);
        return YVEX_ERR_IO;
    }
    count = yvex_native_weight_table_count(table);
    yvex_cli_out_writef(fp, "{\n");
    write_field(fp, "", "schema", "yvex.native_inventory.v1", 1);
    write_field(fp, "", "source_dir", source_dir, 1);
    write_bool_field(fp, "", "payload_loaded", 0, 1);
    yvex_cli_out_writef(fp, "  \"summary\": {\n");
    write_u64_field(fp, "", "shard_count", summary.shard_count, 1);
    write_u64_field(fp, "", "tensor_count", summary.tensor_count, 1);
    write_u64_field(fp, "", "total_tensor_bytes", summary.total_tensor_bytes, 1);
    write_u64_field(fp, "", "unknown_dtype_count", summary.unknown_dtype_count, 1);
    write_u64_field(fp, "", "malformed_shard_count", summary.malformed_shard_count, 0);
    yvex_cli_out_lines(fp, literal_pair_3, sizeof(literal_pair_3) / sizeof(literal_pair_3[0]));
    for (i = 0; i < count; ++i) {
        const yvex_native_weight_info *info = yvex_native_weight_table_at(table, i);
        unsigned int d;

        yvex_cli_out_writef(fp, "    {\n");
        write_field(fp, "", "name", info && info->name ? info->name : "", 1);
        write_field(fp, "", "shard_path", info && info->shard_path ? info->shard_path : "", 1);
        write_field(fp, "", "dtype", info && info->dtype_name ? info->dtype_name : "UNKNOWN", 1);
        yvex_cli_out_writef(fp, "      \"rank\": %u,\n", info ? info->rank : 0u);
        yvex_cli_out_writef(fp, "      \"shape\": [");
        if (info) {
            for (d = 0; d < info->rank; ++d) {
                yvex_cli_out_writef(fp, "%s%llu", d == 0 ? "" : ", ", info->dims[d]);
            }
        }
        yvex_cli_out_writef(fp, "],\n");
        write_u64_field(fp, "", "data_start", info ? info->data_start : 0ull, 1);
        write_u64_field(fp, "", "data_end", info ? info->data_end : 0ull, 1);
        write_u64_field(fp, "", "data_bytes", info ? info->data_bytes : 0ull, 0);
        yvex_cli_out_writef(fp, "    }%s\n", i + 1ull == count ? "" : ",");
    }
    yvex_cli_out_lines(fp, literal_pair_2, sizeof(literal_pair_2) / sizeof(literal_pair_2[0]));
    if (fclose(fp) != 0) {
        yvex_error_setf(err, YVEX_ERR_IO, "models_download_native_inventory",
                        "cannot close native inventory: %s", path);
        return YVEX_ERR_IO;
    }
    return YVEX_OK;
}

int model_download_write_json_sidecar(const char *path, const char *schema,
                                      const yvex_cli_models_download_options *options,
                                      const yvex_model_download_report *report, yvex_error *err) {
    FILE *fp;

    if (!path || !schema || !options || !report) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "models_download_sidecar",
                       "path, schema, options, and report are required");
        return YVEX_ERR_INVALID_ARG;
    }
    fp = fopen(path, "wb");
    if (!fp) {
        yvex_error_setf(err, YVEX_ERR_IO, "models_download_sidecar", "cannot open sidecar: %s",
                        path);
        return YVEX_ERR_IO;
    }
    yvex_cli_out_writef(fp, "{\n");
    write_field(fp, "", "schema", schema, 1);
    write_field(fp, "", "status", report->status, 1);
    write_field(fp, "", "target_id", report->target_id, 1);
    write_field(fp, "", "family", report->family, 1);
    write_field(fp, "", "provider", report->provider, 1);
    write_field(fp, "", "repo_id", report->repo_id, 1);
    write_field(fp, "", "revision", report->revision, 1);
    write_field(fp, "", "local_source_dir", report->local_source_dir, 1);
    write_field(fp, "", "models_root", report->models_root, 1);
    write_field(fp, "", "models_root_source", report->models_root_source, 1);
    model_download_write_pattern_array(fp, "include_patterns", options, 1, 1);
    model_download_write_pattern_array(fp, "exclude_patterns", options, 0, 1);
    write_field(fp, "", "auth_mode", model_download_auth_mode_name(options->auth_mode), 1);
    write_field(fp, "", "provider_cli_path", report->provider_cli_path, 1);
    write_field(fp, "", "provider_cli_status", report->provider_cli_status, 1);
    write_field(fp, "", "account_provider_stage", report->stage_account_provider, 1);
    write_field(fp, "", "credential_source", report->credential_source, 1);
    write_field(fp, "", "account_hint", report->account_hint, 1);
    write_field(fp, "", "accounts_state_path", report->accounts_state_path, 1);
    write_field(fp, "", "hf_cli_path", report->hf_cli_path, 1);
    yvex_cli_out_writef(fp, "  \"hf_exit_code\": %d,\n", report->hf_exit_code);
    yvex_cli_out_writef(fp, "  \"provider_exit_code\": %d,\n", report->provider_exit_code);
    write_field(fp, "", "auth_state", report->auth_state, 1);
    write_field(fp, "", "progress_mode", model_download_progress_mode_name(options->progress_mode),
                1);
    write_u64_field(fp, "", "tick_seconds", options->tick_seconds, 1);
    write_u64_field(fp, "", "tick_count", report->tick_count, 1);
    write_bool_field(fp, "", "stdout_streamed", report->stdout_streamed, 1);
    write_bool_field(fp, "", "stderr_streamed", report->stderr_streamed, 1);
    write_u64_field(fp, "", "stdout_bytes", report->stdout_bytes, 1);
    write_u64_field(fp, "", "stderr_bytes", report->stderr_bytes, 1);
    yvex_cli_out_writef(fp, "  \"provider_pid\": %lld,\n", (long long)report->provider_pid);
    yvex_cli_out_writef(fp, "  \"provider_process_group\": %lld,\n",
                        (long long)report->provider_process_group);
    write_bool_field(fp, "", "interrupted", report->interrupted, 1);
    write_field(fp, "", "interrupt_signal", model_download_signal_name(report->interrupt_signal),
                1);
    write_bool_field(fp, "", "signal_forwarded", report->signal_forwarded, 1);
    write_bool_field(fp, "", "child_terminated", report->child_terminated, 1);
    write_bool_field(fp, "", "child_killed_after_timeout", report->child_killed_after_timeout, 1);
    write_field(fp, "", "child_exit_status", report->child_exit_status, 1);
    write_bool_field(fp, "", "orphan_check_performed", report->orphan_check_performed, 1);
    write_field(fp, "", "orphan_check_status", report->orphan_check_status, 1);
    write_bool_field(fp, "", "partial_source_preserved", report->partial_source_preserved, 1);
    write_bool_field(fp, "", "lock_files_deleted", report->lock_files_deleted, 1);
    write_u64_field(fp, "", "file_count", report->source_scan.file_count, 1);
    write_u64_field(fp, "", "safetensors_count", report->source_scan.safetensors_count, 1);
    write_bool_field(fp, "", "config_present", report->source_scan.config_present, 1);
    write_bool_field(fp, "", "tokenizer_present", report->source_scan.tokenizer_present, 1);
    write_u64_field(fp, "", "total_regular_file_bytes",
                    report->source_scan.total_regular_file_bytes, 1);
    write_field(fp, "", "largest_file_name",
                report->source_scan.largest_file_name[0] ? report->source_scan.largest_file_name
                                                         : "none",
                1);
    write_u64_field(fp, "", "largest_file_bytes", report->source_scan.largest_file_bytes, 1);
    write_field(fp, "", "manifest_path", report->manifest_path, 1);
    write_field(fp, "", "native_inventory_path", report->native_inventory_path, 1);
    write_field(fp, "", "stdout_log", report->stdout_log_path, 1);
    write_field(fp, "", "stderr_log", report->stderr_log_path, 1);
    write_field(fp, "", "created_at", report->created_at, 1);
    write_field(fp, "", "yvex_version", yvex_version_string(), 1);
    write_bool_field(fp, "", "upstream_identity_verified", 0, 1);
    write_bool_field(fp, "", "remote_lookup_performed", 0, 1);
    write_bool_field(fp, "", "payload_hash_verified", 0, 1);
    write_bool_field(fp, "", "force_sidecars", options->force_sidecars, 1);
    write_bool_field(fp, "", "yes", options->yes, 1);
    yvex_cli_out_writef(fp, "  \"boundary\": {\n");
    write_field(fp, "", "source_download",
                options->dry_run
                    ? "dry-run"
                    : ((strcmp(report->status, "model-download-pass") == 0 ||
                        strcmp(report->status, "model-download-resume-pass") == 0)
                           ? "performed"
                           : (strcmp(report->status, "model-download-blocked") == 0 ? "blocked"
                                                                                    : "failed")),
                1);
    write_bool_field(fp, "", "source_manifest_written", report->source_manifest_written, 1);
    write_bool_field(fp, "", "native_inventory_written", report->native_inventory_written, 1);
    write_bool_field(fp, "", "payload_loaded", 0, 1);
    write_bool_field(fp, "", "gguf_created", 0, 1);
    write_bool_field(fp, "", "materialized", 0, 1);
    write_bool_field(fp, "", "runtime_ready", 0, 1);
    write_field(fp, "", "generation", "unsupported", 1);
    write_field(fp, "", "eval", "unsupported", 1);
    write_field(fp, "", "benchmark", "not-measured", 0);
    yvex_cli_out_lines(fp, literal_pair_1, sizeof(literal_pair_1) / sizeof(literal_pair_1[0]));
    if (fclose(fp) != 0) {
        yvex_error_setf(err, YVEX_ERR_IO, "models_download_sidecar", "cannot close sidecar: %s",
                        path);
        return YVEX_ERR_IO;
    }
    return YVEX_OK;
}

int model_download_write_receipt(const char *path, const yvex_cli_models_download_options *options,
                                 const yvex_model_download_report *report, int token_present,
                                 yvex_error *err) {
    FILE *fp;
    unsigned int i;

    if (!path || !options || !report) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "models_download_receipt",
                       "receipt path, options, and report are required");
        return YVEX_ERR_INVALID_ARG;
    }
    fp = fopen(path, "wb");
    if (!fp) {
        yvex_error_setf(err, YVEX_ERR_IO, "models_download_receipt", "cannot open receipt: %s",
                        path);
        return YVEX_ERR_IO;
    }
    yvex_cli_out_writef(fp, "schema: yvex.model_download.receipt.v1\n");
    yvex_cli_out_writef(fp, "target_id: %s\n", report->target_id);
    yvex_cli_out_writef(fp, "family: %s\n", report->family);
    yvex_cli_out_writef(fp, "repo_id: %s\n", report->repo_id);
    yvex_cli_out_writef(fp, "revision: %s\n", report->revision);
    yvex_cli_out_writef(fp, "local_source_dir: %s\n", report->local_source_dir);
    yvex_cli_out_writef(fp, "hf_cli_path: %s\n", report->hf_cli_path);
    yvex_cli_out_writef(fp, "provider_cli_path: %s\n", report->provider_cli_path);
    yvex_cli_out_writef(fp, "provider_cli_status: %s\n", report->provider_cli_status);
    yvex_cli_out_writef(fp, "auth_mode: %s\n", model_download_auth_mode_name(options->auth_mode));
    yvex_cli_out_writef(fp, "auth_state: %s\n", report->auth_state);
    yvex_cli_out_writef(fp, "token_env_name: %s\n", report->token_env_name);
    yvex_cli_out_writef(fp, "token_value_redacted: %s\n", token_present ? "true" : "false");
    if (strcmp(report->provider, "github") == 0) {
        yvex_cli_out_writef(fp, "command: %s release download", report->provider_cli_path);
        if (options->release && options->release[0])
            yvex_cli_out_writef(fp, " %s", options->release);
        yvex_cli_out_writef(fp, " --repo %s --pattern %s --dir %s --skip-existing", report->repo_id,
                            options->asset ? options->asset : "", report->local_source_dir);
    } else {
        yvex_cli_out_writef(fp, "command: %s download %s --revision %s --local-dir %s",
                            report->provider_cli_path, report->repo_id, report->revision,
                            report->local_source_dir);
        for (i = 0; i < model_download_effective_include_count(options); ++i) {
            yvex_cli_out_writef(fp, " --include %s",
                                model_download_effective_include_at(options, i));
        }
        for (i = 0; i < model_download_effective_exclude_count(options); ++i) {
            yvex_cli_out_writef(fp, " --exclude %s",
                                model_download_effective_exclude_at(options, i));
        }
        yvex_cli_out_writef(fp, " --max-workers %llu", options->max_workers);
        if (options->dry_run)
            yvex_cli_out_writef(fp, " --dry-run");
        if (token_present)
            yvex_cli_out_writef(fp, " --token <redacted>");
    }
    yvex_cli_out_writef(fp, "\n");
    if (fclose(fp) != 0) {
        yvex_error_setf(err, YVEX_ERR_IO, "models_download_receipt", "cannot close receipt: %s",
                        path);
        return YVEX_ERR_IO;
    }
    return YVEX_OK;
}

int model_download_write_control_receipt(const char *path,
                                         const yvex_cli_models_download_options *options,
                                         const yvex_model_download_report *report,
                                         const char *status, yvex_error *err) {
    FILE *fp;

    if (!path || !options || !report || !status) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "models_download_control_receipt",
                       "path, options, report, and status are required");
        return YVEX_ERR_INVALID_ARG;
    }
    fp = fopen(path, "wb");
    if (!fp) {
        yvex_error_setf(err, YVEX_ERR_IO, "models_download_control_receipt",
                        "cannot open download receipt: %s", path);
        return YVEX_ERR_IO;
    }
    yvex_cli_out_writef(fp, "{\n");
    write_field(fp, "", "schema", "yvex.model_download.active.v1", 1);
    write_field(fp, "", "target_id", report->target_id, 1);
    write_field(fp, "", "family", report->family, 1);
    write_field(fp, "", "provider", report->provider, 1);
    write_field(fp, "", "repo_id", report->repo_id, 1);
    write_field(fp, "", "revision", report->revision, 1);
    write_field(fp, "", "local_source_dir", report->local_source_dir, 1);
    model_download_write_pattern_array(fp, "include_patterns", options, 1, 1);
    model_download_write_pattern_array(fp, "exclude_patterns", options, 0, 1);
    write_field(fp, "", "auth_mode", model_download_auth_mode_name(options->auth_mode), 1);
    write_field(fp, "", "progress_mode", model_download_progress_mode_name(options->progress_mode),
                1);
    write_field(fp, "", "started_at", report->created_at, 1);
    yvex_cli_out_writef(fp, "  \"yvex_pid\": %lld,\n", (long long)getpid());
    yvex_cli_out_writef(fp, "  \"provider_pid\": %lld,\n", (long long)report->provider_pid);
    yvex_cli_out_writef(fp, "  \"provider_pgid\": %lld,\n",
                        (long long)report->provider_process_group);
    write_field(fp, "", "stdout_log", report->stdout_log_path, 1);
    write_field(fp, "", "stderr_log", report->stderr_log_path, 1);
    write_field(fp, "", "status", status, 0);
    yvex_cli_out_writef(fp, "}\n");
    if (fclose(fp) != 0) {
        yvex_error_setf(err, YVEX_ERR_IO, "models_download_control_receipt",
                        "cannot close download receipt: %s", path);
        return YVEX_ERR_IO;
    }
    return YVEX_OK;
}

int model_download_finalize_control_receipt(const yvex_cli_models_download_options *options,
                                            const yvex_model_download_report *report,
                                            const char *status) {
    yvex_error err;
    int rc = YVEX_OK;

    yvex_error_clear(&err);
    if (report && report->last_receipt_path[0]) {
        rc = model_download_write_control_receipt(report->last_receipt_path, options, report,
                                                  status, &err);
    }
    if (report && report->active_receipt_path[0]) {
        (void)unlink(report->active_receipt_path);
    }
    return rc;
}
