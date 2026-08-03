/*
 * Coordinate model-target report routing and shared bounded evidence probes.
 *
 * The coordinator routes typed requests to specialized model-target modules; it does not render,
 * open operator streams, or contain report-specific static catalogs. Model-target reports are
 * report-only facts. This coordinator does not implement quantization, artifact emission, runtime
 * execution, generation, eval, benchmark, throughput, or release readiness.
 */
#include <yvex/internal/model_target.h>

#include <yvex/internal/families/deepseek_v4.h>

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include <yvex/source.h>

#define MODEL_TARGET_HEADER_CAP (1024ull * 1024ull)

static const char *const output_contract_tail[] = {
    "runtime_claim: unsupported",
    "generation: unsupported-full-model",
    "benchmark_status: not-measured",
    "release_ready: false",
    "boundary: output-contract check only; no runtime/generation claim"
};

typedef struct {
    size_t report_offset;
    size_t profile_offset;
    size_t capacity;
} report_profile_field;

#define REPORT_PROFILE_FIELD(member) \
    { offsetof(yvex_model_target_report, member), \
      offsetof(yvex_model_target_report_profile, member), \
      sizeof(((yvex_model_target_report *)0)->member) }

static const report_profile_field report_profile_fields[] = {
    REPORT_PROFILE_FIELD(target_id),
    REPORT_PROFILE_FIELD(family),
    REPORT_PROFILE_FIELD(model),
    REPORT_PROFILE_FIELD(target_class),
    REPORT_PROFILE_FIELD(stage),
    REPORT_PROFILE_FIELD(eligibility),
    REPORT_PROFILE_FIELD(source_status),
    REPORT_PROFILE_FIELD(artifact_status),
    REPORT_PROFILE_FIELD(tensor_map_status),
    REPORT_PROFILE_FIELD(qtype_policy_status),
    REPORT_PROFILE_FIELD(runtime_status),
    REPORT_PROFILE_FIELD(generation_status),
    REPORT_PROFILE_FIELD(benchmark_status),
    REPORT_PROFILE_FIELD(next_row),
    REPORT_PROFILE_FIELD(boundary),
    REPORT_PROFILE_FIELD(reason)
};

#undef REPORT_PROFILE_FIELD

void yvex_model_target_report_prepare(
    yvex_model_target_report *report,
    const yvex_model_target_request *request,
    const yvex_model_target_report_profile *profile)
{
    const unsigned char *source = (const unsigned char *)profile;
    unsigned char *destination = (unsigned char *)report;
    size_t index;

    if (!report || !request || !profile) return;
    report->kind = request->kind;
    report->mode = request->mode;
    report->status = profile->status;
    report->exit_code = 0;
    for (index = 0u; index < sizeof(report_profile_fields) /
                                 sizeof(report_profile_fields[0]); ++index) {
        const report_profile_field *field = &report_profile_fields[index];
        const char *value = *(const char *const *)(source + field->profile_offset);

        if (value) {
            yvex_core_text_copy((char *)(destination + field->report_offset), field->capacity, value);
        }
    }
}

int yvex_model_target_report_release_coverage(
    const yvex_model_target_request *request,
    yvex_model_target_report *report,
    const char *operation,
    const char *error_where,
    const char *success_status,
    const char *success_boundary,
    yvex_error *err)
{
    const yvex_model_family_api *family = yvex_model_register_deepseek_v4();
    yvex_source_verification verification;
    yvex_deepseek_tensor_coverage *coverage = NULL;
    yvex_deepseek_tensor_coverage_failure failure;
    char models_root[512];
    char source_path[512];
    int rc;

    if (!request || !report || !operation || !error_where || !success_status ||
        !success_boundary || !family) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "model_target_report",
                       "DeepSeek coverage report arguments are required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (!yvex_model_target_release_source_paths(
            request, models_root, sizeof(models_root), source_path,
            sizeof(source_path))) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, error_where,
                       "DeepSeek source path exceeds report bounds");
        return YVEX_ERR_BOUNDS;
    }
    rc = family->coverage.open_verified_source(
        &coverage, &verification, source_path, models_root, &failure, err);
    if (rc != YVEX_OK) {
        report->status = "tensor-coverage-blocked";
        report->exit_code = 5;
        yvex_model_target_report_add_error(
            report,
            "model-target %s: DeepSeek coverage refused: %s tensor=%s",
            operation, family->coverage.failure_name(failure.code),
            failure.tensor_name[0] ? failure.tensor_name : "none");
        yvex_error_clear(err);
        return YVEX_OK;
    }
    report->family_coverage = coverage;
    {
        const yvex_model_target_report_profile profile = {
            .status = success_status, .target_id = request->target_id,
            .family = "deepseek", .stage = "header-only",
            .tensor_map_status = "blocked", .runtime_status = "unsupported",
            .generation_status = "unsupported",
            .next_row = "V010.SOURCE.PAYLOAD.STREAM.0",
            .boundary = success_boundary
        };

        yvex_model_target_report_prepare(report, request, &profile);
    }
    return YVEX_OK;
}

static int model_target_report_refuse(yvex_model_target_report *report,
                                      const char *status,
                                      const char *message)
{
    report->status = status;
    report->exit_code = 2;
    yvex_model_target_report_add_error(report, "%s", message);
    return 0;
}

/*
 * Enforce shared command, target, release, and output-shape rules.
 *
 * Returns false after the first deterministic shape refusal.
 */
int yvex_model_target_validate_request_shape(
    const yvex_model_target_request *request,
    yvex_model_target_report *report,
    const yvex_model_target_request_rules *rules,
    const char *release)
{
    if (!request || !report || !rules) return 0;
    if (request->kind != rules->expected_kind) {
        return model_target_report_refuse(report, rules->kind_failure_status, rules->kind_failure_message);
    }
    if (rules->required_target_operation && !request->target_id[0]) {
        report->status = "parser-error";
        report->exit_code = 2;
        yvex_model_target_report_add_error(
            report, "model-target %s: requires TARGET",
            rules->required_target_operation);
        return 0;
    }
    if (release && release[0] && strcmp(release, "v0.1.0") != 0) {
        report->status = "unsupported-release";
        report->exit_code = 2;
        yvex_model_target_report_add_row(report, "status: unsupported-release");
        yvex_model_target_report_add_row(report, "release: %s", release);
        yvex_model_target_report_add_error(report, "unsupported release: %s",
                                           release);
        return 0;
    }
    if (rules->reject_json &&
        request->mode == YVEX_MODEL_TARGET_OUTPUT_JSON) {
        return model_target_report_refuse(report, "unsupported-output-mode", "JSON output is unsupported");
    }
    return 1;
}

static int model_target_report_store(yvex_model_target_text_value *rows,
                                     unsigned long cap,
                                     unsigned long *count,
                                     const char *fmt,
                                     va_list ap)
{
    int n;

    if (!rows || !count || !fmt || *count >= cap) {
        return 0;
    }
    n = vsnprintf(rows[*count].value, sizeof(rows[*count].value), fmt, ap);
    if (n < 0) {
        rows[*count].value[0] = '\0';
        return 0;
    }
    rows[*count].value[sizeof(rows[*count].value) - 1u] = '\0';
    (*count)++;
    return 1;
}

int yvex_model_target_report_add_row(yvex_model_target_report *report,
                                     const char *fmt,
                                     ...)
{
    va_list ap;
    int ok;

    if (!report || !fmt) {
        return 0;
    }
    va_start(ap, fmt);
    ok = model_target_report_store(report->rows,
                                   YVEX_MODEL_TARGET_ROW_CAP,
                                   &report->row_count,
                                   fmt,
                                   ap);
    va_end(ap);
    return ok;
}

/*
 * Append an immutable ordered row template without duplicating report mechanics.
 *
 * Appends rows in declaration order until the bounded report refuses one.
 */
void yvex_model_target_report_add_rows(yvex_model_target_report *report,
                                       const char *const *rows,
                                       size_t row_count)
{
    size_t row;

    if (!report || (!rows && row_count != 0u)) return;
    for (row = 0u; row < row_count; ++row) {
        if (!yvex_model_target_report_add_row(report, "%s", rows[row])) return;
    }
}

void yvex_model_target_report_project_rows(
    yvex_model_target_report *report,
    const yvex_model_target_row_spec *rows,
    size_t row_count,
    const void *facts)
{
    const unsigned char *base = (const unsigned char *)facts;
    size_t row;

    if (!report || (!rows && row_count != 0u) || (!facts && row_count != 0u)) return;
    for (row = 0u; row < row_count; ++row) {
        const void *value = base + rows[row].value_offset;
        int ok = 0;

        switch (rows[row].kind) {
        case YVEX_MODEL_TARGET_ROW_LITERAL:
            ok = yvex_model_target_report_add_row(report, "%s", rows[row].format);
            break;
        case YVEX_MODEL_TARGET_ROW_STRING:
            ok = yvex_model_target_report_add_row(
                report, rows[row].format, *(const char *const *)value);
            break;
        case YVEX_MODEL_TARGET_ROW_ULONG:
            ok = yvex_model_target_report_add_row(
                report, rows[row].format, *(const unsigned long *)value);
            break;
        case YVEX_MODEL_TARGET_ROW_U64:
            ok = yvex_model_target_report_add_row(
                report, rows[row].format, *(const unsigned long long *)value);
            break;
        case YVEX_MODEL_TARGET_ROW_INT:
            ok = yvex_model_target_report_add_row(report, rows[row].format,
                                                  *(const int *)value);
            break;
        default:
            return;
        }
        if (!ok) return;
    }
}

int yvex_model_target_validate_supported(
    const yvex_model_target_request *request,
    yvex_model_target_report *report,
    const char *operation,
    int contract_refusal_row)
{
    if (!request || !report || !operation) return 0;
    if (!request->target_id[0]) {
        report->exit_code = 2;
        yvex_model_target_report_add_error(
            report, "model-target %s: requires TARGET", operation);
        return 0;
    }
    if (yvex_model_target_supported_source_target(request->target_id)) return 1;
    report->exit_code = 2;
    if (contract_refusal_row && request->output_contract[0]) {
        yvex_model_target_report_add_row(report, "status: unsupported-target");
    } else if (strcmp(request->target_id,
                      YVEX_SOURCE_RETIRED_TARGET_ID) == 0) {
        yvex_model_target_report_add_error(
            report, "model-target %s: unsupported target: %s; use %s",
            operation, YVEX_SOURCE_RETIRED_TARGET_ID,
            YVEX_SOURCE_RELEASE_TARGET_ID);
    } else {
        yvex_model_target_report_add_error(
            report, "model-target %s: unsupported target: %s",
            operation, request->target_id);
    }
    return 0;
}

static unsigned long long probe_le64(const unsigned char bytes[8])
{
    unsigned long long value = 0ull;
    unsigned int byte;

    for (byte = 0u; byte < 8u; ++byte) {
        value |= (unsigned long long)bytes[byte] << (byte * 8u);
    }
    return value;
}

int yvex_model_target_probe_source_path(
    const yvex_model_target_request *request,
    const char *family,
    const char *leaf,
    char *out,
    size_t cap)
{
    int length;

    if (!request || !family || !out || cap == 0u) return 0;
    out[0] = '\0';
    if (request->source_path[0]) {
        length = leaf && leaf[0]
                     ? snprintf(out, cap, "%s/%s", request->source_path, leaf)
                     : snprintf(out, cap, "%s", request->source_path);
    } else if (request->models_root[0]) {
        length = leaf && leaf[0]
                     ? snprintf(out, cap, "%s/hf/%s/%s/%s", request->models_root,
                                family, request->target_id, leaf)
                     : snprintf(out, cap, "%s/hf/%s/%s", request->models_root,
                                family, request->target_id);
    } else {
        return 0;
    }
    if (length < 0 || (size_t)length >= cap) {
        out[0] = '\0';
        return 0;
    }
    return 1;
}

int yvex_model_target_probe_directory(const char *path)
{
    struct stat state;

    return path && path[0] && stat(path, &state) == 0 && S_ISDIR(state.st_mode);
}

int yvex_model_target_probe_file(const char *path)
{
    FILE *file;

    if (!path || !path[0]) return 0;
    file = fopen(path, "rb");
    if (!file) return 0;
    (void)fclose(file);
    return 1;
}

int yvex_model_target_probe_read(const char *path, char *out, size_t cap)
{
    FILE *file;
    size_t read_count;

    if (!path || !out || cap == 0u) return 0;
    out[0] = '\0';
    file = fopen(path, "rb");
    if (!file) return 0;
    read_count = fread(out, 1u, cap - 1u, file);
    out[read_count] = '\0';
    (void)fclose(file);
    return 1;
}

int yvex_model_target_probe_header(const char *path, char **out)
{
    unsigned char length_bytes[8];
    unsigned long long header_length;
    FILE *file;
    char *header;

    if (!path || !out) return 0;
    *out = NULL;
    file = fopen(path, "rb");
    if (!file) return 0;
    if (fread(length_bytes, 1u, sizeof(length_bytes), file) != sizeof(length_bytes)) {
        (void)fclose(file);
        return 0;
    }
    header_length = probe_le64(length_bytes);
    if (header_length == 0ull || header_length > MODEL_TARGET_HEADER_CAP) {
        (void)fclose(file);
        return 0;
    }
    header = (char *)malloc((size_t)header_length + 1u);
    if (!header) {
        (void)fclose(file);
        return 0;
    }
    if (fread(header, 1u, (size_t)header_length, file) != (size_t)header_length) {
        free(header);
        (void)fclose(file);
        return 0;
    }
    (void)fclose(file);
    header[header_length] = '\0';
    *out = header;
    return 1;
}

static const char *const source_profile_metadata[] = {
    "tokenizer.json",
    "config.json",
    "generation_config.json",
    "special_tokens_map.json"
};

static unsigned long source_profile_count(const char *text, const char *needle)
{
    unsigned long count = 0u;
    const char *cursor = text;

    while (cursor && needle && needle[0] &&
           (cursor = strstr(cursor, needle)) != NULL) {
        ++count;
        cursor += strlen(needle);
    }
    return count;
}

/*
 * Gather the shared source/header profile consumed by target reports.
 *
 * Probes one bounded header and four metadata sidecars; payload is untouched. Absent or malformed
 * inputs remain deterministic false/zero facts.
 */
void yvex_model_target_probe_source_profile(
    const yvex_model_target_request *request,
    const char *family,
    yvex_model_target_source_profile *profile)
{
    char directory[1024];
    char path[1200];
    char *header = NULL;
    unsigned long known;
    unsigned long all;
    size_t index;

    if (!profile) return;
    memset(profile, 0, sizeof(*profile));
    if (!request || !family) return;
    (void)yvex_model_target_probe_source_path(
        request, family, NULL, directory, sizeof(directory));
    profile->source_requested = directory[0] != '\0';
    if (!profile->source_requested) return;
    profile->source_directory_present =
        yvex_model_target_probe_directory(directory);
    (void)snprintf(path, sizeof(path), "%s/model.safetensors", directory);
    profile->header_present = yvex_model_target_probe_header(path, &header);
    if (header) {
        profile->f32_count = source_profile_count(header, "\"dtype\":\"F32\"");
        profile->f16_count = source_profile_count(header, "\"dtype\":\"F16\"");
        profile->bf16_count = source_profile_count(header, "\"dtype\":\"BF16\"");
        known = profile->f32_count + profile->f16_count + profile->bf16_count;
        all = source_profile_count(header, "\"dtype\":");
        profile->tensor_count = known;
        profile->other_count = all >= known ? all - known : 0u;
        profile->attention_k_present = strstr(header, "k_proj.weight") != NULL;
        profile->output_head_present = strstr(header, "lm_head.weight") != NULL ||
                                       strstr(header, "output.weight") != NULL;
        profile->output_head_ambiguous = strstr(header, "lm_head.weight") != NULL &&
                                         strstr(header, "output.weight") != NULL;
        free(header);
    }
    profile->metadata_present = 1;
    for (index = 0u; index < sizeof(source_profile_metadata) /
                                sizeof(source_profile_metadata[0]); ++index) {
        (void)snprintf(path, sizeof(path), "%s/%s", directory,
                       source_profile_metadata[index]);
        if (!yvex_model_target_probe_file(path)) {
            profile->metadata_present = 0;
            break;
        }
    }
}

static int source_scan_has(const char *name, const char *needle)
{
    return name && needle && strstr(name, needle) != NULL;
}

static void source_scan_count(yvex_model_target_source_scan *scan,
                              const char *name)
{
    int attention = source_scan_has(name, "self_attn") ||
                    source_scan_has(name, "attention");
    int mlp = source_scan_has(name, ".mlp.") ||
              source_scan_has(name, "feed_forward");

    if (source_scan_has(name, "embed_tokens")) scan->embed = 1;
    if (attention &&
        (source_scan_has(name, "q_proj") || source_scan_has(name, "k_proj") ||
         source_scan_has(name, "v_proj") || source_scan_has(name, "o_proj")))
        scan->attn++;
    if (mlp &&
        (source_scan_has(name, "gate_proj") || source_scan_has(name, "up_proj") ||
         source_scan_has(name, "down_proj") ||
         source_scan_has(name, "mlp.gate.weight") ||
         source_scan_has(name, "experts.gate_up_proj") ||
         source_scan_has(name, "shared_expert.down_proj")))
        scan->mlp++;
    if (source_scan_has(name, "norm") || source_scan_has(name, "layernorm"))
        scan->norm++;
    if (source_scan_has(name, "lm_head") || source_scan_has(name, "output_head"))
        scan->head++;
    if (source_scan_has(name, "router") || source_scan_has(name, "expert"))
        scan->moe++;
}

/*
 * Scan one admitted native-weight inventory into shared lexical report facts.
 *
 * Missing or unreadable sources leave deterministic zero-valued counters.
 */
void yvex_model_target_scan_source(
    const yvex_model_target_request *request,
    const char *family,
    yvex_model_target_source_scan *scan)
{
    yvex_native_weight_options options;
    yvex_native_weight_table *table = NULL;
    yvex_error err;
    unsigned long long index;

    if (!scan) return;
    memset(scan, 0, sizeof(*scan));
    (void)yvex_model_target_probe_source_path(
        request, family, NULL, scan->source_path, sizeof(scan->source_path));
    scan->source_present = yvex_model_target_probe_directory(scan->source_path);
    if (!scan->source_present) return;
    memset(&options, 0, sizeof(options));
    options.source_dir = scan->source_path;
    options.recursive = 1;
    yvex_error_clear(&err);
    if (yvex_native_weight_table_open(&table, &options, &err) != YVEX_OK) return;
    scan->tensors = yvex_native_weight_table_count(table);
    for (index = 0; index < scan->tensors; ++index) {
        const yvex_native_weight_info *info =
            yvex_native_weight_table_at(table, index);
        if (info) source_scan_count(scan, info->name);
    }
    yvex_native_weight_table_close(table);
    scan->layers = scan->attn >= 4 || scan->mlp >= 3 ? 1ull : 0ull;
}

int yvex_model_target_report_add_error(yvex_model_target_report *report,
                                       const char *fmt,
                                       ...)
{
    va_list ap;
    int ok;

    if (!report || !fmt) {
        return 0;
    }
    va_start(ap, fmt);
    ok = model_target_report_store(report->error_rows,
                                   sizeof(report->error_rows) /
                                       sizeof(report->error_rows[0]),
                                   &report->error_row_count,
                                   fmt,
                                   ap);
    va_end(ap);
    return ok;
}

int yvex_model_target_report_add_table_row(yvex_model_target_report *report,
                                           unsigned int column_count,
                                           const char *c0,
                                           const char *c1,
                                           const char *c2,
                                           const char *c3,
                                           const char *c4,
                                           const char *c5,
                                           const char *c6,
                                           const char *c7)
{
    const char *cols[YVEX_MODEL_TARGET_TABLE_COL_CAP] = {
        c0, c1, c2, c3, c4, c5, c6, c7
    };
    yvex_model_target_table_row *row;
    unsigned int i;

    if (!report || column_count > YVEX_MODEL_TARGET_TABLE_COL_CAP ||
        report->table_row_count >= YVEX_MODEL_TARGET_TABLE_ROW_CAP) {
        return 0;
    }
    row = &report->table_rows[report->table_row_count++];
    memset(row, 0, sizeof(*row));
    row->column_count = column_count;
    for (i = 0; i < column_count; ++i) {
        snprintf(row->columns[i], sizeof(row->columns[i]), "%s",
                 cols[i] ? cols[i] : "");
    }
    return 1;
}

void yvex_model_target_report_add_output_contract(yvex_model_target_report *report,
                                                  const char *report_name,
                                                  const char *mode)
{
    yvex_model_target_report_add_row(report, "status: pass");
    yvex_model_target_report_add_row(report, "report: %s",
                                     report_name ? report_name : "unknown");
    yvex_model_target_report_add_row(report, "mode: %s",
                                     mode ? mode : "unknown");
    yvex_model_target_report_add_rows(
        report, output_contract_tail,
        sizeof(output_contract_tail) / sizeof(output_contract_tail[0]));
}

int yvex_model_target_report_build(const yvex_model_target_request *request,
                                   yvex_model_target_report *report,
                                   yvex_error *err)
{
    if (!request || !report) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "model_target_report",
                       "request and report are required");
        return YVEX_ERR_INVALID_ARG;
    }
    memset(report, 0, sizeof(*report));
    report->kind = request->kind;
    report->mode = request->mode;
    report->help_requested = request->help_requested;
    report->exit_code = 0;

    switch (request->kind) {
    case YVEX_MODEL_TARGET_COMMAND_HELP:
        return yvex_model_target_help_report_build(report, err);
    case YVEX_MODEL_TARGET_COMMAND_CLASSES:
    case YVEX_MODEL_TARGET_COMMAND_LIST:
    case YVEX_MODEL_TARGET_COMMAND_INSPECT:
        return yvex_model_target_catalog_report_build(request, report, err);
    case YVEX_MODEL_TARGET_COMMAND_DECISION:
        return yvex_model_target_decision_report_build(request, report, err);
    case YVEX_MODEL_TARGET_COMMAND_CANDIDATE:
    case YVEX_MODEL_TARGET_COMMAND_DENSE_CANDIDATE:
    case YVEX_MODEL_TARGET_COMMAND_QWEN_METAL:
        return yvex_model_target_candidate_report_build(request, report, err);
    case YVEX_MODEL_TARGET_COMMAND_CLASS_PROFILE:
        return yvex_model_class_profile_report_build(request, report, err);
    case YVEX_MODEL_TARGET_COMMAND_TENSOR_COLLECTION:
        return yvex_tensor_collection_report_build(request, report, err);
    case YVEX_MODEL_TARGET_COMMAND_TENSOR_MAP:
        if (request->gate[0]) {
            return yvex_mapping_gate_report_build(request, report, err);
        }
        if (strcmp(request->role, "output-head") == 0) {
            return yvex_output_head_map_report_build(request, report, err);
        }
        if (strcmp(request->role, "tokenizer") == 0) {
            return yvex_tokenizer_map_report_build(request, report, err);
        }
        if (strcmp(request->role, "missing-roles") == 0) {
            return yvex_missing_role_report_build(request, report, err);
        }
        return yvex_tensor_naming_report_build(request, report, err);
    case YVEX_MODEL_TARGET_COMMAND_TOKENIZER_MAP:
        return yvex_tokenizer_map_report_build(request, report, err);
    case YVEX_MODEL_TARGET_COMMAND_MISSING_ROLES:
        return yvex_missing_role_report_build(request, report, err);
    case YVEX_MODEL_TARGET_COMMAND_QUANT_POLICY:
        if (request->gate[0] || request->include_requirements) {
            return yvex_qtype_role_support_report_build(request, report, err);
        }
        return yvex_qtype_policy_report_build(request, report, err);
    case YVEX_MODEL_TARGET_COMMAND_UNKNOWN:
    default:
        return yvex_model_target_catalog_report_build(request, report, err);
    }
}

int yvex_model_target_help_report_build(yvex_model_target_report *report,
                                        yvex_error *err)
{
    return yvex_model_target_catalog_help_report_build(report, err);
}

void yvex_model_target_report_close(yvex_model_target_report *report)
{
    if (!report) {
        return;
    }
    yvex_model_register_deepseek_v4()->lowering.close(
        (yvex_deepseek_gguf_map *)report->family_lowering);
    yvex_model_register_deepseek_v4()->coverage.close(
        (yvex_deepseek_tensor_coverage *)report->family_coverage);
    yvex_model_register_deepseek_v4()->ir.close(
        (yvex_deepseek_v4_ir *)report->family_architecture);
    memset(report, 0, sizeof(*report));
}
