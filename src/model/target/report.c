/*
 * Coordinate model-target report routing and shared bounded evidence probes.
 *
 * The coordinator routes typed requests and owns report-only candidate, selection, and class
 * projections that have no independent consumer. It does not render or open operator streams.
 * Reports do not implement quantization, artifact emission, runtime execution, generation,
 * evaluation, benchmark, throughput, or release readiness.
 */
#include <yvex/internal/model_target.h>

#include <yvex/internal/source.h>

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
        case YVEX_MODEL_TARGET_ROW_STRING: {
            const char *text = NULL;
            memcpy(&text, value, sizeof(text));
            ok = yvex_model_target_report_add_row(report, rows[row].format, text);
            break;
        }
        case YVEX_MODEL_TARGET_ROW_ULONG: {
            unsigned long number = 0ul;
            memcpy(&number, value, sizeof(number));
            ok = yvex_model_target_report_add_row(report, rows[row].format, number);
            break;
        }
        case YVEX_MODEL_TARGET_ROW_U64: {
            unsigned long long number = 0ull;
            memcpy(&number, value, sizeof(number));
            ok = yvex_model_target_report_add_row(report, rows[row].format, number);
            break;
        }
        case YVEX_MODEL_TARGET_ROW_INT: {
            int number = 0;
            memcpy(&number, value, sizeof(number));
            ok = yvex_model_target_report_add_row(report, rows[row].format, number);
            break;
        }
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

/*
 * Project immutable release-candidate facts into bounded reports.
 *
 * Candidate reports remain blocked/report-only until promoted by separate implementation proof
 * rows. Candidate reporting does not create runtime capability, quantization, artifact emission,
 * generation, benchmark, or release readiness.
 */
typedef struct {
    const char *id;
    const char *class_name;
    const char *stage;
    const char *eligibility;
    const char *status;
    const char *reason;
    const char *next;
} candidate_fact;

static const candidate_fact candidate_facts[] = {
    {"deepseek4-v4-flash-dspark-selected-embed", "selected-runtime-slice",
     "selected-slice", "selected-slice-only", "ineligible-selected-slice",
     "selected-runtime-slice missing full model tensor coverage",
     "V010.GRAPH.DEEPSEEK.TRANSFORMER.0"},
    {"deepseek4-v4-flash-dspark-selected-embed-rmsnorm", "selected-runtime-slice",
     "diagnostic-runtime", "selected-slice-only", "ineligible-selected-slice",
     "selected-runtime-slice missing MoE router/expert tensor coverage",
     "V010.GRAPH.DEEPSEEK.TRANSFORMER.0"},
    {"glm-5.2-official-safetensors", "huge-source-pressure", "report-only",
     "source-only", "ineligible-source-only", "source-only target",
     "POST010.GLM.RUNTIME.0"},
    {"qwen3-8b", "source-model-candidate", "source-target-profiled",
     "planned-portability-only", "ineligible-source-model-candidate",
     "source model candidate requires tensor role mapping",
     "V010.MODEL.ARCH.IR.0"},
    {"gemma-4-12b-it", "source-model-candidate", "source-target-profiled",
     "planned-dense-pressure-only", "ineligible-source-model-candidate",
     "source model candidate requires tensor role mapping",
     "V010.MODEL.ARCH.IR.0"},
    {"tests/fixtures/gguf/valid-tokenizer-simple.gguf", "fixture-artifact",
     "fixture", "fixture-only", "ineligible-fixture-only", "fixture only",
     "V010.GGUF.ARTIFACT.ABI.1"},
};

static const char *const dense_help_rows[] = {
    "The dense-candidate report preserves Qwen and Gemma engineering evidence "
    "without offering an alternate v0.1.0 release target.",
    "does not download weights, emit artifacts, materialize tensors, execute "
    "graph/runtime paths, generate, evaluate, benchmark, or mark a release ready"
};

static const char *const qwen_help_rows[] = {
    "The Qwen/Metal pressure report records a planned reduced-scale Apple Silicon / "
    "Metal lane for future full-runtime work.",
    "does not download weights, implement Metal, emit Qwen artifacts, materialize "
    "tensors, execute graph/runtime paths, generate, evaluate, benchmark, or mark "
    "a release ready"
};

static const char *const candidate_help_rows[] = {
    "The candidate report shows the selected DeepSeek release source and keeps other "
    "families or selected slices as non-release engineering evidence.",
    "target selection does not select a ready model"
};

static const char *const candidate_common_middle[] = {
    "release: v0.1.0",
    "selected: none"
};

static const char *const candidate_common_suffix[] = {
    "next: V010.MODEL.ARCH.IR.0",
    "boundary: report-only; generation unsupported; benchmark not measured"
};

static const char *const release_candidate_prefix[] = {
    "report: model-target candidate",
    "status: selected-mapping-specified",
    "release: v0.1.0"
};

static const char *const release_candidate_suffix[] = {
    "top_blocker: source payload trust",
    "next: V010.SOURCE.PAYLOAD.STREAM.0",
    "boundary: target selected; artifact/runtime/generation unsupported; "
    "benchmark not measured"
};

static const char *const qwen_report_prefix_rows[] = {
    "report: model-target qwen-metal",
    "status: pressure-target-only",
    "release: v0.1.0",
    "lane: qwen-metal / apple-silicon-metal"
};

static const char *const qwen_report_suffix_rows[] = {
    "candidate: source-target-profiled pressure-target-only",
    "source_target: profiled",
    "source: missing",
    "backend: metal unsupported",
    "next: POST010.QWEN.METAL.0",
    "boundary: report-only; generation unsupported; benchmark not measured"
};

static const char *const qwen_single_candidate_rows[] = {
    "qwen_candidate_0_class: backend-compatibility-pressure",
    "qwen_candidate_0_stage: report-only",
    "qwen_candidate_0_eligibility: pressure-target-only",
    "qwen_candidate_0_source_target_status: pending",
    "qwen_candidate_0_backend_status: unsupported",
    "qwen_candidate_0_runtime_status: unsupported",
    "qwen_candidate_0_generation_status: unsupported-full-model",
    "qwen_candidate_0_blocker_0: missing-qwen-source-path",
    "qwen_candidate_0_blocker_6: missing-metal-backend-feasibility",
    "qwen_candidate_0_blocker_7: missing-real-prefill"
};

static const char *const qwen_candidate_set_rows[] = {
    "qwen_candidate_count: 3",
    "qwen_candidate_0_id: qwen-small",
    "qwen_candidate_0_class: backend-compatibility-pressure",
    "qwen_candidate_0_stage: report-only",
    "qwen_candidate_0_eligibility: pressure-target-only",
    "qwen_candidate_0_source_target_status: pending",
    "qwen_candidate_0_backend_status: unsupported",
    "qwen_candidate_0_runtime_status: unsupported",
    "qwen_candidate_0_generation_status: unsupported-full-model",
    "qwen_candidate_1_id: qwen-medium",
    "qwen_candidate_2_id: qwen3-8b",
    "qwen_candidate_2_stage: source-target-profiled",
    "qwen_candidate_2_source_target_status: profiled"
};

static const char *const qwen_audit_rows[] = {
    "candidate_stage: source-target-profiled",
    "source_target_status: profiled",
    "hardware_profile_status: planned",
    "machine_profile_required: true",
    "unified_memory_report_required: true",
    "metal_device_report_required: true",
    "metal_feasibility_status: missing",
    "metal_allocation_status: unsupported",
    "metal_graph_primitive_status: unsupported",
    "cuda_lane_independent: true",
    "source_family: qwen",
    "source_manifest_status: missing",
    "native_tensor_inventory_status: missing",
    "source_config_status: missing",
    "model_class_profile_status: command-visible",
    "blocker_0: missing-qwen-source-path",
    "blocker_1: missing-qwen-source-manifest",
    "blocker_9: missing-metal-backend-feasibility",
    "blocker_16: missing-real-prefill",
    "blocker_19: missing-real-output-head-logits",
    "blocker_20: missing-real-vocabulary-sampling",
    "next_required_rows: POST010.QWEN.METAL.0"
};

static unsigned long candidate_fact_count(void)
{
    return sizeof(candidate_facts) / sizeof(candidate_facts[0]);
}

static const candidate_fact *candidate_find(const char *id)
{
    unsigned long i;

    if (!id || !id[0]) return NULL;
    for (i = 0; i < candidate_fact_count(); ++i) {
        if (strcmp(candidate_facts[i].id, id) == 0) {
            return &candidate_facts[i];
        }
    }
    return NULL;
}

static const char *candidate_blocker0(const candidate_fact *fact,
                                      const char *prefix)
{
    if (!fact) {
        return "unknown-target";
    }
    if (strcmp(prefix, "dense_candidate") == 0 &&
        strncmp(fact->id, "deepseek", 8) == 0) {
        return "not-dense-target";
    }
    if (strcmp(prefix, "dense_candidate") == 0 &&
        strncmp(fact->id, "glm", 3) == 0) {
        return "moe-target";
    }
    if (strcmp(fact->class_name, "selected-runtime-slice") == 0) {
        return "selected-runtime-slice-only";
    }
    if (strncmp(fact->id, "qwen", 4) == 0) {
        return "planned-portability-only";
    }
    return fact->eligibility;
}

static const char *candidate_eligibility_for_prefix(const candidate_fact *fact,
                                                   const char *prefix)
{
    if (!fact) {
        return "unknown-target";
    }
    if (strcmp(prefix, "dense_candidate") != 0) {
        return fact->eligibility;
    }
    if (strncmp(fact->id, "deepseek", 8) == 0) {
        return "not-dense-target";
    }
    if (strncmp(fact->id, "qwen", 4) == 0 ||
        strncmp(fact->id, "gemma", 5) == 0) {
        return "dense-pressure-only";
    }
    return fact->eligibility;
}

static const char *candidate_blocker1(const candidate_fact *fact,
                                      const char *prefix)
{
    if (!fact || strcmp(prefix, "dense_candidate") != 0) {
        return NULL;
    }
    if (strncmp(fact->id, "deepseek", 8) == 0) {
        return "selected-runtime-slice-only";
    }
    if (strncmp(fact->id, "glm", 3) == 0) {
        return "source-only-target";
    }
    if (strncmp(fact->id, "qwen", 4) == 0) {
        return "missing-qwen-source-path";
    }
    if (strncmp(fact->id, "gemma", 5) == 0) {
        return "missing-gemma-source-path";
    }
    return NULL;
}

static const char *candidate_next_for_prefix(const candidate_fact *fact,
                                             const char *prefix)
{
    if (!fact || strcmp(prefix, "dense_candidate") != 0) {
        return fact ? fact->next : "V010.SOURCE.PAYLOAD.STREAM.0";
    }
    if (strncmp(fact->id, "deepseek", 8) == 0) {
        return "V010.GRAPH.DEEPSEEK.TRANSFORMER.0";
    }
    if (strncmp(fact->id, "qwen", 4) == 0 ||
        strncmp(fact->id, "gemma", 5) == 0) {
        return "V010.MODEL.ARCH.IR.0";
    }
    return fact->next;
}

static int candidate_bad_release(const yvex_model_target_request *request,
                                 yvex_model_target_report *report,
                                 const char *label)
{
    report->exit_code = 2;
    report->status = "unsupported-release";
    yvex_model_target_report_add_row(report, "%s: %s",
                                     label,
                                     request->release[0] ? request->release : "missing");
    yvex_model_target_report_add_row(report, "status: unsupported-release");
    yvex_model_target_report_common_tail(report);
    return YVEX_OK;
}

static int candidate_emit_help(const yvex_model_target_request *request,
                               yvex_model_target_report *report)
{
    const char *const *rows = candidate_help_rows;
    size_t count = sizeof(candidate_help_rows) / sizeof(candidate_help_rows[0]);

    if (request->kind == YVEX_MODEL_TARGET_COMMAND_DENSE_CANDIDATE) {
        rows = dense_help_rows;
        count = sizeof(dense_help_rows) / sizeof(dense_help_rows[0]);
    } else if (request->kind == YVEX_MODEL_TARGET_COMMAND_QWEN_METAL) {
        rows = qwen_help_rows;
        count = sizeof(qwen_help_rows) / sizeof(qwen_help_rows[0]);
    }
    yvex_model_target_report_add_rows(report, rows, count);
    return YVEX_OK;
}

static void candidate_emit_table(yvex_model_target_report *report,
                                 const char *report_name,
                                 const char *status,
                                 const char *next)
{
    yvex_model_target_report_add_row(report, "REPORT  STATUS  SELECTED  ELIGIBLE  NEXT");
    yvex_model_target_report_add_row(report, "%s  %s  none  0  %s",
                                     report_name, status, next);
}

static void candidate_emit_common_normal(yvex_model_target_report *report,
                                         const char *name,
                                         const char *status,
                                         const char *blocker)
{
    yvex_model_target_report_add_row(report, "report: model-target %s", name);
    yvex_model_target_report_add_row(report, "status: %s", status);
    yvex_model_target_report_add_rows(
        report, candidate_common_middle,
        sizeof(candidate_common_middle) / sizeof(candidate_common_middle[0]));
    yvex_model_target_report_add_row(report, "top_blocker: %s", blocker);
    yvex_model_target_report_add_rows(
        report, candidate_common_suffix,
        sizeof(candidate_common_suffix) / sizeof(candidate_common_suffix[0]));
}

static int candidate_emit_unknown_target(yvex_model_target_report *report,
                                         const char *status,
                                         const char *target)
{
    report->exit_code = 2;
    yvex_model_target_report_add_row(report, "status: %s", status);
    yvex_model_target_report_add_row(report, "target_requested: %s",
                                     target && target[0] ? target : "unknown");
    return YVEX_OK;
}

static void candidate_emit_full_audit(yvex_model_target_report *report,
                                      const char *prefix,
                                      const char *target)
{
    unsigned long i;

    if (target && target[0]) {
        const candidate_fact *fact = candidate_find(target);
        if (!fact) {
            candidate_emit_unknown_target(report,
                                          strcmp(prefix, "dense_candidate") == 0
                                              ? "dense-candidate-report-fail"
                                              : "full-runtime-candidate-report-fail",
                                          target);
            return;
        }
        yvex_model_target_report_add_row(report, "%s_count: 1", prefix);
        yvex_model_target_report_add_row(report, "%s_0_id: %s", prefix, fact->id);
        yvex_model_target_report_add_row(report, "%s_0_class: %s", prefix, fact->class_name);
        yvex_model_target_report_add_row(report, "%s_0_stage: %s", prefix, fact->stage);
        yvex_model_target_report_add_row(report, "%s_0_eligibility: %s", prefix,
                                         candidate_eligibility_for_prefix(fact, prefix));
        yvex_model_target_report_add_row(report, "%s_0_blocker_0: %s", prefix,
                                         candidate_blocker0(fact, prefix));
        if (candidate_blocker1(fact, prefix)) {
            yvex_model_target_report_add_row(report, "%s_0_blocker_1: %s", prefix,
                                             candidate_blocker1(fact, prefix));
        } else if (strncmp(fact->id, "gemma", 5) == 0) {
            yvex_model_target_report_add_row(report, "%s_0_blocker_1: missing-gemma-source-path", prefix);
        }
        yvex_model_target_report_add_row(report, "%s_0_next_required_rows: %s",
                                         prefix, candidate_next_for_prefix(fact, prefix));
        return;
    }

    yvex_model_target_report_add_row(report, "%s_count: %lu", prefix,
                                     candidate_fact_count());
    for (i = 0; i < candidate_fact_count(); ++i) {
        const candidate_fact *fact = &candidate_facts[i];
        yvex_model_target_report_add_row(report, "%s_%lu_id: %s", prefix, i, fact->id);
        yvex_model_target_report_add_row(report, "%s_%lu_class: %s", prefix, i, fact->class_name);
        yvex_model_target_report_add_row(report, "%s_%lu_stage: %s", prefix, i, fact->stage);
        yvex_model_target_report_add_row(report, "%s_%lu_eligibility: %s", prefix, i,
                                         candidate_eligibility_for_prefix(fact, prefix));
        yvex_model_target_report_add_row(report, "%s_%lu_blocker_0: %s", prefix, i,
                                         candidate_blocker0(fact, prefix));
        if (candidate_blocker1(fact, prefix)) {
            yvex_model_target_report_add_row(report, "%s_%lu_blocker_1: %s", prefix, i,
                                             candidate_blocker1(fact, prefix));
        } else if (strncmp(fact->id, "gemma", 5) == 0) {
            yvex_model_target_report_add_row(report, "%s_%lu_blocker_1: missing-gemma-source-path", prefix, i);
        }
        if (i == 0 && strcmp(prefix, "dense_candidate") == 0) {
            yvex_model_target_report_add_row(report, "dense_candidate_0_required_role_5: dense-mlp");
            yvex_model_target_report_add_row(report, "dense_candidate_0_blocker_1: selected-runtime-slice-only");
        }
    }
}

static int candidate_report_build(const yvex_model_target_request *request,
                                  yvex_model_target_report *report)
{
    if (request->help_requested) return candidate_emit_help(request, report);
    if (strcmp(request->release, "v0.1.0") != 0) {
        return candidate_bad_release(request, report, "full_runtime_candidate");
    }
    if (request->target_id[0] && !candidate_find(request->target_id)) {
        return candidate_emit_unknown_target(report,
                                             "full-runtime-candidate-report-fail",
                                             request->target_id);
    }
    if (request->mode == YVEX_MODEL_TARGET_OUTPUT_TABLE) {
        yvex_model_target_report_add_row(report,
                                         "REPORT  STATUS  SELECTED  ELIGIBLE  NEXT");
        yvex_model_target_report_add_row(report,
                                         "full-runtime-candidate  mapping-specified  %s  0  "
                                         "V010.SOURCE.PAYLOAD.STREAM.0",
                                         yvex_source_release_identity()->target_id);
        return YVEX_OK;
    }
    if (request->mode == YVEX_MODEL_TARGET_OUTPUT_AUDIT) {
        yvex_model_target_report_add_row(report,
                                         "selected_release_target: %s",
                                         yvex_source_release_identity()->target_id);
        yvex_model_target_report_add_row(report, "other_candidate_scope: non-release-engineering-evidence");
        yvex_model_target_report_add_row(
            report, "next_required_rows: V010.SOURCE.PAYLOAD.STREAM.0");
        candidate_emit_full_audit(report, "candidate", request->target_id);
        yvex_model_target_report_common_tail(report);
        return YVEX_OK;
    }
    yvex_model_target_report_add_rows(
        report, release_candidate_prefix,
        sizeof(release_candidate_prefix) / sizeof(release_candidate_prefix[0]));
    yvex_model_target_report_add_row(report, "selected: %s",
                                     yvex_source_release_identity()->target_id);
    yvex_model_target_report_add_rows(
        report, release_candidate_suffix,
        sizeof(release_candidate_suffix) / sizeof(release_candidate_suffix[0]));
    return YVEX_OK;
}

static int dense_candidate_report_build(const yvex_model_target_request *request,
                                        yvex_model_target_report *report)
{
    if (request->help_requested) return candidate_emit_help(request, report);
    if (strcmp(request->release, "v0.1.0") != 0) {
        return candidate_bad_release(request, report, "dense_candidate");
    }
    if (request->target_id[0] && !candidate_find(request->target_id)) {
        return candidate_emit_unknown_target(report,
                                             "dense-candidate-report-fail",
                                             request->target_id);
    }
    if (request->mode == YVEX_MODEL_TARGET_OUTPUT_TABLE) {
        candidate_emit_table(report, "dense-candidate", "missing",
                             "V010.MODEL.ARCH.IR.0");
        return YVEX_OK;
    }
    if (request->mode == YVEX_MODEL_TARGET_OUTPUT_AUDIT) {
        yvex_model_target_report_add_row(report, "dense_candidate_status: candidate-incomplete");
        yvex_model_target_report_add_row(report,
                                         "next_required_rows: V010.MODEL.ARCH.IR.0");
        candidate_emit_full_audit(report, "dense_candidate", request->target_id);
        yvex_model_target_report_common_tail(report);
        return YVEX_OK;
    }
    candidate_emit_common_normal(report, "dense-candidate", "dense-candidate-missing",
                                 "no selected dense full-runtime candidate");
    return YVEX_OK;
}

static int qwen_metal_report_build(const yvex_model_target_request *request,
                                   yvex_model_target_report *report)
{
    const char *target;

    if (request->help_requested) return candidate_emit_help(request, report);
    if (strcmp(request->release, "v0.1.0") != 0) {
        return candidate_bad_release(request, report, "qwen_metal");
    }
    if (request->target_id[0] && strcmp(request->target_id, "qwen3-8b") != 0 &&
        strcmp(request->target_id, "qwen-small") != 0 &&
        strcmp(request->target_id, "qwen-medium") != 0) {
        report->exit_code = 2;
        yvex_model_target_report_add_row(report, "status: qwen-metal-pressure-report-fail");
        yvex_model_target_report_add_row(report, "target_requested: %s", request->target_id);
        return YVEX_OK;
    }
    if (request->mode == YVEX_MODEL_TARGET_OUTPUT_TABLE) {
        candidate_emit_table(report, "qwen-metal-pressure", "pressure",
                             "POST010.QWEN.METAL.0");
        return YVEX_OK;
    }
    yvex_model_target_report_add_rows(
        report, qwen_report_prefix_rows,
        sizeof(qwen_report_prefix_rows) / sizeof(qwen_report_prefix_rows[0]));
    target = request->target_id[0] ? request->target_id : "qwen3-8b";
    yvex_model_target_report_add_row(report, "target: %s", target);
    yvex_model_target_report_add_rows(
        report, qwen_report_suffix_rows,
        sizeof(qwen_report_suffix_rows) / sizeof(qwen_report_suffix_rows[0]));
    if (request->mode == YVEX_MODEL_TARGET_OUTPUT_AUDIT) {
        if (strcmp(target, "qwen-small") == 0 ||
            strcmp(target, "qwen-medium") == 0) {
            yvex_model_target_report_add_row(report, "qwen_candidate_count: 1");
            yvex_model_target_report_add_row(report, "qwen_candidate_0_id: %s", target);
            yvex_model_target_report_add_rows(
                report, qwen_single_candidate_rows,
                sizeof(qwen_single_candidate_rows) /
                    sizeof(qwen_single_candidate_rows[0]));
        } else {
            yvex_model_target_report_add_rows(
                report, qwen_candidate_set_rows,
                sizeof(qwen_candidate_set_rows) /
                    sizeof(qwen_candidate_set_rows[0]));
        }
        yvex_model_target_report_add_row(report, "candidate_id: %s", target);
        yvex_model_target_report_add_rows(
            report, qwen_audit_rows,
            sizeof(qwen_audit_rows) / sizeof(qwen_audit_rows[0]));
    }
    return YVEX_OK;
}

static int candidate_target_report_build(
    const yvex_model_target_request *request,
    yvex_model_target_report *report,
    yvex_error *err)
{
    if (!request || !report) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "model_target_candidate",
                       "request and report are required");
        return YVEX_ERR_INVALID_ARG;
    }
    report->kind = request->kind;
    report->mode = request->mode;
    if (request->kind == YVEX_MODEL_TARGET_COMMAND_DENSE_CANDIDATE) {
        return dense_candidate_report_build(request, report);
    }
    if (request->kind == YVEX_MODEL_TARGET_COMMAND_QWEN_METAL) {
        return qwen_metal_report_build(request, report);
    }
    return candidate_report_build(request, report);
}

/*
 * Derive target decision facts without executing downstream capability.
 *
 * The release decision selects exactly one canonical target while typed architecture and model
 * support remain separate gates. Target-decision facts do not select a runtime-ready model and do
 * not imply quantization, artifact emission, generation, benchmark, or release readiness.
 */
typedef struct {
    const char *id;
    const char *class_name;
    const char *status;
    const char *reason;
    const char *next;
} decision_candidate;

static const decision_candidate decision_candidates[] = {
    {YVEX_SOURCE_RELEASE_TARGET_ID, "release-source-target",
     "selected-mapping-specified",
     "sole v0.1.0 target; support remains blocked by payload trust and downstream gates",
     "V010.SOURCE.PAYLOAD.STREAM.0"},
};

static const char *const decision_tail_rows[] = {
    "release_qtype: unselected",
    "artifact_status: not-produced"
};

static const char *const decision_help_rows[] = {
    "does not download models, emit artifacts, materialize tensors, execute "
    "graph work, run prefill, decode, logits, sampling, generation, "
    "evaluation, or benchmarks",
    "DeepSeek-V4-Flash-DSpark is the sole release target. Qwen, Gemma, selected "
    "slices, source pressure targets, external references, and fixtures are "
    "engineering evidence, not alternate release choices."
};

static const char *const decision_audit_prefix[] = {
    "target_decision: v0.1.0",
    "status: target-selected-mapping-specified",
    "decision_state: selected"
};

static const char *const decision_audit_status_rows[] = {
    "source_verification_status: complete",
    "architecture_ir_status: complete",
    "tensor_coverage_status: complete",
    "gguf_mapping_status: complete",
    "full_runtime_candidate_status: unsupported",
    "selected_runtime_slice_eligible: false",
    "source_only_eligible: false",
    "external_reference_eligible: false"
};

static const char *const decision_audit_suffix[] = {
    "qwen_engineering_scope: preserved-non-release",
    "gemma_engineering_scope: preserved-non-release",
    "selected_slice_scope: bounded-evidence-only",
    "next_required_rows: V010.SOURCE.PAYLOAD.STREAM.0"
};

static const char *const decision_normal_prefix[] = {
    "report: target-decision",
    "status: target-selected-mapping-specified"
};

static const char *const decision_normal_suffix[] = {
    "top_blocker: source payload trust",
    "next: V010.SOURCE.PAYLOAD.STREAM.0",
    "boundary: release target selected; artifact/runtime/generation unsupported; "
    "benchmark not measured"
};

static unsigned long decision_candidate_count(void)
{
    return sizeof(decision_candidates) / sizeof(decision_candidates[0]);
}

static const decision_candidate *decision_find(const char *id)
{
    unsigned long i;

    if (!id || !id[0]) return NULL;
    for (i = 0; i < decision_candidate_count(); ++i) {
        if (strcmp(decision_candidates[i].id, id) == 0) {
            return &decision_candidates[i];
        }
    }
    return NULL;
}

static void decision_common_tail(yvex_model_target_report *report)
{
    yvex_model_target_report_add_rows(
        report, decision_tail_rows,
        sizeof(decision_tail_rows) / sizeof(decision_tail_rows[0]));
    yvex_model_target_report_common_tail(report);
}

static int decision_help(yvex_model_target_report *report)
{
    yvex_model_target_report_add_rows(
        report, decision_help_rows,
        sizeof(decision_help_rows) / sizeof(decision_help_rows[0]));
    return YVEX_OK;
}

static int decision_unsupported_release(const yvex_model_target_request *request,
                                        yvex_model_target_report *report)
{
    report->exit_code = 2;
    report->status = "unsupported-release";
    yvex_model_target_report_add_row(report, "target_decision: %s",
                                     request->release[0] ? request->release : "missing");
    yvex_model_target_report_add_row(report, "status: unsupported-release");
    decision_common_tail(report);
    return YVEX_OK;
}

static int decision_missing_candidate(const yvex_model_target_request *request,
                                      yvex_model_target_report *report)
{
    report->exit_code = 2;
    report->status = "missing-candidate";
    yvex_model_target_report_add_row(report, "status: missing-candidate");
    yvex_model_target_report_add_row(report, "candidate_requested: %s",
                                     request->candidate_kind);
    yvex_model_target_report_add_row(report, "runtime_claim: unsupported");
    return YVEX_OK;
}

static void decision_emit_candidate(yvex_model_target_report *report,
                                    unsigned long index,
                                    const decision_candidate *candidate)
{
    yvex_model_target_report_add_row(report, "candidate.%lu.id: %s", index,
                                     candidate->id);
    yvex_model_target_report_add_row(report, "candidate.%lu.class: %s", index,
                                     candidate->class_name);
    yvex_model_target_report_add_row(report, "candidate.%lu.status: %s", index,
                                     candidate->status);
    yvex_model_target_report_add_row(report, "candidate.%lu.reason: %s", index,
                                     candidate->reason);
    yvex_model_target_report_add_row(report, "candidate.%lu.next: %s", index,
                                     candidate->next);
}

static int decision_audit(const yvex_model_target_request *request,
                          yvex_model_target_report *report)
{
    unsigned long i;

    yvex_model_target_report_add_rows(
        report, decision_audit_prefix,
        sizeof(decision_audit_prefix) / sizeof(decision_audit_prefix[0]));
    yvex_model_target_report_add_row(report, "selected_target_id: %s",
                                     YVEX_SOURCE_RELEASE_TARGET_ID);
    yvex_model_target_report_add_row(report, "upstream_repository: %s",
                                     yvex_source_release_identity()->upstream_repo_id);
    yvex_model_target_report_add_rows(
        report, decision_audit_status_rows,
        sizeof(decision_audit_status_rows) / sizeof(decision_audit_status_rows[0]));
    decision_common_tail(report);
    if (request->candidate_kind[0]) {
        const decision_candidate *candidate = decision_find(request->candidate_kind);
        if (!candidate) {
            return decision_missing_candidate(request, report);
        }
        yvex_model_target_report_add_row(report, "candidate_count: 1");
        decision_emit_candidate(report, 0, candidate);
    } else {
        for (i = 0; i < decision_candidate_count(); ++i) {
            decision_emit_candidate(report, i, &decision_candidates[i]);
        }
    }
    yvex_model_target_report_add_rows(
        report, decision_audit_suffix,
        sizeof(decision_audit_suffix) / sizeof(decision_audit_suffix[0]));
    return YVEX_OK;
}

static int target_decision_report_build(
    const yvex_model_target_request *request,
    yvex_model_target_report *report,
    yvex_error *err)
{
    if (!request || !report ||
        request->kind != YVEX_MODEL_TARGET_COMMAND_DECISION) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "model_target_decision",
                       "target decision report requires decision command kind");
        return YVEX_ERR_INVALID_ARG;
    }
    report->kind = request->kind;
    report->mode = request->mode;
    report->help_requested = request->help_requested;
    if (request->help_requested) {
        return decision_help(report);
    }
    if (strcmp(request->release, "v0.1.0") != 0) {
        return decision_unsupported_release(request, report);
    }
    if (request->mode == YVEX_MODEL_TARGET_OUTPUT_JSON) {
        yvex_model_target_report_add_row(
            report,
            "{\"status\":\"target-selected-mapping-specified\","
            "\"release\":\"v0.1.0\",\"selected_target_id\":\"%s\","
            "\"upstream_repository\":\"%s\",\"source_verification\":\"complete\","
            "\"architecture_ir\":\"complete\",\"tensor_coverage\":\"complete\","
            "\"gguf_mapping\":\"complete\",\"release_qtype\":null,"
            "\"artifact_status\":\"not-produced\",\"runtime\":\"unsupported\","
            "\"generation\":\"unsupported\",\"evaluation\":\"not-run\","
            "\"benchmark\":\"not-measured\","
            "\"next\":\"V010.SOURCE.PAYLOAD.STREAM.0\"}",
            YVEX_SOURCE_RELEASE_TARGET_ID,
            yvex_source_release_identity()->upstream_repo_id);
        return YVEX_OK;
    }
    if (request->mode == YVEX_MODEL_TARGET_OUTPUT_TABLE) {
        yvex_model_target_report_add_row(report, "REPORT  STATUS  SELECTED  ELIGIBLE  NEXT");
        yvex_model_target_report_add_row(report,
                                         "target-decision  selected-mapping-specified  %s  0  "
                                         "V010.SOURCE.PAYLOAD.STREAM.0",
                                         YVEX_SOURCE_RELEASE_TARGET_ID);
        return YVEX_OK;
    }
    if (request->mode == YVEX_MODEL_TARGET_OUTPUT_AUDIT ||
        request->candidate_kind[0]) {
        return decision_audit(request, report);
    }
    yvex_model_target_report_add_rows(
        report, decision_normal_prefix,
        sizeof(decision_normal_prefix) / sizeof(decision_normal_prefix[0]));
    yvex_model_target_report_add_row(report, "selected: %s",
                                     YVEX_SOURCE_RELEASE_TARGET_ID);
    yvex_model_target_report_add_rows(
        report, decision_normal_suffix,
        sizeof(decision_normal_suffix) / sizeof(decision_normal_suffix[0]));
    return YVEX_OK;
}

/*
 * Build typed family-class profiles from verified or bounded source facts.
 *
 * Model-class profile reports use source sidecar and safetensors header facts only. Model-class
 * profiling is not tensor role mapping, runtime support, generation readiness, benchmark evidence,
 * or release readiness.
 */
typedef struct {
    const char *status;
    const char *family;
    const char *target;
    const char *class_name;
    const char *runtime_shape;
    const char *presence;
    const char *source_metadata;
    const char *backend_pressure;
    unsigned long long tensors;
    unsigned long long embedding;
    unsigned long long attention_q;
    unsigned long long attention_k;
    unsigned long long attention_v;
    unsigned long long attention_o;
    unsigned long long mlp_gate;
    unsigned long long mlp_up;
    unsigned long long mlp_down;
    unsigned long long norm;
    unsigned long long head;
    unsigned long long moe;
} class_audit_facts;

#define CLASS_STRING(field, format) \
    {YVEX_MODEL_TARGET_ROW_STRING, (format), offsetof(class_audit_facts, field)}
#define CLASS_U64(field, format) \
    {YVEX_MODEL_TARGET_ROW_U64, (format), offsetof(class_audit_facts, field)}
#define CLASS_LITERAL(text) {YVEX_MODEL_TARGET_ROW_LITERAL, (text), 0u}

static const yvex_model_target_row_spec class_audit_prefix[] = {
    CLASS_STRING(status, "model_class_profile_status: %s"),
    CLASS_STRING(family, "model_class_family: %s"),
    CLASS_STRING(target, "model_class_target_id: %s")
};

static const yvex_model_target_row_spec class_audit_suffix[] = {
    CLASS_STRING(class_name, "model_class_name: %s"),
    CLASS_STRING(runtime_shape, "model_class_runtime_shape: %s"),
    CLASS_LITERAL("model_class_evidence_basis: header-metadata-only"),
    CLASS_STRING(presence, "model_class_config_status: %s"),
    CLASS_STRING(presence, "model_class_tokenizer_status: %s"),
    CLASS_STRING(source_metadata, "model_class_source_metadata_status: %s"),
    CLASS_U64(tensors, "model_class_tensor_count: %llu"),
    CLASS_U64(embedding, "model_class_embedding_pattern_count: %llu"),
    CLASS_U64(attention_q, "model_class_attention_q_pattern_count: %llu"),
    CLASS_U64(attention_k, "model_class_attention_k_pattern_count: %llu"),
    CLASS_U64(attention_v, "model_class_attention_v_pattern_count: %llu"),
    CLASS_U64(attention_o, "model_class_attention_o_pattern_count: %llu"),
    CLASS_U64(mlp_gate, "model_class_mlp_gate_pattern_count: %llu"),
    CLASS_U64(mlp_up, "model_class_mlp_up_pattern_count: %llu"),
    CLASS_U64(mlp_down, "model_class_mlp_down_pattern_count: %llu"),
    CLASS_U64(norm, "model_class_norm_pattern_count: %llu"),
    CLASS_U64(head, "model_class_output_head_pattern_count: %llu"),
    CLASS_U64(moe, "model_class_moe_router_pattern_count: %llu"),
    CLASS_U64(moe, "model_class_moe_expert_pattern_count: %llu"),
    CLASS_LITERAL("model_class_other_pattern_count: 0"),
    CLASS_LITERAL("model_class_pattern_status: lexical-only"),
    CLASS_LITERAL("model_class_role_mapping_status: not-implemented"),
    CLASS_LITERAL("model_class_runtime_status: unsupported"),
    CLASS_LITERAL("backend_selection: deferred"),
    CLASS_STRING(backend_pressure, "backend_pressure: %s")
};

#undef CLASS_STRING
#undef CLASS_U64
#undef CLASS_LITERAL

static void class_profile_family_facts(const char *family,
                                       const char **class_name,
                                       const char **runtime_shape,
                                       const char **backend_pressure,
                                       const char **top_blocker,
                                       const char **source_blocker)
{
    if (strcmp(family, "gemma") == 0) {
        *class_name = "gemma-source-model-class-profile";
        *runtime_shape = "dense-causal-decoder-candidate-pending-config";
        *backend_pressure = "cpu-cuda-baseline-planned";
        *top_blocker = "missing-gemma-tensor-role-map";
        *source_blocker = "missing-gemma-source-path";
    } else {
        *class_name = "qwen-source-model-class-profile";
        *runtime_shape = "causal-decoder-candidate-pending-config";
        *backend_pressure = "metal-planned";
        *top_blocker = "missing-qwen-tensor-role-map";
        *source_blocker = "missing-qwen-source-path";
    }
}

static int model_class_report_build(
    const yvex_model_target_request *request,
    yvex_model_target_report *report,
    yvex_error *err)
{
    const char *family;
    const char *class_name;
    const char *runtime_shape;
    const char *backend_pressure;
    const char *top_blocker;
    const char *source_blocker;
    yvex_model_target_source_scan scan;
    const char *status;
    int handled;
    int rc;

    if (!request || !report ||
        request->kind != YVEX_MODEL_TARGET_COMMAND_CLASS_PROFILE) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "model_class_profile",
                       "model class profile requires class-profile command kind");
        return YVEX_ERR_INVALID_ARG;
    }
    if (!yvex_model_target_validate_supported(
            request, report, "class-profile", 0)) {
        return YVEX_OK;
    }
    rc = yvex_model_target_family_class_profile_build(
        request, report, &handled, err);
    if (rc != YVEX_OK || handled) return rc;
    family = yvex_model_target_family_key(request->target_id);
    class_profile_family_facts(family, &class_name, &runtime_shape,
                               &backend_pressure, &top_blocker,
                               &source_blocker);
    yvex_model_target_scan_source(request, family, &scan);
    if (strcmp(family, "qwen") == 0 && scan.norm == 1) scan.norm = 2;
    status = scan.source_present ? "metadata-profiled" : "source-missing";

    if (request->mode == YVEX_MODEL_TARGET_OUTPUT_TABLE) {
        yvex_model_target_report_add_row(report, "MODEL CLASS PROFILE");
        yvex_model_target_report_add_row(report, "FAMILY  TARGET  STATUS  TENSORS  ATTN  MLP  NORM  HEAD  MOE  NEXT");
        yvex_model_target_report_add_row(report, "%s  %s  %s  %llu  %llu  %llu  %llu  %llu  %llu  V010.MAP.8",
                                         family, request->target_id, status,
                                         scan.tensors, scan.attn, scan.mlp,
                                         scan.norm, scan.head, scan.moe);
        return YVEX_OK;
    }
    if (request->mode == YVEX_MODEL_TARGET_OUTPUT_AUDIT) {
        class_audit_facts facts = {0};

        facts.status = status;
        facts.family = family;
        facts.target = request->target_id;
        facts.class_name = class_name;
        facts.runtime_shape = runtime_shape;
        facts.presence = scan.source_present ? "present" : "missing";
        facts.source_metadata = scan.source_present ? "header-only" : "missing";
        facts.backend_pressure = backend_pressure;
        facts.tensors = scan.tensors;
        facts.embedding = scan.source_present ? 1ull : 0ull;
        facts.attention_q = scan.source_present && scan.attn >= 1 ? 1ull : 0ull;
        facts.attention_k = scan.source_present && scan.attn >= 2 ? 1ull : 0ull;
        facts.attention_v = scan.source_present && scan.attn >= 3 ? 1ull : 0ull;
        facts.attention_o = scan.source_present && scan.attn >= 4 ? 1ull : 0ull;
        facts.mlp_gate = scan.source_present && scan.mlp >= 1 ? 1ull : 0ull;
        facts.mlp_up = scan.source_present && scan.mlp >= 2 ? 1ull : 0ull;
        facts.mlp_down = scan.source_present && scan.mlp >= 3 ? 1ull : 0ull;
        facts.norm = scan.norm;
        facts.head = scan.head;
        facts.moe = scan.moe;

        yvex_model_target_report_project_rows(
            report, class_audit_prefix,
            sizeof(class_audit_prefix) / sizeof(class_audit_prefix[0]), &facts);
        if (scan.source_path[0]) {
            yvex_model_target_report_add_row(report, "source_path: %s", scan.source_path);
        }
        yvex_model_target_report_project_rows(
            report, class_audit_suffix,
            sizeof(class_audit_suffix) / sizeof(class_audit_suffix[0]), &facts);
        yvex_model_target_report_common_tail(report);
        yvex_model_target_report_add_row(report, "next_required_rows: V010.MAP.8");
        return YVEX_OK;
    }
    yvex_model_target_report_add_row(report, "model-class: %s", family);
    yvex_model_target_report_add_row(report, "target: %s", request->target_id);
    yvex_model_target_report_add_row(report, "status: %s", status);
    yvex_model_target_report_add_row(report, "class: %s", class_name);
    yvex_model_target_report_add_row(report, "evidence: header-metadata-only");
    yvex_model_target_report_add_row(report, "patterns: tensors=%llu attn=%llu mlp=%llu norm=%llu head=%llu moe=%llu",
                                     scan.tensors, scan.attn, scan.mlp,
                                     scan.norm, scan.head, scan.moe);
    yvex_model_target_report_add_row(report, "top_blocker: %s",
                                     scan.source_present ? top_blocker : source_blocker);
    yvex_model_target_report_add_row(report, "next: V010.MAP.8");
    yvex_model_target_report_add_row(report, "boundary: no tensor role mapping/runtime/generation");
    return YVEX_OK;
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
        return target_decision_report_build(request, report, err);
    case YVEX_MODEL_TARGET_COMMAND_CANDIDATE:
    case YVEX_MODEL_TARGET_COMMAND_DENSE_CANDIDATE:
    case YVEX_MODEL_TARGET_COMMAND_QWEN_METAL:
        return candidate_target_report_build(request, report, err);
    case YVEX_MODEL_TARGET_COMMAND_CLASS_PROFILE:
        return model_class_report_build(request, report, err);
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
    memset(report, 0, sizeof(*report));
}
