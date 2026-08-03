/*
 * Project source-owned release identity into model-target catalog reports.
 *
 * The exact v0.1.0 identity is borrowed from the source owner and remains distinct from support;
 * catalog facts do not claim runtime or generation. Target catalog entries are not capability
 * claims.
 */
#include <yvex/internal/core.h>
#include <yvex/internal/model_target.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const yvex_model_target_class_record catalog_model_target_classes[] = {
    {"release-source-target", "false", "unsupported", "unsupported",
     "exact selected v0.1.0 source target; selection and source verification do not imply artifact or runtime support"},
    {"selected-runtime-slice", "false", "partial-boundary-only", "unsupported",
     "selected real artifact slice used to prove parser, materialization, backend, "
     "graph, reference, and cleanup boundaries"},
    {"official-source-huge-model", "false", "unsupported", "unsupported",
     "official upstream source tensors used to force source manifest, native tensor "
     "inventory, model-class profiling, tensor mapping, quantization policy, and "
     "future YVEX-produced artifacts"},
    {"source-model-candidate", "false", "unsupported", "unsupported",
     "backend-neutral model/source target candidate; backend pressure and runtime "
     "compatibility are reported separately"},
    {"full-runtime-model", "false", "planned", "planned",
     "complete tensor set required for transformer prefill, decode, logits, sampling, "
     "and generation after runtime support exists"},
    {"huge-model-storage-stream", "false", "planned", "unsupported",
     "huge artifact target used to force shard inventory, storage layout, page or "
     "chunk planning, staged residency, and cleanup boundaries"},
    {"external-GGUF-reference", "false", "external-reference-only", "external-reference-only",
     "external GGUF evidence used only to compare artifact layout, qtype choices, "
     "deployment constraints, or external behavior"},
    {"external-runtime-reference", "false", "external-reference-only", "external-reference-only",
     "external runtime evidence used only to compare deployment constraints or external behavior"},
};

static const yvex_model_target_record catalog_model_targets[] = {
    {YVEX_SOURCE_RELEASE_TARGET_ID, YVEX_SOURCE_RELEASE_FAMILY_DISPLAY,
     YVEX_SOURCE_RELEASE_NAME, "release-source-target",
     "official-safetensors", "complete-YVEX-GGUF-not-produced",
     "exact-v0.1.0-release-source", "complete-model-tensor-set-required",
     "canonical-release-source", "verification-required",
     "source verification only; artifact/runtime/generation unsupported",
     "unsupported", "unsupported", "false"},
    {"deepseek4-v4-flash-dspark-selected-embed", YVEX_SOURCE_RELEASE_FAMILY_DISPLAY,
     YVEX_SOURCE_RELEASE_NAME,
     "selected-runtime-slice", "official-safetensors",
     "YVEX-produced-selected-GGUF", "selected-token-embedding-materialization",
     "token_embd.weight", "none", "none",
     "selected materialization and selected graph slice only", "unsupported",
     "unsupported", "false"},
    {"deepseek4-v4-flash-dspark-selected-embed-rmsnorm",
     YVEX_SOURCE_RELEASE_FAMILY_DISPLAY, YVEX_SOURCE_RELEASE_NAME,
     "selected-runtime-slice", "official-safetensors",
     "YVEX-produced-selected-GGUF", "selected-embedding-plus-rmsnorm-segment",
     "token_embd.weight,blk.0.attn_norm.weight", "none", "none",
     "selected segment execution only", "unsupported", "unsupported", "false"},
    {"glm-5.2-official-safetensors", "GLM", "GLM-5.2",
     "official-source-huge-model", "official-safetensors-huge",
     "future-YVEX-produced-GGUF",
     "huge-source-tensor-intake-moe-storage-stream-planning", "none",
     "hf/glm/GLM-5.2", "282 safetensors,1.5T-class",
     "source evidence only", "unsupported", "unsupported", "false"},
    {"qwen3-8b", "Qwen", "Qwen3-8B", "source-model-candidate",
     "official-source-tensors-planned", "future-YVEX-produced-GGUF",
     "backend-neutral-qwen-source-model-target", "pending-source-config",
     "hf/qwen/qwen3-8b", "pending source/config verification",
     "target profile only; no source download/runtime/generation", "unsupported",
     "unsupported", "false"},
    {"gemma-4-12b-it", "Gemma", "Gemma-4-12B-it",
     "source-model-candidate", "official-source-tensors-planned",
     "future-YVEX-produced-GGUF", "backend-neutral-gemma-source-model-target",
     "pending-source-config", "hf/gemma/gemma-4-12b-it",
     "pending source/config verification",
     "target profile only; no source download/runtime/generation", "unsupported",
     "unsupported", "false"},
};

int yvex_model_target_release_source_paths(
    const yvex_model_target_request *request,
    char *models_root,
    size_t models_root_cap,
    char *source_path,
    size_t source_path_cap)
{
    const char *environment;
    const char *root;
    int n;

    if (!request || !models_root || !models_root_cap || !source_path ||
        !source_path_cap) return 0;
    environment = getenv("YVEX_MODELS_ROOT");
    root = request->models_root[0]
               ? request->models_root
               : (environment && environment[0] ? environment : "models");
    n = snprintf(models_root, models_root_cap, "%s", root);
    if (n < 0 || (size_t)n >= models_root_cap) return 0;
    if (request->source_path[0]) {
        n = snprintf(source_path, source_path_cap, "%s", request->source_path);
        return n >= 0 && (size_t)n < source_path_cap;
    }
    return yvex_source_target_path(
        source_path, source_path_cap, models_root,
        yvex_source_release_identity());
}

const yvex_model_target_record *yvex_model_target_find(const char *target_id)
{
    unsigned long i;

    if (!target_id) return NULL;
    for (i = 0; i < sizeof(catalog_model_targets) / sizeof(catalog_model_targets[0]); ++i) {
        if (strcmp(catalog_model_targets[i].target_id, target_id) == 0) {
            return &catalog_model_targets[i];
        }
    }
    return NULL;
}

static unsigned long target_count(void)
{
    return sizeof(catalog_model_targets) / sizeof(catalog_model_targets[0]);
}

static const yvex_model_target_record *target_at(unsigned long index)
{
    return index < target_count() ? &catalog_model_targets[index] : NULL;
}

static unsigned long target_class_count(void)
{
    return sizeof(catalog_model_target_classes) / sizeof(catalog_model_target_classes[0]);
}

static const yvex_model_target_class_record *target_class_at(unsigned long index)
{
    return index < target_class_count()
               ? &catalog_model_target_classes[index]
               : NULL;
}

const char *yvex_model_target_family_key(const char *target_id)
{
    const yvex_model_target_record *record = yvex_model_target_find(target_id);

    if (record) {
        if (strcmp(record->family, "Qwen") == 0) return "qwen";
        if (strcmp(record->family, "Gemma") == 0) return "gemma";
        if (strcmp(record->family, "DeepSeek") == 0) return "deepseek";
        if (strcmp(record->family, "GLM") == 0) return "glm";
    }
    if (target_id && strncmp(target_id, "qwen", 4) == 0) return "qwen";
    if (target_id && strncmp(target_id, "gemma", 5) == 0) return "gemma";
    return "unknown";
}

int yvex_model_target_supported_source_target(const char *target_id)
{
    return target_id && !strstr(target_id, "portability") &&
           (yvex_source_is_release_target(target_id) ||
            strcmp(target_id, "qwen3-8b") == 0 ||
            strcmp(target_id, "gemma-4-12b-it") == 0 ||
            strncmp(target_id, "qwen", 4) == 0 ||
            strncmp(target_id, "gemma", 5) == 0);
}

void yvex_model_target_report_common_tail(yvex_model_target_report *report)
{
    yvex_model_target_report_add_row(report, "runtime_claim: unsupported");
    yvex_model_target_report_add_row(report, "generation: unsupported-full-model");
    yvex_model_target_report_add_row(report, "benchmark_status: not-measured");
    yvex_model_target_report_add_row(report, "release_ready: false");
}

static const char *catalog_models_root(const yvex_model_target_request *request,
                                       const char **source)
{
    const char *env;

    if (source) *source = "default";
    if (request && request->models_root[0]) {
        if (source) *source = "explicit";
        return request->models_root;
    }
    env = getenv("YVEX_MODELS_ROOT");
    if (env && env[0]) {
        if (source) *source = "environment";
        return env;
    }
    return "models";
}

static void catalog_absolute_path(char *out, size_t cap, const char *path)
{
    char cwd[512];

    if (!out || cap == 0u) return;
    out[0] = '\0';
    if (!path || !path[0]) return;
    if (path[0] == '/') {
        yvex_core_text_copy(out, cap, path);
        return;
    }
    if (!getcwd(cwd, sizeof(cwd))) {
        yvex_core_text_copy(out, cap, path);
        return;
    }
    (void)snprintf(out, cap, "%s/%s", cwd, path);
}

static const char *catalog_source_leaf(const yvex_model_target_record *record)
{
    const char *slash;

    if (!record) return "unknown";
    if (yvex_source_is_release_target(record->target_id)) {
        return yvex_source_release_identity()->source_dir_leaf;
    }
    if (record->local_path_class && strcmp(record->local_path_class, "none") != 0) {
        slash = strrchr(record->local_path_class, '/');
        return slash && slash[1] ? slash + 1 : record->local_path_class;
    }
    return record->model ? record->model : record->target_id;
}

static const char *catalog_registry_alias(const yvex_model_target_record *record)
{
    if (!record) return "none";
    return strcmp(record->target_class, "selected-runtime-slice") == 0
               ? record->target_id
               : "none";
}

static const char *catalog_exists_name(const char *path)
{
    return path && access(path, F_OK) == 0 ? "true" : "false";
}

static const char *catalog_source_status(const yvex_model_target_record *rec);
static const char *catalog_artifact_status(const yvex_model_target_record *rec);
static const char *catalog_runtime_status(const yvex_model_target_record *rec);

static void catalog_path_report(const yvex_model_target_request *request,
                                yvex_model_target_report *report,
                                const yvex_model_target_record *record)
{
    const char *root_source;
    const char *root = catalog_models_root(request, &root_source);
    const char *family = yvex_model_target_family_key(record->target_id);
    char root_abs[512];
    char source_path[1024];
    char artifact_path[1024];
    char report_dir[1024];
    char reference_dir[1024];
    char registry_dir[1024];
    int artifact_planned_only =
        strcmp(record->target_class, "official-source-huge-model") == 0;
    int artifact_unselected = yvex_source_is_release_target(record->target_id);

    catalog_absolute_path(root_abs, sizeof(root_abs), root);
    if (artifact_unselected) {
        if (!yvex_source_target_path(
                source_path, sizeof(source_path), root_abs,
                yvex_source_release_identity())) {
            yvex_core_text_copy(source_path, sizeof(source_path), "path-overflow");
        }
    } else {
        (void)snprintf(source_path, sizeof(source_path), "%s/hf/%s/%s",
                       root_abs, family, catalog_source_leaf(record));
    }
    if (artifact_unselected) {
        (void)snprintf(artifact_path, sizeof(artifact_path), "not-selected");
    } else if (artifact_planned_only) {
        (void)snprintf(artifact_path, sizeof(artifact_path), "planned");
    } else if (strcmp(family, "deepseek") == 0) {
        (void)snprintf(artifact_path, sizeof(artifact_path),
                       "%s/gguf/%s/%s-F16-noimatrix-yvex-v1.gguf",
                       root_abs, family, record->target_id);
    } else {
        (void)snprintf(artifact_path, sizeof(artifact_path), "%s/gguf/%s/%s",
                       root_abs, family, record->target_id);
    }
    (void)snprintf(report_dir, sizeof(report_dir), "%s/reports/%s", root_abs, family);
    (void)snprintf(reference_dir, sizeof(reference_dir), "%s/reference/%s",
                   root_abs, family);
    (void)snprintf(registry_dir, sizeof(registry_dir), "%s/registry", root_abs);

    if (request->mode == YVEX_MODEL_TARGET_OUTPUT_AUDIT) {
        yvex_model_target_report_add_row(report, "target_id: %s", record->target_id);
        yvex_model_target_report_add_row(report, "models_root_source: %s", root_source);
        yvex_model_target_report_add_row(report, "models_root: %s", root_abs);
        yvex_model_target_report_add_row(report, "source_path: %s", source_path);
        yvex_model_target_report_add_row(report, "source_exists: %s",
                                         catalog_exists_name(source_path));
        yvex_model_target_report_add_row(report, "artifact_path: %s", artifact_path);
        yvex_model_target_report_add_row(report, "artifact_exists: %s",
                                         artifact_planned_only || artifact_unselected
                                             ? "false"
                                             : catalog_exists_name(artifact_path));
        yvex_model_target_report_add_row(report, "report_dir: %s", report_dir);
        yvex_model_target_report_add_row(report, "report_dir_exists: %s",
                                         catalog_exists_name(report_dir));
        yvex_model_target_report_add_row(report, "reference_dir: %s", reference_dir);
        yvex_model_target_report_add_row(report, "reference_dir_exists: %s",
                                         catalog_exists_name(reference_dir));
        yvex_model_target_report_add_row(report, "registry_dir: %s", registry_dir);
        yvex_model_target_report_add_row(report, "registry_dir_exists: %s",
                                         catalog_exists_name(registry_dir));
        yvex_model_target_report_add_row(report, "registry_alias: %s",
                                         catalog_registry_alias(record));
        yvex_model_target_report_add_row(report, "source_artifact_class: %s", record->source_artifact_class);
        yvex_model_target_report_add_row(report, "source_artifact_status: %s",
                                         catalog_source_status(record));
        yvex_model_target_report_add_row(report, "source_tensor_payload_status: not-present");
        yvex_model_target_report_add_row(report, "target_artifact_class: %s", record->target_artifact_class);
        yvex_model_target_report_add_row(report, "target_artifact_status: %s",
                                         catalog_artifact_status(record));
        yvex_model_target_report_add_row(report, "yvex_produced_artifact_status: %s",
                                         catalog_artifact_status(record));
        yvex_model_target_report_add_row(report, "runtime_execution: %s",
                                         catalog_runtime_status(record));
        yvex_model_target_report_add_row(report, "generation: unsupported");
    } else {
        yvex_model_target_report_add_row(report, "target: %s", record->target_id);
        yvex_model_target_report_add_row(report, "source: %s  %s",
                                         access(source_path, F_OK) == 0
                                             ? "present-unverified"
                                             : "missing",
                                         source_path);
        yvex_model_target_report_add_row(report, "source_class: %s", record->source_artifact_class);
        yvex_model_target_report_add_row(report, "artifact: %s  %s",
                                         artifact_unselected
                                             ? "not-produced"
                                             : "planned",
                                         artifact_path);
        yvex_model_target_report_add_row(report, "artifact_class: %s", record->target_artifact_class);
        yvex_model_target_report_add_row(report, "reports: %s", report_dir);
        yvex_model_target_report_add_row(report, "registry: %s", registry_dir);
        yvex_model_target_report_add_row(report, "boundary: path report only, no runtime execution");
    }
}

static int catalog_unknown_subcommand(const yvex_model_target_request *request,
                                      yvex_model_target_report *report)
{
    (void)request;
    report->exit_code = 2;
    yvex_model_target_report_add_error(report, "model-target: unknown subcommand");
    return YVEX_OK;
}

typedef enum {
    CATALOG_PROJECTION_SOURCE,
    CATALOG_PROJECTION_NEXT,
    CATALOG_PROJECTION_BOUNDARY,
    CATALOG_PROJECTION_COUNT
} catalog_projection_kind;

typedef struct {
    const char *target_class;
    const char *values[CATALOG_PROJECTION_COUNT];
} catalog_projection_row;

static const char *const catalog_release_values[CATALOG_PROJECTION_COUNT] = {
    "verification-required",
    "V010.SOURCE.PAYLOAD.STREAM.0",
    "selected release source only; artifact, runtime, and generation unsupported",
};

static const char *const catalog_default_values[CATALOG_PROJECTION_COUNT] = {
    "missing",
    "",
    "target/source profile only; no source download/runtime/generation",
};

static const catalog_projection_row catalog_projection_rows[] = {
    {"selected-runtime-slice",
     {"unknown", "", "selected-slice only; no full-runtime generation"}},
    {"official-source-huge-model",
     {"planned", "V010.SOURCE.8",
      "source/storage pressure only; no GLM runtime/generation"}},
    {"source-model-candidate", {"missing", "V010.MAP.8",
                                "target/source profile only; no source download/runtime/generation"}},
};

static const char *catalog_projection(const yvex_model_target_record *rec,
                                      catalog_projection_kind kind)
{
    size_t index;

    if (yvex_source_is_release_target(rec->target_id)) return catalog_release_values[kind];
    for (index = 0u; index < sizeof(catalog_projection_rows) /
                                      sizeof(catalog_projection_rows[0]); ++index)
        if (strcmp(rec->target_class, catalog_projection_rows[index].target_class) == 0)
            return catalog_projection_rows[index].values[kind];
    return catalog_default_values[kind];
}

static const char *catalog_source_status(const yvex_model_target_record *rec)
{
    return catalog_projection(rec, CATALOG_PROJECTION_SOURCE);
}

static const char *catalog_artifact_status(const yvex_model_target_record *rec)
{
    if (yvex_source_is_release_target(rec->target_id)) {
        return "not-produced";
    }
    return strcmp(rec->target_class, "selected-runtime-slice") == 0
               ? "present"
               : "planned";
}

static const char *catalog_runtime_status(const yvex_model_target_record *rec)
{
    if (strcmp(rec->target_id, "deepseek4-v4-flash-dspark-selected-embed") == 0) {
        return "selected-boundary-only";
    }
    if (strcmp(rec->target_id, "deepseek4-v4-flash-dspark-selected-embed-rmsnorm") == 0) {
        return "selected-segment-boundary-only";
    }
    return "unsupported";
}

static const char *catalog_next_row(const yvex_model_target_record *rec)
{
    return catalog_projection(rec, CATALOG_PROJECTION_NEXT);
}

static const char *catalog_boundary(const yvex_model_target_record *rec)
{
    return catalog_projection(rec, CATALOG_PROJECTION_BOUNDARY);
}

static const char *catalog_runtime_shape(const yvex_model_target_record *rec)
{
    if (yvex_source_is_release_target(rec->target_id)) {
        return "typed-deepseek-v4-architecture-ir";
    }
    if (strcmp(yvex_model_target_family_key(rec->target_id), "gemma") == 0) {
        return "dense-causal-decoder-candidate-pending-config";
    }
    if (strcmp(yvex_model_target_family_key(rec->target_id), "qwen") == 0) {
        return "causal-decoder-candidate-pending-config";
    }
    return "selected-slice-only";
}

typedef struct {
    const char *target_id;
    const char *family;
    const char *model;
    const char *target_class;
    const char *source_class;
    const char *source_status;
    const char *identity_status;
    const char *profile_status;
    const char *runtime_shape;
    const char *evidence;
    const char *pattern;
    const char *role_mapping;
    const char *output_status;
    const char *output_next;
    const char *tokenizer_status;
    const char *tokenizer_next;
    const char *missing_next;
    const char *gate_status;
    const char *gate_next;
    const char *artifact_class;
    const char *artifact_status;
} catalog_audit_facts;

#define CATALOG_STRING_ROW(format_, member_) \
    {YVEX_MODEL_TARGET_ROW_STRING, format_, offsetof(catalog_audit_facts, member_)}
#define CATALOG_LITERAL_ROW(text_) \
    {YVEX_MODEL_TARGET_ROW_LITERAL, text_, 0u}

static const yvex_model_target_row_spec catalog_inspect_prefix_rows[] = {
    CATALOG_STRING_ROW("target_id: %s", target_id),
    CATALOG_STRING_ROW("family: %s", family),
    CATALOG_STRING_ROW("model: %s", model),
    CATALOG_STRING_ROW("target_class: %s", target_class)
};

static const yvex_model_target_row_spec catalog_inspect_rows[] = {
    CATALOG_STRING_ROW("source_artifact_class: %s", source_class),
    CATALOG_STRING_ROW("source_artifact_status: %s", source_status),
    CATALOG_STRING_ROW("source_provenance_status: %s", source_status),
    CATALOG_LITERAL_ROW("source_origin: planned-official"),
    CATALOG_LITERAL_ROW("source_authority: upstream-official-planned"),
    CATALOG_LITERAL_ROW("source_revision_status: unknown"),
    CATALOG_STRING_ROW("source_identity_status: %s", identity_status),
    CATALOG_LITERAL_ROW("source_hash_status: not-computed"),
    CATALOG_LITERAL_ROW("source_verification_status: not-verified"),
    CATALOG_STRING_ROW("native_inventory_status: %s", source_status),
    CATALOG_LITERAL_ROW("native_tensor_count: 0"),
    CATALOG_LITERAL_ROW("native_safetensors_payload_loaded: false"),
    CATALOG_STRING_ROW("source_tensor_metadata_status: %s", source_status),
    CATALOG_LITERAL_ROW("source_tensor_count: 0"),
    CATALOG_LITERAL_ROW("source_tensor_metadata_payload_loaded: false"),
    CATALOG_STRING_ROW("model_class_profile_status: %s", profile_status),
    CATALOG_STRING_ROW("model_class_target_id: %s", target_id),
    CATALOG_STRING_ROW("model_class_runtime_shape: %s", runtime_shape),
    CATALOG_STRING_ROW("model_class_evidence_basis: %s", evidence),
    CATALOG_STRING_ROW("model_class_pattern_status: %s", pattern),
    CATALOG_STRING_ROW("model_class_role_mapping_status: %s", role_mapping),
    CATALOG_LITERAL_ROW("model_class_runtime_status: unsupported"),
    CATALOG_LITERAL_ROW("tensor_collection_status: command-visible"),
    CATALOG_STRING_ROW("tensor_collection_family: %s", family),
    CATALOG_STRING_ROW("tensor_collection_target_id: %s", target_id),
    CATALOG_LITERAL_ROW("tensor_collection_stage: header-collection-inventory"),
    CATALOG_LITERAL_ROW("tensor_collection_evidence_basis: header-metadata-only"),
    CATALOG_LITERAL_ROW("tensor_collection_validation_status: lexical-and-header-only"),
    CATALOG_STRING_ROW("tensor_collection_role_mapping_status: %s", role_mapping),
    CATALOG_LITERAL_ROW("tensor_collection_runtime_descriptor_status: not-implemented"),
    CATALOG_LITERAL_ROW("tensor_collection_graph_consumer_status: not-implemented"),
    CATALOG_STRING_ROW("output_head_map_status: %s", output_status),
    CATALOG_STRING_ROW("output_head_map_family: %s", family),
    CATALOG_STRING_ROW("output_head_map_target_id: %s", target_id),
    CATALOG_LITERAL_ROW("output_head_map_stage: header-output-head-map"),
    CATALOG_STRING_ROW("output_head_map_next: %s", output_next),
    CATALOG_STRING_ROW("tokenizer_map_status: %s", tokenizer_status),
    CATALOG_STRING_ROW("tokenizer_map_family: %s", family),
    CATALOG_STRING_ROW("tokenizer_map_target_id: %s", target_id),
    CATALOG_LITERAL_ROW("tokenizer_map_stage: metadata-tokenizer-map"),
    CATALOG_LITERAL_ROW("tokenizer_runtime_status: not-implemented"),
    CATALOG_STRING_ROW("tokenizer_map_next: %s", tokenizer_next),
    CATALOG_LITERAL_ROW("missing_role_report_status: not-run"),
    CATALOG_LITERAL_ROW("missing_role_report_stage: missing-role-blocker-report"),
    CATALOG_STRING_ROW("missing_role_next_required_row: %s", missing_next),
    CATALOG_STRING_ROW("tensor_mapping_gate_status: %s", gate_status),
    CATALOG_LITERAL_ROW("tensor_mapping_gate: v0.1.0-tensor-mapping"),
    CATALOG_STRING_ROW("tensor_mapping_gate_next_required_row: %s", gate_next),
    CATALOG_STRING_ROW("target_artifact_class: %s", artifact_class),
    CATALOG_STRING_ROW("target_artifact_status: %s", artifact_status),
    CATALOG_LITERAL_ROW("benchmark_status: not-measured"),
    CATALOG_LITERAL_ROW("runtime_execution: unsupported"),
    CATALOG_LITERAL_ROW("generation: unsupported")
};

static const yvex_model_target_row_spec catalog_list_audit_rows[] = {
    CATALOG_STRING_ROW("target: %s", target_id),
    CATALOG_STRING_ROW("target_class: %s", target_class),
    CATALOG_STRING_ROW("model_class_profile_status: %s", profile_status),
    CATALOG_STRING_ROW("model_class_target_id: %s", target_id),
    CATALOG_STRING_ROW("model_class_evidence_basis: %s", evidence),
    CATALOG_STRING_ROW("model_class_pattern_status: %s", pattern),
    CATALOG_STRING_ROW("model_class_role_mapping_status: %s", role_mapping),
    CATALOG_LITERAL_ROW("tensor_collection_status: command-visible"),
    CATALOG_STRING_ROW("tensor_collection_family: %s", family),
    CATALOG_STRING_ROW("tensor_collection_target_id: %s", target_id),
    CATALOG_LITERAL_ROW("tensor_collection_stage: header-collection-inventory"),
    CATALOG_LITERAL_ROW("tensor_collection_validation_status: lexical-and-header-only"),
    CATALOG_STRING_ROW("tensor_collection_role_mapping_status: %s", role_mapping),
    CATALOG_STRING_ROW("output_head_map_status: %s", output_status),
    CATALOG_STRING_ROW("output_head_map_family: %s", family),
    CATALOG_STRING_ROW("output_head_map_target_id: %s", target_id),
    CATALOG_STRING_ROW("output_head_map_next: %s", output_next),
    CATALOG_STRING_ROW("tokenizer_map_status: %s", tokenizer_status),
    CATALOG_STRING_ROW("tokenizer_map_family: %s", family),
    CATALOG_STRING_ROW("tokenizer_map_target_id: %s", target_id),
    CATALOG_LITERAL_ROW("tokenizer_runtime_status: not-implemented"),
    CATALOG_STRING_ROW("tokenizer_map_next: %s", tokenizer_next),
    CATALOG_LITERAL_ROW("missing_role_report_status: not-run"),
    CATALOG_STRING_ROW("missing_role_report_target_id: %s", target_id),
    CATALOG_STRING_ROW("missing_role_next_required_row: %s", missing_next),
    CATALOG_STRING_ROW("tensor_mapping_gate_status: %s", gate_status),
    CATALOG_STRING_ROW("tensor_mapping_gate_target_id: %s", target_id),
    CATALOG_STRING_ROW("tensor_mapping_gate_next_required_row: %s", gate_next)
};

static catalog_audit_facts catalog_audit_project(const yvex_model_target_record *rec)
{
    int release = yvex_source_is_release_target(rec->target_id);
    catalog_audit_facts facts = {
        rec->target_id, yvex_model_target_family_key(rec->target_id), rec->model,
        rec->target_class, rec->source_artifact_class, catalog_source_status(rec),
        NULL, release ? "typed-architecture-ir" : "command-visible",
        catalog_runtime_shape(rec),
        release ? "strict-source-verification-to-typed-ir" : "header-metadata-only",
        release ? "not-used-for-release-architecture" : "lexical-only",
        release ? "canonical-logical-map-complete" : "not-implemented",
        release ? "mapped-logical-plan" : "not-run",
        release ? "V010.SOURCE.PAYLOAD.STREAM.0" : "V010.MAP.8",
        release ? "mapped-logical-plan" : "not-run",
        release ? "V010.SOURCE.PAYLOAD.STREAM.0" : "V010.MAP.7",
        release ? "V010.SOURCE.PAYLOAD.STREAM.0" : "V010.MAP.9",
        release ? "complete-logical-plan" : "not-run",
        release ? "V010.SOURCE.PAYLOAD.STREAM.0" : "V010.QUANT.0",
        rec->target_artifact_class, catalog_artifact_status(rec)
    };

    facts.identity_status = strcmp(facts.source_status, "missing") == 0
                                ? "not-present" : "not-verified";
    return facts;
}

static void catalog_emit_inspect_audit(const yvex_model_target_record *rec,
                                       yvex_model_target_report *report)
{
    catalog_audit_facts facts = catalog_audit_project(rec);
    const char *family_key = facts.family;

    facts.family = rec->family;
    yvex_model_target_report_project_rows(
        report, catalog_inspect_prefix_rows,
        sizeof(catalog_inspect_prefix_rows) /
            sizeof(catalog_inspect_prefix_rows[0]), &facts);
    facts.family = family_key;
    if (yvex_source_is_release_target(rec->target_id)) {
        const yvex_source_target_identity *identity =
            yvex_source_release_identity();
        yvex_model_target_report_add_row(report, "release_selected: true");
        yvex_model_target_report_add_row(report, "upstream_repository: %s",
                                         identity->upstream_repo_id);
        yvex_model_target_report_add_row(report, "source_directory_leaf: %s",
                                         identity->source_dir_leaf);
        yvex_model_target_report_add_row(report, "config_model_type: %s",
                                         identity->config_model_type);
        yvex_model_target_report_add_row(report, "config_architecture: %s",
                                         identity->config_architecture);
        yvex_model_target_report_add_row(report,
                                         "architecture_ir_owner: src/model/families/deepseek_v4.c");
        yvex_model_target_report_add_row(report,
                                         "architecture_ir_consumer: canonical-deepseek-gguf-map");
        yvex_model_target_report_add_row(report,
                                         "release_target_next: V010.SOURCE.PAYLOAD.STREAM.0");
    }
    yvex_model_target_report_project_rows(
        report, catalog_inspect_rows,
        sizeof(catalog_inspect_rows) / sizeof(catalog_inspect_rows[0]), &facts);
}

static void catalog_emit_list_audit_target(
    const yvex_model_target_record *rec,
    yvex_model_target_report *report)
{
    catalog_audit_facts facts = catalog_audit_project(rec);

    yvex_model_target_report_project_rows(
        report, catalog_list_audit_rows,
        sizeof(catalog_list_audit_rows) / sizeof(catalog_list_audit_rows[0]), &facts);
}

#undef CATALOG_LITERAL_ROW
#undef CATALOG_STRING_ROW

int yvex_model_target_catalog_report_build(
    const yvex_model_target_request *request,
    yvex_model_target_report *report,
    yvex_error *err)
{
    unsigned long i;

    if (!request || !report) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "model_target_catalog",
                       "request and report are required");
        return YVEX_ERR_INVALID_ARG;
    }
    report->kind = request->kind;
    report->mode = request->mode;
    report->exit_code = 0;

    if (request->kind == YVEX_MODEL_TARGET_COMMAND_CLASSES) {
        report->status = "model-target-classes";
        yvex_model_target_report_add_row(report, "status: model-target-classes");
        for (i = 0; i < target_class_count(); ++i) {
            const yvex_model_target_class_record *cls = target_class_at(i);
            yvex_model_target_report_add_row(report, "class: %s", cls->class_id);
            yvex_model_target_report_add_row(report, "capability_claim: %s", cls->capability_claim);
            yvex_model_target_report_add_row(report, "runtime_execution: %s", cls->runtime_execution);
            yvex_model_target_report_add_row(report, "generation: %s", cls->generation);
        }
        return YVEX_OK;
    }
    if (request->kind == YVEX_MODEL_TARGET_COMMAND_LIST) {
        report->status = "model-target-list";
        if (request->mode == YVEX_MODEL_TARGET_OUTPUT_JSON) {
            yvex_model_target_report_add_row(report,
                                             "{\"status\":\"model-target-list\",\"targets\":[");
            for (i = 0; i < target_count(); ++i) {
                const yvex_model_target_record *rec = target_at(i);
                yvex_model_target_report_add_row(
                    report,
                    "%s{\"target_id\":\"%s\",\"family\":\"%s\","
                    "\"class\":\"%s\",\"release_selected\":%s,"
                    "\"runtime\":\"%s\",\"generation\":\"%s\"}",
                    i ? "," : "", rec->target_id, rec->family,
                    rec->target_class,
                    yvex_source_is_release_target(rec->target_id)
                        ? "true"
                        : "false",
                    rec->runtime_execution, rec->generation);
            }
            yvex_model_target_report_add_row(report, "]}");
            return YVEX_OK;
        }
        yvex_model_target_report_add_row(report, "MODEL TARGETS  count=%lu",
                                         target_count());
        yvex_model_target_report_add_row(report, "TARGET  FAMILY  CLASS  RUNTIME  GENERATION");
        for (i = 0; i < target_count(); ++i) {
            const yvex_model_target_record *rec = target_at(i);
            yvex_model_target_report_add_row(report, "%s  %s  %s  %s  %s",
                                             rec->target_id, rec->family,
                                             rec->target_class,
                                             rec->runtime_execution,
                                             rec->generation);
        }
        yvex_model_target_report_add_row(report, "status: model-target-list");
        if (request->mode == YVEX_MODEL_TARGET_OUTPUT_AUDIT) {
            for (i = 0; i < target_count(); ++i) {
                const yvex_model_target_record *rec = target_at(i);
                catalog_emit_list_audit_target(rec, report);
            }
            yvex_model_target_report_add_row(report, "source_provenance_status:");
            yvex_model_target_report_add_row(report, "source_origin:");
            yvex_model_target_report_add_row(report, "source_authority:");
            yvex_model_target_report_add_row(report, "source_revision_status: unknown");
            yvex_model_target_report_add_row(report, "source_identity_status:");
            yvex_model_target_report_add_row(report, "source_hash_status: not-computed");
            yvex_model_target_report_add_row(report, "source_verification_status: not-verified");
            yvex_model_target_report_add_row(report, "native_inventory_status:");
            yvex_model_target_report_add_row(report, "native_tensor_count: 0");
            yvex_model_target_report_add_row(report, "native_safetensors_payload_loaded: false");
            yvex_model_target_report_add_row(report, "source_tensor_metadata_status:");
            yvex_model_target_report_add_row(report, "source_tensor_count: 0");
            yvex_model_target_report_add_row(report, "source_tensor_metadata_payload_loaded: false");
            yvex_model_target_report_add_row(report, "runtime_execution: unsupported");
            yvex_model_target_report_add_row(report, "generation: unsupported");
        }
        return YVEX_OK;
    }
    if (request->kind == YVEX_MODEL_TARGET_COMMAND_INSPECT) {
        const yvex_model_target_record *rec;
        if (!request->target_id[0]) {
            report->exit_code = 2;
            yvex_model_target_report_add_error(report, "model-target inspect: requires TARGET");
            return YVEX_OK;
        }
        rec = yvex_model_target_find(request->target_id);
        if (!rec) {
            report->exit_code = 2;
            yvex_model_target_report_add_error(report, "model-target: unknown target: %s", request->target_id);
            return YVEX_OK;
        }
        if (request->mode == YVEX_MODEL_TARGET_OUTPUT_JSON) {
            const yvex_source_target_identity *identity =
                yvex_source_is_release_target(rec->target_id)
                    ? yvex_source_release_identity()
                    : NULL;
            yvex_model_target_report_add_row(
                report,
                "{\"status\":\"model-target\",\"target_id\":\"%s\","
                "\"family\":\"%s\",\"class\":\"%s\","
                "\"release_selected\":%s,\"upstream_repository\":%s%s%s,"
                "\"source_status\":\"%s\",\"artifact_status\":\"%s\","
                "\"runtime\":\"%s\",\"generation\":\"%s\",\"next\":\"%s\"}",
                rec->target_id, rec->family, rec->target_class,
                identity ? "true" : "false",
                identity ? "\"" : "", identity ? identity->upstream_repo_id : "null",
                identity ? "\"" : "", catalog_source_status(rec),
                catalog_artifact_status(rec), catalog_runtime_status(rec),
                rec->generation, catalog_next_row(rec));
            return YVEX_OK;
        }
        if (request->include_paths) {
            catalog_path_report(request, report, rec);
            return YVEX_OK;
        }
        if (request->mode == YVEX_MODEL_TARGET_OUTPUT_AUDIT) {
            catalog_emit_inspect_audit(rec, report);
            return YVEX_OK;
        }
        report->status = "model-target";
        yvex_model_target_report_add_row(report, "status: model-target");
        yvex_model_target_report_add_row(report, "target: %s", rec->target_id);
        yvex_model_target_report_add_row(report, "family: %s  class=%s",
                                         rec->family, rec->target_class);
        yvex_model_target_report_add_row(report, "model: %s", rec->model);
        yvex_model_target_report_add_row(report, "source: %s  status=%s",
                                         rec->source_artifact_class,
                                         catalog_source_status(rec));
        yvex_model_target_report_add_row(report, "artifact: %s  status=%s",
                                         rec->target_artifact_class,
                                         catalog_artifact_status(rec));
        yvex_model_target_report_add_row(report, "runtime: %s",
                                         catalog_runtime_status(rec));
        yvex_model_target_report_add_row(report, "generation: %s", rec->generation);
        if (catalog_next_row(rec)[0]) {
            yvex_model_target_report_add_row(report, "next: %s",
                                             catalog_next_row(rec));
        }
        yvex_model_target_report_add_row(report, "boundary: %s",
                                         catalog_boundary(rec));
        return YVEX_OK;
    }
    return catalog_unknown_subcommand(request, report);
}

int yvex_model_target_catalog_help_report_build(
    yvex_model_target_report *report,
    yvex_error *err)
{
    if (!report) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "model_target_help",
                       "report is required");
        return YVEX_ERR_INVALID_ARG;
    }
    memset(report, 0, sizeof(*report));
    report->kind = YVEX_MODEL_TARGET_COMMAND_HELP;
    report->mode = YVEX_MODEL_TARGET_OUTPUT_NORMAL;
    report->help_requested = 1;
    report->status = "model-target-help";
    yvex_model_target_report_add_row(
        report, "The candidate report shows the selected DeepSeek release source and "
                "subordinate non-release engineering evidence.");
    yvex_model_target_report_add_row(
        report, "The dense-candidate report preserves Qwen and Gemma engineering evidence "
                "without offering an alternate v0.1.0 release target.");
    yvex_model_target_report_add_row(
        report, "The Qwen/Metal pressure report records a planned reduced-scale Apple "
                "Silicon / Metal lane for future full-runtime work.");
    yvex_model_target_report_add_row(
        report, "This command records the v0.1.0 target decision without promoting "
                "runtime support.");
    yvex_model_target_report_add_row(report, "The tensor naming map reads safetensors headers only");
    yvex_model_target_report_add_row(
        report, "The tensor mapping gate aggregates model-class, tensor-collection, tensor "
                "naming, output-head, tokenizer metadata, and missing-role reports.");
    yvex_model_target_report_add_row(
        report, "Release-target selection and engineering target evidence are not "
                "model-support claims.");
    yvex_model_target_report_add_row(report,
                                     "External GGUFs and external %s%s are reference evidence only.",
                                     "run", "ners");
    yvex_model_target_report_add_row(
        report, "Model-target path reporting does not read model payloads, create "
                "artifacts, register aliases, or claim runtime support.");
    yvex_model_target_report_add_row(
        report, "boundary: report-only; no runtime execution, generation, eval, "
                "benchmark, or release readiness");
    return YVEX_OK;
}
