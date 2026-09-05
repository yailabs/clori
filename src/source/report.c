/*
 * Assemble typed source facts from canonical metadata owners.
 *
 * Reports consume canonical facts and never load tensor payload bytes. Reports project evidence
 * and never promote capability.
 */
#include <ctype.h>
#include <dirent.h>
#include <limits.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <yvex/internal/core.h>
#include <yvex/internal/source_payload.h>

#define YVEX_SOURCE_MANIFEST_PROBE_CAP 8192u

static const char *source_report_tail_blockers[] = {
    "missing-qwen-tensor-role-map",
    "missing-qwen-tensor-map",
    "missing-qwen-yvex-artifact",
    "missing-qwen-artifact-identity",
    "missing-metal-hardware-profile",
    "missing-metal-backend-feasibility",
    "missing-unified-memory-residency-plan",
    "missing-real-prefill",
    "missing-real-kv-path",
    "missing-real-decode",
    "missing-real-output-head-logits",
    "missing-real-vocabulary-sampling",
    "missing-generation-loop-over-real-state",
    "missing-eval-path",
    "missing-benchmark-path",
};

static const char *gemma_source_tail_blockers[] = {
    "missing-gemma-tensor-role-map",
    "missing-gemma-tensor-map",
    "missing-gemma-yvex-artifact",
    "missing-gemma-artifact-identity",
    "missing-gemma-real-prefill",
    "missing-gemma-real-kv-path",
    "missing-gemma-real-decode",
    "missing-gemma-real-output-head-logits",
    "missing-gemma-real-vocabulary-sampling",
    "missing-gemma-generation-loop-over-real-state",
    "missing-gemma-eval-path",
    "missing-gemma-benchmark-path",
};

static const char *deepseek_source_tail_blockers[] = {
    "missing-deepseek-architecture-ir",
    "reopened-gguf-qtype-abi",
    "missing-deepseek-tensor-role-map",
    "missing-complete-deepseek-gguf",
    "unsupported-deepseek-runtime",
    "unsupported-deepseek-generation",
};

static const yvex_source_family_profile source_family_profiles[] = {
    {
        YVEX_SOURCE_RELEASE_FAMILY_KEY, YVEX_SOURCE_RELEASE_FAMILY_DISPLAY,
        "deepseek-source-verification", YVEX_SOURCE_RELEASE_TARGET_ID,
        "release-source-target", YVEX_SOURCE_RELEASE_NAME,
        "selected-unverified", "registered",
        "official-safetensors", "safetensors+structured-sidecars",
        "official", "upstream-official",
        "safetensors", "complete-YVEX-GGUF-not-produced",
        "not-produced", "true",
        "false", "not-produced",
        "exact-v0.1.0-release-source", "raw-config-facts-only",
        "dgx-spark", "cuda-release-lane-unsupported",
        "verification-required", "missing-source-path",
        "missing-source-manifest", "invalid-safetensors-header",
        "missing-source-config", "missing-tokenizer-json",
        "missing-deepseek-architecture-ir", "V010.SOURCE.PAYLOAD.STREAM.0",
        deepseek_source_tail_blockers,
        sizeof(deepseek_source_tail_blockers) / sizeof(deepseek_source_tail_blockers[0]),
    },
    {
        "qwen", "Qwen",
        "qwen-source-pressure", "qwen3-8b",
        "source-model-candidate", "Qwen3-8B",
        "profiled", "present",
        "official-source-tensors-planned", "safetensors+config-tokenizer-sidecars",
        "official", "upstream-official",
        "safetensors", "future-YVEX-produced-GGUF",
        "planned", "true",
        "false", "planned",
        "backend-neutral-qwen-source-model-target", "causal-decoder-candidate-pending-config",
        "backend-selection-deferred", "metal-planned",
        "pending source/config verification", "missing-qwen-source-path",
        "missing-qwen-source-manifest", "missing-qwen-native-inventory",
        "missing-qwen-source-config", "missing-qwen-tokenizer-files",
        "missing-qwen-tensor-role-map", "V010.MAP.8",
        source_report_tail_blockers,
        sizeof(source_report_tail_blockers) / sizeof(source_report_tail_blockers[0]),
    },
    {
        "gemma", "Gemma",
        "gemma-source-pressure", "gemma-4-12b-it",
        "source-model-candidate", "Gemma-4-12B-it",
        "profiled", "present",
        "official-source-tensors-planned", "safetensors+config-tokenizer-sidecars",
        "official", "upstream-official",
        "safetensors", "future-YVEX-produced-GGUF",
        "planned", "true",
        "false", "planned",
        "backend-neutral-gemma-source-model-target", "dense-candidate-pending-source-config",
        "backend-selection-deferred", "cpu-cuda-baseline-planned",
        "pending source/config verification", "missing-gemma-source-path",
        "missing-gemma-source-manifest", "missing-gemma-native-inventory",
        "missing-gemma-source-config", "missing-gemma-tokenizer-files",
        "missing-gemma-tensor-role-map", "V010.MAP.8",
        gemma_source_tail_blockers,
        sizeof(gemma_source_tail_blockers) / sizeof(gemma_source_tail_blockers[0]),
    },
};

static const unsigned long source_family_profile_count =
    sizeof(source_family_profiles) / sizeof(source_family_profiles[0]);

typedef struct {
    yvex_native_dtype dtype;
    size_t native_offset;
    size_t metadata_offset;
} source_dtype_counter;

typedef struct {
    const char *first;
    const char *second;
    const char *third;
    size_t offset;
} source_name_counter;

typedef struct {
    const char *suffix;
    size_t path_offset;
    size_t exists_offset;
    int registry_root;
} source_sidecar_fact;

typedef struct {
    size_t input;
    size_t outputs[4];
} source_u64_projection;

static const source_dtype_counter source_dtype_counters[] = {
    {YVEX_NATIVE_DTYPE_F16, offsetof(yvex_source_report, native_dtype_f16_count),
     offsetof(yvex_source_report, source_tensor_dtype_f16_count)},
    {YVEX_NATIVE_DTYPE_BF16, offsetof(yvex_source_report, native_dtype_bf16_count),
     offsetof(yvex_source_report, source_tensor_dtype_bf16_count)},
    {YVEX_NATIVE_DTYPE_F32, offsetof(yvex_source_report, native_dtype_f32_count),
     offsetof(yvex_source_report, source_tensor_dtype_f32_count)},
    {YVEX_NATIVE_DTYPE_I8, offsetof(yvex_source_report, native_dtype_i8_count),
     offsetof(yvex_source_report, source_tensor_dtype_i8_count)},
    {YVEX_NATIVE_DTYPE_I16, offsetof(yvex_source_report, native_dtype_i16_count),
     offsetof(yvex_source_report, source_tensor_dtype_i16_count)},
    {YVEX_NATIVE_DTYPE_I32, offsetof(yvex_source_report, native_dtype_i32_count),
     offsetof(yvex_source_report, source_tensor_dtype_i32_count)},
    {YVEX_NATIVE_DTYPE_I64, offsetof(yvex_source_report, native_dtype_i64_count),
     offsetof(yvex_source_report, source_tensor_dtype_i64_count)},
    {YVEX_NATIVE_DTYPE_U8, offsetof(yvex_source_report, native_dtype_u8_count),
     offsetof(yvex_source_report, source_tensor_dtype_u8_count)},
};

static const size_t source_rank_offsets[] = {
    offsetof(yvex_source_report, source_tensor_rank_0_count),
    offsetof(yvex_source_report, source_tensor_rank_1_count),
    offsetof(yvex_source_report, source_tensor_rank_2_count),
    offsetof(yvex_source_report, source_tensor_rank_3_count),
    offsetof(yvex_source_report, source_tensor_rank_4_count),
    offsetof(yvex_source_report, source_tensor_rank_other_count),
};

static const source_name_counter source_name_counters[] = {
    {"embed", "embd", NULL, offsetof(yvex_source_report, source_tensor_name_embed_count)},
    {"attn", "attention", NULL, offsetof(yvex_source_report, source_tensor_name_attn_count)},
    {"mlp", "ffn", "feed_forward", offsetof(yvex_source_report, source_tensor_name_mlp_count)},
    {"norm", NULL, NULL, offsetof(yvex_source_report, source_tensor_name_norm_count)},
    {"lm_head", "output.weight", NULL,
     offsetof(yvex_source_report, source_tensor_name_lm_head_count)},
};

static const source_sidecar_fact source_sidecars[] = {
    {".download.json", offsetof(yvex_source_report, download_registry_path),
     offsetof(yvex_source_report, download_registry_exists), 1},
    {".download-report.json", offsetof(yvex_source_report, download_report_path),
     offsetof(yvex_source_report, download_report_exists), 0},
    {".tensor-map.json", offsetof(yvex_source_report, tensor_map_path),
     offsetof(yvex_source_report, tensor_map_exists), 0},
    {".output-head-map.json", offsetof(yvex_source_report, output_head_map_path),
     offsetof(yvex_source_report, output_head_map_exists), 0},
    {".tokenizer-map.json", offsetof(yvex_source_report, tokenizer_map_path),
     offsetof(yvex_source_report, tokenizer_map_exists), 0},
};

static const source_u64_projection native_report_projections[] = {
    {offsetof(yvex_native_weight_table, header_read_count),
     {offsetof(yvex_source_report, native_safetensors_header_read_count), SIZE_MAX, SIZE_MAX,
      SIZE_MAX}},
    {offsetof(yvex_native_weight_table, header_error_count),
     {offsetof(yvex_source_report, native_safetensors_header_error_count),
      offsetof(yvex_source_report, native_invalid_file_count),
      offsetof(yvex_source_report, native_inventory_error_count),
      offsetof(yvex_source_report, source_tensor_metadata_error_count)}},
    {offsetof(yvex_native_weight_table, header_bytes),
     {offsetof(yvex_source_report, native_safetensors_header_bytes), SIZE_MAX, SIZE_MAX, SIZE_MAX}},
    {offsetof(yvex_native_weight_table, count),
     {offsetof(yvex_source_report, native_tensor_count),
      offsetof(yvex_source_report, source_tensor_count),
      offsetof(yvex_source_report, source_tensor_name_count), SIZE_MAX}},
    {offsetof(yvex_native_weight_table, summary) +
         offsetof(yvex_native_weight_summary, total_tensor_bytes),
     {offsetof(yvex_source_report, native_declared_tensor_bytes),
      offsetof(yvex_source_report, source_tensor_declared_tensor_bytes), SIZE_MAX, SIZE_MAX}},
};

static const source_u64_projection deepseek_report_projections[] = {
    {offsetof(yvex_source_verification, source_file_count),
     {offsetof(yvex_source_report, source_file_count),
      offsetof(yvex_source_report, source_regular_file_count), SIZE_MAX, SIZE_MAX}},
    {offsetof(yvex_source_verification, shard_count),
     {offsetof(yvex_source_report, safetensors_count),
      offsetof(yvex_source_report, native_safetensors_count),
      offsetof(yvex_source_report, native_safetensors_opened), SIZE_MAX}},
    {offsetof(yvex_source_verification, source_total_bytes),
     {offsetof(yvex_source_report, total_size_bytes), SIZE_MAX, SIZE_MAX, SIZE_MAX}},
    {offsetof(yvex_source_verification, shard_bytes),
     {offsetof(yvex_source_report, safetensors_size_bytes), SIZE_MAX, SIZE_MAX, SIZE_MAX}},
    {offsetof(yvex_source_verification, header_shard_count),
     {offsetof(yvex_source_report, native_safetensors_header_read_count),
      offsetof(yvex_source_report, source_tensor_file_count), SIZE_MAX, SIZE_MAX}},
    {offsetof(yvex_source_verification, header_bytes),
     {offsetof(yvex_source_report, native_safetensors_header_bytes), SIZE_MAX, SIZE_MAX, SIZE_MAX}},
    {offsetof(yvex_source_verification, header_tensor_count),
     {offsetof(yvex_source_report, native_tensor_count),
      offsetof(yvex_source_report, source_tensor_count),
      offsetof(yvex_source_report, source_tensor_name_count), SIZE_MAX}},
    {offsetof(yvex_source_verification, declared_tensor_bytes),
     {offsetof(yvex_source_report, native_declared_tensor_bytes),
      offsetof(yvex_source_report, source_tensor_declared_tensor_bytes), SIZE_MAX, SIZE_MAX}},
    {offsetof(yvex_source_verification, max_tensor_rank),
     {offsetof(yvex_source_report, native_max_rank),
      offsetof(yvex_source_report, source_tensor_max_rank), SIZE_MAX, SIZE_MAX}},
    {offsetof(yvex_source_verification, dtype_f16_count),
     {offsetof(yvex_source_report, native_dtype_f16_count),
      offsetof(yvex_source_report, source_tensor_dtype_f16_count), SIZE_MAX, SIZE_MAX}},
    {offsetof(yvex_source_verification, dtype_bf16_count),
     {offsetof(yvex_source_report, native_dtype_bf16_count),
      offsetof(yvex_source_report, source_tensor_dtype_bf16_count), SIZE_MAX, SIZE_MAX}},
    {offsetof(yvex_source_verification, dtype_f32_count),
     {offsetof(yvex_source_report, native_dtype_f32_count),
      offsetof(yvex_source_report, source_tensor_dtype_f32_count), SIZE_MAX, SIZE_MAX}},
    {offsetof(yvex_source_verification, dtype_i64_count),
     {offsetof(yvex_source_report, native_dtype_i64_count),
      offsetof(yvex_source_report, source_tensor_dtype_i64_count), SIZE_MAX, SIZE_MAX}},
    {offsetof(yvex_source_verification, dtype_i8_count),
     {offsetof(yvex_source_report, native_dtype_i8_count),
      offsetof(yvex_source_report, source_tensor_dtype_i8_count), SIZE_MAX, SIZE_MAX}},
};

static void source_report_project_u64(yvex_source_report *report,
                                      const void *input,
                                      const source_u64_projection *rows,
                                      size_t count) {
    const unsigned char *source = (const unsigned char *)input;
    unsigned char *target = (unsigned char *)report;
    size_t row, output;

    for (row = 0u; row < count; ++row)
        for (output = 0u; output < 4u && rows[row].outputs[output] != SIZE_MAX; ++output)
            *(unsigned long long *)(void *)(target + rows[row].outputs[output]) =
                *(const unsigned long long *)(const void *)(source + rows[row].input);
}

const yvex_source_family_profile *yvex_source_report_find_profile(const char *family) {
    unsigned long i;

    if (!family) {
        return NULL;
    }
    for (i = 0; i < source_family_profile_count; ++i) {
        if (strcmp(source_family_profiles[i].family_key, family) == 0) {
            return &source_family_profiles[i];
        }
    }
    return NULL;
}

static int source_report_target_is_supported(const yvex_source_family_profile *profile,
                                             const char *target) {
    if (!profile || !target) {
        return 0;
    }
    if (strcmp(profile->family_key, "deepseek") == 0) {
        return yvex_source_is_release_target(target);
    }
    if ((strcmp(profile->family_key, "qwen") == 0 && strncmp(target, "qwen", 4) == 0) ||
        (strcmp(profile->family_key, "gemma") == 0 && strncmp(target, "gemma", 5) == 0)) {
        return 1;
    }
    return strcmp(target, profile->target_id) == 0;
}

static int source_report_path_join(char *out, size_t cap, const char *left, const char *right) {
    int n;

    if (!out || cap == 0u || !left || !right) {
        return 0;
    }
    n = snprintf(out, cap, "%s/%s", left, right);
    if (n < 0 || (size_t)n >= cap) {
        out[cap - 1] = '\0';
        return 0;
    }
    return 1;
}

static int
source_report_json_u64_field(const char *text, const char *key, unsigned long long *out) {
    const char *p;
    unsigned long long value = 0;
    int seen = 0;

    if (out)
        *out = 0;
    if (!text || !key || !out)
        return 0;
    p = yvex_json_probe_field_value(text, key);
    if (!p)
        return 0;
    while (*p && isdigit((unsigned char)*p)) {
        value = value * 10ull + (unsigned long long)(*p - '0');
        seen = 1;
        p++;
    }
    if (!seen)
        return 0;
    *out = value;
    return 1;
}

static void source_report_copy_model_display(char *out,
                                             size_t cap,
                                             const char *family,
                                             const char *model_name) {
    if (!out || cap == 0u)
        return;
    out[0] = '\0';
    if (!model_name || !model_name[0])
        return;
    yvex_core_text_copy(out, cap, model_name);
    if (family && strcmp(family, "gemma") == 0 && strncmp(out, "gemma-", 6) == 0) {
        out[0] = 'G';
    }
}

/*
 * Project report probe download identity file facts while preserving the canonical source report
 * invariants.
 */
static int source_report_probe_download_identity_file(const char *path,
                                                      const char *target,
                                                      const char *family,
                                                      yvex_source_report *report) {
    char buf[YVEX_SOURCE_MANIFEST_PROBE_CAP + 1u];
    char parsed_target[128] = {0};
    char parsed_family[32] = {0};
    char repo_id[256] = {0};
    char revision[128] = {0};
    char source_dir[YVEX_PATH_CAP] = {0};
    const char *model_name;

    if (!path || !path[0] || !target || !family || !report)
        return 0;
    if (!yvex_core_file_read_text_prefix(path, buf, sizeof(buf)))
        return 0;

    yvex_json_probe_string_field(buf, "target_id", parsed_target, sizeof(parsed_target));
    if (parsed_target[0] && strcmp(parsed_target, target) != 0)
        return 0;
    yvex_json_probe_string_field(buf, "family", parsed_family, sizeof(parsed_family));
    if (parsed_family[0] && strcmp(parsed_family, family) != 0)
        return 0;
    if (!yvex_json_probe_string_field(buf, "repo_id", repo_id, sizeof(repo_id)) || !repo_id[0])
        yvex_json_probe_string_field(buf, "repo", repo_id, sizeof(repo_id));
    yvex_json_probe_string_field(buf, "revision", revision, sizeof(revision));
    if (!yvex_json_probe_string_field(
            buf, "local_source_dir", source_dir, sizeof(source_dir)) ||
        !source_dir[0])
        yvex_json_probe_string_field(buf, "path", source_dir, sizeof(source_dir));

    yvex_core_text_copy(report->identity_target_id, sizeof(report->identity_target_id),
                        parsed_target[0] ? parsed_target : target);
    yvex_core_text_copy(report->identity_family, sizeof(report->identity_family),
                        parsed_family[0] ? parsed_family : family);
    yvex_core_text_copy(report->identity_repo_id, sizeof(report->identity_repo_id),
                        repo_id[0] ? repo_id : "unknown");
    yvex_core_text_copy(report->identity_revision, sizeof(report->identity_revision),
                        revision[0] ? revision : "main");
    if (source_dir[0])
        yvex_core_text_copy(report->identity_local_source_dir,
                            sizeof(report->identity_local_source_dir), source_dir);
    model_name = yvex_source_path_basename(repo_id);
    if (model_name && model_name[0]) {
        source_report_copy_model_display(
            report->identity_model, sizeof(report->identity_model), family, model_name);
    }
    report->source_identity_from_download_sidecar = 1;
    return 1;
}

static void source_report_probe_map_sidecars(yvex_source_report *report) {
    char buf[YVEX_SOURCE_MANIFEST_PROBE_CAP + 1u];
    char status[64];
    char coverage[64];
    unsigned long long unmapped = 0;

    if (!report)
        return;
    if (report->tensor_map_exists &&
        yvex_core_file_read_text_prefix(report->tensor_map_path, buf, sizeof(buf))) {
        if (yvex_json_probe_string_field(
                buf, "required_role_coverage_status", coverage, sizeof(coverage))) {
            if (strcmp(coverage, "required-groups-present") != 0) {
                report->tensor_map_incomplete = 1;
            }
        } else if (source_report_json_u64_field(buf, "unmapped_unknown_count", &unmapped) &&
                   unmapped > 0ull) {
            report->tensor_map_incomplete = 1;
        }
    }
    if (report->output_head_map_exists &&
        yvex_core_file_read_text_prefix(report->output_head_map_path, buf, sizeof(buf)) &&
        yvex_json_probe_string_field(buf, "output_head_status", status, sizeof(status)) &&
        strcmp(status, "present") != 0) {
        report->output_head_map_missing = 1;
    }
}

static int source_report_stat_kind(const char *path, int want_dir) {
    struct stat st;

    if (!path || path[0] == '\0') {
        return 0;
    }
    if (stat(path, &st) != 0) {
        return 0;
    }
    return want_dir ? S_ISDIR(st.st_mode) : S_ISREG(st.st_mode);
}

static int source_report_manifest_file_exists(char *out, size_t cap, const char *dir,
                                              const char *name) {
    char candidate[YVEX_PATH_CAP];

    if (!out || cap == 0 || !dir || dir[0] == '\0' || !name) {
        return 0;
    }
    if (!source_report_path_join(candidate, sizeof(candidate), dir, name)) {
        return 0;
    }
    if (!source_report_stat_kind(candidate, 0)) {
        return 0;
    }
    yvex_core_text_copy(out, cap, candidate);
    return 1;
}

static int source_report_check_file(const char *dir, const char *name) {
    char path[YVEX_PATH_CAP];
    return source_report_manifest_file_exists(path, sizeof(path), dir, name);
}

static void source_report_native_collect_table(yvex_source_report *report,
                                               const yvex_native_weight_table *table);

static int source_report_scan_local(const char *dir, yvex_source_report *report) {
    const unsigned long long maximum_shards = 1024u;
    yvex_native_weight_table *table;
    DIR *dp;
    struct dirent *ent;
    unsigned long long scan_errors = 0u;
    int fatal = YVEX_OK;

    if (!dir || !dir[0] || !report || !report->source_exists)
        return YVEX_OK;
    table = (yvex_native_weight_table *)calloc(1u, sizeof(*table));
    if (!table)
        return YVEX_ERR_NOMEM;
    dp = opendir(dir);
    if (!dp) {
        free(table);
        return YVEX_OK;
    }
    while ((ent = readdir(dp)) != NULL) {
        char path[YVEX_PATH_CAP];
        struct stat st;
        unsigned long long size_bytes;
        unsigned long long *size_total;
        yvex_error header_error;
        int is_safetensors;
        int is_sidecar;
        int is_config;
        int is_tokenizer;
        int rc;

        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0)
            continue;
        if (!source_report_path_join(path, sizeof(path), dir, ent->d_name))
            continue;
        if (stat(path, &st) != 0 || !S_ISREG(st.st_mode))
            continue;
        size_bytes = st.st_size > 0 ? (unsigned long long)st.st_size : 0ull;
        is_safetensors = yvex_source_ends_with(ent->d_name, ".safetensors");
        is_config = strcmp(ent->d_name, "config.json") == 0 ||
                    strcmp(ent->d_name, "generation_config.json") == 0;
        is_tokenizer = strcmp(ent->d_name, "tokenizer.json") == 0 ||
                       strcmp(ent->d_name, "tokenizer_config.json") == 0;
        is_sidecar = is_config || is_tokenizer || strcmp(ent->d_name, "README.md") == 0 ||
                     yvex_source_ends_with(ent->d_name, ".json");
        report->source_file_count++;
        report->source_regular_file_count++;
        report->total_size_bytes += size_bytes;
        size_total = is_safetensors ? &report->safetensors_size_bytes
                     : is_sidecar   ? &report->sidecar_size_bytes
                                    : &report->other_size_bytes;
        *size_total += size_bytes;
        report->safetensors_count += (unsigned long long)is_safetensors;
        report->bin_count += (unsigned long long)yvex_source_ends_with(ent->d_name, ".bin");
        report->dat_count += (unsigned long long)yvex_source_ends_with(ent->d_name, ".dat");
        report->json_count += (unsigned long long)yvex_source_ends_with(ent->d_name, ".json");
        report->tokenizer_file_count += (unsigned long long)is_tokenizer;
        report->config_file_count += (unsigned long long)is_config;
        if (size_bytes > report->largest_source_file_bytes) {
            report->largest_source_file_bytes = size_bytes;
            yvex_core_text_copy(report->largest_source_file_name,
                                sizeof(report->largest_source_file_name), ent->d_name);
        }
        if (!is_safetensors || fatal != YVEX_OK)
            continue;
        if (report->native_safetensors_count >= maximum_shards) {
            scan_errors = 1u;
            continue;
        }
        report->native_safetensors_count++;
        report->native_safetensors_opened++;
        table->summary.shard_count++;
        yvex_error_clear(&header_error);
        {
            unsigned long long errors_before = table->header_error_count;

            rc = yvex_safetensors_read_header_file(path, ent->d_name, table, &header_error);
            if (rc != YVEX_OK && table->header_error_count == errors_before)
                table->header_error_count++;
        }
        if (rc == YVEX_ERR_NOMEM)
            fatal = rc;
    }
    closedir(dp);
    table->header_error_count += scan_errors;
    source_report_native_collect_table(report, table);
    yvex_native_weight_table_close(table);
    return fatal;
}

static unsigned long long
source_report_native_tensor_elements(const yvex_native_weight_info *info) {
    unsigned long long elements = 1;
    unsigned int i;

    if (!info) {
        return 0;
    }
    for (i = 0; i < info->rank; ++i) {
        if (info->dims[i] == 0) {
            return 0;
        }
        if (!yvex_core_u64_mul(elements, info->dims[i], &elements)) {
            return ULLONG_MAX;
        }
    }
    return elements;
}

static unsigned long long source_report_u64_max(unsigned long long left,
                                                unsigned long long right) {
    return left > right ? left : right;
}

static void
source_report_tensor_shape_string(const yvex_native_weight_info *info, char *out, size_t cap) {
    size_t used = 0;
    unsigned int i;

    if (!out || cap == 0) {
        return;
    }
    out[0] = '\0';
    if (!info) {
        snprintf(out, cap, "[]");
        return;
    }
    used += (size_t)snprintf(out + used, cap - used, "[");
    for (i = 0; i < info->rank && used < cap; ++i) {
        int n = snprintf(out + used, cap - used, "%s%llu", i == 0 ? "" : ",", info->dims[i]);
        if (n < 0) {
            break;
        }
        if ((size_t)n >= cap - used) {
            out[cap - 1] = '\0';
            return;
        }
        used += (size_t)n;
    }
    if (used < cap) {
        snprintf(out + used, cap - used, "]");
    } else {
        out[cap - 1] = '\0';
    }
}

static int source_report_tensor_shape_first_seen(const yvex_native_weight_table *table,
                                                 unsigned long long index) {
    unsigned long long i;

    if (!table || index >= table->count) {
        return 0;
    }
    for (i = 0; i < index; ++i) {
        const yvex_native_weight_info *previous = &table->items[i];
        const yvex_native_weight_info *current = &table->items[index];
        unsigned int dimension;

        if (previous->rank != current->rank)
            continue;
        for (dimension = 0u; dimension < current->rank; ++dimension) {
            if (previous->dims[dimension] != current->dims[dimension])
                break;
        }
        if (dimension == current->rank)
            return 0;
    }
    return 1;
}

static int source_report_name_contains_ci(const char *name, const char *needle) {
    size_t needle_len;
    size_t i;

    if (!name || !needle) {
        return 0;
    }
    needle_len = strlen(needle);
    if (needle_len == 0) {
        return 1;
    }
    for (i = 0; name[i] != '\0'; ++i) {
        size_t j;
        for (j = 0; j < needle_len; ++j) {
            if (name[i + j] == '\0' ||
                tolower((unsigned char)name[i + j]) != tolower((unsigned char)needle[j])) {
                break;
            }
        }
        if (j == needle_len) {
            return 1;
        }
    }
    return 0;
}

static const char *source_report_file_label(const char *path) {
    const char *label = yvex_source_path_basename(path);

    return label ? label : "unknown";
}

static void source_report_count_dtype(yvex_source_report *report, yvex_native_dtype dtype) {
    size_t index;
    size_t native_offset = offsetof(yvex_source_report, native_dtype_other_count);
    size_t metadata_offset = offsetof(yvex_source_report, source_tensor_dtype_other_count);
    unsigned char *base = (unsigned char *)report;

    for (index = 0u; index < sizeof(source_dtype_counters) / sizeof(source_dtype_counters[0]);
         ++index) {
        if (source_dtype_counters[index].dtype == dtype) {
            native_offset = source_dtype_counters[index].native_offset;
            metadata_offset = source_dtype_counters[index].metadata_offset;
            break;
        }
    }
    (*(unsigned long long *)(void *)(base + native_offset))++;
    (*(unsigned long long *)(void *)(base + metadata_offset))++;
}

static void source_report_metadata_count_rank(yvex_source_report *report,
                                              unsigned long long rank) {
    size_t bucket = rank < 5u ? (size_t)rank : 5u;
    unsigned char *base = (unsigned char *)report;

    (*(unsigned long long *)(void *)(base + source_rank_offsets[bucket]))++;
}

static void source_report_metadata_count_name(yvex_source_report *report, const char *name) {
    size_t index;
    size_t offset = offsetof(yvex_source_report, source_tensor_name_other_count);

    for (index = 0u; index < sizeof(source_name_counters) / sizeof(source_name_counters[0]); ++index) {
        const source_name_counter *row = &source_name_counters[index];

        if (source_report_name_contains_ci(name, row->first) ||
            (row->second && source_report_name_contains_ci(name, row->second)) ||
            (row->third && source_report_name_contains_ci(name, row->third))) {
            offset = row->offset;
            break;
        }
    }
    (*(unsigned long long *)(void *)((unsigned char *)report + offset))++;
}

static void source_report_count_distinct(const yvex_source_report *report,
                                         unsigned long long *dtype_count,
                                         unsigned long long *rank_count) {
    const unsigned char *base = (const unsigned char *)report;
    size_t index;

    *dtype_count = 0u;
    *rank_count = 0u;
    for (index = 0u; index < sizeof(source_dtype_counters) / sizeof(source_dtype_counters[0]);
         ++index) {
        if (*(const unsigned long long *)(const void *)(base +
                                                       source_dtype_counters[index].metadata_offset))
            (*dtype_count)++;
    }
    if (report->source_tensor_dtype_other_count)
        (*dtype_count)++;
    for (index = 0u; index < sizeof(source_rank_offsets) / sizeof(source_rank_offsets[0]); ++index) {
        if (*(const unsigned long long *)(const void *)(base + source_rank_offsets[index]))
            (*rank_count)++;
    }
}

static void source_report_metadata_add_sample(yvex_source_report *report,
                                              const yvex_native_weight_info *info,
                                              unsigned long long elements) {
    yvex_source_tensor_sample *sample;

    if (!report || !info || report->source_tensor_sample_count >= YVEX_SOURCE_TENSOR_SAMPLE_CAP) {
        return;
    }
    sample = &report->source_tensor_samples[report->source_tensor_sample_count++];
    yvex_core_text_copy(sample->name, sizeof(sample->name),
                        info->name ? info->name : "unknown");
    yvex_core_text_copy(sample->file, sizeof(sample->file),
                        source_report_file_label(info->shard_path));
    yvex_core_text_copy(sample->dtype, sizeof(sample->dtype),
                        yvex_native_dtype_name(info->dtype));
    source_report_tensor_shape_string(info, sample->shape, sizeof(sample->shape));
    sample->rank = info->rank;
    sample->elements = elements;
    sample->declared_bytes = info->data_bytes;
}

static void source_report_native_collect_table(yvex_source_report *report,
                                               const yvex_native_weight_table *table) {
    const char *active_shard = NULL;
    unsigned long long active_shard_end = 0u;
    unsigned long long i;

    if (!report || !table) {
        return;
    }
    source_report_project_u64(report,
                              table,
                              native_report_projections,
                              sizeof(native_report_projections) /
                                  sizeof(native_report_projections[0]));

    for (i = 0; i < table->count; ++i) {
        const yvex_native_weight_info *info = &table->items[i];
        unsigned long long elements = source_report_native_tensor_elements(info);
        char shape[YVEX_SOURCE_TENSOR_SHAPE_CAP];

        source_report_count_dtype(report, info->dtype);
        source_report_metadata_count_rank(report, info->rank);
        source_report_metadata_count_name(report, info->name);
        if (!active_shard || strcmp(active_shard, info->shard_path) != 0) {
            if (active_shard) {
                report->native_declared_data_bytes += active_shard_end;
                report->source_tensor_declared_data_bytes += active_shard_end;
                report->source_tensor_file_count++;
            }
            active_shard = info->shard_path;
            active_shard_end = 0u;
        }
        if (info->data_end > active_shard_end)
            active_shard_end = info->data_end;
        if (elements == ULLONG_MAX ||
            !yvex_core_u64_add(report->source_tensor_total_elements, elements, &report->source_tensor_total_elements)) {
            report->source_tensor_total_elements = ULLONG_MAX;
        }
        report->native_max_rank = source_report_u64_max(report->native_max_rank, info->rank);
        report->source_tensor_max_rank =
            source_report_u64_max(report->source_tensor_max_rank, info->rank);
        report->native_max_tensor_elements =
            source_report_u64_max(report->native_max_tensor_elements, elements);
        report->source_tensor_max_elements =
            source_report_u64_max(report->source_tensor_max_elements, elements);
        if (source_report_tensor_shape_first_seen(table, i)) {
            report->source_tensor_shape_count++;
        }
        source_report_metadata_add_sample(report, info, elements);
        if (info->data_bytes > report->native_largest_tensor_bytes) {
            report->native_largest_tensor_bytes = info->data_bytes;
            yvex_core_text_copy(report->native_largest_tensor_name,
                                sizeof(report->native_largest_tensor_name),
                                info->name ? info->name : "unknown");
        }
        if (info->data_bytes > report->source_tensor_largest_declared_bytes) {
            source_report_tensor_shape_string(info, shape, sizeof(shape));
            report->source_tensor_largest_declared_bytes = info->data_bytes;
            report->source_tensor_largest_rank = info->rank;
            report->source_tensor_largest_elements = elements;
            yvex_core_text_copy(report->source_tensor_largest_name,
                                sizeof(report->source_tensor_largest_name),
                                info->name ? info->name : "unknown");
            yvex_core_text_copy(report->source_tensor_largest_file,
                                sizeof(report->source_tensor_largest_file),
                                source_report_file_label(info->shard_path));
            yvex_core_text_copy(report->source_tensor_largest_dtype,
                                sizeof(report->source_tensor_largest_dtype),
                                yvex_native_dtype_name(info->dtype));
            yvex_core_text_copy(report->source_tensor_largest_shape,
                                sizeof(report->source_tensor_largest_shape), shape);
        }
    }
    if (active_shard) {
        report->native_declared_data_bytes += active_shard_end;
        report->source_tensor_declared_data_bytes += active_shard_end;
        report->source_tensor_file_count++;
    }
    source_report_count_distinct(
        report, &report->source_tensor_dtype_count, &report->source_tensor_rank_count);
}

static void source_report_choose_report_file(char *out,
                                             size_t cap,
                                             const yvex_source_family_profile *profile,
                                             const char *reports_root,
                                             const char *source_path,
                                             const char *target,
                                             const char *kind,
                                             int *out_exists) {
    char family_dir[YVEX_PATH_CAP];
    char target_name[192];
    char family_name[192];
    const char *suffix;
    int manifest, target_valid, family_valid;

    if (out_exists) {
        *out_exists = 0;
    }
    if (!out || cap == 0 || !profile || !kind) {
        return;
    }
    out[0] = '\0';

    manifest = strcmp(kind, "manifest") == 0;
    suffix = manifest ? "source-manifest.json" : "native-inventory.json";
    if (source_path && source_path[0] &&
        (source_report_manifest_file_exists(
             out, cap, source_path, manifest ? "source_manifest.json" : suffix) ||
         (manifest &&
          source_report_manifest_file_exists(out, cap, source_path, suffix))))
        goto found;
    if (!reports_root || !reports_root[0] ||
        !source_report_path_join(family_dir, sizeof(family_dir), reports_root, profile->family_key))
        return;
    target_valid = snprintf(target_name, sizeof(target_name), "%s.%s", target, suffix);
    family_valid = snprintf(family_name, sizeof(family_name), "%s-%s", profile->family_key, suffix);
    target_valid = target_valid >= 0 && (size_t)target_valid < sizeof(target_name);
    family_valid = family_valid >= 0 && (size_t)family_valid < sizeof(family_name);
    if (target_valid && source_report_manifest_file_exists(out, cap, family_dir, target_name))
        goto found;
    if (family_valid && source_report_manifest_file_exists(out, cap, family_dir, family_name))
        goto found;
    if (!target_valid || !source_report_path_join(out, cap, family_dir, target_name))
        out[0] = '\0';
    return;

found:
    if (out_exists)
        *out_exists = 1;
}

static int source_report_manifest_blob_has_field(const char *blob, const char *field) {
    char quoted[96];
    int n;

    if (!blob || !field) {
        return 0;
    }
    n = snprintf(quoted, sizeof(quoted), "\"%s\"", field);
    if (n < 0 || (size_t)n >= sizeof(quoted)) {
        return 0;
    }
    return strstr(blob, quoted) != NULL;
}

static int source_report_manifest_blob_has_value(const char *blob, const char *value) {
    return blob && value && value[0] != '\0' && strstr(blob, value) != NULL;
}

static void source_report_probe_manifest(yvex_source_report *report,
                                         const yvex_source_family_profile *profile,
                                         const char *target) {
    char buf[YVEX_SOURCE_MANIFEST_PROBE_CAP + 1u];
    FILE *fp;
    size_t nread;

    if (!report || !profile || !target || !report->manifest_exists ||
        report->manifest_path[0] == '\0') {
        return;
    }
    fp = fopen(report->manifest_path, "rb");
    if (!fp) {
        report->manifest_probe_error = 1;
        return;
    }
    nread = fread(buf, 1, YVEX_SOURCE_MANIFEST_PROBE_CAP, fp);
    if (ferror(fp)) {
        report->manifest_probe_error = 1;
        fclose(fp);
        return;
    }
    fclose(fp);
    buf[nread] = '\0';
    report->manifest_probe_checked = 1;
    report->manifest_has_schema = source_report_manifest_blob_has_field(buf, "schema");
    report->manifest_schema_matches =
        source_report_manifest_blob_has_value(buf, "yvex.source_manifest.v1");
    yvex_core_text_copy(report->manifest_schema_version,
                        sizeof(report->manifest_schema_version),
                        report->manifest_schema_matches ? "yvex.source_manifest.v1" : "unknown");
    report->manifest_has_family =
        source_report_manifest_blob_has_field(buf, "family") ||
        source_report_manifest_blob_has_value(buf, profile->family_key) ||
        source_report_manifest_blob_has_value(buf, profile->display_family);
    report->manifest_family_matches =
        source_report_manifest_blob_has_value(buf, profile->family_key) ||
        source_report_manifest_blob_has_value(buf, profile->display_family);
    report->manifest_has_target = source_report_manifest_blob_has_field(buf, "target") ||
                                  source_report_manifest_blob_has_value(buf, target);
    report->manifest_target_matches = source_report_manifest_blob_has_value(buf, target);
    report->manifest_has_artifact_class =
        source_report_manifest_blob_has_field(buf, "artifact_class") ||
        source_report_manifest_blob_has_field(buf, "source_artifact_class") ||
        source_report_manifest_blob_has_value(buf, profile->source_artifact_class);
    report->manifest_has_footprint = source_report_manifest_blob_has_field(buf, "footprint") ||
                                     source_report_manifest_blob_has_field(buf, "summary");
    report->manifest_has_provenance = source_report_manifest_blob_has_field(buf, "provenance") ||
                                      source_report_manifest_blob_has_field(buf, "source");
    report->manifest_has_native_inventory =
        source_report_manifest_blob_has_field(buf, "native_inventory") ||
        source_report_manifest_blob_has_field(buf, "native");
    report->manifest_has_tensor_metadata =
        source_report_manifest_blob_has_field(buf, "tensor_metadata") ||
        source_report_manifest_blob_has_field(buf, "tensors");
}

static void source_report_add_blocker(yvex_source_report *report, const char *blocker) {
    if (!report || !blocker || report->blocker_count >= 32) {
        return;
    }
    report->blockers[report->blocker_count++] = blocker;
}

static int source_report_tail_blocker_is_tensor_map(const char *blocker) {
    return blocker &&
           (strstr(blocker, "tensor-role-map") != NULL || strstr(blocker, "tensor-map") != NULL);
}

static void source_report_apply_deepseek_verification(yvex_source_report *report) {
    const yvex_source_verification *verification = &report->verification;

    if (verification->repository_id[0]) {
        yvex_core_text_copy(report->identity_repo_id, sizeof(report->identity_repo_id),
                            verification->repository_id);
    }
    if (verification->revision[0]) {
        yvex_core_text_copy(report->identity_revision, sizeof(report->identity_revision),
                            verification->revision);
    }
    yvex_core_text_copy(report->identity_local_source_dir,
                        sizeof(report->identity_local_source_dir),
                        verification->resolved_source_path);
    yvex_core_text_copy(report->manifest_path, sizeof(report->manifest_path),
                        verification->manifest_path);
    report->manifest_exists = verification->manifest_path[0] != '\0';
    report->source_identity_from_path = verification->path_verified;
    report->config_exists = verification->config_valid;
    report->tokenizer_json_exists = verification->tokenizer_json_valid;
    report->tokenizer_config_exists = verification->tokenizer_config_valid;
    source_report_project_u64(report,
                              verification,
                              deepseek_report_projections,
                              sizeof(deepseek_report_projections) /
                                  sizeof(deepseek_report_projections[0]));
    report->sidecar_size_bytes = verification->source_total_bytes >= verification->shard_bytes
                                     ? verification->source_total_bytes - verification->shard_bytes
                                     : 0u;
    report->native_safetensors_header_error_count =
        verification->header_shard_count <= verification->shard_count
            ? verification->shard_count - verification->header_shard_count
            : 0u;
    report->native_dtype_other_count =
        verification->dtype_fp4_count + verification->dtype_f8_count +
        verification->dtype_f8_e8m0_count + verification->dtype_other_count;
    report->source_tensor_dtype_other_count =
        verification->dtype_fp4_count + verification->dtype_f8_count +
        verification->dtype_f8_e8m0_count + verification->dtype_other_count;
    report->source_tensor_dtype_count =
        (verification->dtype_f16_count ? 1u : 0u) + (verification->dtype_bf16_count ? 1u : 0u) +
        (verification->dtype_f32_count ? 1u : 0u) + (verification->dtype_i64_count ? 1u : 0u) +
        (verification->dtype_i8_count ? 1u : 0u) + (verification->dtype_fp4_count ? 1u : 0u) +
        (verification->dtype_f8_count ? 1u : 0u) + (verification->dtype_f8_e8m0_count ? 1u : 0u) +
        (verification->dtype_other_count ? 1u : 0u);
}

static const char *source_report_presence(int present) {
    return present ? "present" : "missing";
}

static const char *source_report_verification_state(int release,
                                                    int verified,
                                                    const char *other) {
    return release ? verified ? "verified" : "blocked" : other;
}

static const char *source_report_header_status(int source_exists,
                                               unsigned long long shard_count,
                                               unsigned long long error_count,
                                               unsigned long long evidence_count) {
    if (!source_exists)
        return "missing";
    if (!shard_count)
        return "no-safetensors";
    if (error_count)
        return "header-error";
    return evidence_count ? "header-only" : "unknown";
}

static const char *source_report_native_metadata_status(const char *inventory_status,
                                                        unsigned long long tensor_count) {
    if (strcmp(inventory_status, "header-only") == 0)
        return "header-only";
    if (strcmp(inventory_status, "header-error") == 0)
        return tensor_count ? "partial-header-only" : "header-error";
    if (strcmp(inventory_status, "missing") == 0 ||
        strcmp(inventory_status, "no-safetensors") == 0)
        return "not-present";
    return "unknown";
}

static const char *
source_report_manifest_match(const yvex_source_report *report, int has_field, int matches) {
    if (!report->manifest_exists)
        return "not-checked";
    if (report->manifest_probe_error)
        return "unreadable";
    if (matches)
        return "matched";
    return has_field ? "mismatch" : "not-declared";
}

static const char *source_report_manifest_decl(const yvex_source_report *report, int present) {
    if (!report->manifest_exists)
        return "not-checked";
    if (report->manifest_probe_error)
        return "unreadable";
    return present ? "declared" : "not-declared";
}

/* Append canonical source report fields to a deterministic identity stream. */
static void source_report_finalize_identity(yvex_source_report *report,
                                            yvex_source_report_semantics *facts,
                                            int deepseek,
                                            int has_tokenizer) {
    facts->verification_status =
        deepseek ? yvex_source_verification_status(&report->verification) : "not-verified";
    facts->repository_status = source_report_verification_state(
        deepseek, report->verification.repository_verified, "not-verified");
    facts->revision_status = source_report_verification_state(
        deepseek, report->verification.revision_verified, "unknown");
    facts->config_identity_status = source_report_verification_state(
        deepseek, report->verification.config_valid, "not-verified");
    facts->tokenizer_verified = deepseek && report->verification.tokenizer_json_valid &&
                                report->verification.tokenizer_config_valid;
    facts->tokenizer_verification_status = source_report_verification_state(
        deepseek, facts->tokenizer_verified, "not-verified");
    facts->generation_config_status = source_report_verification_state(
        deepseek, report->verification.generation_config_valid, "not-verified");
    facts->shard_index_status = !deepseek                                  ? "not-verified"
                                : report->verification.shard_index_valid   ? "verified"
                                : report->verification.shard_index_present ? "malformed"
                                                                           : "missing";
    facts->inventory_authority = deepseek && report->verification.inventory_authority[0]
                                     ? report->verification.inventory_authority
                                     : "blocked";
    facts->upstream_index_identity_status =
        !deepseek                                                   ? "not-verified"
        : strcmp(facts->inventory_authority, "header-derived") == 0 ? "not-applicable"
        : report->verification.upstream_index_identity_verified     ? "verified"
                                                                    : "not-verified";
    facts->model_name = report->identity_model[0]           ? report->identity_model
                        : report->request.resolved_model[0] ? report->request.resolved_model
                                                            : report->profile->model;
    facts->config_presence = source_report_presence(report->config_exists);
    facts->generation_config_presence = source_report_presence(report->generation_config_exists);
    facts->tokenizer_json_presence = source_report_presence(report->tokenizer_json_exists);
    facts->tokenizer_config_presence = source_report_presence(report->tokenizer_config_exists);
    facts->tokenizer_status = has_tokenizer ? "present" : "missing";
    facts->safetensors_status = report->safetensors_count ? "present" : "missing";
    facts->manifest_status = deepseek && report->verification.manifest_status[0]
                                 ? report->verification.manifest_status
                             : report->manifest_exists ? "present"
                                                       : "missing";
    facts->native_inventory_report_status =
        report->native_inventory_exists ? "available-report-only" : "missing";
    facts->tensor_map_report_status = !report->tensor_map_exists      ? "missing"
                                      : report->tensor_map_incomplete ? "incomplete-report-only"
                                                                      : "available-report-only";
    facts->tensor_role_map_report_status = facts->tensor_map_report_status;
    facts->output_head_map_report_status = !report->output_head_map_exists ? "missing"
                                           : report->output_head_map_missing
                                               ? "missing-in-report"
                                               : "available-report-only";
    facts->tokenizer_map_report_status =
        report->tokenizer_map_exists ? "available-report-only" : "missing";
}

static void source_report_finalize_inventory(yvex_source_report *report,
                                             yvex_source_report_semantics *facts,
                                             int deepseek,
                                             int has_config,
                                             int has_tokenizer) {
    const unsigned long long mib = 1024ull * 1024ull;
    const unsigned long long gib = 1024ull * mib;

    facts->native_inventory_status =
        source_report_header_status(report->source_exists,
                                    report->native_safetensors_count,
                                    report->native_safetensors_header_error_count,
                                    report->native_safetensors_header_read_count);
    facts->native_inventory_source = report->source_exists ? "source-path" : "not-present";
    facts->tensor_metadata_status =
        source_report_header_status(report->source_exists,
                                    report->native_safetensors_count,
                                    report->source_tensor_metadata_error_count,
                                    report->source_tensor_count ||
                                        report->native_safetensors_header_read_count);
    facts->tensor_metadata_source = report->source_exists ? "source-path" : "not-present";
    facts->native_tensor_metadata_status = source_report_native_metadata_status(
        facts->native_inventory_status, report->native_tensor_count);
    facts->native_tensor_payload_status =
        strcmp(facts->native_inventory_status, "missing") == 0 ||
                strcmp(facts->native_inventory_status, "no-safetensors") == 0
            ? "not-present"
            : "not-loaded";
    facts->sidecar_status = !report->source_exists        ? "missing"
                            : has_config && has_tokenizer ? "present"
                            : has_config || has_tokenizer ? "partial"
                                                          : "missing";
    facts->tensor_payload_status =
        !report->source_exists || !report->safetensors_count ? "not-present" : "present-not-loaded";
    facts->target_artifact_status = report->profile->yvex_produced_artifact_status
                                        ? report->profile->yvex_produced_artifact_status
                                        : "planned";
    if (!report->source_exists)
        facts->footprint_class = "missing";
    else if (!report->source_regular_file_count)
        facts->footprint_class = "empty";
    else if (report->total_size_bytes < 100ull * mib)
        facts->footprint_class = "tiny";
    else if (report->total_size_bytes < 5ull * gib)
        facts->footprint_class = "small";
    else if (report->total_size_bytes < 30ull * gib)
        facts->footprint_class = "medium";
    else if (report->total_size_bytes < 200ull * gib)
        facts->footprint_class = "large";
    else
        facts->footprint_class = "huge";
    facts->footprint_status =
        !report->source_exists ? "missing"
        : deepseek ? report->verification.footprint_overflow ? "overflow" : "metadata-verified"
                   : "report-only";
    facts->provenance_origin_normal = report->source_exists ? "local-path" : "planned-official";
    facts->provenance_origin_audit = !report->source_exists   ? "planned-official"
                                     : report->request.source ? "explicit-source-path"
                                                              : "configured-models-root";
    facts->provenance_status = !report->source_exists ? "missing"
                               : deepseek ? yvex_source_verification_status(&report->verification)
                                          : "local-unverified";
    facts->identity_status = !report->source_exists ? "not-present"
                             : deepseek             ? report->verification.repository_verified &&
                                                  report->verification.revision_verified
                                                          ? "verified"
                                                          : "not-verified"
                             : report->source_identity_from_download_sidecar ? "download-sidecar"
                             : report->source_identity_from_path             ? "inferred-from-path"
                                                                             : "not-verified";
    facts->authority = deepseek                ? report->verification.repository_verified
                                                     ? "upstream-repository-manifest"
                                                     : "unverified"
                       : report->source_exists ? "local-unverified"
                                               : "upstream-official-planned";
    facts->authority_status =
        deepseek                ? report->verification.repository_verified ? "verified" : "blocked"
        : report->source_exists ? "local-unverified"
                                : "planned";
}

static void source_report_finalize_manifest(yvex_source_report *report,
                                            yvex_source_report_semantics *facts,
                                            int deepseek) {
    facts->manifest_provenance_status =
        report->manifest_exists ? "manifest-present" : "manifest-missing";
    facts->manifest_authority = deepseek && report->verification.manifest_verified
                                    ? "verifier-owned"
                                : report->manifest_exists ? "local-unverified"
                                                          : "unknown";
    if (!report->manifest_exists)
        facts->manifest_schema_status = "not-checked";
    else if (deepseek)
        facts->manifest_schema_status =
            report->verification.manifest_verified ? "verified" : "blocked";
    else if (report->manifest_probe_error)
        facts->manifest_schema_status = "unreadable";
    else if (report->manifest_schema_matches)
        facts->manifest_schema_status = "matched";
    else if (report->manifest_has_schema)
        facts->manifest_schema_status = "present-unrecognized";
    else
        facts->manifest_schema_status = "not-declared";
    facts->manifest_family_status =
        deepseek ? report->verification.repository_verified ? "matched" : "blocked"
                 : source_report_manifest_match(
                       report, report->manifest_has_family, report->manifest_family_matches);
    facts->manifest_target_status =
        deepseek ? report->verification.manifest_target_id[0] &&
                           strcmp(report->verification.manifest_target_id,
                                  report->identity_target_id) == 0
                       ? "matched"
                       : "blocked"
                 : source_report_manifest_match(
                       report, report->manifest_has_target, report->manifest_target_matches);
    facts->manifest_artifact_class_status =
        source_report_manifest_decl(report, report->manifest_has_artifact_class);
    facts->manifest_footprint_status =
        deepseek ? report->verification.manifest_verified ? "declared" : "blocked"
                 : source_report_manifest_decl(report, report->manifest_has_footprint);
    facts->manifest_native_inventory_status =
        deepseek ? report->verification.manifest_verified ? "declared" : "blocked"
                 : source_report_manifest_decl(report, report->manifest_has_native_inventory);
    facts->manifest_tensor_metadata_status =
        deepseek ? report->verification.manifest_verified ? "declared" : "blocked"
                 : source_report_manifest_decl(report, report->manifest_has_tensor_metadata);
    facts->manifest_consistency_status =
        !report->manifest_exists ? "not-checked"
        : deepseek               ? report->verification.manifest_verified ? "verified" : "blocked"
        : report->manifest_probe_error ? "report-only"
        : report->manifest_schema_matches && report->manifest_family_matches &&
                report->manifest_target_matches
            ? "partial"
            : "report-only";
    facts->manifest_hardening_status =
        deepseek ? report->verification.manifest_verified ? "verified" : "blocked" : "report-only";
    facts->manifest_creation_performed = deepseek && report->verification.manifest_published;
}

static void source_report_finalize_semantics(yvex_source_report *report) {
    yvex_source_report_semantics *facts = &report->semantics;
    int deepseek = report->profile && strcmp(report->profile->family_key, "deepseek") == 0;
    int has_config = report->config_exists || report->generation_config_exists;
    int has_tokenizer = report->tokenizer_json_exists || report->tokenizer_config_exists;

    source_report_finalize_identity(report, facts, deepseek, has_tokenizer);
    source_report_finalize_inventory(report, facts, deepseek, has_config, has_tokenizer);
    source_report_finalize_manifest(report, facts, deepseek);
}

static void source_report_probe_sidecars(const yvex_source_report_request *options,
                                         const yvex_operator_paths *paths,
                                         int deepseek,
                                         yvex_source_report *report) {
    char registry_family_dir[YVEX_PATH_CAP];
    char reports_family_dir[YVEX_PATH_CAP];
    char file_name[192];
    size_t index;

    if (!source_report_path_join(registry_family_dir,
                                 sizeof(registry_family_dir),
                                 paths->registry_root,
                                 options->profile->family_key) ||
        !source_report_path_join(reports_family_dir,
                                 sizeof(reports_family_dir),
                                 paths->reports_root,
                                 options->profile->family_key))
        return;
    for (index = 0u; index < sizeof(source_sidecars) / sizeof(source_sidecars[0]); ++index) {
        const source_sidecar_fact *fact = &source_sidecars[index];
        char *path = (char *)report + fact->path_offset;
        int *exists = (int *)(void *)((unsigned char *)report + fact->exists_offset);

        snprintf(file_name, sizeof(file_name), "%s%s", options->target, fact->suffix);
        (void)source_report_path_join(path,
                                      YVEX_PATH_CAP,
                                      fact->registry_root ? registry_family_dir
                                                          : reports_family_dir,
                                      file_name);
        *exists = path[0] && access(path, F_OK) == 0;
    }
    source_report_probe_map_sidecars(report);
    if (!deepseek && report->download_registry_exists)
        (void)source_report_probe_download_identity_file(
            report->download_registry_path, options->target, options->profile->family_key, report);
    if (!deepseek && !report->source_identity_from_download_sidecar &&
        report->download_report_exists)
        (void)source_report_probe_download_identity_file(
            report->download_report_path, options->target, options->profile->family_key, report);
}

static int source_report_scan_source(const yvex_source_report_request *options,
                                     const yvex_operator_paths *paths,
                                     int deepseek,
                                     yvex_source_report *report,
                                     yvex_error *err) {
    int rc;

    report->source_exists = source_report_stat_kind(report->source_path, 1);
    if (report->source_exists) {
        report->config_exists = source_report_check_file(report->source_path, "config.json");
        report->generation_config_exists =
            source_report_check_file(report->source_path, "generation_config.json");
        report->tokenizer_json_exists =
            source_report_check_file(report->source_path, "tokenizer.json");
        report->tokenizer_config_exists =
            source_report_check_file(report->source_path, "tokenizer_config.json");
        report->readme_exists = source_report_check_file(report->source_path, "README.md");
        report->license_exists = source_report_check_file(report->source_path, "LICENSE") ||
                                 source_report_check_file(report->source_path, "LICENSE.txt") ||
                                 source_report_check_file(report->source_path, "COPYING");
        if (!deepseek)
            return source_report_scan_local(report->source_path, report);
    }
    if (deepseek) {
        yvex_source_verify_options verify_options;

        memset(&verify_options, 0, sizeof(verify_options));
        verify_options.identity = yvex_source_release_identity();
        verify_options.source_path = report->source_path;
        verify_options.models_root = paths->models_root;
        verify_options.promote_manifest = options->strict;
        rc = yvex_source_verify(&verify_options, &report->verification, err);
        if (rc != YVEX_OK)
            return rc;
        source_report_apply_deepseek_verification(report);
    }
    return YVEX_OK;
}

int yvex_source_report_build(const yvex_source_report_request *request,
                             yvex_source_report *report,
                             yvex_error *err) {
    yvex_paths paths;
    yvex_operator_paths operator_paths;
    int rc, n, deepseek;
    unsigned long i;
    const yvex_source_report_request *options = request;

    if (!options || !report || !options->profile) {
        yvex_error_set(err,
                       YVEX_ERR_INVALID_ARG,
                       "source_report_build",
                       "request, profile, and report are required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (!options->target ||
        !source_report_target_is_supported(options->profile, options->target)) {
        if (options->target &&
            strcmp(options->target, YVEX_SOURCE_RETIRED_TARGET_ID) == 0)
            yvex_error_setf(
                err, YVEX_ERR_INVALID_ARG, "source_report_build",
                "unsupported target: %s; use %s",
                YVEX_SOURCE_RETIRED_TARGET_ID, YVEX_SOURCE_RELEASE_TARGET_ID);
        else
            yvex_error_setf(err, YVEX_ERR_INVALID_ARG,
                            "source_report_build", "unsupported target: %s",
                            options->target ? options->target : "");
        return YVEX_ERR_INVALID_ARG;
    }
    deepseek = yvex_source_is_release_target(options->target);

    memset(report, 0, sizeof(*report));
    report->profile = options->profile;
    report->request = *options;
    if (options->target == options->resolved_target) {
        report->request.target = report->request.resolved_target;
    }
    yvex_error_clear(err);
    rc = yvex_paths_default(&paths, err);
    if (rc != YVEX_OK) {
        return rc;
    }
    rc = yvex_operator_paths_resolve(&paths, options->models_root, &operator_paths, err);
    if (rc != YVEX_OK) {
        return rc;
    }

    if (options->source) {
        if (strlen(options->source) >= sizeof(report->source_path)) {
            yvex_error_set(err, YVEX_ERR_BOUNDS, "source_report_build", "source path is too long");
            return YVEX_ERR_BOUNDS;
        }
        yvex_core_text_copy(report->source_path, sizeof(report->source_path), options->source);
        yvex_core_text_copy(report->source_path_source,
                            sizeof(report->source_path_source), "explicit-source");
    } else {
        const yvex_source_target_identity *identity =
            yvex_source_target_identity_find(options->target);
        if (identity) {
            n = yvex_source_target_path(report->source_path,
                                        sizeof(report->source_path),
                                        operator_paths.models_root,
                                        identity)
                    ? 0
                    : -1;
        } else {
            /* Unqualified names do not imply a source location. The caller
             * must select an acquired catalog record or supply a local path. */
            report->source_path[0] = '\0';
            n = 0;
        }
        if (n < 0 || (!deepseek && (size_t)n >= sizeof(report->source_path))) {
            yvex_error_set(err, YVEX_ERR_BOUNDS, "source_report_build", "source path is too long");
            return YVEX_ERR_BOUNDS;
        }
        yvex_core_text_copy(report->source_path_source,
                            sizeof(report->source_path_source),
                            operator_paths.models_root_source);
    }
    yvex_core_text_copy(report->identity_target_id, sizeof(report->identity_target_id),
                        options->target);
    yvex_core_text_copy(report->identity_family, sizeof(report->identity_family),
                        options->profile->family_key);
    if (deepseek) {
        const yvex_source_target_identity *identity = yvex_source_release_identity();

        yvex_core_text_copy(report->identity_model, sizeof(report->identity_model),
                            identity->model_name);
        yvex_core_text_copy(report->identity_repo_id, sizeof(report->identity_repo_id),
                            identity->upstream_repo_id);
    }
    if (options->source) {
        const char *base = yvex_source_path_basename(report->source_path);
        if (base && strcmp(base, options->target) == 0 &&
            yvex_source_target_matches_family_name(options->profile->family_key, base)) {
            report->source_identity_from_path = 1;
        }
    }
    if (!report->identity_model[0] && strcmp(options->target, options->profile->target_id) != 0) {
        yvex_core_text_copy(report->identity_model, sizeof(report->identity_model), options->target);
    }
    source_report_probe_sidecars(options, &operator_paths, deepseek, report);
    rc = source_report_scan_source(options, &operator_paths, deepseek, report, err);
    if (rc != YVEX_OK)
        return rc;

    if (!deepseek) {
        source_report_choose_report_file(report->manifest_path,
                                         sizeof(report->manifest_path),
                                         options->profile,
                                         operator_paths.reports_root,
                                         report->source_path,
                                         options->target,
                                         "manifest",
                                         &report->manifest_exists);
        source_report_choose_report_file(report->native_inventory_path,
                                         sizeof(report->native_inventory_path),
                                         options->profile,
                                         operator_paths.reports_root,
                                         report->source_path,
                                         options->target,
                                         "inventory",
                                         &report->native_inventory_exists);
        source_report_probe_manifest(report, options->profile, options->target);
    }

    report->source_state = report->source_exists ? "present" : "missing";
    if (deepseek) {
        report->status =
            report->verification.verified ? "exact-source-verified" : "exact-source-blocked";
        report->top_blocker =
            report->verification.blocker_count ? report->verification.blockers[0] : "none";
        report->next_row = "V010.REBASE.DEEPSEEK.DSPARK.0";
        for (i = 0; i < report->verification.blocker_count; ++i) {
            source_report_add_blocker(report, report->verification.blockers[i]);
        }
        for (i = 0; i < options->profile->tail_blocker_count; ++i) {
            source_report_add_blocker(report, options->profile->tail_blockers[i]);
        }
        report->exit_code = options->strict && !report->verification.verified ? 5 : 0;
        source_report_finalize_semantics(report);
        yvex_error_clear(err);
        return YVEX_OK;
    }
    report->status = !report->source_exists    ? "source-target-profiled"
                     : !report->manifest_exists ? "source-present-report-only"
                                                : "source-profile-incomplete";
    report->top_blocker = report->manifest_exists ? options->profile->model_class_blocker
                                                   : options->profile->source_manifest_blocker;
    report->next_row = options->profile->model_class_next;
    if (report->source_exists && report->tensor_map_exists && !report->tensor_map_incomplete &&
        report->output_head_map_exists && !report->output_head_map_missing) {
        report->top_blocker = report->tokenizer_map_exists
                                  ? "quant-policy-or-artifact-emitter"
                                  : options->profile->tokenizer_blocker;
        report->next_row = report->tokenizer_map_exists ? "V010.QUANT.1" : "V010.MAP.7";
    }

    source_report_add_blocker(
        report, report->source_exists ? NULL : options->profile->source_path_blocker);
    source_report_add_blocker(
        report, report->manifest_exists ? NULL : options->profile->source_manifest_blocker);
    source_report_add_blocker(
        report, report->config_exists ? NULL : options->profile->source_config_blocker);
    source_report_add_blocker(report,
                              report->tokenizer_json_exists || report->tokenizer_config_exists
                                  ? NULL
                                  : options->profile->tokenizer_blocker);
    for (i = 0; i < options->profile->tail_blocker_count; ++i) {
        if (report->source_exists && report->tensor_map_exists && !report->tensor_map_incomplete &&
            source_report_tail_blocker_is_tensor_map(options->profile->tail_blockers[i])) {
            continue;
        }
        source_report_add_blocker(report, options->profile->tail_blockers[i]);
    }
    source_report_finalize_semantics(report);
    yvex_error_clear(err);
    return YVEX_OK;
}
