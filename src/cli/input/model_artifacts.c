/*
 * Provide bounded model-download and fullmodel command arguments.
 *
 * Parsers perform no artifact IO and call no domain builders. Argument parsing is not artifact
 * emission or runtime support.
 */
#include "src/cli/input/private.h"
#include "src/cli/model_artifacts/private.h"

#include <ctype.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const char *const literal_lines_0[] = {
    "yvex: fullmodel requires report, materialization-plan, materialize, descriptor, or "
    "family-runtime",
    "usage: yvex inspect model full materialization-plan --model FILE_OR_ALIAS [--backend cpu|cuda] "
    "[--residency "
    "resident|host-staged|ssd-staged|hybrid] [--limit-tensors N]",
    "usage: yvex inspect model full materialize --model FILE_OR_ALIAS [--backend cpu|cuda] [--dry-run] "
    "[--plan-"
    "only] [--limit-bytes N]",
    "usage: yvex inspect model full descriptor --model FILE_OR_ALIAS [--backend cpu|cuda] [--target TARGET] "
    "[--"
    "format text] [--limit-tensors N]",
    "usage: yvex inspect model full family-runtime --model FILE_OR_ALIAS [--family auto|deepseek|glm|qwen] "
    "[--backend cpu|cuda]"};

static const char *const literal_lines_1[] = {
    "usage: yvex inspect model full report --model FILE_OR_ALIAS [--backend cpu|cuda] [--target TARGET] "
    "[--limit-tensors N]",
    "usage: yvex inspect model full materialization-plan --model FILE_OR_ALIAS [--backend cpu|cuda] "
    "[--residency "
    "resident|host-staged|ssd-staged|hybrid] [--limit-tensors N]",
    "usage: yvex inspect model full materialize --model FILE_OR_ALIAS [--backend cpu|cuda] [--dry-run] "
    "[--plan-"
    "only] [--limit-bytes N]",
    "usage: yvex inspect model full descriptor --model FILE_OR_ALIAS [--backend cpu|cuda] [--target TARGET] "
    "[--"
    "format text] [--limit-tensors N]",
    "usage: yvex inspect model full family-runtime --model FILE_OR_ALIAS [--family auto|deepseek|glm|qwen] "
    "[--backend cpu|cuda]"};

typedef enum {
    INPUT_OPTION_TEXT = 0,
    INPUT_OPTION_FLAG,
    INPUT_OPTION_CHOICE_TEXT,
    INPUT_OPTION_CHOICE_INT,
    INPUT_OPTION_MIRRORED_TEXT,
    INPUT_OPTION_PATTERN,
    INPUT_OPTION_POSITIVE_U64,
    INPUT_OPTION_OUTPUT,
    INPUT_OPTION_FIXED_INT,
    INPUT_OPTION_REJECT,
    INPUT_OPTION_PROVIDER,
    INPUT_OPTION_SOURCE
} input_option_kind;

typedef enum {
    INPUT_VALUE_STANDARD = 0,
    INPUT_VALUE_NONEMPTY
} input_value_policy;

typedef enum {
    INPUT_ERROR_NONE = 0,
    INPUT_ERROR_FLAG,
    INPUT_ERROR_VALUE
} input_error_detail;

typedef struct {
    const char *name;
    const char *canonical;
    int number;
} input_option_choice;

typedef struct {
    const char *flag;
    input_option_kind kind;
    size_t offset;
    size_t auxiliary_offset;
    const input_option_choice *choices;
    unsigned int command_mask;
    unsigned long long maximum;
    int fixed_value;
    input_value_policy value_policy;
    const char *invalid_error;
    input_error_detail invalid_detail;
    const char *scope_error;
    input_error_detail scope_detail;
} input_option_spec;

enum {
    FULLMODEL_REPORT_MASK = 1u << YVEX_FULLMODEL_COMMAND_REPORT,
    FULLMODEL_PLAN_MASK = 1u << YVEX_FULLMODEL_COMMAND_MATERIALIZATION_PLAN,
    FULLMODEL_MATERIALIZE_MASK = 1u << YVEX_FULLMODEL_COMMAND_MATERIALIZE,
    FULLMODEL_DESCRIPTOR_MASK = 1u << YVEX_FULLMODEL_COMMAND_DESCRIPTOR,
    FULLMODEL_FAMILY_RUNTIME_MASK = 1u << YVEX_FULLMODEL_COMMAND_FAMILY_RUNTIME
};

static const input_option_choice source_choices[] = {{"hf", "hf", 0}, {NULL, NULL, 0}};
static const input_option_choice github_source_choices[] = {
    {"release-asset", "release-asset", 0}, {NULL, NULL, 0}};
static const input_option_choice auth_choices[] = {
    {"auto", NULL, YVEX_MODEL_DOWNLOAD_AUTH_AUTO},
    {"required", NULL, YVEX_MODEL_DOWNLOAD_AUTH_REQUIRED},
    {"never", NULL, YVEX_MODEL_DOWNLOAD_AUTH_NEVER},
    {NULL, NULL, 0}};
static const input_option_choice progress_choices[] = {
    {"auto", NULL, YVEX_MODEL_DOWNLOAD_PROGRESS_AUTO},
    {"live", NULL, YVEX_MODEL_DOWNLOAD_PROGRESS_LIVE},
    {"plain", NULL, YVEX_MODEL_DOWNLOAD_PROGRESS_PLAIN},
    {"log", NULL, YVEX_MODEL_DOWNLOAD_PROGRESS_LOG},
    {"off", NULL, YVEX_MODEL_DOWNLOAD_PROGRESS_OFF},
    {NULL, NULL, 0}};
static const input_option_choice backend_choices[] = {
    {"cpu", "cpu", 0}, {"cuda", "cuda", 0}, {NULL, NULL, 0}};
static const input_option_choice residency_choices[] = {
    {"resident", "resident", 0},
    {"host-staged", "host-staged", 0},
    {"ssd-staged", "ssd-staged", 0},
    {"hybrid", "hybrid", 0},
    {"ssd-streamed", "ssd-streamed", 0},
    {"managed-memory", "managed-memory", 0},
    {"distributed", "distributed", 0},
    {NULL, NULL, 0}};
static const input_option_choice format_choices[] = {
    {"text", "text", 0}, {NULL, NULL, 0}};
static const input_option_choice phase_choices[] = {
    {"preflight", "preflight", 0},
    {"resolve-model", "resolve-model", 0},
    {"artifact-identity", "artifact-identity", 0},
    {"tensor-inventory", "tensor-inventory", 0},
    {"role-coverage", "role-coverage", 0},
    {"placement-plan", "placement-plan", 0},
    {"memory-budget", "memory-budget", 0},
    {"backend-preflight", "backend-preflight", 0},
    {"materialize-embedding", "materialize-embedding", 0},
    {"materialize-normalization", "materialize-normalization", 0},
    {"materialize-attention", "materialize-attention", 0},
    {"materialize-mlp", "materialize-mlp", 0},
    {"materialize-moe", "materialize-moe", 0},
    {"materialize-output", "materialize-output", 0},
    {"materialize-tokenizer", "materialize-tokenizer", 0},
    {"cleanup", "cleanup", 0},
    {"complete", "complete", 0},
    {"failed", "failed", 0},
    {NULL, NULL, 0}};

static const input_option_spec download_options[] = {
    {.flag = "--models-root", INPUT_OPTION_TEXT, offsetof(yvex_cli_models_download_options, models_root)},
    {.flag = "--repo", INPUT_OPTION_TEXT, offsetof(yvex_cli_models_download_options, repo)},
    {.flag = "--family", INPUT_OPTION_TEXT, offsetof(yvex_cli_models_download_options, family)},
    {.flag = "--name", INPUT_OPTION_TEXT, offsetof(yvex_cli_models_download_options, name)},
    {.flag = "--revision", INPUT_OPTION_TEXT, offsetof(yvex_cli_models_download_options, revision)},
    {.flag = "--asset", INPUT_OPTION_TEXT, offsetof(yvex_cli_models_download_options, asset)},
    {.flag = "--asset-name", INPUT_OPTION_TEXT, offsetof(yvex_cli_models_download_options, asset_name)},
    {.flag = "--token-env", INPUT_OPTION_TEXT, offsetof(yvex_cli_models_download_options, token_env)},
    {.flag = "--cli", INPUT_OPTION_TEXT, offsetof(yvex_cli_models_download_options, cli)},
    {.flag = "--dry-run", INPUT_OPTION_FLAG, offsetof(yvex_cli_models_download_options, dry_run)},
    {.flag = "--no-manifest", INPUT_OPTION_FLAG,
     offsetof(yvex_cli_models_download_options, no_manifest)},
    {.flag = "--no-native-inventory", INPUT_OPTION_FLAG,
     offsetof(yvex_cli_models_download_options, no_native_inventory)},
    {.flag = "--force-sidecars", INPUT_OPTION_FLAG,
     offsetof(yvex_cli_models_download_options, force_sidecars)},
    {.flag = "--yes", INPUT_OPTION_FLAG, offsetof(yvex_cli_models_download_options, yes)},
    {.flag = "--clear-stale-locks", INPUT_OPTION_FLAG,
     offsetof(yvex_cli_models_download_options, clear_stale_locks)},
    {.flag = "--force", INPUT_OPTION_FLAG, offsetof(yvex_cli_models_download_options, force)},
    {.flag = "--match-provider-process", INPUT_OPTION_FLAG,
     offsetof(yvex_cli_models_download_options, match_provider_process)},
    {.flag = "--stale-locks", INPUT_OPTION_FLAG,
     offsetof(yvex_cli_models_download_options, cleanup_stale_locks)},
    {.flag = "--logs", INPUT_OPTION_FLAG, offsetof(yvex_cli_models_download_options, cleanup_logs)},
    {.flag = "--receipts", INPUT_OPTION_FLAG,
     offsetof(yvex_cli_models_download_options, cleanup_receipts)},
    {.flag = "--failed-partials", INPUT_OPTION_FLAG,
     offsetof(yvex_cli_models_download_options, cleanup_failed_partials)},
    {.flag = "--all-provider-cache", INPUT_OPTION_FLAG,
     offsetof(yvex_cli_models_download_options, cleanup_all_provider_cache)},
    {.flag = "--source", INPUT_OPTION_SOURCE, offsetof(yvex_cli_models_download_options, source),
     offsetof(yvex_cli_models_download_options, provider), source_choices, 0u, 0ull, 0,
     INPUT_VALUE_STANDARD, "yvex: models download --source supports hf only\n"},
    {.flag = "--provider", INPUT_OPTION_PROVIDER, offsetof(yvex_cli_models_download_options, provider), 0u,
     NULL, 0u, 0ull, 0, INPUT_VALUE_STANDARD,
     "yvex: models download --provider requires hf|huggingface|gh|github\n"},
    {.flag = "--release", INPUT_OPTION_MIRRORED_TEXT,
     offsetof(yvex_cli_models_download_options, release),
     offsetof(yvex_cli_models_download_options, revision)},
    {.flag = "--github-source", INPUT_OPTION_CHOICE_TEXT,
     offsetof(yvex_cli_models_download_options, github_source), 0u, github_source_choices, 0u, 0ull,
     0, INPUT_VALUE_STANDARD,
     "yvex: models download --github-source supports release-asset only\n"},
    {.flag = "--auth", INPUT_OPTION_CHOICE_INT, offsetof(yvex_cli_models_download_options, auth_mode), 0u,
     auth_choices, 0u, 0ull, 0, INPUT_VALUE_STANDARD,
     "yvex: models download --auth requires auto|required|never\n"},
    {.flag = "--include", INPUT_OPTION_PATTERN,
     offsetof(yvex_cli_models_download_options, include_patterns),
     offsetof(yvex_cli_models_download_options, include_count), NULL, 0u,
     YVEX_MODEL_DOWNLOAD_PATTERN_CAP, 0, INPUT_VALUE_STANDARD,
     "yvex: models download too many --include patterns\n"},
    {.flag = "--exclude", INPUT_OPTION_PATTERN,
     offsetof(yvex_cli_models_download_options, exclude_patterns),
     offsetof(yvex_cli_models_download_options, exclude_count), NULL, 0u,
     YVEX_MODEL_DOWNLOAD_PATTERN_CAP, 0, INPUT_VALUE_STANDARD,
     "yvex: models download too many --exclude patterns\n"},
    {.flag = "--max-workers", INPUT_OPTION_POSITIVE_U64,
     offsetof(yvex_cli_models_download_options, max_workers), 0u, NULL, 0u, 0ull, 0,
     INPUT_VALUE_STANDARD,
     "yvex: models download --max-workers requires a positive integer\n"},
    {.flag = "--progress", INPUT_OPTION_CHOICE_INT,
     offsetof(yvex_cli_models_download_options, progress_mode), 0u, progress_choices, 0u, 0ull, 0,
     INPUT_VALUE_STANDARD,
     "yvex: models download --progress requires auto|live|plain|log|off\n"},
    {.flag = "--tick-seconds", INPUT_OPTION_POSITIVE_U64,
     offsetof(yvex_cli_models_download_options, tick_seconds), 0u, NULL, 0u, 0ull, 0,
     INPUT_VALUE_STANDARD,
     "yvex: models download --tick-seconds requires a positive integer\n"},
    {.flag = "--timeout-seconds", INPUT_OPTION_POSITIVE_U64,
     offsetof(yvex_cli_models_download_options, timeout_seconds), 0u, NULL, 0u, 0ull, 0,
     INPUT_VALUE_STANDARD,
     "yvex: models download --timeout-seconds requires a positive integer\n"},
    {.flag = "--output", INPUT_OPTION_OUTPUT, offsetof(yvex_cli_models_download_options, output_mode), 0u,
     NULL, 0u, 0ull, 0, INPUT_VALUE_STANDARD,
     "yvex: models download unsupported output mode: %s\n", INPUT_ERROR_VALUE},
    {.flag = "--audit", INPUT_OPTION_FIXED_INT, offsetof(yvex_cli_models_download_options, output_mode), 0u,
     NULL, 0u, 0ull, YVEX_MODELS_OUTPUT_AUDIT},
    {.flag = "--no-progress", INPUT_OPTION_FIXED_INT,
     offsetof(yvex_cli_models_download_options, progress_mode), 0u, NULL, 0u, 0ull,
     YVEX_MODEL_DOWNLOAD_PROGRESS_OFF},
    {.flag = "--json", INPUT_OPTION_REJECT, 0u, 0u, NULL, 0u, 0ull, 0, INPUT_VALUE_STANDARD,
     "yvex: models download JSON output is unsupported; use --output normal|table|audit\n"},
    {.flag = NULL, INPUT_OPTION_TEXT, 0u}};

static const input_option_spec fullmodel_options[] = {
    {.flag = "--model", INPUT_OPTION_TEXT, offsetof(yvex_cli_fullmodel_options, model)},
    {.flag = "--backend", INPUT_OPTION_CHOICE_TEXT, offsetof(yvex_cli_fullmodel_options, backend), 0u,
     backend_choices, 0u, 0ull, 0, INPUT_VALUE_STANDARD,
     "yvex: fullmodel --backend must be cpu or cuda\n"},
    {.flag = "--target", INPUT_OPTION_TEXT, offsetof(yvex_cli_fullmodel_options, target)},
    {.flag = "--registry", INPUT_OPTION_TEXT, offsetof(yvex_cli_fullmodel_options, registry_path)},
    {.flag = "--family", INPUT_OPTION_TEXT, offsetof(yvex_cli_fullmodel_options, family), 0u, NULL,
     FULLMODEL_FAMILY_RUNTIME_MASK, 0ull, 0, INPUT_VALUE_STANDARD, NULL, INPUT_ERROR_NONE,
     "yvex: fullmodel --family is only valid with family-runtime\n"},
    {.flag = "--residency", INPUT_OPTION_CHOICE_TEXT, offsetof(yvex_cli_fullmodel_options, residency), 0u,
     residency_choices, 0u, 0ull, 0, INPUT_VALUE_STANDARD,
     "yvex: fullmodel --residency must be resident, host-staged, ssd-staged, hybrid, "
     "ssd-streamed, managed-memory, or distributed\n"},
    {.flag = "--require-role", INPUT_OPTION_TEXT, offsetof(yvex_cli_fullmodel_options, require_role), 0u,
     NULL, FULLMODEL_MATERIALIZE_MASK | FULLMODEL_DESCRIPTOR_MASK, 0ull, 0,
     INPUT_VALUE_STANDARD, NULL, INPUT_ERROR_NONE,
     "yvex: fullmodel %s is only valid with materialize or descriptor\n", INPUT_ERROR_FLAG},
    {.flag = "--require-collection", INPUT_OPTION_TEXT,
     offsetof(yvex_cli_fullmodel_options, require_collection), 0u, NULL,
     FULLMODEL_MATERIALIZE_MASK | FULLMODEL_DESCRIPTOR_MASK, 0ull, 0,
     INPUT_VALUE_STANDARD, NULL, INPUT_ERROR_NONE,
     "yvex: fullmodel %s is only valid with materialize or descriptor\n", INPUT_ERROR_FLAG},
    {.flag = "--fail-after-phase", INPUT_OPTION_CHOICE_TEXT,
     offsetof(yvex_cli_fullmodel_options, fail_after_phase), 0u, phase_choices,
     FULLMODEL_MATERIALIZE_MASK, 0ull, 0, INPUT_VALUE_STANDARD,
     "yvex: fullmodel --fail-after-phase value is not a known materialize phase\n",
     INPUT_ERROR_NONE, "yvex: fullmodel %s is only valid with materialize\n", INPUT_ERROR_FLAG},
    {.flag = "--report-dir", INPUT_OPTION_TEXT, offsetof(yvex_cli_fullmodel_options, report_dir), 0u, NULL,
     FULLMODEL_MATERIALIZE_MASK, 0ull, 0, INPUT_VALUE_STANDARD, NULL, INPUT_ERROR_NONE,
     "yvex: fullmodel %s is only valid with materialize\n", INPUT_ERROR_FLAG},
    {.flag = "--format", INPUT_OPTION_CHOICE_TEXT, offsetof(yvex_cli_fullmodel_options, format), 0u,
     format_choices, FULLMODEL_DESCRIPTOR_MASK, 0ull, 0, INPUT_VALUE_STANDARD,
     "yvex: fullmodel descriptor currently supports --format text only\n", INPUT_ERROR_NONE,
     "yvex: fullmodel --format is only valid with descriptor\n"},
    {.flag = "--dry-run", INPUT_OPTION_FLAG, offsetof(yvex_cli_fullmodel_options, dry_run), 0u, NULL,
     FULLMODEL_MATERIALIZE_MASK, 0ull, 0, INPUT_VALUE_STANDARD, NULL, INPUT_ERROR_NONE,
     "yvex: fullmodel %s is only valid with materialize\n", INPUT_ERROR_FLAG},
    {.flag = "--plan-only", INPUT_OPTION_FLAG, offsetof(yvex_cli_fullmodel_options, plan_only), 0u, NULL,
     FULLMODEL_MATERIALIZE_MASK, 0ull, 0, INPUT_VALUE_STANDARD, NULL, INPUT_ERROR_NONE,
     "yvex: fullmodel %s is only valid with materialize\n", INPUT_ERROR_FLAG},
    {.flag = "--include-blockers", INPUT_OPTION_FLAG,
     offsetof(yvex_cli_fullmodel_options, include_blockers), 0u, NULL,
     FULLMODEL_DESCRIPTOR_MASK | FULLMODEL_FAMILY_RUNTIME_MASK, 0ull, 0,
     INPUT_VALUE_STANDARD, NULL, INPUT_ERROR_NONE,
     "yvex: fullmodel %s is only valid with descriptor or family-runtime\n", INPUT_ERROR_FLAG},
    {.flag = "--include-roles", INPUT_OPTION_FLAG, offsetof(yvex_cli_fullmodel_options, include_roles), 0u,
     NULL, FULLMODEL_FAMILY_RUNTIME_MASK, 0ull, 0, INPUT_VALUE_STANDARD, NULL, INPUT_ERROR_NONE,
     "yvex: fullmodel %s is only valid with family-runtime\n", INPUT_ERROR_FLAG},
    {.flag = "--include-placement", INPUT_OPTION_FLAG,
     offsetof(yvex_cli_fullmodel_options, include_placement), 0u, NULL,
     FULLMODEL_DESCRIPTOR_MASK | FULLMODEL_FAMILY_RUNTIME_MASK, 0ull, 0,
     INPUT_VALUE_STANDARD, NULL, INPUT_ERROR_NONE,
     "yvex: fullmodel %s is only valid with descriptor or family-runtime\n", INPUT_ERROR_FLAG},
    {.flag = "--include-graph", INPUT_OPTION_FLAG, offsetof(yvex_cli_fullmodel_options, include_graph), 0u,
     NULL, FULLMODEL_DESCRIPTOR_MASK | FULLMODEL_FAMILY_RUNTIME_MASK, 0ull, 0,
     INPUT_VALUE_STANDARD, NULL, INPUT_ERROR_NONE,
     "yvex: fullmodel %s is only valid with descriptor or family-runtime\n", INPUT_ERROR_FLAG},
    {.flag = "--include-kv", INPUT_OPTION_FLAG, offsetof(yvex_cli_fullmodel_options, include_kv), 0u, NULL,
     FULLMODEL_DESCRIPTOR_MASK | FULLMODEL_FAMILY_RUNTIME_MASK, 0ull, 0,
     INPUT_VALUE_STANDARD, NULL, INPUT_ERROR_NONE,
     "yvex: fullmodel %s is only valid with descriptor or family-runtime\n", INPUT_ERROR_FLAG},
    {.flag = "--include-logits", INPUT_OPTION_FLAG, offsetof(yvex_cli_fullmodel_options, include_logits),
     0u, NULL, FULLMODEL_DESCRIPTOR_MASK | FULLMODEL_FAMILY_RUNTIME_MASK, 0ull, 0,
     INPUT_VALUE_STANDARD, NULL, INPUT_ERROR_NONE,
     "yvex: fullmodel %s is only valid with descriptor or family-runtime\n", INPUT_ERROR_FLAG},
    {.flag = "--include-moe", INPUT_OPTION_FLAG, offsetof(yvex_cli_fullmodel_options, include_moe), 0u,
     NULL, FULLMODEL_FAMILY_RUNTIME_MASK, 0ull, 0, INPUT_VALUE_STANDARD, NULL, INPUT_ERROR_NONE,
     "yvex: fullmodel %s is only valid with family-runtime\n", INPUT_ERROR_FLAG},
    {.flag = "--include-output", INPUT_OPTION_FLAG, offsetof(yvex_cli_fullmodel_options, include_output),
     0u, NULL, FULLMODEL_FAMILY_RUNTIME_MASK, 0ull, 0, INPUT_VALUE_STANDARD, NULL,
     INPUT_ERROR_NONE, "yvex: fullmodel %s is only valid with family-runtime\n", INPUT_ERROR_FLAG},
    {.flag = "--output", INPUT_OPTION_OUTPUT, offsetof(yvex_cli_fullmodel_options, output_mode), 0u, NULL,
     0u, 0ull, 0, INPUT_VALUE_STANDARD, "yvex: fullmodel unsupported output mode: %s\n",
     INPUT_ERROR_VALUE},
    {.flag = "--limit-tensors", INPUT_OPTION_POSITIVE_U64,
     offsetof(yvex_cli_fullmodel_options, limit_tensors), 0u, NULL, 0u, 16ull, 0,
     INPUT_VALUE_NONEMPTY,
     "yvex: fullmodel --limit-tensors requires a positive integer\n"},
    {.flag = "--limit-bytes", INPUT_OPTION_POSITIVE_U64,
     offsetof(yvex_cli_fullmodel_options, limit_bytes),
     offsetof(yvex_cli_fullmodel_options, has_limit_bytes), NULL, FULLMODEL_MATERIALIZE_MASK, 0ull,
     0, INPUT_VALUE_NONEMPTY, "yvex: fullmodel --limit-bytes requires a positive integer\n",
     INPUT_ERROR_NONE, "yvex: fullmodel --limit-bytes is only valid with materialize\n"},
    {.flag = "--audit", INPUT_OPTION_FIXED_INT, offsetof(yvex_cli_fullmodel_options, output_mode), 0u,
     NULL, 0u, 0ull, YVEX_MODELS_OUTPUT_AUDIT},
    {.flag = NULL, INPUT_OPTION_TEXT, 0u}};

int fullmodel_string_is_empty(const char *text) {
    return !text || !text[0];
}

_Static_assert(sizeof(yvex_model_download_auth_mode) == sizeof(int),
               "download auth enum must use int storage");
_Static_assert(sizeof(yvex_model_download_progress_mode) == sizeof(int),
               "download progress enum must use int storage");
_Static_assert(sizeof(yvex_models_output_mode) == sizeof(int),
               "models output enum must use int storage");

static const input_option_spec *input_option_find(const input_option_spec *specs,
                                                  const char *flag) {
    while (specs->flag) {
        if (strcmp(specs->flag, flag) == 0)
            return specs;
        ++specs;
    }
    return NULL;
}

static const input_option_choice *input_option_choice_find(const input_option_choice *choices,
                                                           const char *value) {
    if (!choices)
        return NULL;
    while (choices->name) {
        if (strcmp(choices->name, value) == 0)
            return choices;
        ++choices;
    }
    return NULL;
}

static void input_option_error(const char *message, input_error_detail detail,
                               const input_option_spec *spec, const char *value) {
    if (detail == INPUT_ERROR_FLAG)
        yvex_cli_out_writef(stderr, message, spec->flag);
    else if (detail == INPUT_ERROR_VALUE)
        yvex_cli_out_writef(stderr, message, value);
    else
        yvex_cli_out_fputs(message, stderr);
}

static int input_option_value(const char *command, const input_option_spec *spec, int arg_count,
                              char **args, int *index, const char **value) {
    if (*index + 1 >= arg_count) {
        yvex_cli_out_writef(stderr, "yvex: %s %s requires a value\n", command, spec->flag);
        return 2;
    }
    *value = args[++(*index)];
    if (spec->value_policy == INPUT_VALUE_NONEMPTY ? fullmodel_string_is_empty(*value)
                                                    : !cli_arg_value_valid(*value)) {
        yvex_cli_out_writef(stderr, "yvex: %s %s value is empty%s\n", command, spec->flag,
                            spec->value_policy == INPUT_VALUE_NONEMPTY ? "" : " or invalid");
        return 2;
    }
    return 0;
}

static int input_option_apply(const char *command, const input_option_spec *spec, int option_command,
                              int arg_count, char **args, int *index, void *options) {
    unsigned char *base = options;
    unsigned char *field = base + spec->offset;
    const input_option_choice *choice;
    const char *value = NULL;
    unsigned long long parsed = 0ull;

    if (spec->kind == INPUT_OPTION_REJECT) {
        input_option_error(spec->invalid_error, spec->invalid_detail, spec, NULL);
        return 2;
    }
    if (spec->kind == INPUT_OPTION_FLAG)
        *(int *)field = 1;
    else if (spec->kind == INPUT_OPTION_FIXED_INT)
        memcpy(field, &spec->fixed_value, sizeof(spec->fixed_value));
    else if (input_option_value(command, spec, arg_count, args, index, &value) != 0)
        return 2;

    if (spec->command_mask != 0u &&
        (option_command < 0 || !(spec->command_mask & (1u << (unsigned int)option_command)))) {
        input_option_error(spec->scope_error, spec->scope_detail, spec, value);
        return 2;
    }
    if (spec->kind == INPUT_OPTION_FLAG || spec->kind == INPUT_OPTION_FIXED_INT)
        return 0;
    if (spec->kind == INPUT_OPTION_TEXT) {
        *(const char **)field = value;
        return 0;
    }
    if (spec->kind == INPUT_OPTION_MIRRORED_TEXT) {
        *(const char **)field = value;
        *(const char **)(base + spec->auxiliary_offset) = value;
        return 0;
    }
    if (spec->kind == INPUT_OPTION_PROVIDER) {
        yvex_account_provider provider;
        if (!yvex_account_provider_from_name(value, &provider)) {
            input_option_error(spec->invalid_error, spec->invalid_detail, spec, value);
            return 2;
        }
        *(const char **)field = yvex_account_provider_name(provider);
        return 0;
    }
    choice = input_option_choice_find(spec->choices, value);
    if ((spec->kind == INPUT_OPTION_SOURCE || spec->kind == INPUT_OPTION_CHOICE_TEXT ||
         spec->kind == INPUT_OPTION_CHOICE_INT) &&
        !choice) {
        input_option_error(spec->invalid_error, spec->invalid_detail, spec, value);
        return 2;
    }
    if (spec->kind == INPUT_OPTION_SOURCE) {
        *(const char **)field = choice->canonical;
        if (!*(const char **)(base + spec->auxiliary_offset))
            *(const char **)(base + spec->auxiliary_offset) = "huggingface";
        return 0;
    }
    if (spec->kind == INPUT_OPTION_CHOICE_TEXT) {
        *(const char **)field = choice->canonical;
        return 0;
    }
    if (spec->kind == INPUT_OPTION_CHOICE_INT) {
        memcpy(field, &choice->number, sizeof(choice->number));
        return 0;
    }
    if (spec->kind == INPUT_OPTION_PATTERN) {
        unsigned int *count = (unsigned int *)(base + spec->auxiliary_offset);
        if (*count >= spec->maximum) {
            input_option_error(spec->invalid_error, spec->invalid_detail, spec, value);
            return 2;
        }
        ((const char **)field)[(*count)++] = value;
        return 0;
    }
    if (spec->kind == INPUT_OPTION_POSITIVE_U64) {
        if (!parse_positive_ull(value, &parsed) || parsed == 0ull) {
            input_option_error(spec->invalid_error, spec->invalid_detail, spec, value);
            return 2;
        }
        *(unsigned long long *)field = spec->maximum && parsed > spec->maximum ? spec->maximum
                                                                                : parsed;
        if (spec->auxiliary_offset)
            *(int *)(base + spec->auxiliary_offset) = 1;
        return 0;
    }
    if (spec->kind == INPUT_OPTION_OUTPUT &&
        parse_models_output_mode(value, (yvex_models_output_mode *)field))
        return 0;
    input_option_error(spec->invalid_error, spec->invalid_detail, spec, value);
    return 2;
}

const char *model_download_progress_mode_name(yvex_model_download_progress_mode mode) {
    switch (mode) {
    case YVEX_MODEL_DOWNLOAD_PROGRESS_AUTO:
        return "auto";
    case YVEX_MODEL_DOWNLOAD_PROGRESS_LIVE:
        return "live";
    case YVEX_MODEL_DOWNLOAD_PROGRESS_PLAIN:
        return "plain";
    case YVEX_MODEL_DOWNLOAD_PROGRESS_LOG:
        return "log";
    case YVEX_MODEL_DOWNLOAD_PROGRESS_OFF:
        return "off";
    }
    return "auto";
}

const char *model_download_signal_name(int signo) {
    switch (signo) {
    case SIGINT:
        return "SIGINT";
    case SIGTERM:
        return "SIGTERM";
    case SIGKILL:
        return "SIGKILL";
    case 0:
        return "none";
    }
    return "unknown";
}

yvex_model_download_progress_mode
model_download_effective_progress_mode(yvex_model_download_progress_mode mode) {
    if (mode != YVEX_MODEL_DOWNLOAD_PROGRESS_AUTO) {
        return mode;
    }
    return isatty(STDOUT_FILENO) && isatty(STDERR_FILENO) ? YVEX_MODEL_DOWNLOAD_PROGRESS_LIVE
                                                          : YVEX_MODEL_DOWNLOAD_PROGRESS_PLAIN;
}

const char *model_download_auth_mode_name(yvex_model_download_auth_mode mode) {
    switch (mode) {
    case YVEX_MODEL_DOWNLOAD_AUTH_REQUIRED:
        return "required";
    case YVEX_MODEL_DOWNLOAD_AUTH_NEVER:
        return "never";
    case YVEX_MODEL_DOWNLOAD_AUTH_AUTO:
    default:
        return "auto";
    }
}

static int model_download_options_validate(yvex_cli_models_download_options *options) {
    if (!options->target && !options->repo) {
        yvex_cli_out_writef(stderr, "yvex: models download requires TARGET or --repo OWNER/NAME\n");
        return 2;
    }
    if (options->target && options->repo) {
        yvex_cli_out_writef(stderr,
                            "yvex: models download accepts either TARGET or --repo, not both\n");
        return 2;
    }
    if (!options->provider) {
        options->provider = "huggingface";
    }
    if (options->repo && !model_download_repo_valid(options->repo)) {
        yvex_cli_out_writef(stderr, "yvex: models download --repo requires OWNER/NAME\n");
        return 2;
    }
    if (strcmp(options->provider, "github") == 0 && !options->repo) {
        yvex_cli_out_writef(stderr,
                            "yvex: models download --provider github requires --repo OWNER/NAME\n");
        return 2;
    }
    if (strcmp(options->provider, "github") == 0 && !options->asset) {
        yvex_cli_out_writef(stderr,
                            "yvex: models download --provider github requires --asset GLOB\n");
        return 2;
    }
    if (strcmp(options->provider, "github") == 0 && options->target) {
        yvex_cli_out_writef(
            stderr,
            "yvex: models download catalog targets use Hugging Face provider in this wave\n");
        return 2;
    }
    if (strcmp(options->provider, "github") != 0 && options->repo &&
        (!options->family || !model_download_family_valid(options->family))) {
        yvex_cli_out_writef(
            stderr,
            "yvex: models download --repo requires --family "
            "deepseek|glm|qwen|gemma|minimax-h3\n");
        return 2;
    }
    if (options->repo && !options->name) {
        options->name =
            options->asset_name ? options->asset_name : yvex_source_path_basename(options->repo);
    }
    if (options->repo && !model_download_local_name_valid(options->name)) {
        yvex_cli_out_writef(
            stderr, "yvex: models download --name is required and must be a local model name\n");
        return 2;
    }
    return 0;
}

int parse_models_download_options_from(int arg_count, char **args, int start_index,
                                       yvex_cli_models_download_options *options) {
    int i;

    if (!options)
        return 2;
    memset(options, 0, sizeof(*options));
    options->source = "hf";
    options->revision = "main";
    options->max_workers = 8ull;
    options->auth_mode = YVEX_MODEL_DOWNLOAD_AUTH_AUTO;
    options->output_mode = YVEX_MODELS_OUTPUT_NORMAL;
    options->progress_mode = YVEX_MODEL_DOWNLOAD_PROGRESS_AUTO;
    options->tick_seconds = 2ull;
    options->timeout_seconds = 5ull;

    if (arg_count > start_index &&
        (strcmp(args[start_index], "--help") == 0 || strcmp(args[start_index], "-h") == 0)) {
        return 1;
    }

    for (i = start_index; i < arg_count; ++i) {
        const input_option_spec *spec = input_option_find(download_options, args[i]);

        if (spec) {
            int rc =
                input_option_apply("models download", spec, -1, arg_count, args, &i, options);
            if (rc != 0)
                return rc;
            continue;
        }
        if (args[i][0] == '-') {
            yvex_cli_out_writef(stderr, "yvex: unknown models download option: %s\n", args[i]);
            return 2;
        }
        if (!options->target) {
            options->target = args[i];
            if (!cli_arg_value_valid(options->target)) {
                yvex_cli_out_writef(stderr, "yvex: models download target is empty or invalid\n");
                return 2;
            }
            continue;
        }
        yvex_cli_out_writef(stderr,
                            "yvex: models download received extra positional argument: %s\n",
                            args[i]);
        return 2;
    }

    return model_download_options_validate(options);
}

static int model_catalog_output_parse(const char *value,
                                      yvex_model_catalog_output_mode *mode)
{
    if (!value || !mode) return 0;
    if (strcmp(value, "normal") == 0 || strcmp(value, "table") == 0)
        *mode = YVEX_MODEL_CATALOG_OUTPUT_TABLE;
    else if (strcmp(value, "audit") == 0)
        *mode = YVEX_MODEL_CATALOG_OUTPUT_AUDIT;
    else if (strcmp(value, "json") == 0)
        *mode = YVEX_MODEL_CATALOG_OUTPUT_JSON;
    else
        return 0;
    return 1;
}

static int model_catalog_value(const char *command,
                               const char *flag,
                               int arg_count,
                               char **args,
                               int *index,
                               const char **value)
{
    if (*index + 1 >= arg_count || !cli_arg_value_valid(args[*index + 1])) {
        yvex_cli_out_writef(stderr, "yvex: %s %s requires a value\n", command, flag);
        return 0;
    }
    *value = args[++(*index)];
    return 1;
}

static int model_catalog_provider(const char *value, const char **provider)
{
    yvex_account_provider parsed;

    if (!yvex_account_provider_from_name(value, &parsed) ||
        parsed != YVEX_ACCOUNT_PROVIDER_HUGGINGFACE)
        return 0;
    *provider = yvex_account_provider_name(parsed);
    return 1;
}

int model_search_options_parse(int arg_count,
                               char **args,
                               int start,
                               yvex_cli_model_search_options *options)
{
    int index;

    if (!options) return 2;
    memset(options, 0, sizeof(*options));
    options->provider = "huggingface";
    options->page = 1u;
    options->page_size = 20u;
    options->output_mode = YVEX_MODEL_CATALOG_OUTPUT_TABLE;
    for (index = start; index < arg_count; ++index) {
        const char *value = NULL;
        unsigned long long number;

        if (strcmp(args[index], "--interactive") == 0) {
            options->interactive = 1;
        } else if (strcmp(args[index], "--json") == 0) {
            options->output_mode = YVEX_MODEL_CATALOG_OUTPUT_JSON;
        } else if (strcmp(args[index], "--audit") == 0) {
            options->output_mode = YVEX_MODEL_CATALOG_OUTPUT_AUDIT;
        } else if (strcmp(args[index], "--provider") == 0) {
            if (!model_catalog_value("model search", args[index], arg_count, args, &index,
                                     &value) ||
                !model_catalog_provider(value, &options->provider)) {
                yvex_cli_out_fputs(
                    "yvex: model search --provider currently requires hf|huggingface\n", stderr);
                return 2;
            }
        } else if (strcmp(args[index], "--author") == 0) {
            if (!model_catalog_value("model search", args[index], arg_count, args, &index,
                                     &options->author))
                return 2;
        } else if (strcmp(args[index], "--filter") == 0) {
            if (!model_catalog_value("model search", args[index], arg_count, args, &index,
                                     &options->filter))
                return 2;
        } else if (strcmp(args[index], "--cli") == 0) {
            if (!model_catalog_value("model search", args[index], arg_count, args, &index,
                                     &options->cli))
                return 2;
        } else if (strcmp(args[index], "--models-root") == 0) {
            if (!model_catalog_value("model search", args[index], arg_count, args, &index,
                                     &options->models_root))
                return 2;
        } else if (strcmp(args[index], "--page") == 0 ||
                   strcmp(args[index], "--limit") == 0) {
            const char *flag = args[index];
            if (!model_catalog_value("model search", flag, arg_count, args, &index, &value) ||
                !parse_positive_ull(value, &number) || number == 0ull ||
                (strcmp(flag, "--page") == 0 ? number > 20ull : number > 50ull)) {
                yvex_cli_out_writef(stderr,
                                    "yvex: model search %s requires an integer in range 1..%u\n",
                                    flag, strcmp(flag, "--page") == 0 ? 20u : 50u);
                return 2;
            }
            if (strcmp(flag, "--page") == 0)
                options->page = (unsigned int)number;
            else
                options->page_size = (unsigned int)number;
        } else if (strcmp(args[index], "--output") == 0) {
            if (!model_catalog_value("model search", args[index], arg_count, args, &index,
                                     &value) ||
                !model_catalog_output_parse(value, &options->output_mode)) {
                yvex_cli_out_fputs(
                    "yvex: model search --output requires table|audit|json\n", stderr);
                return 2;
            }
        } else if (args[index][0] == '-') {
            yvex_cli_out_writef(stderr, "yvex: unknown model search option: %s\n", args[index]);
            return 2;
        } else if (!options->query) {
            options->query = args[index];
        } else {
            yvex_cli_out_writef(stderr,
                                "yvex: model search received extra query argument: %s\n",
                                args[index]);
            return 2;
        }
    }
    if (options->interactive && options->output_mode == YVEX_MODEL_CATALOG_OUTPUT_JSON) {
        yvex_cli_out_fputs("yvex: model search --interactive cannot be combined with JSON\n",
                           stderr);
        return 2;
    }
    return 0;
}

int model_remote_inspect_options_parse(int arg_count,
                                       char **args,
                                       int start,
                                       yvex_cli_model_inspect_options *options)
{
    int index;

    if (!options) return 2;
    memset(options, 0, sizeof(*options));
    options->provider = "huggingface";
    options->output_mode = YVEX_MODEL_CATALOG_OUTPUT_TABLE;
    for (index = start; index < arg_count; ++index) {
        const char *value = NULL;
        if (strcmp(args[index], "--json") == 0) {
            options->output_mode = YVEX_MODEL_CATALOG_OUTPUT_JSON;
        } else if (strcmp(args[index], "--audit") == 0) {
            options->output_mode = YVEX_MODEL_CATALOG_OUTPUT_AUDIT;
        } else if (strcmp(args[index], "--provider") == 0) {
            if (!model_catalog_value("model inspect", args[index], arg_count, args, &index,
                                     &value) ||
                !model_catalog_provider(value, &options->provider)) {
                yvex_cli_out_fputs(
                    "yvex: model inspect --provider currently requires hf|huggingface\n", stderr);
                return 2;
            }
        } else if (strcmp(args[index], "--revision") == 0) {
            if (!model_catalog_value("model inspect", args[index], arg_count, args, &index,
                                     &options->revision))
                return 2;
        } else if (strcmp(args[index], "--cli") == 0) {
            if (!model_catalog_value("model inspect", args[index], arg_count, args, &index,
                                     &options->cli))
                return 2;
        } else if (strcmp(args[index], "--models-root") == 0) {
            if (!model_catalog_value("model inspect", args[index], arg_count, args, &index,
                                     &options->models_root))
                return 2;
        } else if (strcmp(args[index], "--output") == 0) {
            if (!model_catalog_value("model inspect", args[index], arg_count, args, &index,
                                     &value) ||
                !model_catalog_output_parse(value, &options->output_mode)) {
                yvex_cli_out_fputs(
                    "yvex: model inspect --output requires table|audit|json\n", stderr);
                return 2;
            }
        } else if (args[index][0] == '-') {
            yvex_cli_out_writef(stderr, "yvex: unknown model inspect option: %s\n", args[index]);
            return 2;
        } else if (!options->repository) {
            options->repository = args[index];
        } else {
            yvex_cli_out_writef(stderr,
                                "yvex: model inspect received extra repository argument: %s\n",
                                args[index]);
            return 2;
        }
    }
    if (!options->repository) {
        yvex_cli_out_fputs("yvex: model inspect requires OWNER/REPOSITORY\n", stderr);
        return 2;
    }
    return 0;
}

int model_local_list_options_parse(int arg_count,
                                   char **args,
                                   int start,
                                   yvex_cli_model_list_options *options)
{
    int index;

    if (!options) return 2;
    memset(options, 0, sizeof(*options));
    options->output_mode = YVEX_MODEL_CATALOG_OUTPUT_TABLE;
    for (index = start; index < arg_count; ++index) {
        const char *value = NULL;
        if (strcmp(args[index], "--json") == 0) {
            options->output_mode = YVEX_MODEL_CATALOG_OUTPUT_JSON;
        } else if (strcmp(args[index], "--audit") == 0) {
            options->output_mode = YVEX_MODEL_CATALOG_OUTPUT_AUDIT;
        } else if (strcmp(args[index], "--models-root") == 0) {
            if (!model_catalog_value("model list", args[index], arg_count, args, &index,
                                     &options->models_root))
                return 2;
        } else if (strcmp(args[index], "--registry") == 0) {
            if (!model_catalog_value("model list", args[index], arg_count, args, &index,
                                     &options->registry_path))
                return 2;
        } else if (strcmp(args[index], "--output") == 0) {
            if (!model_catalog_value("model list", args[index], arg_count, args, &index,
                                     &value) ||
                !model_catalog_output_parse(value, &options->output_mode)) {
                yvex_cli_out_fputs("yvex: model list --output requires table|audit|json\n",
                                   stderr);
                return 2;
            }
        } else {
            yvex_cli_out_writef(stderr, "yvex: unknown model list option: %s\n", args[index]);
            return 2;
        }
    }
    return 0;
}

long long model_download_json_i64_field(const char *text, const char *key) {
    const char *p;

    p = yvex_json_probe_field_value(text, key);
    if (!p)
        return -1;
    return strtoll(p, NULL, 10);
}

static int model_download_identity_family_hint(const char *target, char *family,
                                               size_t family_cap) {
    if (family && family_cap > 0u)
        family[0] = '\0';
    if (!target || !family || family_cap == 0u)
        return 0;
    if (model_download_name_starts_with(target, "qwen")) {
        snprintf(family, family_cap, "qwen");
        return 1;
    }
    if (model_download_name_starts_with(target, "gemma")) {
        snprintf(family, family_cap, "gemma");
        return 1;
    }
    if (model_download_name_starts_with(target, "deepseek")) {
        snprintf(family, family_cap, "deepseek");
        return 1;
    }
    if (model_download_name_starts_with(target, "glm")) {
        snprintf(family, family_cap, "glm");
        return 1;
    }
    return 0;
}

int model_download_identity_paths(const char *target, const char *family,
                                  const yvex_operator_paths *operator_paths,
                                  yvex_model_download_resolved_target *out, yvex_error *err) {
    char reports_family_dir[YVEX_PATH_CAP];
    char registry_family_dir[YVEX_PATH_CAP];
    char file_name[256];
    int rc;

    if (!target || !family || !operator_paths || !out)
        return 0;
    rc = path_join2(reports_family_dir, sizeof(reports_family_dir), operator_paths->reports_root,
                    family, err, "models_download_identity");
    if (rc == YVEX_OK) {
        rc = path_join2(registry_family_dir, sizeof(registry_family_dir),
                        operator_paths->registry_root, family, err, "models_download_identity");
    }
    if (rc != YVEX_OK)
        return 0;

    snprintf(file_name, sizeof(file_name), "%s.download.json", target);
    rc = path_join2(out->registry_path, sizeof(out->registry_path), registry_family_dir, file_name,
                    err, "models_download_identity");
    snprintf(file_name, sizeof(file_name), "%s.download-report.json", target);
    if (rc == YVEX_OK) {
        rc = path_join2(out->download_report_path, sizeof(out->download_report_path),
                        reports_family_dir, file_name, err, "models_download_identity");
    }
    snprintf(file_name, sizeof(file_name), "%s.source-manifest.json", target);
    if (rc == YVEX_OK) {
        rc = path_join2(out->manifest_path, sizeof(out->manifest_path), reports_family_dir,
                        file_name, err, "models_download_identity");
    }
    snprintf(file_name, sizeof(file_name), "%s.native-inventory.json", target);
    if (rc == YVEX_OK) {
        rc = path_join2(out->native_inventory_path, sizeof(out->native_inventory_path),
                        reports_family_dir, file_name, err, "models_download_identity");
    }
    return rc == YVEX_OK;
}

int model_download_read_identity_file(const char *path, const char *target, const char *family,
                                      yvex_model_download_resolved_target *out) {
    char buf[16384];
    char parsed_target[128];
    char parsed_family[32];
    char parsed_repo[256];
    char parsed_provider[32];
    char parsed_revision[128];
    char parsed_source[YVEX_PATH_CAP];

    if (!path || !path[0] || !target || !family || !out)
        return 0;
    if (access(path, F_OK) != 0)
        return 0;
    if (!yvex_core_file_read_text_prefix(path, buf, sizeof(buf)))
        return 0;

    memset(parsed_target, 0, sizeof(parsed_target));
    memset(parsed_family, 0, sizeof(parsed_family));
    memset(parsed_repo, 0, sizeof(parsed_repo));
    memset(parsed_provider, 0, sizeof(parsed_provider));
    memset(parsed_revision, 0, sizeof(parsed_revision));
    memset(parsed_source, 0, sizeof(parsed_source));

    yvex_json_probe_string_field(buf, "target_id", parsed_target, sizeof(parsed_target));
    if (parsed_target[0] && strcmp(parsed_target, target) != 0)
        return 0;
    yvex_json_probe_string_field(buf, "family", parsed_family, sizeof(parsed_family));
    if (parsed_family[0] && strcmp(parsed_family, family) != 0)
        return 0;
    yvex_json_probe_string_field(buf, "repo_id", parsed_repo, sizeof(parsed_repo));
    if (!parsed_repo[0]) {
        yvex_json_probe_string_field(buf, "repo", parsed_repo, sizeof(parsed_repo));
    }
    yvex_json_probe_string_field(buf, "provider", parsed_provider, sizeof(parsed_provider));
    yvex_json_probe_string_field(buf, "revision", parsed_revision, sizeof(parsed_revision));
    yvex_json_probe_string_field(buf, "local_source_dir", parsed_source, sizeof(parsed_source));
    if (!parsed_source[0]) {
        yvex_json_probe_string_field(buf, "path", parsed_source, sizeof(parsed_source));
    }

    snprintf(out->target_id, sizeof(out->target_id), "%s",
             parsed_target[0] ? parsed_target : target);
    snprintf(out->family, sizeof(out->family), "%s", parsed_family[0] ? parsed_family : family);
    snprintf(out->repo_id, sizeof(out->repo_id), "%s", parsed_repo[0] ? parsed_repo : "unknown");
    snprintf(out->provider, sizeof(out->provider), "%s",
             parsed_provider[0] ? parsed_provider : "huggingface");
    snprintf(out->revision, sizeof(out->revision), "%s",
             parsed_revision[0] ? parsed_revision : "main");
    snprintf(out->local_name, sizeof(out->local_name), "%s", target);
    if (parsed_source[0]) {
        snprintf(out->local_source_dir, sizeof(out->local_source_dir), "%s", parsed_source);
    }
    out->found = 1;
    return 1;
}

int model_download_resolve_downloaded_target(const char *target,
                                             const yvex_operator_paths *operator_paths,
                                             yvex_model_download_resolved_target *out,
                                             yvex_error *err) {
    static const char *families[] = {"qwen", "gemma", "deepseek", "glm", "github"};
    char hinted_family[32];
    unsigned long pass;

    if (!out)
        return 0;
    memset(out, 0, sizeof(*out));
    if (!target || !target[0] || !operator_paths)
        return 0;
    model_download_identity_family_hint(target, hinted_family, sizeof(hinted_family));

    for (pass = 0; pass < 2u; ++pass) {
        unsigned long i;
        for (i = 0; i < sizeof(families) / sizeof(families[0]); ++i) {
            const char *family = families[i];
            char hf_family_dir[YVEX_PATH_CAP];
            yvex_model_download_resolved_target candidate;

            if (pass == 0) {
                if (!hinted_family[0] || strcmp(family, hinted_family) != 0) {
                    continue;
                }
            } else if (hinted_family[0] && strcmp(family, hinted_family) == 0) {
                continue;
            }

            memset(&candidate, 0, sizeof(candidate));
            if (!model_download_identity_paths(target, family, operator_paths, &candidate, err)) {
                continue;
            }
            if (model_download_read_identity_file(candidate.registry_path, target, family,
                                                  &candidate) ||
                model_download_read_identity_file(candidate.download_report_path, target, family,
                                                  &candidate) ||
                model_download_read_identity_file(candidate.manifest_path, target, family,
                                                  &candidate)) {
                if (!candidate.local_source_dir[0] && strcmp(candidate.provider, "github") != 0 &&
                    path_join2(hf_family_dir, sizeof(hf_family_dir), operator_paths->hf_root,
                               family, err, "models_download_identity") == YVEX_OK) {
                    (void)path_join2(candidate.local_source_dir, sizeof(candidate.local_source_dir),
                                     hf_family_dir, target, err, "models_download_identity");
                }
                if (!candidate.target_id[0]) {
                    snprintf(candidate.target_id, sizeof(candidate.target_id), "%s", target);
                }
                if (!candidate.family[0]) {
                    snprintf(candidate.family, sizeof(candidate.family), "%s", family);
                }
                if (!candidate.provider[0]) {
                    snprintf(candidate.provider, sizeof(candidate.provider), "huggingface");
                }
                if (!candidate.revision[0]) {
                    snprintf(candidate.revision, sizeof(candidate.revision), "main");
                }
                if (!candidate.local_name[0]) {
                    snprintf(candidate.local_name, sizeof(candidate.local_name), "%s", target);
                }
                *out = candidate;
                out->found = 1;
                return 1;
            }
        }
    }
    return 0;
}

static int fullmodel_options_begin(int arg_count, char **args,
                                   yvex_cli_fullmodel_options *options) {
    memset(options, 0, sizeof(*options));
    options->backend = "cpu";
    options->residency = "resident";
    options->format = "text";
    options->family = "auto";
    options->limit_tensors = 5ull;
    options->output_mode = YVEX_MODELS_OUTPUT_NORMAL;
    options->command = YVEX_FULLMODEL_COMMAND_REPORT;

    if (arg_count >= 3 && (strcmp(args[2], "--help") == 0 || strcmp(args[2], "-h") == 0)) {
        yvex_fullmodel_help(stdout);
        return 1;
    }
    if (arg_count < 3) {
        yvex_cli_out_lines(stderr, literal_lines_0,
                           sizeof(literal_lines_0) / sizeof(literal_lines_0[0]));
        return 2;
    }
    if (strcmp(args[2], "report") == 0) {
        options->command = YVEX_FULLMODEL_COMMAND_REPORT;
    } else if (strcmp(args[2], "materialization-plan") == 0 || strcmp(args[2], "plan") == 0) {
        options->command = YVEX_FULLMODEL_COMMAND_MATERIALIZATION_PLAN;
    } else if (strcmp(args[2], "materialize") == 0) {
        options->command = YVEX_FULLMODEL_COMMAND_MATERIALIZE;
    } else if (strcmp(args[2], "descriptor") == 0) {
        options->command = YVEX_FULLMODEL_COMMAND_DESCRIPTOR;
    } else if (strcmp(args[2], "family-runtime") == 0) {
        options->command = YVEX_FULLMODEL_COMMAND_FAMILY_RUNTIME;
    } else {
        yvex_cli_out_writef(stderr, "yvex: unknown fullmodel subcommand: %s\n", args[2]);
        yvex_cli_out_lines(stderr, literal_lines_1,
                           sizeof(literal_lines_1) / sizeof(literal_lines_1[0]));
        return 2;
    }
    return 0;
}

static int fullmodel_options_finish(const yvex_cli_fullmodel_options *options) {
    const char *name = "report";

    if (options->model)
        return 0;
    if (options->command == YVEX_FULLMODEL_COMMAND_MATERIALIZATION_PLAN) {
        name = "materialization-plan";
    } else if (options->command == YVEX_FULLMODEL_COMMAND_MATERIALIZE) {
        name = "materialize";
    } else if (options->command == YVEX_FULLMODEL_COMMAND_DESCRIPTOR) {
        name = "descriptor";
    } else if (options->command == YVEX_FULLMODEL_COMMAND_FAMILY_RUNTIME) {
        name = "family-runtime";
    }
    yvex_cli_out_writef(stderr, "yvex: fullmodel %s requires --model FILE_OR_ALIAS\n", name);
    return 2;
}

int model_artifacts_fullmodel_options_parse(int arg_count, char **args,
                                            yvex_cli_fullmodel_options *options) {
    int i;
    int begin_rc;

    if (!options)
        return 2;
    begin_rc = fullmodel_options_begin(arg_count, args, options);
    if (begin_rc != 0)
        return begin_rc;

    for (i = 3; i < arg_count; ++i) {
        const input_option_spec *spec = input_option_find(fullmodel_options, args[i]);

        if (!spec) {
            yvex_cli_out_writef(stderr, "yvex: unknown fullmodel option: %s\n", args[i]);
            return 2;
        }
        begin_rc = input_option_apply("fullmodel", spec, options->command, arg_count, args, &i,
                                      options);
        if (begin_rc != 0)
            return begin_rc;
    }
    return fullmodel_options_finish(options);
}
