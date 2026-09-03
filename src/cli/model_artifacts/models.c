/*
 * Provide models namespace routing and registry-backed models commands.
 *
 * CLI-only and excluded from libyvex.a. Registry command output is operator projection, not domain
 * ownership.
 */
#include "src/cli/model_artifacts/private.h"

#include <yvex/artifact.h>
#include <yvex/internal/deployment_compatibility.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    const char *registry_path;
    const char *path;
    const char *alias;
    const char *family;
    const char *model;
    const char *scope;
    const char *artifact_class;
    const char *qprofile;
    const char *calibration;
    const char *sha256;
    const char *support_level;
    const char *runtime_profile;
    const char *runtime_installation;
    const char *runtime_binding;
    const char *runtime_target;
    const char *runtime_backend;
    const char *runtime_execution_strategy;
    const char *runtime_context;
} models_add_options;

typedef struct {
    const char *identity_status;
    const char *metadata_status;
    const char *readiness_status;
    const char *status;
    const char *reason;
    int pass;
    int metadata_checked;
} models_verify_result;

typedef struct {
    const char *name;
    yvex_cli_field_kind kind;
    size_t offset;
} models_verify_pair;

typedef enum {
    VERIFY_AUDIT_REGISTERED = 0,
    VERIFY_AUDIT_IDENTITY,
    VERIFY_AUDIT_RESULT
} models_verify_source;

typedef struct {
    models_verify_source source;
    yvex_cli_field_spec field;
} models_verify_field;

#define REGISTRY_FIELD(key_, kind_, member_, fallback_)                                            \
    {key_, kind_, offsetof(yvex_model_registry_entry, member_), fallback_}

static const yvex_cli_field_spec registry_add_fields[] = {
    REGISTRY_FIELD("alias", YVEX_CLI_FIELD_TEXT, alias, ""),
    REGISTRY_FIELD("path", YVEX_CLI_FIELD_TEXT, path, ""),
    REGISTRY_FIELD("registered_file_size", YVEX_CLI_FIELD_U64, file_size, NULL),
    REGISTRY_FIELD("registered_sha256", YVEX_CLI_FIELD_TEXT, sha256, ""),
    REGISTRY_FIELD("registered_format", YVEX_CLI_FIELD_TEXT, format, ""),
    REGISTRY_FIELD("registered_architecture", YVEX_CLI_FIELD_TEXT, architecture, ""),
    REGISTRY_FIELD("registered_tensor_count", YVEX_CLI_FIELD_U64, tensor_count, NULL),
    REGISTRY_FIELD("registered_known_tensor_bytes", YVEX_CLI_FIELD_U64, known_tensor_bytes, NULL),
    REGISTRY_FIELD("registered_primary_tensor", YVEX_CLI_FIELD_TEXT, primary_tensor_name, ""),
    REGISTRY_FIELD("registered_primary_role", YVEX_CLI_FIELD_TEXT, primary_tensor_role, ""),
    REGISTRY_FIELD("registered_primary_dtype", YVEX_CLI_FIELD_TEXT, primary_tensor_dtype, ""),
    REGISTRY_FIELD("registered_primary_rank", YVEX_CLI_FIELD_U32, primary_tensor_rank, NULL),
    REGISTRY_FIELD("registered_primary_dims", YVEX_CLI_FIELD_TEXT, primary_tensor_dims, ""),
    REGISTRY_FIELD("registered_primary_bytes", YVEX_CLI_FIELD_U64, primary_tensor_bytes, NULL),
    REGISTRY_FIELD("registered_selected_embedding_ready", YVEX_CLI_FIELD_BOOL,
                   selected_embedding_ready, NULL),
    REGISTRY_FIELD("registered_selected_embedding_hidden_size", YVEX_CLI_FIELD_U64,
                   selected_embedding_hidden_size, NULL),
    REGISTRY_FIELD("registered_selected_embedding_vocab_size", YVEX_CLI_FIELD_U64,
                   selected_embedding_vocab_size, NULL),
    REGISTRY_FIELD("registered_selected_embedding_output_count", YVEX_CLI_FIELD_U64,
                   selected_embedding_output_count, NULL),
    REGISTRY_FIELD("registered_selected_embedding_slice_bytes", YVEX_CLI_FIELD_U64,
                   selected_embedding_slice_bytes, NULL),
    REGISTRY_FIELD("runtime_profile", YVEX_CLI_FIELD_TEXT, runtime_profile, ""),
    REGISTRY_FIELD("runtime_installation", YVEX_CLI_FIELD_TEXT, runtime_installation, ""),
    REGISTRY_FIELD("runtime_binding", YVEX_CLI_FIELD_TEXT, runtime_binding, ""),
    REGISTRY_FIELD("runtime_target", YVEX_CLI_FIELD_TEXT, runtime_target, ""),
    REGISTRY_FIELD("runtime_backend", YVEX_CLI_FIELD_TEXT, runtime_backend, ""),
    REGISTRY_FIELD("runtime_engine_kind", YVEX_CLI_FIELD_TEXT,
                   runtime_engine_kind, ""),
    REGISTRY_FIELD("runtime_execution_strategy", YVEX_CLI_FIELD_TEXT,
                   runtime_execution_strategy, ""),
    REGISTRY_FIELD("runtime_context", YVEX_CLI_FIELD_U64, runtime_context, NULL),
};

static const yvex_cli_field_spec registry_inspect_fields[] = {
    REGISTRY_FIELD("alias", YVEX_CLI_FIELD_TEXT, alias, ""),
    REGISTRY_FIELD("path", YVEX_CLI_FIELD_TEXT, path, ""),
    REGISTRY_FIELD("family", YVEX_CLI_FIELD_TEXT, family, ""),
    REGISTRY_FIELD("model", YVEX_CLI_FIELD_TEXT, model, ""),
    REGISTRY_FIELD("scope", YVEX_CLI_FIELD_TEXT, scope, ""),
    REGISTRY_FIELD("artifact_class", YVEX_CLI_FIELD_TEXT, artifact_class, ""),
    REGISTRY_FIELD("qprofile", YVEX_CLI_FIELD_TEXT, qprofile, ""),
    REGISTRY_FIELD("calibration", YVEX_CLI_FIELD_TEXT, calibration, ""),
    REGISTRY_FIELD("artifact_support_level", YVEX_CLI_FIELD_TEXT, support_level, ""),
    REGISTRY_FIELD("registered_file_size", YVEX_CLI_FIELD_U64, file_size, NULL),
    REGISTRY_FIELD("registered_sha256", YVEX_CLI_FIELD_TEXT, sha256, "absent"),
    REGISTRY_FIELD("registered_format", YVEX_CLI_FIELD_TEXT, format, ""),
    REGISTRY_FIELD("registered_architecture", YVEX_CLI_FIELD_TEXT, architecture, ""),
    REGISTRY_FIELD("registered_tensor_count", YVEX_CLI_FIELD_U64, tensor_count, NULL),
    REGISTRY_FIELD("registered_known_tensor_bytes", YVEX_CLI_FIELD_U64, known_tensor_bytes, NULL),
    REGISTRY_FIELD("primary_tensor_name", YVEX_CLI_FIELD_TEXT, primary_tensor_name, ""),
    REGISTRY_FIELD("primary_tensor_role", YVEX_CLI_FIELD_TEXT, primary_tensor_role, ""),
    REGISTRY_FIELD("primary_tensor_dtype", YVEX_CLI_FIELD_TEXT, primary_tensor_dtype, ""),
    REGISTRY_FIELD("primary_tensor_rank", YVEX_CLI_FIELD_U32, primary_tensor_rank, NULL),
    REGISTRY_FIELD("primary_tensor_dims", YVEX_CLI_FIELD_TEXT, primary_tensor_dims, ""),
    REGISTRY_FIELD("primary_tensor_bytes", YVEX_CLI_FIELD_U64, primary_tensor_bytes, NULL),
    REGISTRY_FIELD("selected_embedding_ready", YVEX_CLI_FIELD_BOOL, selected_embedding_ready, NULL),
    REGISTRY_FIELD("selected_embedding_hidden_size", YVEX_CLI_FIELD_U64,
                   selected_embedding_hidden_size, NULL),
    REGISTRY_FIELD("selected_embedding_vocab_size", YVEX_CLI_FIELD_U64,
                   selected_embedding_vocab_size, NULL),
    REGISTRY_FIELD("selected_embedding_output_count", YVEX_CLI_FIELD_U64,
                   selected_embedding_output_count, NULL),
    REGISTRY_FIELD("selected_embedding_slice_bytes", YVEX_CLI_FIELD_U64,
                   selected_embedding_slice_bytes, NULL),
    REGISTRY_FIELD("artifact_execution_ready", YVEX_CLI_FIELD_BOOL, execution_ready, NULL),
    REGISTRY_FIELD("runtime_profile", YVEX_CLI_FIELD_TEXT, runtime_profile, ""),
    REGISTRY_FIELD("runtime_installation", YVEX_CLI_FIELD_TEXT, runtime_installation, ""),
    REGISTRY_FIELD("runtime_binding", YVEX_CLI_FIELD_TEXT, runtime_binding, ""),
    REGISTRY_FIELD("runtime_target", YVEX_CLI_FIELD_TEXT, runtime_target, ""),
    REGISTRY_FIELD("runtime_backend", YVEX_CLI_FIELD_TEXT, runtime_backend, ""),
    REGISTRY_FIELD("runtime_engine_kind", YVEX_CLI_FIELD_TEXT,
                   runtime_engine_kind, ""),
    REGISTRY_FIELD("runtime_execution_strategy", YVEX_CLI_FIELD_TEXT,
                   runtime_execution_strategy, ""),
    REGISTRY_FIELD("runtime_context", YVEX_CLI_FIELD_U64, runtime_context, NULL),
};

static const yvex_model_registry_entry empty_registry_entry = {
    .schema_version = YVEX_MODEL_REGISTRY_ENTRY_SCHEMA_CURRENT,
    .alias = "", .family = "", .model = "", .scope = "", .artifact_class = "",
    .qprofile = "", .calibration = "", .producer = "yvex", .artifact_schema = "v1",
    .path = "", .sha256 = "", .format = "", .architecture = "",
    .primary_tensor_name = "", .primary_tensor_role = "", .primary_tensor_dtype = "",
    .primary_tensor_dims = "", .support_level = "", .runtime_profile = "",
    .runtime_installation = "", .runtime_binding = "",
    .runtime_target = "", .runtime_backend = "", .runtime_engine_kind = "",
    .runtime_execution_strategy = ""
};

static const models_verify_pair verify_audit_pairs[] = {
    {"artifact_support_level", YVEX_CLI_FIELD_TEXT,
     offsetof(yvex_model_registry_entry, support_level)},
    {"architecture", YVEX_CLI_FIELD_TEXT,
     offsetof(yvex_model_registry_entry, architecture)},
    {"tensor_count", YVEX_CLI_FIELD_U64, offsetof(yvex_model_registry_entry, tensor_count)},
    {"known_tensor_bytes", YVEX_CLI_FIELD_U64,
     offsetof(yvex_model_registry_entry, known_tensor_bytes)},
    {"primary_tensor", YVEX_CLI_FIELD_TEXT,
     offsetof(yvex_model_registry_entry, primary_tensor_name)},
    {"primary_role", YVEX_CLI_FIELD_TEXT,
     offsetof(yvex_model_registry_entry, primary_tensor_role)},
    {"primary_dtype", YVEX_CLI_FIELD_TEXT,
     offsetof(yvex_model_registry_entry, primary_tensor_dtype)},
    {"primary_rank", YVEX_CLI_FIELD_U32,
     offsetof(yvex_model_registry_entry, primary_tensor_rank)},
    {"primary_dims", YVEX_CLI_FIELD_TEXT,
     offsetof(yvex_model_registry_entry, primary_tensor_dims)},
    {"primary_bytes", YVEX_CLI_FIELD_U64,
     offsetof(yvex_model_registry_entry, primary_tensor_bytes)},
    {"selected_embedding_ready", YVEX_CLI_FIELD_BOOL,
     offsetof(yvex_model_registry_entry, selected_embedding_ready)},
    {"selected_embedding_hidden_size", YVEX_CLI_FIELD_U64,
     offsetof(yvex_model_registry_entry, selected_embedding_hidden_size)},
    {"selected_embedding_vocab_size", YVEX_CLI_FIELD_U64,
     offsetof(yvex_model_registry_entry, selected_embedding_vocab_size)},
    {"selected_embedding_output_count", YVEX_CLI_FIELD_U64,
     offsetof(yvex_model_registry_entry, selected_embedding_output_count)},
    {"selected_embedding_slice_bytes", YVEX_CLI_FIELD_U64,
     offsetof(yvex_model_registry_entry, selected_embedding_slice_bytes)},
};

static const models_verify_field verify_audit_head[] = {
    {VERIFY_AUDIT_REGISTERED,
     {"alias", YVEX_CLI_FIELD_TEXT, offsetof(yvex_model_registry_entry, alias), ""}},
    {VERIFY_AUDIT_REGISTERED,
     {"path", YVEX_CLI_FIELD_TEXT, offsetof(yvex_model_registry_entry, path), ""}},
    {VERIFY_AUDIT_REGISTERED,
     {"registered_sha256", YVEX_CLI_FIELD_TEXT,
      offsetof(yvex_model_registry_entry, sha256), "absent"}},
    {VERIFY_AUDIT_IDENTITY,
     {"current_sha256", YVEX_CLI_FIELD_TEXT_ARRAY,
      offsetof(yvex_artifact_file_identity, sha256), "unavailable"}},
    {VERIFY_AUDIT_REGISTERED,
     {"registered_file_size", YVEX_CLI_FIELD_U64,
      offsetof(yvex_model_registry_entry, file_size), NULL}},
    {VERIFY_AUDIT_IDENTITY,
     {"current_file_size", YVEX_CLI_FIELD_U64,
      offsetof(yvex_artifact_file_identity, file_size), NULL}},
    {VERIFY_AUDIT_RESULT,
     {"digest_status", YVEX_CLI_FIELD_TEXT,
      offsetof(models_verify_result, identity_status), "unknown"}},
    {VERIFY_AUDIT_RESULT,
     {"identity_status", YVEX_CLI_FIELD_TEXT,
      offsetof(models_verify_result, identity_status), "unknown"}},
};

#undef REGISTRY_FIELD

static const char *const literal_pair_1[] = { "gguf:",
    "  status: unavailable"};

static const char *const literal_pair_4[] = { "identity_status: recorded",
    "status: models-added"};

static const char *const models_help_lines[] = {
    "usage: yvex model search [QUERY] [--author NAME] [--filter TAG] [--page N] [--limit N | --all] "
        "[--interactive] [--output table|audit|json]",
    "       yvex source inspect OWNER/NAME [--revision REVISION] [--output table|audit|json]",
    "usage: yvex profile scan --root DIR [--registry FILE]",
    "       yvex profile create --path FILE [--alias ALIAS] [--support-level LEVEL] "
        "[--startup-profile single-artifact --runtime-binding FILE --target ID "
        "--backend cpu|cuda --execution-strategy target-only|speculative --ctx N] "
        "[--startup-profile composite --installation-root DIR --target ID --backend cuda "
        "] [--registry FILE]",
    "       yvex source acquire TARGET [--models-root DIR] [--auth auto|required|never] [--dry-run] "
        "[--progress auto|live|plain|log|off] [--tick-seconds N] [--no-progress] [--audit | --output "
        "normal|table|audit]",
    "       yvex source status TARGET [--models-root DIR] [--audit | --output "
        "normal|table|audit]",
    "       yvex source stop TARGET [--models-root DIR] [--force] [--timeout-seconds N] "
        "[--match-provider-process] [--dry-run] [--audit]",
    "       yvex source resume TARGET [--models-root DIR] [--auth auto|required|never] "
        "[--progress auto|live|plain|log|off] [--tick-seconds N] [--clear-stale-locks] [--audit]",
    "       yvex source cleanup TARGET [--models-root DIR] [--stale-locks] [--logs] "
        "[--receipts] [--failed-partials] [--all-provider-cache] [--dry-run] [--yes] [--audit]",
    "       yvex source acquire --repo OWNER/NAME --family deepseek|glm|qwen|gemma|minimax-h3 "
        "[--name LOCAL_NAME] [--revision REVISION] [--include GLOB ...] [--exclude GLOB ...] "
        "[--models-root DIR] [--auth auto|required|never] [--progress auto|live|plain|log|off]",
    "       yvex source acquire --provider github --repo OWNER/NAME [--release TAG] --asset GLOB "
        "[--models-root DIR] [--auth auto|required|never] [--progress auto|live|plain|log|off]",
    "       yvex inspect artifact registry [--models-root DIR] [--family deepseek|glm|qwen|gemma] "
        "[--output normal|table|audit|json]",
    "       yvex inspect artifact status TARGET [--models-root DIR] [--audit | --output "
        "normal|table|audit|json]",
    "       yvex compile TARGET [--overwrite] [--source DIR] [--out FILE | --out-dir DIR] "
        "[--models-root DIR] [--registry FILE] [--dry-run] [--no-register] "
        "[--audit | --output normal|table|audit]",
    "       yvex artifact status TARGET [--backend cpu|cuda] [--level quick|runtime|full] "
        "[--models-root DIR] [--registry FILE] [--report-dir DIR] [--no-materialize] [--no-graph] "
        "[--audit | --output normal|table|audit]",
    "       yvex model list [--models-root DIR] [--registry FILE] "
        "[--output table|audit|json]",
    "       yvex profile verify|show ALIAS [--registry FILE] [--audit | --output normal|table|audit]",
    "       yvex profile remove ALIAS [--registry FILE]",
    "\nExamples:",
    "  yvex model search \"MiniMax H3\" --output table",
    "  yvex source inspect MiniMaxAI/MiniMax-H3 --output table",
    "  yvex source acquire --repo OWNER/GGUF_REPOSITORY --family FAMILY --name LOCAL_NAME "
        "--revision REVISION --include 'selected-file.gguf' --models-root ~/lab/models",
    "  yvex artifact status deepseek4-v4-flash-dspark-selected-embed",
    "  yvex source acquire gemma-4-12b-it --models-root ~/lab/models --dry-run --audit",
    "  yvex source status gemma-4-12b-it --models-root ~/lab/models --audit",
    "  yvex source stop gemma-4-12b-it --models-root ~/lab/models --audit",
    "  yvex source resume gemma-4-12b-it --models-root ~/lab/models --auth required "
        "--progress live --tick-seconds 2 --audit",
    "  yvex source cleanup gemma-4-12b-it --models-root ~/lab/models --stale-locks "
        "--dry-run --audit",
    "  yvex source acquire gemma-4-12b-it --models-root ~/lab/models --auth required "
        "--progress live --tick-seconds 2 --audit",
    "  yvex source acquire qwen3-8b --models-root ~/lab/models --auth auto --audit",
    "  yvex source status qwen3-32b --models-root ~/lab/models",
    "  yvex inspect artifact registry --models-root ~/lab/models --output table",
    "  yvex inspect artifact status qwen3-6-35b-a3b --models-root ~/lab/models --audit",
    "  yvex source acquire --provider github --repo OWNER/REPO --release TAG --asset \"*.gguf\" "
        "--models-root ~/lab/models --auth auto --audit",
    "  yvex artifact status deepseek4-v4-flash-dspark-selected-embed --backend cpu --level runtime",
    "  yvex artifact status deepseek4-v4-flash-dspark-selected-embed --backend cuda --level runtime --no-graph",
    "  yvex artifact status deepseek4-v4-flash-dspark-selected-embed --level full --report-dir build/reports",
    "\nModels separates remote provider discovery, physical representations, acquired sources, admitted "
        "packages, the local catalog, and live engines. Search and remote inspect normalize Hugging Face "
        "metadata without downloading payloads. Download uses the local accounts/provider preflight for "
        "Hugging Face and GitHub provider CLIs, writes source intake reports only, and does not register "
        "runtime artifacts. "
        "Artifacts list/status reads operator paths, GGUF filenames, and source sidecars only; it does not "
        "hash files, load tensor payloads, emit GGUF, materialize, execute runtime paths, generate, evaluate, "
        "or benchmark. Prepare currently supports deepseek4-v4-flash-dspark-selected-embed only and does not "
        "materialize, run graph execution, decode, logits, sampling, generation, evaluation, or benchmarks.",
    "Default report output is compact. Use --audit for full diagnostic fields.",
    "Check composes implemented artifact, identity, integrity, selected materialization, "
        "engine/session, plan, selected graph, and selected gates only; it does not create artifacts, run "
        "source conversion, run prefill, decode, produce logits, sample, generate, evaluate, or benchmark."
};

static int parse_models_registry_options(int arg_count,
                                         char **args,
                                         int start,
                                         const char **registry_path,
                                         yvex_models_output_mode *output_mode)
{
    int i;

    if (output_mode) {
        *output_mode = YVEX_MODELS_OUTPUT_NORMAL;
    }
    for (i = start; i < arg_count; ++i) {
        if (strcmp(args[i], "--registry") == 0) {
            if (i + 1 >= arg_count) {
                yvex_cli_out_writef(stderr, "yvex: models --registry requires a file\n");
                return 2;
            }
            *registry_path = args[++i];
        } else if (output_mode && strcmp(args[i], "--audit") == 0) {
            *output_mode = YVEX_MODELS_OUTPUT_AUDIT;
        } else if (output_mode && strcmp(args[i], "--output") == 0) {
            if (i + 1 >= arg_count) {
                yvex_cli_out_writef(stderr,
                                    "yvex: models --output requires normal|table|audit|json\n");
                return 2;
            }
            if (!parse_models_output_mode(args[++i], output_mode)) {
                yvex_cli_out_writef(stderr, "yvex: models unsupported output mode: %s\n", args[i]);
                return 2;
            }
        } else if (output_mode && strcmp(args[i], "--json") == 0) {
            *output_mode = YVEX_MODELS_OUTPUT_JSON;
        } else if (strcmp(args[i], "--json") != 0) {
            yvex_cli_out_writef(stderr, "yvex: unknown models option: %s\n", args[i]);
            return 2;
        }
    }
    return 0;
}

static int parse_models_add_options(int arg_count, char **args,
                                    models_add_options *options)
{
    int i;

    memset(options, 0, sizeof(*options));
    for (i = 3; i < arg_count; ++i) {
        if (i + 1 >= arg_count) {
            yvex_cli_out_writef(stderr, "yvex: models add option requires a value: %s\n", args[i]);
            return 2;
        }
        if (strcmp(args[i], "--registry") == 0) options->registry_path = args[++i];
        else if (strcmp(args[i], "--path") == 0) options->path = args[++i];
        else if (strcmp(args[i], "--alias") == 0) options->alias = args[++i];
        else if (strcmp(args[i], "--family") == 0) options->family = args[++i];
        else if (strcmp(args[i], "--model") == 0) options->model = args[++i];
        else if (strcmp(args[i], "--scope") == 0) options->scope = args[++i];
        else if (strcmp(args[i], "--class") == 0) options->artifact_class = args[++i];
        else if (strcmp(args[i], "--qprofile") == 0) options->qprofile = args[++i];
        else if (strcmp(args[i], "--calibration") == 0) options->calibration = args[++i];
        else if (strcmp(args[i], "--sha256") == 0) options->sha256 = args[++i];
        else if (strcmp(args[i], "--support-level") == 0) options->support_level = args[++i];
        else if (strcmp(args[i], "--startup-profile") == 0)
            options->runtime_profile = args[++i];
        else if (strcmp(args[i], "--installation-root") == 0)
            options->runtime_installation = args[++i];
        else if (strcmp(args[i], "--runtime-binding") == 0) options->runtime_binding = args[++i];
        else if (strcmp(args[i], "--target") == 0) options->runtime_target = args[++i];
        else if (strcmp(args[i], "--backend") == 0) options->runtime_backend = args[++i];
        else if (strcmp(args[i], "--execution-strategy") == 0)
            options->runtime_execution_strategy = args[++i];
        else if (strcmp(args[i], "--generation-mode") == 0) {
            yvex_cli_out_writef(
                stderr,
                "yvex: --generation-mode is retired; use --execution-strategy "
                "target-only|speculative (composite media profiles need neither)\n");
            return 2;
        }
        else if (strcmp(args[i], "--ctx") == 0) options->runtime_context = args[++i];
        else {
            yvex_cli_out_writef(stderr, "yvex: unknown models add option: %s\n", args[i]);
            return 2;
        }
    }
    return 0;
}

static int command_models_scan(int arg_count, char **args)
{
    yvex_model_registry_entry *entries = NULL;
    yvex_error err;
    const char *root = NULL;
    const char *registry_path = NULL;
    unsigned long long count = 0;
    unsigned long long i;
    int rc;

    yvex_error_clear(&err);
    for (i = 3; (int)i < arg_count; ++i) {
        if (strcmp(args[i], "--root") == 0) {
            if ((int)i + 1 >= arg_count) {
                yvex_cli_out_writef(stderr, "yvex: models scan --root requires a directory\n");
                return 2;
            }
            root = args[++i];
        } else if (strcmp(args[i], "--registry") == 0) {
            if ((int)i + 1 >= arg_count) {
                yvex_cli_out_writef(stderr, "yvex: models scan --registry requires a file\n");
                return 2;
            }
            registry_path = args[++i];
        } else if (strcmp(args[i], "--json") == 0) {
            /* Reserved for the registry scan JSON projection; text remains canonical. */
        } else {
            yvex_cli_out_writef(stderr, "yvex: unknown models scan option: %s\n", args[i]);
            return 2;
        }
    }
    (void)registry_path;
    if (!root) {
        yvex_cli_out_writef(stderr, "yvex: models scan requires --root DIR\n");
        return 2;
    }
    rc = yvex_model_registry_scan_root(root, &entries, &count, &err);
    if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));
    yvex_cli_out_writef(stdout, "models: scan\n");
    yvex_cli_out_writef(stdout, "root: %s\n", root);
    for (i = 0; i < count; ++i) {
        if (i > 0) yvex_cli_out_writef(stdout, "\n");
        print_model_registry_scan_entry_cli(&entries[i]);
    }
    yvex_cli_out_writef(stdout, "candidates: %llu\n", count);
    yvex_cli_out_writef(stdout, "status: models-scan\n");
    yvex_model_registry_scan_free(entries, count);
    return 0;
}

int yvex_model_profile_create_adapter(int arg_count, char **args,
                                      int render_result, int replace_existing)
{
    models_add_options cli_options;
    yvex_model_registry *registry = NULL;
    yvex_model_registry_entry derived = empty_registry_entry;
    yvex_model_registry_entry entry = empty_registry_entry;
    yvex_error err;
    char registered_sha256[YVEX_SHA256_HEX_CAP] = {0};
    char registered_format[16] = {0};
    char registered_architecture[64] = {0};
    char primary_tensor_name[128] = {0};
    char primary_tensor_role[64] = {0};
    char primary_tensor_dtype[32] = {0};
    char primary_tensor_dims[128] = {0};
    unsigned int single_fields = 0u, composite_fields = 0u;
    char *context_end = NULL;
    int have_derived = 0;
    int rc;

    yvex_error_clear(&err);
    rc = parse_models_add_options(arg_count, args, &cli_options);
    if (rc != 0) return rc;
    if (!cli_options.path) {
        yvex_cli_out_writef(stderr, "yvex: models add requires --path FILE\n");
        return 2;
    }
    if (yvex_model_registry_entry_derive_from_path(&derived, cli_options.path, &err) == YVEX_OK) {
        have_derived = 1;
    } else {
        yvex_error_clear(&err);
    }
    if (!cli_options.alias && !have_derived) {
        yvex_cli_out_writef(stderr, "yvex: models add requires --alias when filename is not canonical\n");
        return 2;
    }
    entry = have_derived ? derived : empty_registry_entry;
    entry.alias = cli_options.alias ? cli_options.alias : entry.alias;
    entry.family = cli_options.family ? cli_options.family : entry.family;
    entry.model = cli_options.model ? cli_options.model : entry.model;
    entry.scope = cli_options.scope ? cli_options.scope : entry.scope;
    entry.artifact_class = cli_options.artifact_class
        ? cli_options.artifact_class
        : entry.artifact_class;
    entry.qprofile = cli_options.qprofile ? cli_options.qprofile : entry.qprofile;
    entry.calibration = cli_options.calibration ? cli_options.calibration : entry.calibration;
    entry.path = cli_options.path;
    entry.support_level = cli_options.support_level ? cli_options.support_level : "";
    single_fields += cli_options.runtime_binding != NULL;
    single_fields += cli_options.runtime_target != NULL;
    single_fields += cli_options.runtime_backend != NULL;
    single_fields += cli_options.runtime_execution_strategy != NULL;
    single_fields += cli_options.runtime_context != NULL;
    composite_fields += cli_options.runtime_installation != NULL;
    composite_fields += cli_options.runtime_target != NULL;
    composite_fields += cli_options.runtime_backend != NULL;
    if (cli_options.runtime_profile &&
        strcmp(cli_options.runtime_profile, "single-artifact") != 0 &&
        strcmp(cli_options.runtime_profile, "composite") != 0) {
        yvex_cli_out_writef(stderr,
            "yvex: --startup-profile must be single-artifact or composite\n");
        return 2;
    }
    if (cli_options.runtime_profile &&
        strcmp(cli_options.runtime_profile, "composite") == 0) {
        if (composite_fields != 3u || cli_options.runtime_binding ||
            cli_options.runtime_context || cli_options.runtime_execution_strategy) {
            yvex_cli_out_writef(stderr,
                "yvex: a composite startup profile requires --installation-root, --target, "
                "and --backend without --execution-strategy, --runtime-binding, or --ctx\n");
            return 2;
        }
        entry.runtime_profile = "composite";
        entry.runtime_installation = cli_options.runtime_installation;
        entry.runtime_target = cli_options.runtime_target;
        entry.runtime_backend = cli_options.runtime_backend;
        entry.runtime_engine_kind = "media";
        entry.runtime_execution_strategy = "not-applicable";
    } else if (single_fields != 0u || cli_options.runtime_profile) {
        if (single_fields != 5u || cli_options.runtime_installation) {
            yvex_cli_out_writef(stderr,
                "yvex: a startup profile requires --runtime-binding, --target, --backend, "
                "--execution-strategy, and --ctx together\n");
            return 2;
        }
        errno = 0;
        entry.runtime_context = strtoull(cli_options.runtime_context, &context_end, 10);
        if (errno || !context_end || *context_end || entry.runtime_context == 0ull) {
            yvex_cli_out_writef(stderr, "yvex: model registry add --ctx requires a positive integer\n");
            return 2;
        }
        entry.runtime_profile = "single-artifact";
        entry.runtime_binding = cli_options.runtime_binding;
        entry.runtime_target = cli_options.runtime_target;
        entry.runtime_backend = cli_options.runtime_backend;
        entry.runtime_engine_kind = "text";
        entry.runtime_execution_strategy = cli_options.runtime_execution_strategy;
    }

    rc = populate_registry_identity(&entry,
                                    registered_sha256,
                                    registered_format,
                                    registered_architecture,
                                    primary_tensor_name,
                                    primary_tensor_role,
                                    primary_tensor_dtype,
                                    primary_tensor_dims,
                                    &err);
    if (rc != YVEX_OK) {
        return print_yvex_error(&err, exit_for_status(rc));
    }
    if (cli_options.sha256 && cli_options.sha256[0] &&
        strcmp(cli_options.sha256, registered_sha256) != 0) {
        yvex_error_setf(&err, YVEX_ERR_STATE, "models_add_identity",
                        "sha256 mismatch: expected %s got %s",
                        cli_options.sha256, registered_sha256);
        return print_yvex_error(&err, exit_for_status(YVEX_ERR_STATE));
    }
    if ((entry.runtime_profile && entry.runtime_profile[0]) || single_fields != 0u) {
        rc = yvex_model_registry_startup_validate(&entry, &err);
        if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));
    }

    rc = models_registry_open(&registry, cli_options.registry_path, 1, &err);
    if (rc == YVEX_OK && replace_existing &&
        yvex_model_registry_find(registry, entry.alias))
        rc = yvex_model_registry_remove(registry, entry.alias, &err);
    if (rc == YVEX_OK) rc = yvex_model_registry_add(registry, &entry, &err);
    if (rc == YVEX_OK) rc = yvex_model_registry_save(registry, cli_options.registry_path, &err);
    if (rc != YVEX_OK) {
        yvex_model_registry_close(registry);
        return print_yvex_error(&err, exit_for_status(rc));
    }
    if (render_result) {
        yvex_cli_out_writef(stdout, "models: add\n");
        (void)yvex_cli_out_fields(
            stdout, &entry, registry_add_fields,
            sizeof(registry_add_fields) / sizeof(registry_add_fields[0]));
        yvex_cli_out_lines(stdout, literal_pair_4,
                           sizeof(literal_pair_4) / sizeof(literal_pair_4[0]));
    }
    yvex_model_registry_close(registry);
    return 0;
}

static int command_models_add(int arg_count, char **args)
{
    return yvex_model_profile_create_adapter(arg_count, args, 1, 0);
}

static int command_models_remote(int arg_count, char **args)
{
    yvex_cli_model_inspect_options cli;
    yvex_remote_inspect_options options;
    yvex_local_catalog_options local_options;
    yvex_remote_catalog *catalog = NULL;
    yvex_local_catalog *local_catalog = NULL;
    yvex_error err;
    yvex_account_provider provider;
    int rc;

    rc = model_remote_inspect_options_parse(arg_count, args, 3, &cli);
    if (rc != 0) return rc;
    if (!yvex_account_provider_from_name(cli.provider, &provider)) return 2;
    memset(&options, 0, sizeof(options));
    options.provider = provider;
    options.repository = cli.repository;
    options.revision = cli.revision;
    yvex_error_clear(&err);
    rc = yvex_remote_model_inspect(&catalog, &options, &err);
    memset(&local_options, 0, sizeof(local_options));
    local_options.models_root = cli.models_root;
    if (rc == YVEX_OK)
        rc = yvex_local_catalog_open(&local_catalog, &local_options, &err);
    if (rc != YVEX_OK) {
        yvex_remote_catalog_close(catalog);
        return print_yvex_error(&err, exit_for_status(rc));
    }
    rc = yvex_remote_catalog_render(stdout, catalog, local_catalog,
                                    cli.output_mode, 1);
    yvex_local_catalog_close(local_catalog);
    yvex_remote_catalog_close(catalog);
    if (rc != YVEX_OK) {
        yvex_error_set(&err, rc, "model_inspect", "cannot render remote model catalog");
        return print_yvex_error(&err, exit_for_status(rc));
    }
    return 0;
}

static int command_models_search(int arg_count, char **args)
{
    yvex_cli_model_search_options cli;
    yvex_remote_search_options options;
    yvex_local_catalog_options local_options;
    yvex_remote_catalog *catalog = NULL;
    yvex_local_catalog *local_catalog = NULL;
    yvex_error err;
    yvex_account_provider provider;
    int rc;

    rc = model_search_options_parse(arg_count, args, 3, &cli);
    if (rc != 0) return rc;
    if (!strcmp(cli.provider, "local")) {
        if (cli.author || cli.filter) {
            yvex_cli_out_fputs(
                "yvex: local model search supports QUERY, paging, and output options only\n",
                stderr);
            return 2;
        }
        return yvex_model_catalog_search_local(&cli);
    }
    if (!yvex_account_provider_from_name(cli.provider, &provider)) return 2;
    memset(&options, 0, sizeof(options));
    options.provider = provider;
    options.query = cli.query;
    options.author = cli.author;
    options.filter = cli.filter;
    options.page = cli.page;
    options.page_size = cli.page_size;
    yvex_error_clear(&err);
    rc = yvex_remote_model_search(&catalog, &options, &err);
    memset(&local_options, 0, sizeof(local_options));
    local_options.models_root = cli.models_root;
    if (rc == YVEX_OK)
        rc = yvex_local_catalog_open(&local_catalog, &local_options, &err);
    if (rc != YVEX_OK) {
        yvex_remote_catalog_close(catalog);
        return print_yvex_error(&err, exit_for_status(rc));
    }
    rc = yvex_remote_catalog_render(stdout, catalog, local_catalog,
                                    cli.output_mode, 0);
    yvex_local_catalog_close(local_catalog);
    yvex_remote_catalog_close(catalog);
    if (rc != YVEX_OK) {
        yvex_error_set(&err, rc, "model_search", "cannot render remote model catalog");
        return print_yvex_error(&err, exit_for_status(rc));
    }
    return 0;
}

static int library_open_cli(int arg_count, char **args, int option_start,
                            yvex_model_library **library,
                            yvex_model_catalog_output_mode *mode)
{
    yvex_cli_model_list_options cli;
    yvex_local_catalog_options options = {0};
    yvex_error err;
    int rc = model_local_list_options_parse(
        arg_count, args, option_start, "model plumbing",
        YVEX_MODEL_LOCAL_OPTIONS_DETAIL | YVEX_MODEL_LOCAL_OPTIONS_LEGACY_OUTPUT,
        &cli);
    if (rc != 0) return rc;
    options.models_root = cli.models_root;
    options.registry_path = cli.registry_path;
    yvex_error_clear(&err);
    rc = yvex_model_library_open(library, &options, &err);
    if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));
    *mode = cli.output_mode;
    return 0;
}

static int model_has_deployment(const yvex_model_library_entry *model)
{
    return model && (model->artifact_count || model->profile_count);
}

static void source_identity(const yvex_local_source_record *source,
                            char output[YVEX_MODEL_LIBRARY_ID_CAP])
{
    (void)snprintf(output, YVEX_MODEL_LIBRARY_ID_CAP, "%s:%s@%s",
                   source->provider, source->repository, source->revision);
}

static void source_render(const yvex_model_library_entry *model,
                          const yvex_local_source_record *source,
                          yvex_model_catalog_output_mode mode)
{
    char identity[YVEX_MODEL_LIBRARY_ID_CAP];
    source_identity(source, identity);
    if (mode == YVEX_MODEL_CATALOG_OUTPUT_TABLE) {
        yvex_cli_out_writef(stdout, "%-28s %-16s verify=%s\n",
                            source->name, source->acquisition_state,
                            source->verification_state);
        yvex_cli_out_writef(stdout, "  source: %s\n  model binding: %s\n", identity,
                            model_has_deployment(model) ? model->display_name : "none");
        if (source->blocker[0])
            yvex_cli_out_writef(stdout, "  action: %s\n", source->blocker);
        return;
    }
    if (mode == YVEX_MODEL_CATALOG_OUTPUT_AUDIT) {
        yvex_cli_out_writef(
            stdout,
            "%s\n  state: %s\n  verification: %s\n  size: %s\n"
            "  model binding: %s\n  model identity: %s\n  path: %s\n",
            identity, source->acquisition_state, source->verification_state,
            source->size_known ? "known" : "unknown",
            model_has_deployment(model) ? model->display_name : "none",
            model->identity, source->path);
        if (source->blocker[0])
            yvex_cli_out_writef(stdout, "  blocker: %s\n", source->blocker);
        return;
    }
    yvex_cli_out_fputs("{\"identity\":", stdout);
    yvex_cli_out_json_string(stdout, identity);
    yvex_cli_out_fputs(",\"model_identity\":", stdout);
    yvex_cli_out_json_string(stdout, model->identity);
    yvex_cli_out_fputs(",\"name\":", stdout);
    yvex_cli_out_json_string(stdout, source->name);
    yvex_cli_out_fputs(",\"family\":", stdout);
    yvex_cli_out_json_string(stdout, source->family);
    yvex_cli_out_fputs(",\"provider\":", stdout);
    yvex_cli_out_json_string(stdout, source->provider);
    yvex_cli_out_fputs(",\"repository\":", stdout);
    yvex_cli_out_json_string(stdout, source->repository);
    yvex_cli_out_fputs(",\"revision\":", stdout);
    yvex_cli_out_json_string(stdout, source->revision);
    yvex_cli_out_fputs(",\"representation\":", stdout);
    yvex_cli_out_json_string(stdout, source->representation);
    yvex_cli_out_fputs(",\"acquisition_state\":", stdout);
    yvex_cli_out_json_string(stdout, source->acquisition_state);
    yvex_cli_out_fputs(",\"verification_state\":", stdout);
    yvex_cli_out_json_string(stdout, source->verification_state);
    yvex_cli_out_fputs(",\"blocker\":", stdout);
    yvex_cli_out_json_string(stdout, source->blocker);
    yvex_cli_out_fputs(",\"path\":", stdout);
    yvex_cli_out_json_string(stdout, source->path);
    yvex_cli_out_writef(stdout, ",\"size_bytes\":%llu,\"size_known\":%s}",
                        source->size_bytes,
                        source->size_known ? "true" : "false");
}

static void artifact_render(const yvex_model_library_entry *model,
                            const yvex_model_artifact_fact *artifact,
                            unsigned long long profile_count,
                            unsigned long long launchable_count,
                            yvex_model_catalog_output_mode mode)
{
    char size[32];
    const char *identity = artifact->identity[0] ? artifact->identity : artifact->path;
    model_download_format_bytes(size, sizeof(size), artifact->file_size);
    if (mode == YVEX_MODEL_CATALOG_OUTPUT_TABLE) {
        yvex_cli_out_writef(
            stdout, "  %.20s  %-28s %8s · %llu tensors · %llu/%llu runnable profiles\n",
            identity, artifact->artifact_class[0] ? artifact->artifact_class
                                                 : artifact->format,
            size, artifact->tensor_count, launchable_count, profile_count);
        return;
    }
    if (mode == YVEX_MODEL_CATALOG_OUTPUT_AUDIT) {
        yvex_cli_out_writef(
            stdout,
            "%s\n  model: %s\n  class: %s\n  format: %s\n  variant: %s\n"
            "  size: %llu\n  tensors: %llu\n  legacy execution capability: %s\n"
            "  profiles: %llu (%llu runnable)\n  path: %s\n",
            identity, model->identity, artifact->artifact_class, artifact->format,
            artifact->physical_variant, artifact->file_size,
            artifact->tensor_count, artifact->execution_ready ? "recorded" : "not-recorded",
            profile_count, launchable_count, artifact->path);
        return;
    }
    yvex_cli_out_fputs("{\"identity\":", stdout);
    yvex_cli_out_json_string(stdout, identity);
    yvex_cli_out_fputs(",\"model_identity\":", stdout);
    yvex_cli_out_json_string(stdout, model->identity);
    yvex_cli_out_fputs(",\"path\":", stdout);
    yvex_cli_out_json_string(stdout, artifact->path);
    yvex_cli_out_fputs(",\"artifact_class\":", stdout);
    yvex_cli_out_json_string(stdout, artifact->artifact_class);
    yvex_cli_out_fputs(",\"format\":", stdout);
    yvex_cli_out_json_string(stdout, artifact->format);
    yvex_cli_out_fputs(",\"physical_variant\":", stdout);
    yvex_cli_out_json_string(stdout, artifact->physical_variant);
    yvex_cli_out_writef(stdout,
                        ",\"file_size\":%llu,\"tensor_count\":%llu,"
                        "\"execution_ready\":%s,\"profile_count\":%llu,"
                        "\"launchable_profile_count\":%llu}",
                        artifact->file_size, artifact->tensor_count,
                        artifact->execution_ready ? "true" : "false",
                        profile_count, launchable_count);
}

static void profile_binding_label(const yvex_model_runtime_profile_fact *profile,
                                  char *out, size_t capacity)
{
    const char *path = profile->runtime_binding;
    const char *end, *start;
    size_t length;

    if (!path[0]) {
        snprintf(out, capacity, "%s", profile->installation[0] ? "installed" : "direct");
        return;
    }
    end = strrchr(path, '/');
    if (!end || end == path) {
        snprintf(out, capacity, "binding");
        return;
    }
    start = end;
    while (start > path && start[-1] != '/') --start;
    length = (size_t)(end - start);
    if (!length || length >= capacity) snprintf(out, capacity, "binding");
    else {
        memcpy(out, start, length);
        out[length] = '\0';
    }
}

static void profile_render(const yvex_model_library_entry *model,
                           const yvex_model_runtime_profile_fact *profile,
                           yvex_model_catalog_output_mode mode)
{
    const char *deployment = profile->artifact_class[0] ? profile->artifact_class
                                                        : profile->profile[0]
                                                              ? profile->profile : "default";
    char binding[128];
    profile_binding_label(profile, binding, sizeof(binding));
    if (mode == YVEX_MODEL_CATALOG_OUTPUT_TABLE) {
        yvex_cli_out_writef(
            stdout,
            "  %s\n    deployment=%s · artifact=%.16s · binding=%s\n"
            "    %s/%s/%s · context=%llu · %s\n",
            profile->alias, deployment, profile->artifact_identity, binding,
            profile->backend, profile->engine_kind, profile->execution_strategy,
            profile->context_capacity, profile->launchable ? "runnable" : "blocked");
        if (!profile->launchable)
            yvex_cli_out_writef(stdout, "  blocker: %s\n", profile->blocker);
        return;
    }
    if (mode == YVEX_MODEL_CATALOG_OUTPUT_AUDIT) {
        yvex_cli_out_writef(
            stdout,
            "%s\n  model: %s\n  deployment: %s\n  artifact: %s\n"
            "  binding label: %s\n  binding: %s\n  backend: %s\n  engine: %s\n"
            "  strategy: %s\n  context: %llu\n  status: %s\n",
            profile->alias, model->identity, deployment, profile->artifact_identity,
            binding, profile->runtime_binding, profile->backend, profile->engine_kind,
            profile->execution_strategy, profile->context_capacity,
            profile->launchable ? "runnable" : "blocked");
        if (!profile->launchable)
            yvex_cli_out_writef(stdout, "  blocker: %s\n", profile->blocker);
        return;
    }
    yvex_cli_out_fputs("{\"identity\":", stdout);
    yvex_cli_out_json_string(stdout, profile->alias);
    yvex_cli_out_fputs(",\"model_identity\":", stdout);
    yvex_cli_out_json_string(stdout, model->identity);
    yvex_cli_out_fputs(",\"profile_class\":", stdout);
    yvex_cli_out_json_string(stdout, profile->profile);
    yvex_cli_out_fputs(",\"artifact_identity\":", stdout);
    yvex_cli_out_json_string(stdout, profile->artifact_identity);
    yvex_cli_out_fputs(",\"artifact_path\":", stdout);
    yvex_cli_out_json_string(stdout, profile->artifact_path);
    yvex_cli_out_fputs(",\"runtime_binding\":", stdout);
    yvex_cli_out_json_string(stdout, profile->runtime_binding);
    yvex_cli_out_fputs(",\"runtime_target\":", stdout);
    yvex_cli_out_json_string(stdout, profile->runtime_target);
    yvex_cli_out_fputs(",\"deployment_class\":", stdout);
    yvex_cli_out_json_string(stdout, deployment);
    yvex_cli_out_fputs(",\"backend\":", stdout);
    yvex_cli_out_json_string(stdout, profile->backend);
    yvex_cli_out_fputs(",\"engine_kind\":", stdout);
    yvex_cli_out_json_string(stdout, profile->engine_kind);
    yvex_cli_out_fputs(",\"execution_strategy\":", stdout);
    yvex_cli_out_json_string(stdout, profile->execution_strategy);
    yvex_cli_out_fputs(",\"blocker\":", stdout);
    yvex_cli_out_json_string(stdout, profile->blocker);
    yvex_cli_out_writef(stdout, ",\"context_capacity\":%llu,\"launchable\":%s}",
                        profile->context_capacity,
                        profile->launchable ? "true" : "false");
}

static int command_library_list(int arg_count, char **args)
{
    return yvex_model_catalog_list_command(arg_count, args);
}

static int command_library_show(int arg_count, char **args)
{
    return yvex_model_catalog_show_command(arg_count, args);
}

static int command_source_list(int arg_count, char **args)
{
    yvex_model_library *library = NULL;
    yvex_model_catalog_output_mode mode = YVEX_MODEL_CATALOG_OUTPUT_TABLE;
    unsigned long long model_index, source_index, emitted = 0u;
    int rc = library_open_cli(arg_count, args, 3, &library, &mode);
    if (rc) return rc;
    if (mode == YVEX_MODEL_CATALOG_OUTPUT_JSON)
        yvex_cli_out_fputs("{\"schema\":\"yvex.source.list.v1\",\"sources\":[", stdout);
    else if (mode == YVEX_MODEL_CATALOG_OUTPUT_TABLE)
        yvex_cli_out_fputs("SOURCES\n", stdout);
    for (model_index = 0u; model_index < yvex_model_library_count(library); ++model_index) {
        const yvex_model_library_entry *model = yvex_model_library_at(library, model_index);
        for (source_index = 0u;
             source_index < yvex_model_library_source_count(library, model_index);
             ++source_index) {
            if (mode == YVEX_MODEL_CATALOG_OUTPUT_JSON && emitted)
                yvex_cli_out_writef(stdout, "%c", ',');
            source_render(model, yvex_model_library_source_at(
                                     library, model_index, source_index), mode);
            emitted++;
        }
    }
    if (mode == YVEX_MODEL_CATALOG_OUTPUT_JSON) yvex_cli_out_fputs("]}\n", stdout);
    else if (!emitted) yvex_cli_out_fputs("no acquired sources known locally\n", stdout);
    yvex_model_library_close(library);
    return 0;
}

static int command_source_show(int arg_count, char **args)
{
    yvex_model_library *library = NULL;
    yvex_model_catalog_output_mode mode = YVEX_MODEL_CATALOG_OUTPUT_TABLE;
    unsigned long long model_index, source_index;
    int rc;
    if (arg_count < 4) {
        yvex_cli_out_fputs("yvex: source show requires SOURCE\n", stderr);
        return 2;
    }
    rc = library_open_cli(arg_count, args, 4, &library, &mode);
    if (rc) return rc;
    for (model_index = 0u; model_index < yvex_model_library_count(library); ++model_index) {
        const yvex_model_library_entry *model = yvex_model_library_at(library, model_index);
        for (source_index = 0u;
             source_index < yvex_model_library_source_count(library, model_index);
             ++source_index) {
            const yvex_local_source_record *source = yvex_model_library_source_at(
                library, model_index, source_index);
            char identity[YVEX_MODEL_LIBRARY_ID_CAP];
            source_identity(source, identity);
            if (strcmp(identity, args[3])) continue;
            if (mode == YVEX_MODEL_CATALOG_OUTPUT_JSON)
                yvex_cli_out_fputs("{\"schema\":\"yvex.source.v1\",\"source\":", stdout);
            source_render(model, source, mode == YVEX_MODEL_CATALOG_OUTPUT_JSON
                                            ? mode : YVEX_MODEL_CATALOG_OUTPUT_AUDIT);
            if (mode == YVEX_MODEL_CATALOG_OUTPUT_JSON) yvex_cli_out_fputs("}\n", stdout);
            yvex_model_library_close(library);
            return 0;
        }
    }
    yvex_model_library_close(library);
    yvex_cli_out_writef(stderr, "yvex: source not found: %s\n", args[3]);
    return 2;
}

static unsigned long long artifact_profile_count(
    const yvex_model_library *library, unsigned long long model_index,
    const yvex_model_artifact_fact *artifact, int launchable_only)
{
    unsigned long long index, count = 0u;

    for (index = 0u; index < yvex_model_library_profile_count(library, model_index);
         ++index) {
        const yvex_model_runtime_profile_fact *profile =
            yvex_model_library_profile_at(library, model_index, index);
        int same = artifact->identity[0]
                       ? strcmp(artifact->identity, profile->artifact_identity) == 0
                       : strcmp(artifact->path, profile->artifact_path) == 0;
        if (same && (!launchable_only || profile->launchable)) count++;
    }
    return count;
}

static int command_artifact_list(int arg_count, char **args)
{
    yvex_model_library *library = NULL;
    yvex_model_catalog_output_mode mode = YVEX_MODEL_CATALOG_OUTPUT_TABLE;
    unsigned long long model_index, artifact_index, emitted = 0u;
    int rc = library_open_cli(arg_count, args, 3, &library, &mode);
    if (rc) return rc;
    if (mode == YVEX_MODEL_CATALOG_OUTPUT_JSON)
        yvex_cli_out_fputs("{\"schema\":\"yvex.artifact.list.v1\",\"artifacts\":[", stdout);
    else if (mode == YVEX_MODEL_CATALOG_OUTPUT_TABLE)
        yvex_cli_out_fputs("ARTIFACTS\n", stdout);
    for (model_index = 0u; model_index < yvex_model_library_count(library); ++model_index) {
        const yvex_model_library_entry *model = yvex_model_library_at(library, model_index);
        if (mode == YVEX_MODEL_CATALOG_OUTPUT_TABLE &&
            yvex_model_library_artifact_count(library, model_index))
            yvex_cli_out_writef(stdout, "\n%s\n", model->display_name);
        for (artifact_index = 0u;
             artifact_index < yvex_model_library_artifact_count(library, model_index);
             ++artifact_index) {
            const yvex_model_artifact_fact *artifact = yvex_model_library_artifact_at(
                library, model_index, artifact_index);
            unsigned long long profiles = artifact_profile_count(
                library, model_index, artifact, 0);
            unsigned long long launchable = artifact_profile_count(
                library, model_index, artifact, 1);
            if (mode == YVEX_MODEL_CATALOG_OUTPUT_JSON && emitted)
                yvex_cli_out_writef(stdout, "%c", ',');
            artifact_render(model, artifact, profiles, launchable, mode);
            emitted++;
        }
    }
    if (mode == YVEX_MODEL_CATALOG_OUTPUT_JSON) yvex_cli_out_fputs("]}\n", stdout);
    else if (!emitted) yvex_cli_out_fputs("no compiled artifacts known locally\n", stdout);
    yvex_model_library_close(library);
    return 0;
}

static int command_profile_list(int arg_count, char **args)
{
    yvex_model_library *library = NULL;
    yvex_model_catalog_output_mode mode = YVEX_MODEL_CATALOG_OUTPUT_TABLE;
    unsigned long long model_index, profile_index, emitted = 0u;
    int rc = library_open_cli(arg_count, args, 3, &library, &mode);
    if (rc) return rc;
    if (mode == YVEX_MODEL_CATALOG_OUTPUT_JSON)
        yvex_cli_out_fputs("{\"schema\":\"yvex.profile.list.v1\",\"profiles\":[", stdout);
    else if (mode == YVEX_MODEL_CATALOG_OUTPUT_TABLE)
        yvex_cli_out_fputs(
            "DEPLOYMENT PROFILES\nEach entry is a distinct physical artifact/binding.\n",
            stdout);
    for (model_index = 0u; model_index < yvex_model_library_count(library); ++model_index) {
        const yvex_model_library_entry *model = yvex_model_library_at(library, model_index);
        if (mode == YVEX_MODEL_CATALOG_OUTPUT_TABLE &&
            yvex_model_library_profile_count(library, model_index))
            yvex_cli_out_writef(stdout, "\n%s · %llu/%llu runnable\n",
                                model->display_name,
                                model->launchable_profile_count,
                                model->profile_count);
        for (profile_index = 0u;
             profile_index < yvex_model_library_profile_count(library, model_index);
             ++profile_index) {
            if (mode == YVEX_MODEL_CATALOG_OUTPUT_JSON && emitted)
                yvex_cli_out_writef(stdout, "%c", ',');
            profile_render(model, yvex_model_library_profile_at(
                                      library, model_index, profile_index), mode);
            emitted++;
        }
    }
    if (mode == YVEX_MODEL_CATALOG_OUTPUT_JSON) yvex_cli_out_fputs("]}\n", stdout);
    else if (!emitted) yvex_cli_out_fputs("no runtime profiles known locally\n", stdout);
    yvex_model_library_close(library);
    return 0;
}

static int command_profile_show(int arg_count, char **args)
{
    yvex_model_library *library = NULL;
    yvex_model_catalog_output_mode mode = YVEX_MODEL_CATALOG_OUTPUT_TABLE;
    unsigned long long model_index, profile_index;
    int rc;
    if (arg_count < 4) {
        yvex_cli_out_fputs("yvex: profile show requires PROFILE\n", stderr);
        return 2;
    }
    rc = library_open_cli(arg_count, args, 4, &library, &mode);
    if (rc) return rc;
    for (model_index = 0u; model_index < yvex_model_library_count(library); ++model_index) {
        const yvex_model_library_entry *model = yvex_model_library_at(library, model_index);
        for (profile_index = 0u;
             profile_index < yvex_model_library_profile_count(library, model_index);
             ++profile_index) {
            const yvex_model_runtime_profile_fact *profile =
                yvex_model_library_profile_at(library, model_index, profile_index);
            if (strcmp(profile->alias, args[3])) continue;
            if (mode == YVEX_MODEL_CATALOG_OUTPUT_JSON)
                yvex_cli_out_fputs("{\"schema\":\"yvex.profile.v1\",\"profile\":", stdout);
            profile_render(model, profile, mode == YVEX_MODEL_CATALOG_OUTPUT_JSON
                                             ? mode : YVEX_MODEL_CATALOG_OUTPUT_AUDIT);
            if (mode == YVEX_MODEL_CATALOG_OUTPUT_JSON) yvex_cli_out_fputs("}\n", stdout);
            yvex_model_library_close(library);
            return 0;
        }
    }
    yvex_model_library_close(library);
    yvex_cli_out_writef(stderr, "yvex: profile not found: %s\n", args[3]);
    return 2;
}

/*
 * Verify one registered file and its metadata without deriving CLI presentation.
 *
 * Immutable registry entry and caller-owned identity, metadata, error, and result storage. Records
 * the exact identity or metadata refusal without mutating the registry.
 */
static void verify_registered_artifact(const yvex_model_registry_entry *entry,
                                       yvex_artifact_file_identity *identity,
                                       yvex_model_metadata_snapshot *current,
                                       yvex_model_metadata_drift_report *report,
                                       yvex_error *err,
                                       models_verify_result *result)
{
    int rc;

    memset(identity, 0, sizeof(*identity));
    rc = yvex_artifact_identity_read(entry->path, identity, err);
    if (rc != YVEX_OK) {
        result->identity_status = "fail";
        result->reason = yvex_error_message(err);
        return;
    }
    if (!entry->sha256 || !entry->sha256[0] || !yvex_sha256_hex_is_valid(entry->sha256)) {
        result->identity_status = "missing";
        result->status = "models-identity-missing";
        result->reason = "registered alias lacks digest identity; re-add model";
        return;
    }
    if (strcmp(entry->sha256, identity->sha256) != 0 ||
        (entry->file_size != 0ull && entry->file_size != identity->file_size)) {
        result->identity_status = "fail";
        result->reason = "digest mismatch for registered alias";
        return;
    }

    result->identity_status = "pass";
    result->reason = "current file identity matches registered alias";
    memset(current, 0, sizeof(*current));
    memset(report, 0, sizeof(*report));
    rc = yvex_model_metadata_snapshot_read(current, entry->path, err);
    if (rc != YVEX_OK) {
        result->metadata_status = "fail";
        result->readiness_status = "fail";
        result->reason = "current artifact metadata could not be parsed";
        result->status = "models-metadata-drift";
        return;
    }
    rc = yvex_model_registry_compare_metadata(entry, &current->entry, report, err);
    result->metadata_checked = 1;
    if (rc != YVEX_OK) {
        result->metadata_status = "fail";
        result->readiness_status = "fail";
        result->reason = yvex_error_message(err);
        result->status = "models-metadata-drift";
        return;
    }
    result->metadata_status = report->metadata_status;
    result->readiness_status = report->readiness_status;
    if (strcmp(result->metadata_status, "pass") == 0 &&
        strcmp(result->readiness_status, "pass") == 0) {
        result->pass = 1;
        result->status = "models-identity-pass";
    } else if (strcmp(result->metadata_status, "missing") == 0 ||
               strcmp(result->readiness_status, "missing") == 0) {
        result->reason = "registered alias lacks metadata summary; re-add model";
        result->status = "models-metadata-missing";
    } else {
        result->reason = "registered alias metadata does not match current artifact facts";
        result->status = "models-metadata-drift";
    }
}

static void print_verify_pair(const models_verify_pair *pair,
                              const yvex_model_registry_entry *registered,
                              const yvex_model_registry_entry *current,
                              int metadata_checked)
{
    yvex_cli_field_spec field;
    char key[96];

    field.kind = pair->kind;
    field.offset = pair->offset;
    snprintf(key, sizeof(key), "registered_%s", pair->name);
    field.key = key;
    field.fallback = "";
    (void)yvex_cli_out_fields(stdout, registered, &field, 1u);
    snprintf(key, sizeof(key), "current_%s", pair->name);
    field.key = key;
    field.fallback = metadata_checked ? "" : "not-checked";
    (void)yvex_cli_out_fields(stdout, current, &field, 1u);
}

/*
 * Render the complete verification audit from one typed result.
 *
 * Immutable registry, file identity, metadata, drift, and verification facts.
 */
static void print_models_verify_audit(const yvex_model_registry_entry *entry,
                                      const yvex_artifact_file_identity *identity,
                                      const yvex_model_metadata_snapshot *current,
                                      const yvex_model_metadata_drift_report *report,
                                      const models_verify_result *result)
{
    const yvex_model_registry_entry *current_entry =
        result->metadata_checked ? &current->entry : &empty_registry_entry;
    const void *head_objects[] = {entry, identity, result};
    unsigned long i;

    yvex_cli_out_writef(stdout, "models: verify\n");
    for (i = 0; i < sizeof(verify_audit_head) / sizeof(verify_audit_head[0]); ++i) {
        const models_verify_field *projection = &verify_audit_head[i];
        (void)yvex_cli_out_fields(stdout, head_objects[projection->source],
                                  &projection->field, 1u);
    }
    for (i = 0; i < sizeof(verify_audit_pairs) / sizeof(verify_audit_pairs[0]); ++i) {
        print_verify_pair(&verify_audit_pairs[i], entry, current_entry,
                          result->metadata_checked);
    }
    if (result->metadata_checked) {
        print_metadata_drift_cli(report);
    } else {
        yvex_cli_out_writef(stdout, "metadata_status: %s\n", result->metadata_status);
        yvex_cli_out_writef(stdout, "readiness_status: %s\n", result->readiness_status);
    }
    yvex_cli_out_writef(stdout, "reason: %s\n", result->reason);
    yvex_cli_out_writef(stdout, "status: %s\n", result->status);
}

static int command_models_verify(int arg_count, char **args)
{
    yvex_model_registry *registry = NULL;
    yvex_artifact_file_identity identity;
    yvex_model_metadata_snapshot current_metadata;
    yvex_model_metadata_drift_report metadata_report;
    models_verify_result result = {
        .identity_status = "unknown",
        .metadata_status = "not-checked", .readiness_status = "not-checked",
        .status = "models-identity-fail", .reason = ""
    };
    yvex_error err;
    const char *registry_path = NULL;
    yvex_models_output_mode output_mode = YVEX_MODELS_OUTPUT_NORMAL;
    const yvex_model_registry_entry *entry;
    const char *alias;
    int rc;

    if (arg_count < 4) {
        yvex_cli_out_writef(stderr, "yvex: models verify requires ALIAS\n");
        return 2;
    }
    alias = args[3];
    rc = parse_models_registry_options(arg_count, args, 4, &registry_path, &output_mode);
    if (rc != 0) return rc;
    yvex_error_clear(&err);
    rc = models_registry_open(&registry, registry_path, 1, &err);
    if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));
    entry = yvex_model_registry_find(registry, alias);
    if (!entry) {
        yvex_model_registry_close(registry);
        yvex_cli_out_writef(stderr, "yvex: model alias not found: %s\n", alias);
        return 2;
    }

    verify_registered_artifact(entry, &identity, &current_metadata, &metadata_report,
                               &err, &result);

    if (output_mode != YVEX_MODELS_OUTPUT_AUDIT) {
        yvex_cli_out_writef(stdout, "verify: %s alias=%s\n",
                            result.pass ? "pass" : "fail", entry->alias);
        yvex_cli_out_writef(stdout, "identity: %s digest: %s metadata: %s\n",
                            result.identity_status, result.identity_status,
                            result.metadata_status);
        yvex_cli_out_writef(stdout, "artifact: %s tensors=%llu size=%llu\n",
                            entry->format ? entry->format : "unknown",
                            result.metadata_checked ? current_metadata.entry.tensor_count
                                                    : entry->tensor_count,
                            identity.file_size);
        yvex_cli_out_writef(stdout, "top_blocker: %s\n",
                            result.pass ? "none" : result.reason);
        yvex_cli_out_writef(stdout, "boundary: identity verified, runtime generation unsupported\n");
        yvex_cli_out_writef(stdout, "status: %s\n", result.status);
    } else {
        print_models_verify_audit(entry, &identity, &current_metadata, &metadata_report,
                                  &result);
    }
    yvex_model_registry_close(registry);
    return result.pass ? 0 : exit_for_status(YVEX_ERR_STATE);
}

static int command_models_remove(int arg_count, char **args)
{
    yvex_model_registry *registry = NULL;
    yvex_error err;
    const char *registry_path = NULL;
    const char *alias;
    int rc;

    if (arg_count < 4) {
        yvex_cli_out_writef(stderr, "yvex: models remove requires ALIAS\n");
        return 2;
    }
    alias = args[3];
    rc = parse_models_registry_options(arg_count, args, 4, &registry_path, NULL);
    if (rc != 0) return rc;
    yvex_error_clear(&err);
    rc = models_registry_open(&registry, registry_path, 1, &err);
    if (rc == YVEX_OK) rc = yvex_model_registry_remove(registry, alias, &err);
    if (rc == YVEX_OK) rc = yvex_model_registry_save(registry, registry_path, &err);
    if (rc != YVEX_OK) {
        yvex_model_registry_close(registry);
        return print_yvex_error(&err, exit_for_status(rc));
    }
    yvex_cli_out_writef(stdout, "models: remove\n");
    yvex_cli_out_writef(stdout, "removed: %s\n", alias);
    yvex_cli_out_writef(stdout, "status: models-removed\n");
    yvex_model_registry_close(registry);
    return 0;
}

static int command_models_inspect(int arg_count, char **args)
{
    yvex_model_registry *registry = NULL;
    yvex_model_context ctx;
    yvex_error err;
    const yvex_model_registry_entry *entry;
    const yvex_gguf_header *header;
    const char *registry_path = NULL;
    yvex_models_output_mode output_mode = YVEX_MODELS_OUTPUT_NORMAL;
    const char *alias;
    int rc;

    if (arg_count < 4) {
        yvex_cli_out_writef(stderr, "yvex: models inspect requires ALIAS\n");
        return 2;
    }
    alias = args[3];
    rc = parse_models_registry_options(arg_count, args, 4, &registry_path, &output_mode);
    if (rc != 0) return rc;
    yvex_error_clear(&err);
    rc = models_registry_open(&registry, registry_path, 1, &err);
    if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));
    entry = yvex_model_registry_find(registry, alias);
    if (!entry) {
        yvex_model_registry_close(registry);
        yvex_cli_out_writef(stderr, "yvex: model alias not found: %s\n", alias);
        return 2;
    }
    if (output_mode != YVEX_MODELS_OUTPUT_AUDIT) {
        yvex_deployment_compatibility compatibility = {0};
        yvex_error startup_error;
        int startup_ready;
        yvex_error_clear(&startup_error);
        startup_ready = yvex_deployment_compatibility_evaluate(
                            entry, &compatibility, &startup_error) == YVEX_OK &&
                        compatibility.current;
        yvex_cli_out_writef(stdout, "model: %s\n", entry->alias);
        yvex_cli_out_writef(stdout, "family: %s class=%s tensors=%llu size=%llu\n",
               entry->family ? entry->family : "",
               entry->artifact_class ? entry->artifact_class : "",
               entry->tensor_count,
               entry->file_size);
        yvex_cli_out_writef(stdout, "primary: %s %s %s\n",
               entry->primary_tensor_name ? entry->primary_tensor_name : "",
               entry->primary_tensor_dtype ? entry->primary_tensor_dtype : "",
               entry->primary_tensor_dims ? entry->primary_tensor_dims : "");
        yvex_cli_out_writef(stdout, "artifact: support=%s execution=%s\n",
               entry->support_level ? entry->support_level : "",
               entry->execution_ready ? "ready" : "not-established-by-inspection");
        if (startup_ready)
            yvex_cli_out_writef(
                stdout, "runtime profile: ready kind=%s backend=%s strategy=%s context=%llu\n",
                entry->runtime_engine_kind, entry->runtime_backend,
                entry->runtime_execution_strategy,
                entry->runtime_context);
        else
            yvex_cli_out_writef(
                stdout, "runtime profile: unavailable (%s: %s)\n",
                yvex_deployment_compatibility_status_name(compatibility.status),
                compatibility.reason[0] ? compatibility.reason
                                        : yvex_error_message(&startup_error));
        yvex_cli_out_writef(stdout, "status: models-inspect\n");
        yvex_model_registry_close(registry);
        return 0;
    }
    yvex_cli_out_writef(stdout, "models: inspect\n");
    (void)yvex_cli_out_fields(stdout, entry, registry_inspect_fields,
                              sizeof(registry_inspect_fields) /
                                  sizeof(registry_inspect_fields[0]));
    {
        yvex_deployment_compatibility compatibility = {0};
        yvex_error startup_error;
        yvex_error_clear(&startup_error);
        yvex_cli_out_writef(
            stdout, "startup_profile_status: %s\n",
            yvex_deployment_compatibility_evaluate(
                entry, &compatibility, &startup_error) == YVEX_OK &&
                    compatibility.current
                ? "ready" : "unavailable");
        yvex_cli_out_writef(
            stdout, "deployment_compatibility: %s\n",
            yvex_deployment_compatibility_status_name(compatibility.status));
    }
    rc = yvex_model_context_open(entry->path, &ctx, &err);
    if (rc == YVEX_OK) {
        header = yvex_gguf_header_view(ctx.gguf);
        yvex_cli_out_writef(stdout, "gguf:\n");
        yvex_cli_out_writef(stdout, "  version: %u\n", header->version);
        yvex_cli_out_writef(stdout, "  tensor_count: %llu\n", header->tensor_count);
        yvex_model_context_close(&ctx);
    } else {
        yvex_cli_out_lines(stdout, literal_pair_1, sizeof(literal_pair_1) / sizeof(literal_pair_1[0]));
        yvex_cli_out_writef(stdout, "  reason: %s\n", yvex_error_message(&err));
        yvex_error_clear(&err);
    }
    yvex_cli_out_writef(stdout, "status: models-inspect\n");
    yvex_model_registry_close(registry);
    return 0;
}

/* Full model inventory and placement reporting. */

typedef int (*yvex_models_subcommand_fn)(int arg_count, char **args);

typedef struct {
    const char *name;
    yvex_models_subcommand_fn run;
} yvex_models_subcommand;

static const yvex_models_subcommand model_subcommands[] = {
    { "library-list", command_library_list },
    { "library-show", command_library_show },
    { "pull", yvex_model_pull_command },
    { "pull-status", yvex_model_pull_lifecycle_command },
    { "pull-stop", yvex_model_pull_lifecycle_command },
    { "push", yvex_model_push_command },
    { "prepare-product", yvex_model_prepare_command },
    { "source-list", command_source_list },
    { "source-show", command_source_show },
    { "artifact-list", command_artifact_list },
    { "profile-list", command_profile_list },
    { "profile-show", command_profile_show },
    { "scan", command_models_scan },
    { "add", command_models_add },
    { "download", yvex_models_download_surface_command },
    { "artifacts", yvex_models_artifacts_surface_command },
    { "prepare", yvex_models_prepare_surface_command },
    { "check", yvex_models_check_surface_command },
    { "search", command_models_search },
    { "remote", command_models_remote },
    { "verify", command_models_verify },
    { "inspect", command_models_inspect },
    { "remove", command_models_remove }
};

static int command_models(int arg_count, char **args)
{
    unsigned long i;

    if (arg_count >= 3 && (strcmp(args[2], "--help") == 0 || strcmp(args[2], "-h") == 0)) {
        yvex_models_help(stdout);
        return 0;
    }
    if (arg_count < 3) {
        yvex_cli_out_writef(stderr,
            "yvex: internal models adapter requires an admitted subcommand\n");
        return 2;
    }
    for (i = 0; i < sizeof(model_subcommands) / sizeof(model_subcommands[0]); ++i) {
        if (strcmp(args[2], model_subcommands[i].name) == 0) {
            return model_subcommands[i].run(arg_count, args);
        }
    }
    yvex_cli_out_writef(stderr, "yvex: unknown models subcommand: %s\n", args[2]);
    return 2;
}

int yvex_models_command(int arg_count, char **args)
{
    return command_models(arg_count, args);
}

void yvex_models_help(FILE *fp)
{
    yvex_cli_out_lines(fp, models_help_lines,
                       sizeof(models_help_lines) / sizeof(models_help_lines[0]));
}
