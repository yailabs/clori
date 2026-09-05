/* Project pull, prepare, and push over the existing source, compiler, and catalog owners. */
#define _POSIX_C_SOURCE 200809L
#include "src/cli/model_artifacts/private.h"

#include <yvex/internal/source_catalog.h>
#include <yvex/internal/source_distribution.h>
#include <yvex/quant.h>

#include <ctype.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>

typedef struct {
    const char *source, *models_root, *name, *family, *revision;
    const char *format, *variant, *quant, *auth, *progress;
    const char *include[YVEX_MODEL_DOWNLOAD_PATTERN_CAP];
    const char *exclude[YVEX_MODEL_DOWNLOAD_PATTERN_CAP];
    unsigned int include_count, exclude_count;
    int reference, managed, prepare, stream, resume, dry_run, verbose, json;
    int clear_stale_locks;
} model_pull_options;

typedef struct {
    const char *model, *destination, *models_root, *registry_path;
    const char *variant, *representation;
    int json;
} model_push_options;

static int workflow_value(const char *command, const char *flag, int argc,
                          char **argv, int *index, const char **out)
{
    if (*index + 1 >= argc || !argv[*index + 1][0]) {
        yvex_cli_out_writef(stderr, "yvex: %s %s requires a value\n", command, flag);
        return 0;
    }
    *out = argv[++*index];
    return 1;
}

static int pull_repeat(const char *command, const char *flag, int argc,
                       char **argv, int *index, const char **values,
                       unsigned int *count)
{
    const char *value;
    if (*count >= YVEX_MODEL_DOWNLOAD_PATTERN_CAP) {
        yvex_cli_out_writef(stderr, "yvex: %s has too many %s values\n", command, flag);
        return 0;
    }
    if (!workflow_value(command, flag, argc, argv, index, &value)) return 0;
    values[(*count)++] = value;
    return 1;
}

static int pull_options_parse(int argc, char **argv, model_pull_options *out)
{
    int index;
    memset(out, 0, sizeof(*out));
    for (index = 3; index < argc; ++index) {
        const char *flag = argv[index];
        const char **field = NULL;
        if (!strcmp(flag, "--models-root")) field = &out->models_root;
        else if (!strcmp(flag, "--name")) field = &out->name;
        else if (!strcmp(flag, "--family")) field = &out->family;
        else if (!strcmp(flag, "--revision")) field = &out->revision;
        else if (!strcmp(flag, "--format")) field = &out->format;
        else if (!strcmp(flag, "--variant")) field = &out->variant;
        else if (!strcmp(flag, "--quant")) field = &out->quant;
        else if (!strcmp(flag, "--auth")) field = &out->auth;
        else if (!strcmp(flag, "--progress")) field = &out->progress;
        if (field) {
            if (!workflow_value("model pull", flag, argc, argv, &index, field)) return 2;
        } else if (!strcmp(flag, "--include")) {
            if (!pull_repeat("model pull", flag, argc, argv, &index,
                             out->include, &out->include_count)) return 2;
        } else if (!strcmp(flag, "--exclude")) {
            if (!pull_repeat("model pull", flag, argc, argv, &index,
                             out->exclude, &out->exclude_count)) return 2;
        } else if (!strcmp(flag, "--reference")) out->reference = 1;
        else if (!strcmp(flag, "--managed")) out->managed = 1;
        else if (!strcmp(flag, "--prepare")) out->prepare = 1;
        else if (!strcmp(flag, "--stream")) out->stream = 1;
        else if (!strcmp(flag, "--resume")) out->resume = 1;
        else if (!strcmp(flag, "--clear-stale-locks")) out->clear_stale_locks = 1;
        else if (!strcmp(flag, "--dry-run")) out->dry_run = 1;
        else if (!strcmp(flag, "--verbose")) out->verbose = 1;
        else if (!strcmp(flag, "--json")) out->json = 1;
        else if (flag[0] == '-') {
            yvex_cli_out_writef(stderr, "yvex: unknown model pull option: %s\n", flag);
            return 2;
        } else if (!out->source) out->source = flag;
        else {
            yvex_cli_out_writef(stderr, "yvex: model pull received extra source: %s\n", flag);
            return 2;
        }
    }
    if (!out->source) {
        yvex_cli_out_fputs("yvex: model pull requires SOURCE\n", stderr);
        return 2;
    }
    if (out->reference && out->managed) {
        yvex_cli_out_fputs("yvex: model pull --reference and --managed conflict\n", stderr);
        return 2;
    }
    if (out->stream && !out->prepare) {
        yvex_cli_out_fputs("yvex: model pull --stream requires --prepare\n", stderr);
        return 2;
    }
    if (out->quant && !out->prepare) {
        yvex_cli_out_fputs("yvex: model pull --quant requires --prepare\n", stderr);
        return 2;
    }
    if (out->reference && out->resume) {
        yvex_cli_out_fputs("yvex: model pull --reference and --resume conflict\n", stderr);
        return 2;
    }
    if (out->prepare && out->json) {
        yvex_cli_out_fputs(
            "yvex: model pull --prepare and --json conflict; run pull and prepare "
            "separately so each operation emits one JSON document\n",
            stderr);
        return 2;
    }
    return 0;
}

static int workflow_models_root(const char *requested, char out[YVEX_PATH_CAP])
{
    yvex_paths paths;
    yvex_operator_paths operator_paths;
    yvex_error err;
    int rc;
    memset(&paths, 0, sizeof(paths));
    memset(&operator_paths, 0, sizeof(operator_paths));
    yvex_error_clear(&err);
    rc = yvex_operator_paths_resolve(&paths, requested, &operator_paths, &err);
    if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));
    snprintf(out, YVEX_PATH_CAP, "%s", operator_paths.models_root);
    return 0;
}

static int revision_immutable(const char *revision)
{
    size_t index, length;
    if (!revision) return 0;
    length = strlen(revision);
    if (length != 40u && length != 64u) return 0;
    for (index = 0u; index < length; ++index)
        if (!isxdigit((unsigned char)revision[index])) return 0;
    return 1;
}

static void pull_product_name(const char *repository,
                              char out[YVEX_REMOTE_NAME_CAP])
{
    const unsigned char *source = (const unsigned char *)strrchr(repository, '/');
    size_t used = 0u;
    source = source ? source + 1u : (const unsigned char *)repository;
    while (*source && used + 1u < YVEX_REMOTE_NAME_CAP) {
        unsigned char value = *source++;
        if (isalnum(value)) out[used++] = (char)tolower(value);
        else if ((value == '-' || value == '_' || value == '.') && used &&
                 out[used - 1u] != '-')
            out[used++] = value == '_' ? '-' : (char)value;
        else if (used && out[used - 1u] != '-') out[used++] = '-';
    }
    while (used && (out[used - 1u] == '-' || out[used - 1u] == '.')) used--;
    out[used] = '\0';
    if (!out[0]) snprintf(out, YVEX_REMOTE_NAME_CAP, "%s", "model");
}

static const char *pull_family(const model_pull_options *options,
                               const yvex_remote_model *remote)
{
    if (options->family && model_download_family_valid(options->family))
        return options->family;
    if (remote->family[0] && model_download_family_valid(remote->family))
        return remote->family;
    return "unknown";
}

typedef struct {
    char ordinal[12];
    char files[24];
    yvex_cli_table_cell cells[5];
    yvex_cli_table_row row;
} representation_choice_row;

static int representation_read(unsigned int maximum, unsigned int *choice)
{
    char line[64], *end, *newline;
    unsigned long value;
    yvex_cli_out_writef(stdout, "Representation [1-%u, q to cancel]: ", maximum);
    fflush(stdout);
    if (!fgets(line, sizeof(line), stdin)) return 0;
    newline = strpbrk(line, "\r\n");
    if (newline) *newline = '\0';
    if (!line[0] || !strcasecmp(line, "q")) return 0;
    errno = 0;
    value = strtoul(line, &end, 10);
    if (errno || end == line || *end || !value || value > maximum) return 0;
    *choice = (unsigned int)value;
    return 1;
}

static int representation_select(const yvex_remote_catalog *catalog,
                                 const yvex_remote_model *model,
                                 const model_pull_options *options,
                                 const yvex_model_representation **selected)
{
    static const yvex_cli_table_column columns[] = {
        {"#", 1u, 3u, YVEX_CLI_TABLE_RIGHT, 0},
        {"FORMAT", 8u, 14u, YVEX_CLI_TABLE_LEFT, 0},
        {"QUANT / DTYPE", 10u, 30u, YVEX_CLI_TABLE_LEFT, 0},
        {"FILES", 5u, 8u, YVEX_CLI_TABLE_RIGHT, 0},
        {"SIZE", 8u, 14u, YVEX_CLI_TABLE_RIGHT, 0}
    };
    unsigned int eligible[YVEX_REMOTE_MAX_REPRESENTATIONS], count = 0u, index, choice = 1u;
    representation_choice_row *storage;
    yvex_cli_table_row *rows;
    char (*sizes)[32];
    for (index = 0u; index < model->representation_count; ++index) {
        const yvex_model_representation *item =
            yvex_remote_catalog_representation_at(catalog, 0u, index);
        if (!item) continue;
        if (options->format && strcasecmp(options->format, item->format)) continue;
        if (options->variant && strcmp(options->variant, item->identity) &&
            strcasecmp(options->variant, item->precision)) continue;
        eligible[count++] = index;
    }
    if (!count) {
        yvex_cli_out_fputs("yvex: no remote representation matches --format/--variant\n", stderr);
        return 2;
    }
    if (count == 1u) {
        *selected = yvex_remote_catalog_representation_at(catalog, 0u, eligible[0]);
        return 0;
    }
    if (!isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO)) {
        yvex_cli_out_fputs(
            "yvex: multiple representations are available; non-TTY use requires --format and --variant\n",
            stderr);
        return 2;
    }
    storage = calloc(count, sizeof(*storage));
    rows = calloc(count, sizeof(*rows));
    sizes = calloc(count, sizeof(*sizes));
    if (!storage || !rows || !sizes) { free(storage); free(rows); free(sizes); return 1; }
    yvex_cli_out_fputs("Select representation\n\n", stdout);
    for (index = 0u; index < count; ++index) {
        const yvex_model_representation *item =
            yvex_remote_catalog_representation_at(catalog, 0u, eligible[index]);
        snprintf(storage[index].ordinal, sizeof(storage[index].ordinal), "%u", index + 1u);
        model_download_format_bytes(sizes[index], 32u, item->size_bytes);
        storage[index].cells[0] = (yvex_cli_table_cell){storage[index].ordinal, YVEX_CLI_TABLE_ACCENT};
        storage[index].cells[1] = (yvex_cli_table_cell){item->format, YVEX_CLI_TABLE_PLAIN};
        storage[index].cells[2] = (yvex_cli_table_cell){item->precision, YVEX_CLI_TABLE_PLAIN};
        snprintf(storage[index].files, sizeof(storage[index].files), "%llu",
                 item->file_count);
        storage[index].cells[3] = (yvex_cli_table_cell){storage[index].files,
                                                        YVEX_CLI_TABLE_PLAIN};
        storage[index].cells[4] = (yvex_cli_table_cell){sizes[index], YVEX_CLI_TABLE_PLAIN};
        storage[index].row.cells = storage[index].cells;
        rows[index] = storage[index].row;
    }
    (void)yvex_cli_table_render(stdout, columns, 5u, rows, count);
    if (!representation_read(count, &choice)) {
        yvex_cli_out_fputs("model pull cancelled\n", stdout);
        free(storage); free(rows); free(sizes);
        return 2;
    }
    *selected = yvex_remote_catalog_representation_at(catalog, 0u, eligible[choice - 1u]);
    free(storage); free(rows); free(sizes);
    return 0;
}

static int pull_prepare_followup(const char *argv0, const char *model,
                                 const char *models_root, const char *quant)
{
    char *prepare_argv[9];
    int count = 0;
    prepare_argv[count++] = (char *)argv0;
    prepare_argv[count++] = "models";
    prepare_argv[count++] = "prepare-product";
    prepare_argv[count++] = (char *)model;
    prepare_argv[count++] = "--models-root";
    prepare_argv[count++] = (char *)models_root;
    if (quant) {
        prepare_argv[count++] = "--quant";
        prepare_argv[count++] = (char *)quant;
    }
    return yvex_model_prepare_command(count, prepare_argv);
}

static int pull_remote_download(int argc, char **argv,
                                const model_pull_options *options,
                                const yvex_source_locator *locator,
                                const yvex_remote_model *remote,
                                const yvex_remote_catalog *catalog,
                                const yvex_model_representation *representation,
                                const char *models_root)
{
    char *download_argv[180];
    char expected[32];
    char derived_name[YVEX_REMOTE_NAME_CAP];
    const char *name;
    const char *prepare_selector;
    const yvex_source_target_identity *target =
        yvex_source_target_identity_find_repository(locator->repository);
    int count = 0, rc;
    unsigned int index, excludes = options->exclude_count;
    (void)argc;
    pull_product_name(locator->repository, derived_name);
    name = options->name ? options->name : derived_name;
    prepare_selector = name;
    download_argv[count++] = argv[0];
    download_argv[count++] = "models";
    download_argv[count++] = "download";
    if (options->resume) download_argv[count++] = "resume";
#define PULL_ARG(flag_, value_) do { download_argv[count++] = (flag_); \
                                     download_argv[count++] = (char *)(value_); } while (0)
    PULL_ARG("--repo", locator->repository);
    PULL_ARG("--family", pull_family(options, remote));
    PULL_ARG("--name", name);
    PULL_ARG("--revision", remote->resolved_revision);
    PULL_ARG("--models-root", models_root);
    if (representation->file_pattern[0]) PULL_ARG("--include", representation->file_pattern);
    PULL_ARG("--include", "*.json");
    PULL_ARG("--include", "tokenizer*");
    if (options->include_count + 3u > YVEX_MODEL_DOWNLOAD_PATTERN_CAP) {
        yvex_cli_out_fputs("yvex: too many acquisition include patterns\n", stderr);
        return 2;
    }
    for (index = 0u; index < remote->available_file_count; ++index) {
        const yvex_remote_file *file = yvex_remote_catalog_file_at(catalog, 0u, index);
        int index_file = yvex_source_ends_with(file->path, ".safetensors.index.json");
        int alternate = file->kind == YVEX_REMOTE_FILE_SAFETENSORS &&
                        strcmp(file->representation, representation->identity);
        if (!alternate && !(index_file &&
            !strncmp(representation->identity, "safetensors-file-", 17u))) continue;
        if (++excludes > YVEX_MODEL_DOWNLOAD_PATTERN_CAP) {
            yvex_cli_out_fputs("yvex: acquisition exclusion population exceeds bounded contract\n", stderr);
            return 2;
        }
        PULL_ARG("--exclude", file->path);
    }
    for (index = 0u; index < options->include_count; ++index)
        PULL_ARG("--include", options->include[index]);
    for (index = 0u; index < options->exclude_count; ++index)
        PULL_ARG("--exclude", options->exclude[index]);
    if (options->auth) PULL_ARG("--auth", options->auth);
    if (options->progress && !options->json) PULL_ARG("--progress", options->progress);
    else if (options->dry_run && !options->verbose && !options->json)
        PULL_ARG("--progress", "log");
    if (options->json) {
        PULL_ARG("--output", "json");
        PULL_ARG("--progress", "off");
    }
    if (options->dry_run) download_argv[count++] = "--dry-run";
    if (options->clear_stale_locks) download_argv[count++] = "--clear-stale-locks";
#undef PULL_ARG
    model_download_format_bytes(expected, sizeof(expected),
                                representation->size_bytes);
    if (!options->json)
        yvex_cli_out_writef(stdout,
                            "model       %s\nprovider    Hugging Face\nrepository  %s\n"
                            "revision    %s\nformat      %s\nprecision   %s\nexpected    %s\n"
                            "representation %s\n\n",
                            name, locator->repository, remote->resolved_revision,
                            representation->format, representation->precision,
                            representation->size_known ? expected : "unknown",
                            representation->identity);
    if (!options->json)
        yvex_cli_out_writef(stdout, "family      %s%s%s\n\n",
                            pull_family(options, remote),
                            remote->family_evidence[0] ? " · " : "",
                            remote->family_evidence);
    rc = yvex_models_download_surface_command(count, download_argv);
    if (target && !strcmp(target->upstream_revision, remote->resolved_revision))
        prepare_selector = target->target_id;
    if (!rc && options->prepare && !options->dry_run)
        rc = pull_prepare_followup(argv[0], prepare_selector, models_root,
                                   options->quant);
    else if (!rc && options->prepare && options->dry_run)
        yvex_cli_out_fputs(
            "prepare     planned after acquisition; no source or build state changed\n",
            stdout);
    return rc;
}

static int pull_remote(int argc, char **argv, const model_pull_options *options,
                       const yvex_source_locator *locator, const char *models_root)
{
    yvex_remote_inspect_options inspect = {0};
    yvex_remote_catalog *catalog = NULL;
    const yvex_remote_model *remote;
    const yvex_model_representation *representation = NULL;
    yvex_error err;
    int rc;
    inspect.provider = YVEX_ACCOUNT_PROVIDER_HUGGINGFACE;
    inspect.repository = locator->repository;
    inspect.revision = options->revision ? options->revision
                                         : locator->revision_present ? locator->revision : NULL;
    yvex_error_clear(&err);
    rc = yvex_remote_model_inspect(&catalog, &inspect, &err);
    if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));
    remote = yvex_remote_catalog_count(catalog) ? yvex_remote_catalog_at(catalog, 0u) : NULL;
    if (!remote || !revision_immutable(remote->resolved_revision)) {
        yvex_cli_out_fputs("yvex: provider did not resolve an immutable model revision\n", stderr);
        yvex_remote_catalog_close(catalog);
        return 1;
    }
    if (options->family && !model_download_family_valid(options->family)) {
        yvex_cli_out_fputs(
            "yvex: model pull --family override must be a canonical lower-case key\n",
            stderr);
        yvex_remote_catalog_close(catalog);
        return 2;
    }
    rc = representation_select(catalog, remote, options, &representation);
    if (rc) { yvex_remote_catalog_close(catalog); return rc; }
    if (!representation->file_pattern[0] && !options->reference) {
        yvex_cli_out_fputs(
            "yvex: provider representation has no deterministic payload selection\n",
            stderr);
        yvex_remote_catalog_close(catalog);
        return 3;
    }
    if (options->reference && options->prepare) {
        yvex_cli_out_fputs(
            "yvex: a remote reference cannot be prepared without an admitted "
            "streamed transport; pull the payload first\n",
            stderr);
        yvex_remote_catalog_close(catalog);
        return 3;
    }
    if (options->reference) {
        yvex_source_reference_options reference = {0};
        yvex_source_reference_result result;
        char derived_name[YVEX_REMOTE_NAME_CAP];
        const char *name;
        pull_product_name(locator->repository, derived_name);
        name = options->name ? options->name : derived_name;
        if (options->dry_run) {
            if (options->json) {
                yvex_cli_out_fputs(
                    "{\"schema\":\"yvex.model.pull.v1\",\"dry_run\":true,\"state\":\"REMOTE\",\"model\":",
                    stdout);
                yvex_cli_out_json_string(stdout, name);
                yvex_cli_out_fputs(",\"origin\":", stdout);
                yvex_cli_out_json_string(stdout, locator->canonical);
                yvex_cli_out_fputs(",\"resolved_revision\":", stdout);
                yvex_cli_out_json_string(stdout, remote->resolved_revision);
                yvex_cli_out_fputs(",\"format\":", stdout);
                yvex_cli_out_json_string(stdout, representation->format);
                yvex_cli_out_fputs("}\n", stdout);
            } else {
                yvex_cli_out_writef(
                    stdout,
                    "model       %s\nstate       REMOTE\nrepository  %s\nrevision    %s\n"
                    "format      %s\naction      dry-run; no reference registered\n",
                    name, locator->repository, remote->resolved_revision,
                    representation->format);
            }
            yvex_remote_catalog_close(catalog);
            return 0;
        }
        reference.locator = locator;
        reference.models_root = models_root;
        reference.name = name;
        reference.family = pull_family(options, remote);
        reference.resolved_revision = remote->resolved_revision;
        reference.format = representation->format;
        reference.precision = representation->precision;
        reference.size_bytes = representation->size_bytes;
        reference.size_known = representation->size_known;
        yvex_error_clear(&err);
        rc = yvex_source_register_reference(&reference, &result, &err);
        if (rc == YVEX_OK && options->json) {
            yvex_cli_out_fputs(
                "{\"schema\":\"yvex.model.pull.v1\",\"state\":\"REMOTE\",\"model\":",
                stdout);
            yvex_cli_out_json_string(stdout, name);
            yvex_cli_out_fputs(",\"origin\":", stdout);
            yvex_cli_out_json_string(stdout, result.immutable_uri);
            yvex_cli_out_fputs(",\"revision\":", stdout);
            yvex_cli_out_json_string(stdout, remote->resolved_revision);
            yvex_cli_out_fputs(",\"format\":", stdout);
            yvex_cli_out_json_string(stdout, representation->format);
            yvex_cli_out_fputs(",\"precision\":", stdout);
            yvex_cli_out_json_string(stdout, representation->precision);
            yvex_cli_out_fputs(",\"record\":", stdout);
            yvex_cli_out_json_string(stdout, result.record_path);
            yvex_cli_out_fputs("}\n", stdout);
        } else if (rc == YVEX_OK)
            yvex_cli_out_writef(
                stdout,
                "model       %s\nstate       REMOTE\norigin      %s\nrevision    %s\n"
                "format      %s\nprecision   %s\nrecord      %s\n",
                name, result.immutable_uri, remote->resolved_revision,
                representation->format, representation->precision,
                result.record_path);
        else rc = print_yvex_error(&err, exit_for_status(rc));
    } else {
        rc = pull_remote_download(argc, argv, options, locator, remote, catalog,
                                  representation, models_root);
    }
    yvex_remote_catalog_close(catalog);
    return rc;
}

static int pull_local_storage(const model_pull_options *options,
                              yvex_source_storage_kind *storage)
{
    char line[32];
    if (options->reference) { *storage = YVEX_SOURCE_STORAGE_EXTERNAL; return 0; }
    if (options->managed) { *storage = YVEX_SOURCE_STORAGE_MANAGED; return 0; }
    if (!isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO)) {
        yvex_cli_out_fputs(
            "yvex: local model pull requires --managed or --reference when input is not a terminal\n",
            stderr);
        return 2;
    }
    yvex_cli_out_fputs(
        "Import into YVEX storage?\n  [1] managed copy\n  [2] external reference\nChoice [1-2]: ",
        stdout);
    fflush(stdout);
    if (!fgets(line, sizeof(line), stdin)) return 2;
    *storage = line[0] == '1' ? YVEX_SOURCE_STORAGE_MANAGED
                              : line[0] == '2' ? YVEX_SOURCE_STORAGE_EXTERNAL
                                               : (yvex_source_storage_kind)-1;
    if (*storage != YVEX_SOURCE_STORAGE_MANAGED &&
        *storage != YVEX_SOURCE_STORAGE_EXTERNAL) {
        yvex_cli_out_fputs("model pull cancelled\n", stdout);
        return 2;
    }
    return 0;
}

static int pull_local(char **argv, const model_pull_options *options,
                      const yvex_source_locator *locator, const char *models_root)
{
    const char *model_name;
    yvex_source_storage_kind storage;
    yvex_source_representation_fact inspected;
    yvex_source_import_options import = {0};
    yvex_source_import_result result;
    yvex_error err;
    int rc;
    if (options->resume) {
        yvex_cli_out_fputs(
            "yvex: local representations are copied atomically and have no resumable acquisition\n",
            stderr);
        return 3;
    }
    yvex_error_clear(&err);
    rc = yvex_source_representation_inspect_local(locator, &inspected, &err);
    if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));
    if ((options->format && strcasecmp(options->format, inspected.format)) ||
        (options->variant && strcasecmp(options->variant, inspected.precision) &&
         strcmp(options->variant, inspected.digest))) {
        yvex_cli_out_writef(stderr,
                            "yvex: local representation is %s/%s, not the requested selection\n",
                            inspected.format,
                            inspected.precision[0] ? inspected.precision : "not recorded");
        return 2;
    }
    rc = pull_local_storage(options, &storage);
    if (rc) return rc;
    if (options->dry_run) {
        if (options->json) {
            yvex_cli_out_fputs(
                "{\"schema\":\"yvex.model.pull.v1\",\"dry_run\":true,\"model\":",
                stdout);
            yvex_cli_out_json_string(stdout, options->name ? options->name
                                                           : inspected.name);
            yvex_cli_out_fputs(",\"origin\":", stdout);
            yvex_cli_out_json_string(stdout, locator->canonical);
            yvex_cli_out_fputs(",\"format\":", stdout);
            yvex_cli_out_json_string(stdout, inspected.format);
            yvex_cli_out_fputs(",\"digest\":", stdout);
            yvex_cli_out_json_string(stdout, inspected.digest);
            yvex_cli_out_writef(stdout, ",\"size_bytes\":%llu}\n",
                                inspected.size_bytes);
        } else {
            yvex_cli_out_writef(
                stdout,
                "model       %s\norigin      %s\nstorage     %s\nformat      %s\n"
                "size        %llu bytes\ndigest      %s\naction      dry-run; no files copied or registered\n",
                options->name ? options->name : inspected.name,
                locator->canonical,
                storage == YVEX_SOURCE_STORAGE_EXTERNAL ? "external" : "managed",
                inspected.format, inspected.size_bytes, inspected.digest);
        }
        if (options->prepare)
            yvex_cli_out_fputs(
                "prepare     planned after acquisition; no source or build state changed\n",
                stdout);
        return 0;
    }
    import.locator = locator;
    import.models_root = models_root;
    import.name = options->name;
    import.family = options->family;
    import.storage = storage;
    yvex_error_clear(&err);
    rc = yvex_source_import_local(&import, &result, &err);
    if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));
    model_name = options->name ? options->name : result.representation.name;
    if (options->json) {
        yvex_cli_out_fputs("{\"schema\":\"yvex.model.pull.v1\",\"model\":", stdout);
        yvex_cli_out_json_string(stdout, model_name);
        yvex_cli_out_fputs(",\"origin\":", stdout);
        yvex_cli_out_json_string(stdout, result.origin_uri);
        yvex_cli_out_fputs(",\"location\":", stdout);
        yvex_cli_out_json_string(stdout, result.source_path);
        yvex_cli_out_fputs(",\"storage\":", stdout);
        yvex_cli_out_json_string(stdout, result.storage);
        yvex_cli_out_fputs(",\"format\":", stdout);
        yvex_cli_out_json_string(stdout, result.representation.format);
        yvex_cli_out_fputs(",\"digest\":", stdout);
        yvex_cli_out_json_string(stdout, result.representation.digest);
        yvex_cli_out_writef(stdout, ",\"size_bytes\":%llu}\n",
                            result.representation.size_bytes);
    } else {
        yvex_cli_out_writef(stdout,
            "model       %s\norigin      %s\nstorage     %s\nformat      %s\n"
            "size        %llu bytes\ndigest      %s\nlocation    %s\nstate       %s\n",
            model_name, result.origin_uri, result.storage,
            result.representation.format, result.representation.size_bytes,
            result.representation.digest, result.source_path,
            storage == YVEX_SOURCE_STORAGE_EXTERNAL ? "EXTERNAL" : "VERIFIED");
    }
    if (options->prepare && !options->dry_run)
        rc = pull_prepare_followup(argv[0], model_name, models_root,
                                   options->quant);
    return rc;
}

int yvex_model_pull_command(int arg_count, char **args)
{
    model_pull_options options;
    yvex_source_locator locator;
    yvex_error err;
    char models_root[YVEX_PATH_CAP];
    int rc = pull_options_parse(arg_count, args, &options);
    if (rc) return rc;
    if (options.stream) {
        yvex_cli_out_fputs(
            "yvex: streamed preparation is unavailable: no admitted transport "
            "currently provides authenticated range/retry semantics\n",
            stderr);
        return 3;
    }
    rc = workflow_models_root(options.models_root, models_root);
    if (rc) return rc;
    yvex_error_clear(&err);
    rc = yvex_source_locator_parse(options.source, &locator, &err);
    if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));
    if (locator.kind == YVEX_SOURCE_LOCATOR_HUGGINGFACE)
        return pull_remote(arg_count, args, &options, &locator, models_root);
    if (locator.kind == YVEX_SOURCE_LOCATOR_LOCAL_PATH)
        return pull_local(args, &options, &locator, models_root);
    yvex_cli_out_writef(stderr, "yvex: %s transport unavailable\n",
                        yvex_source_locator_kind_name(locator.kind));
    return 3;
}

int yvex_model_pull_lifecycle_command(int arg_count, char **args)
{
    char *plumbing[32];
    yvex_model_library *library = NULL;
    yvex_local_catalog_options open = {0};
    const yvex_model_library_entry *model;
    const yvex_local_source_record *source = NULL;
    yvex_error err;
    unsigned long long model_index = 0u, source_index, provider_sources = 0u;
    const char *action;
    int count = 0, index;
    if (arg_count < 4) {
        yvex_cli_out_fputs("yvex: model acquisition command requires MODEL\n", stderr);
        return 2;
    }
    for (index = 4; index < arg_count; ++index) {
        if (!strcmp(args[index], "--models-root") && index + 1 < arg_count)
            open.models_root = args[++index];
    }
    yvex_error_clear(&err);
    if (yvex_model_library_open(&library, &open, &err) != YVEX_OK)
        return print_yvex_error(&err, exit_for_status(yvex_error_code(&err)));
    index = yvex_cli_model_find(library, args[3], &model_index);
    if (index != 1) {
        yvex_cli_out_writef(stderr, "yvex: model selector %s: %s\n", args[3],
                            index ? "ambiguous" : "not found");
        yvex_model_library_close(library);
        return 2;
    }
    model = yvex_model_library_at(library, model_index);
    for (source_index = 0u;
         source_index < yvex_model_library_source_count(library, model_index);
         ++source_index) {
        const yvex_local_source_record *candidate =
            yvex_model_library_source_at(library, model_index, source_index);
        if (strcmp(candidate->provider, "huggingface") ||
            !candidate->repository[0] || !candidate->revision[0]) continue;
        source = candidate;
        provider_sources++;
    }
    if (provider_sources != 1u) {
        yvex_cli_out_writef(
            stderr,
            "yvex: model %s has %llu provider acquisitions; "
            "model-level lifecycle requires exactly one\n",
            yvex_cli_model_selector(model), provider_sources);
        yvex_model_library_close(library);
        return 2;
    }
    action = !strcmp(args[2], "pull-stop") ? "stop" : "status";
    plumbing[count++] = args[0];
    plumbing[count++] = "models";
    plumbing[count++] = "download";
    plumbing[count++] = (char *)action;
    plumbing[count++] = "--provider";
    plumbing[count++] = "huggingface";
    plumbing[count++] = "--repo";
    plumbing[count++] = (char *)source->repository;
    plumbing[count++] = "--family";
    plumbing[count++] = (char *)source->family;
    plumbing[count++] = "--name";
    plumbing[count++] = (char *)source->name;
    plumbing[count++] = "--revision";
    plumbing[count++] = (char *)source->revision;
    for (index = 4; index < arg_count; ++index) {
        if (count == (int)(sizeof(plumbing) / sizeof(plumbing[0]))) {
            yvex_model_library_close(library);
            return 2;
        }
        plumbing[count++] = args[index];
    }
    index = yvex_models_download_surface_command(count, plumbing);
    yvex_model_library_close(library);
    return index;
}


static int push_options_parse(int argc, char **argv, model_push_options *out)
{
    int index;
    memset(out, 0, sizeof(*out));
    for (index = 3; index < argc; ++index) {
        const char *flag = argv[index];
        const char **field = NULL;
        if (!strcmp(flag, "--models-root")) field = &out->models_root;
        else if (!strcmp(flag, "--registry")) field = &out->registry_path;
        else if (!strcmp(flag, "--variant")) field = &out->variant;
        else if (!strcmp(flag, "--representation")) field = &out->representation;
        if (field) {
            if (!workflow_value("model push", flag, argc, argv, &index, field)) return 2;
        } else if (!strcmp(flag, "--json")) out->json = 1;
        else if (flag[0] == '-') {
            yvex_cli_out_writef(stderr, "yvex: unknown model push option: %s\n", flag);
            return 2;
        } else if (!out->model) out->model = flag;
        else if (!out->destination) out->destination = flag;
        else return 2;
    }
    if (!out->model || !out->destination) {
        yvex_cli_out_fputs("yvex: model push requires MODEL DESTINATION\n", stderr);
        return 2;
    }
    if (out->representation && strcmp(out->representation, "artifact") &&
        strcmp(out->representation, "source")) {
        yvex_cli_out_fputs(
            "yvex: model push --representation requires artifact or source\n",
            stderr);
        return 2;
    }
    return 0;
}

typedef struct {
    char ordinal[16], precision[YVEX_REMOTE_PRECISION_CAP], size[32], digest[20];
    yvex_cli_table_cell cells[5];
    yvex_cli_table_row row;
} push_choice_row;

static int push_choice_read(unsigned long long count, unsigned long long *choice)
{
    char line[64], *end = NULL, *newline;
    unsigned long long value;
    yvex_cli_out_writef(stdout, "Representation [1-%llu, q to cancel] > ", count);
    yvex_cli_out_flush(stdout);
    if (!fgets(line, sizeof(line), stdin)) return 0;
    newline = strpbrk(line, "\r\n");
    if (newline) *newline = '\0';
    if (!line[0] || !strcasecmp(line, "q")) return 0;
    errno = 0;
    value = strtoull(line, &end, 10);
    if (errno || end == line || *end || !value || value > count) return 0;
    *choice = value - 1u;
    return 1;
}

static const yvex_model_artifact_fact *push_artifact_choose_tty(
    const yvex_model_artifact_fact *const *items, unsigned long long count)
{
    static const yvex_cli_table_column columns[] = {
        {"#", 1u, 3u, YVEX_CLI_TABLE_RIGHT, 0},
        {"FORMAT", 6u, 12u, YVEX_CLI_TABLE_LEFT, 0},
        {"QUANT/PRECISION", 10u, 28u, YVEX_CLI_TABLE_LEFT, 0},
        {"SIZE", 8u, 14u, YVEX_CLI_TABLE_RIGHT, 0},
        {"DIGEST", 12u, 18u, YVEX_CLI_TABLE_LEFT, 0}
    };
    push_choice_row *storage = calloc((size_t)count, sizeof(*storage));
    yvex_cli_table_row *rows = calloc((size_t)count, sizeof(*rows));
    unsigned long long index, choice;
    if (!storage || !rows) { free(storage); free(rows); return NULL; }
    yvex_cli_out_fputs("Select representation to publish\n\n", stdout);
    for (index = 0u; index < count; ++index) {
        snprintf(storage[index].ordinal, sizeof(storage[index].ordinal), "%llu",
                 index + 1u);
        yvex_cli_precision_format(
            storage[index].precision, sizeof(storage[index].precision),
            items[index]->physical_variant[0] ? items[index]->physical_variant
                                              : items[index]->artifact_class);
        model_download_format_bytes(storage[index].size,
                                    sizeof(storage[index].size),
                                    items[index]->file_size);
        snprintf(storage[index].digest, sizeof(storage[index].digest), "%.16s",
                 items[index]->identity);
        storage[index].cells[0] = (yvex_cli_table_cell){storage[index].ordinal,
                                                        YVEX_CLI_TABLE_DIM};
        storage[index].cells[1] = (yvex_cli_table_cell){items[index]->format,
                                                        YVEX_CLI_TABLE_PLAIN};
        storage[index].cells[2] = (yvex_cli_table_cell){storage[index].precision,
                                                        YVEX_CLI_TABLE_PLAIN};
        storage[index].cells[3] = (yvex_cli_table_cell){storage[index].size,
                                                        YVEX_CLI_TABLE_PLAIN};
        storage[index].cells[4] = (yvex_cli_table_cell){storage[index].digest,
                                                        YVEX_CLI_TABLE_DIM};
        storage[index].row.cells = storage[index].cells;
        rows[index] = storage[index].row;
    }
    (void)yvex_cli_table_render(stdout, columns, 5u, rows, (size_t)count);
    if (!push_choice_read(count, &choice)) {
        free(storage); free(rows);
        return NULL;
    }
    free(storage); free(rows);
    return items[choice];
}

static const yvex_model_artifact_fact *push_artifact_select(
    const yvex_model_library *library, unsigned long long model_index,
    const model_push_options *options)
{
    unsigned long long index, matches = 0u;
    const yvex_model_artifact_fact *items[YVEX_MODELS_ARTIFACT_ROWS_CAP];
    const yvex_model_artifact_fact *selected = NULL;
    for (index = 0u; index < yvex_model_library_artifact_count(library, model_index); ++index) {
        const yvex_model_artifact_fact *item =
            yvex_model_library_artifact_at(library, model_index, index);
        if (options->variant && strcmp(options->variant, item->identity) &&
            strcasecmp(options->variant, item->physical_variant) &&
            strcasecmp(options->variant, item->artifact_class)) continue;
        selected = item;
        if (matches < YVEX_MODELS_ARTIFACT_ROWS_CAP) items[matches] = item;
        matches++;
    }
    if (matches == 1u) return selected;
    if (!matches)
        yvex_cli_out_fputs("yvex: no publishable representation matches --variant\n",
                           stderr);
    else if (!options->variant && matches <= YVEX_MODELS_ARTIFACT_ROWS_CAP &&
             isatty(STDIN_FILENO) && isatty(STDOUT_FILENO))
        return push_artifact_choose_tty(items, matches);
    else
        yvex_cli_out_fputs(
            "yvex: multiple publishable representations exist; pass an exact --variant from `yvex model show`\n",
            stderr);
    return NULL;
}

int yvex_model_push_command(int arg_count, char **args)
{
    model_push_options options;
    yvex_local_catalog_options open;
    yvex_model_library *library = NULL;
    const yvex_model_artifact_fact *artifact = NULL;
    const yvex_local_source_record *source = NULL;
    yvex_source_export_options export = {0};
    yvex_source_export_result result;
    yvex_error err;
    unsigned long long model_index = 0u;
    int matches, rc = push_options_parse(arg_count, args, &options);
    if (rc) return rc;
    if (!strncmp(options.destination, "hf://", 5u)) {
        yvex_cli_out_fputs("yvex: Hugging Face provider write unavailable\n", stderr);
        return 3;
    }
    if (strstr(options.destination, "://") && strncmp(options.destination, "file://", 7u)) {
        const char *separator = strstr(options.destination, "://");
        yvex_cli_out_writef(stderr, "yvex: %.*s transport unavailable\n",
                            (int)(separator - options.destination), options.destination);
        return 3;
    }
    memset(&open, 0, sizeof(open));
    open.models_root = options.models_root;
    open.registry_path = options.registry_path;
    yvex_error_clear(&err);
    rc = yvex_model_library_open(&library, &open, &err);
    if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));
    matches = yvex_cli_model_find(library, options.model, &model_index);
    if (matches != 1) {
        yvex_cli_out_writef(stderr, "yvex: model selector %s is %s\n", options.model,
                            matches ? "ambiguous" : "not found");
        yvex_model_library_close(library);
        return 2;
    }
    if ((!options.representation || !strcmp(options.representation, "artifact")) &&
        yvex_model_library_artifact_count(library, model_index))
        artifact = push_artifact_select(library, model_index, &options);
    if (!artifact && (!options.representation ||
                      !strcmp(options.representation, "artifact")) &&
        yvex_model_library_artifact_count(library, model_index)) {
        yvex_model_library_close(library);
        return 2;
    }
    if (!artifact && options.representation &&
        !strcmp(options.representation, "artifact")) {
        yvex_cli_out_fputs("yvex: model has no local artifact to publish\n", stderr);
        yvex_model_library_close(library);
        return 1;
    }
    if (!artifact && yvex_model_library_source_count(library, model_index))
        source = yvex_model_library_source_at(library, model_index, 0u);
    if (!artifact && (!source || !source->path[0])) {
        yvex_cli_out_fputs("yvex: model has no local publishable representation\n", stderr);
        yvex_model_library_close(library);
        return 1;
    }
    export.source_path = artifact ? artifact->path : source->path;
    export.expected_digest = artifact ? artifact->identity : source->digest;
    export.destination_path = options.destination;
    yvex_error_clear(&err);
    rc = yvex_source_export_local(&export, &result, &err);
    if (rc != YVEX_OK) {
        yvex_model_library_close(library);
        return print_yvex_error(&err, exit_for_status(rc));
    }
    if (options.json) {
        yvex_cli_out_fputs("{\"schema\":\"yvex.model.push.v1\",\"model\":", stdout);
        yvex_cli_out_json_string(stdout, yvex_cli_model_selector(
                                          yvex_model_library_at(library, model_index)));
        yvex_cli_out_fputs(",\"representation_identity\":", stdout);
        yvex_cli_out_json_string(stdout, result.digest);
        yvex_cli_out_fputs(",\"destination\":", stdout);
        yvex_cli_out_json_string(stdout, result.destination_path);
        yvex_cli_out_writef(stdout, ",\"bytes\":%llu}\n", result.bytes);
    } else {
        yvex_cli_out_writef(stdout,
            "model           %s\nrepresentation  %s\nbytes           %llu\n"
            "digest          %s\ndestination     %s\nstate           published\n",
            yvex_cli_model_selector(yvex_model_library_at(library, model_index)),
            artifact ? artifact->format : source->format, result.bytes,
            result.digest, result.destination_path);
    }
    yvex_model_library_close(library);
    return 0;
}
