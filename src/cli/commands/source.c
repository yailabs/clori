/*
 * Bind source-manifest report CLI input to the typed source report API.
 *
 * Adapter stays thin and does not hide domain behavior. Command dispatch is not source
 * verification or runtime readiness.
 */
#include "src/cli/input/private.h"
#include "src/cli/io/private.h"
#include "src/cli/render/private.h"

#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <yvex/source.h>

static const char *const literal_lines_0[] = {
    "       yvex compile source manifest report --family deepseek|qwen|gemma --release v0.1.0 [options]\n",
    "Source manifest scans a local official-weight source directory and writes provenance JSON. It "
    "does "
    "not download, parse safetensors payloads, quantize, emit GGUF, materialize, or infer.\n",
    "The DeepSeek report verifies exact source identity and metadata without reading tensor "
    "payloads. Qwen "
    "and Gemma remain bounded engineering reports. No report emits artifacts, executes runtime "
    "paths, "
    "generates, evaluates, benchmarks, or marks a release ready."};

typedef struct {
    const char *name;
    yvex_source_status status;
} source_status_arg;

static const source_status_arg source_status_args[] = {
    {"unknown", YVEX_SOURCE_STATUS_UNKNOWN},       {"in-progress", YVEX_SOURCE_STATUS_IN_PROGRESS},
    {"incomplete", YVEX_SOURCE_STATUS_INCOMPLETE}, {"complete", YVEX_SOURCE_STATUS_COMPLETE},
    {"failed", YVEX_SOURCE_STATUS_FAILED},
};

typedef struct {
    const char *name;
    size_t offset;
} source_manifest_arg;

static const source_manifest_arg source_manifest_args[] = {
    {"--hf-repo", offsetof(yvex_source_manifest_options, repo)},
    {"--revision", offsetof(yvex_source_manifest_options, revision)},
    {"--license", offsetof(yvex_source_manifest_options, license)},
    {"--model-card", offsetof(yvex_source_manifest_options, model_card)},
    {"--local-path", offsetof(yvex_source_manifest_options, local_path)},
    {"--node", offsetof(yvex_source_manifest_options, node_name)},
    {"--dry-run-log", offsetof(yvex_source_manifest_options, dry_run_log)},
    {"--download-log", offsetof(yvex_source_manifest_options, download_log)},
    {"--pid-file", offsetof(yvex_source_manifest_options, pid_file)},
    {"--download-command", offsetof(yvex_source_manifest_options, download_command)},
};

int yvex_source_manifest_report_command(int argc, char **argv);
void yvex_source_manifest_help(FILE *fp);

static int source_cli_parse_status(const char *text, yvex_source_status *out) {
    size_t index;

    if (!text || !out)
        return 0;
    for (index = 0; index < sizeof(source_status_args) / sizeof(source_status_args[0]); ++index) {
        if (strcmp(text, source_status_args[index].name) == 0) {
            *out = source_status_args[index].status;
            return 1;
        }
    }
    return 0;
}

static const source_manifest_arg *source_manifest_arg_find(const char *name) {
    size_t index;

    for (index = 0; index < sizeof(source_manifest_args) / sizeof(source_manifest_args[0]); ++index)
        if (strcmp(name, source_manifest_args[index].name) == 0)
            return &source_manifest_args[index];
    return NULL;
}

static int source_cli_create_manifest(int argc, char **argv) {
    yvex_source_manifest_options options;
    yvex_source_manifest_summary summary;
    yvex_error err;
    const char *out_path = NULL;
    int i;
    int rc;

    memset(&options, 0, sizeof(options));
    memset(&summary, 0, sizeof(summary));
    yvex_error_clear(&err);
    options.status = YVEX_SOURCE_STATUS_UNKNOWN;
    options.include_files = 1;

    i = 3;
    while (i < argc) {
        const char *name = argv[i];
        const char *value;
        const source_manifest_arg *argument;

        if (i + 1 >= argc) {
            yvex_cli_out_writef(stderr, "yvex: option requires a value: %s\n", name);
            return 2;
        }
        value = argv[i + 1];
        argument = source_manifest_arg_find(name);

        if (argument) {
            const char **field = (const char **)((unsigned char *)&options + argument->offset);

            *field = value;
        } else if (strcmp(name, "--status") == 0) {
            if (!source_cli_parse_status(value, &options.status)) {
                yvex_cli_out_writef(stderr, "yvex: unknown source status: %s\n", value);
                return 2;
            }
            if (options.status == YVEX_SOURCE_STATUS_COMPLETE) {
                yvex_cli_out_writef(stderr, "yvex: source status complete is verifier-owned; run "
                                            "strict exact-source verification\n");
                return 2;
            }
        } else if (strcmp(name, "--out") == 0) {
            out_path = value;
        } else {
            yvex_cli_out_writef(stderr, "yvex: unknown source-manifest option: %s\n", name);
            return 2;
        }
        i += 2;
    }

    if (!options.repo || !options.revision || !options.local_path || !out_path) {
        yvex_cli_out_writef(stderr,
                            "yvex: --hf-repo, --revision, --local-path, and --out are required\n");
        return 2;
    }

    rc = yvex_source_manifest_write_json(out_path, &options, &summary, &err);
    if (rc != YVEX_OK) {
        int exit_code = exit_for_status(rc);

        return print_yvex_error(&err, exit_code == 1 ? 3 : exit_code);
    }

    yvex_cli_out_writef(stdout, "compile source manifest: written\n");
    yvex_cli_out_writef(stdout, "repo: %s\n", options.repo);
    yvex_cli_out_writef(stdout, "revision: %s\n", options.revision);
    yvex_cli_out_writef(stdout, "local_path: %s\n", options.local_path);
    yvex_cli_out_writef(stdout, "status: %s\n", yvex_source_status_name(options.status));
    yvex_cli_out_writef(stdout, "files: %llu\n", summary.file_count);
    yvex_cli_out_writef(stdout, "safetensors: %llu\n", summary.safetensors_count);
    yvex_cli_out_writef(stdout, "total_size_bytes: %llu\n", summary.total_size_bytes);
    yvex_cli_out_writef(stdout, "out: %s\n", out_path);
    yvex_cli_out_writef(stdout, "status: source-manifest-written\n");
    return 0;
}

int yvex_source_manifest_command(int argc, char **argv) {
    if (argc >= 3 && (strcmp(argv[2], "--help") == 0 || strcmp(argv[2], "-h") == 0)) {
        yvex_source_manifest_help(stdout);
        return 0;
    }

    if (argc < 3) {
        yvex_cli_out_writef(stderr, "yvex: source-manifest requires a subcommand\n");
        yvex_cli_out_writef(stderr, "usage: yvex compile source manifest create --hf-repo REPO --revision "
                                    "REV --local-path DIR --status "
                                    "STATUS --out FILE\n");
        yvex_cli_out_writef(
            stderr,
            "       yvex compile source manifest report --family qwen --release v0.1.0 [options]\n");
        return 2;
    }

    if (strcmp(argv[2], "report") == 0) {
        return yvex_source_manifest_report_command(argc, argv);
    }
    if (strcmp(argv[2], "inspect") == 0) {
        yvex_cli_out_writef(
            stderr, "yvex: source-manifest inspect is not implemented in open-weight intake\n");
        return 5;
    }
    if (strcmp(argv[2], "create") == 0) {
        return source_cli_create_manifest(argc, argv);
    }

    yvex_cli_out_writef(stderr, "yvex: unknown source-manifest subcommand: %s\n", argv[2]);
    return 2;
}

void yvex_source_manifest_help(FILE *fp) {
    yvex_cli_out_writef(fp, "usage: yvex compile source manifest create --hf-repo REPO --revision REV "
                            "--local-path DIR --status STATUS "
                            "--out FILE [--license TEXT] [--model-card URL] [--node NAME] "
                            "[--dry-run-log FILE] [--download-log "
                            "FILE] [--pid-file FILE] [--download-command TEXT]\n");
    yvex_cli_out_lines(fp, literal_lines_0, sizeof(literal_lines_0) / sizeof(literal_lines_0[0]));
    yvex_cli_out_writef(fp, "Report options: --source DIR --models-root DIR --target TARGET "
                            "--include-files --include-config --"
                            "include-blockers --include-next --strict --audit --json --output "
                            "normal|table|audit|json\n");
}

int yvex_source_manifest_report_command(int argc, char **argv) {
    yvex_source_args args;
    yvex_source_report_request request;
    yvex_source_report report;
    yvex_error err;
    int rc;

    yvex_error_clear(&err);
    rc = yvex_source_args_parse(argc, argv, &args, &err);
    if (rc != YVEX_OK) {
        yvex_cli_out_writef(stderr, "%s\n", yvex_error_message(&err));
        rc = exit_for_status(rc);
        return rc == 1 ? 3 : rc;
    }
    if (args.help) {
        yvex_source_render_help(stdout);
        return 0;
    }

    yvex_source_report_request_from_parsed(&request, &args);
    rc = yvex_source_report_build(&request, &report, &err);
    if (rc != YVEX_OK) {
        int exit_code = exit_for_status(rc);

        return print_yvex_error(&err, exit_code == 1 ? 3 : exit_code);
    }

    return yvex_source_render(stdout, args.render_mode, &report);
}
