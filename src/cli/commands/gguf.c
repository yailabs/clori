#include "src/cli/input/private.h"
#include "src/cli/io/private.h"
#include <yvex/artifact.h>
#include <yvex/gguf.h>
#include <yvex/qtype.h>
#include <yvex/core.h>
#include <yvex/quant.h>
#include <yvex/model.h>
#include <yvex/source.h>
#include <yvex/tokenizer.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *const literal_pair_0[] = {
    "       yvex compile quant policy derive --template FILE --arch NAME --out FILE",
    "\nQuant policy handles declarative qtype policy manifests. It does not quantize tensors or infer."};

static const char *const literal_pair_1[] = { "       yvex compile quant job inspect|validate --manifest FILE",
    "\nQuant job records an external quantization/conversion job manifest without running arbitrary tools."};

static const char *const literal_pair_2[] = { "       yvex compile quant imatrix inspect|validate --manifest FILE",
    "\nImatrix handles calibration artifact manifests. It does not generate imatrix data, calibrate, "
        "quantize, emit GGUF, materialize, or infer."
};

static const char *const literal_pair_3[] = {
    "       yvex compile artifact template compare --template FILE --native-source DIR",
    "\nGGUF template validates metadata, tokenizer metadata, tensor directory, tensor roles, and optional "
        "exact-name native inventory comparison."
};

static const char *const literal_pair_4[] = {
    "       yvex compile quant convert emit --arch ARCH --native-source DIR --tensor NAME --target-qtype QTYPE --out "
        "FILE [--overwrite]",
    "\nConvert plans or emits selected open-weight GGUF tensor artifacts. It does not infer, execute a "
        "full model, or claim generation support."
};

static const char *const literal_pair_5[] = { "{",
    "  \"schema\": \"yvex.tensor_map.v1\","};

static const char *const literal_pair_6[] = { "  }",
    "}"};

static const char *const literal_pair_7[] = { "{",
    "  \"schema\": \"yvex.native_weights.v1\","};

static const char *const literal_pair_8[] = { "execution_ready: false",
    "status: conversion-gguf-written"};

static int cli_parse_gguf_template_options(int arg_count, char **args, int start,
                                           const char **template_path,
                                           const char **native_source,
                                           int *require_all)
{
    int i = start;

    *template_path = NULL;
    *native_source = NULL;
    *require_all = 0;
    while (i < arg_count) {
        if (strcmp(args[i], "--require-all-template-tensors-in-native") == 0) {
            *require_all = 1;
            i++;
            continue;
        }
        if (i + 1 >= arg_count) {
            yvex_cli_out_writef(stderr, "yvex: gguf-template option requires a value: %s\n", args[i]);
            return 2;
        }
        if (strcmp(args[i], "--template") == 0) {
            *template_path = args[i + 1];
        } else if (strcmp(args[i], "--native-source") == 0) {
            *native_source = args[i + 1];
        } else {
            yvex_cli_out_writef(stderr, "yvex: unknown gguf-template option: %s\n", args[i]);
            return 2;
        }
        i += 2;
    }
    return 0;
}

static void cli_print_template_issues(const yvex_gguf_template *tmpl)
{
    unsigned long long i;
    unsigned long long count = yvex_gguf_template_issue_count(tmpl);

    for (i = 0; i < count; ++i) {
        const yvex_gguf_template_issue *issue = yvex_gguf_template_issue_at(tmpl, i);
        if (!issue) {
            continue;
        }
        if (issue->tensor_name && issue->tensor_name[0] != '\0') {
            yvex_cli_out_writef(stdout, "issue %llu %s tensor=\"%s\" message=\"%s\"\n",
                   i,
                   yvex_gguf_template_issue_kind_name(issue->kind),
                   issue->tensor_name,
                   issue->message ? issue->message : "");
        } else {
            yvex_cli_out_writef(stdout, "issue %llu %s message=\"%s\"\n",
                   i,
                   yvex_gguf_template_issue_kind_name(issue->kind),
                   issue->message ? issue->message : "");
        }
    }
}

static int command_gguf_template(int arg_count, char **args)
{
    yvex_gguf_template_options options;
    yvex_gguf_template *tmpl = NULL;
    yvex_gguf_template_summary summary;
    yvex_error err;
    const char *template_path;
    const char *native_source;
    int require_all;
    int rc;

    yvex_error_clear(&err);
    if (arg_count >= 3 && (strcmp(args[2], "--help") == 0 || strcmp(args[2], "-h") == 0)) {
        yvex_gguf_template_help(stdout);
        return 0;
    }
    if (arg_count < 3) {
        yvex_cli_out_writef(stderr, "yvex: gguf-template requires inspect, validate, or compare\n");
        yvex_cli_out_writef(stderr, "usage: yvex compile artifact template inspect|validate --template FILE\n");
        return 2;
    }
    if (strcmp(args[2], "inspect") != 0 && strcmp(args[2], "validate") != 0 &&
        strcmp(args[2], "compare") != 0) {
        yvex_cli_out_writef(stderr, "yvex: unknown gguf-template subcommand: %s\n", args[2]);
        return 2;
    }
    rc = cli_parse_gguf_template_options(arg_count, args, 3, &template_path, &native_source, &require_all);
    if (rc != 0) {
        return rc;
    }
    if (!template_path) {
        yvex_cli_out_writef(stderr, "yvex: gguf-template requires --template FILE\n");
        return 2;
    }
    if (strcmp(args[2], "compare") == 0 && !native_source) {
        yvex_cli_out_writef(stderr, "yvex: gguf-template compare requires --native-source DIR\n");
        return 2;
    }

    memset(&options, 0, sizeof(options));
    options.template_path = template_path;
    options.native_source_dir = native_source;
    options.compare_native = strcmp(args[2], "compare") == 0;
    options.require_all_template_tensors_in_native = require_all;

    rc = yvex_gguf_template_open(&tmpl, &options, &err);
    if (rc != YVEX_OK) {
        return print_yvex_error(&err, exit_for_status(rc));
    }
    rc = yvex_gguf_template_get_summary(tmpl, &summary, &err);
    if (rc != YVEX_OK) {
        yvex_gguf_template_close(tmpl);
        return print_yvex_error(&err, exit_for_status(rc));
    }

    if (strcmp(args[2], "compare") == 0) {
        yvex_cli_out_writef(stdout, "gguf template: compare\n");
        yvex_cli_out_writef(stdout, "template: %s\n", template_path);
        yvex_cli_out_writef(stdout, "native_source: %s\n", native_source);
        yvex_cli_out_writef(stdout, "template_tensors: %llu\n", summary.tensor_count);
        yvex_cli_out_writef(stdout, "native_tensors: %llu\n", summary.native_tensor_count);
        yvex_cli_out_writef(stdout, "matched_exact: %llu\n", summary.matched_exact);
        yvex_cli_out_writef(stdout, "missing_in_native: %llu\n", summary.missing_in_native);
        yvex_cli_out_writef(stdout, "shape_mismatch: %llu\n", summary.shape_mismatch);
        yvex_cli_out_writef(stdout, "status: template-compare-%s\n",
               summary.status == YVEX_GGUF_TEMPLATE_STATUS_VALID ? "valid" :
               summary.status == YVEX_GGUF_TEMPLATE_STATUS_INVALID ? "invalid" : "partial");
        if (summary.missing_in_native > 0 || summary.shape_mismatch > 0) {
            yvex_cli_out_writef(stdout, "reason: architecture adapter mapping requires open-weight intake\n");
        }
        cli_print_template_issues(tmpl);
    } else if (strcmp(args[2], "validate") == 0) {
        yvex_cli_out_writef(stdout, "gguf template: validate\n");
        yvex_cli_out_writef(stdout, "template: %s\n", template_path);
        yvex_cli_out_writef(stdout, "status: %s\n", yvex_gguf_template_status_name(summary.status));
        yvex_cli_out_writef(stdout, "issues: %llu\n", summary.issue_count);
        cli_print_template_issues(tmpl);
    } else {
        yvex_cli_out_writef(stdout, "gguf template: inspect\n");
        yvex_cli_out_writef(stdout, "template: %s\n", template_path);
        yvex_cli_out_writef(stdout, "architecture: %s\n", summary.architecture ? summary.architecture : "");
        yvex_cli_out_writef(stdout, "model_name: %s\n", summary.model_name ? summary.model_name : "");
        yvex_cli_out_writef(stdout, "metadata_count: %llu\n", summary.metadata_count);
        yvex_cli_out_writef(stdout, "tensor_count: %llu\n", summary.tensor_count);
        yvex_cli_out_writef(stdout, "has_tokenizer: %s\n", summary.has_tokenizer ? "yes" : "no");
        yvex_cli_out_writef(stdout, "known_roles: %llu\n", summary.known_role_count);
        yvex_cli_out_writef(stdout, "unknown_roles: %llu\n", summary.unknown_role_count);
        yvex_cli_out_writef(stdout, "status: %s\n", yvex_gguf_template_status_name(summary.status));
    }

    yvex_gguf_template_close(tmpl);
    return 0;
}

static int command_gguf_emit(int arg_count, char **args)
{
    yvex_gguf_emit_options options;
    yvex_gguf_emit_summary summary;
    yvex_error err;
    int i;
    int rc;

    yvex_error_clear(&err);
    memset(&options, 0, sizeof(options));
    memset(&summary, 0, sizeof(summary));

    if (arg_count >= 3 && (strcmp(args[2], "--help") == 0 || strcmp(args[2], "-h") == 0)) {
        yvex_gguf_emit_help(stdout);
        return 0;
    }
    if (arg_count < 3 || strcmp(args[2], "controlled") != 0) {
        yvex_cli_out_writef(stderr, "yvex: gguf-emit requires subcommand controlled\n");
        yvex_cli_out_writef(stderr,
            "usage: yvex compile artifact emit --out FILE [--template FILE] [--model-name NAME] [--arch "
                "ARCH] [--target-qtype F32|F16] [--overwrite]\n");
        return 2;
    }

    options.tensor_name = "embed.weight";
    options.target_name = "token_embd.weight";
    options.model_name = "yvex-owned-gguf-test";
    options.architecture = "llama";
    options.transpose_2d = 1;

    for (i = 3; i < arg_count; ++i) {
        if (strcmp(args[i], "--overwrite") == 0) {
            options.overwrite = 1;
            continue;
        }
        if (i + 1 >= arg_count) {
            yvex_cli_out_writef(stderr, "yvex: gguf-emit option requires a value: %s\n", args[i]);
            return 2;
        }
        if (strcmp(args[i], "--out") == 0) {
            options.out_path = args[++i];
        } else if (strcmp(args[i], "--template") == 0) {
            options.template_path = args[++i];
        } else if (strcmp(args[i], "--native-source") == 0) {
            options.native_source_dir = args[++i];
        } else if (strcmp(args[i], "--tensor-name") == 0) {
            options.tensor_name = args[++i];
        } else if (strcmp(args[i], "--target-name") == 0) {
            options.target_name = args[++i];
        } else if (strcmp(args[i], "--target-qtype") == 0) {
            options.target_qtype = args[++i];
        } else if (strcmp(args[i], "--model-name") == 0) {
            options.model_name = args[++i];
        } else if (strcmp(args[i], "--arch") == 0) {
            options.architecture = args[++i];
        } else {
            yvex_cli_out_writef(stderr, "yvex: unknown gguf-emit option: %s\n", args[i]);
            yvex_cli_out_writef(stderr, "Try 'yvex help gguf-emit' for usage.\n");
            return 2;
        }
    }

    if (!options.out_path) {
        yvex_cli_out_writef(stderr, "yvex: gguf-emit controlled requires --out FILE\n");
        yvex_cli_out_writef(stderr,
            "usage: yvex compile artifact emit --out FILE [--template FILE] [--model-name NAME] [--arch "
                "ARCH] [--target-qtype F32|F16] [--overwrite]\n");
        return 2;
    }

    rc = yvex_gguf_emit_controlled(&options, &summary, &err);
    if (rc != YVEX_OK) {
        return print_yvex_error(&err, exit_for_status(rc));
    }

    yvex_cli_out_writef(stdout, "gguf emit: controlled\n");
    yvex_cli_out_writef(stdout, "out: %s\n", summary.out_path ? summary.out_path : "");
    yvex_cli_out_writef(stdout, "architecture: %s\n", summary.architecture ? summary.architecture : "");
    yvex_cli_out_writef(stdout, "model_name: %s\n", summary.model_name ? summary.model_name : "");
    yvex_cli_out_writef(stdout, "metadata_count: %llu\n", summary.metadata_count);
    yvex_cli_out_writef(stdout, "tensor_count: %llu\n", summary.tensor_count);
    yvex_cli_out_writef(stdout, "tensor_payload_bytes: %llu\n", summary.tensor_payload_bytes);
    yvex_cli_out_writef(stdout, "alignment: %llu\n", summary.alignment);
    yvex_cli_out_writef(stdout, "roundtrip_validated: %s\n", summary.roundtrip_validated ? "yes" : "no");
    yvex_cli_out_writef(stdout, "status: %s\n", yvex_gguf_emit_status_name(summary.status));
    return 0;
}

static int command_qtype_support(int arg_count, char **args)
{
    unsigned long long i;

    if (arg_count == 3 && (strcmp(args[2], "--help") == 0 || strcmp(args[2], "-h") == 0)) {
        yvex_qtype_support_help(stdout);
        return 0;
    }
    if (arg_count != 2) {
        yvex_cli_out_writef(stderr, "yvex: qtype-support takes no arguments\n");
        return 2;
    }
    yvex_cli_out_writef(stdout, "qtype support:\n");
    for (i = 0; i < yvex_qtype_support_count(); ++i) {
        const yvex_qtype_support_info *row = yvex_qtype_support_at(i);
        yvex_cli_out_writef(stdout, "  %s policy=%s storage=%s emit=%s quantize=%s compute=%s notes=%s\n",
               yvex_qtype_support_name(row),
               row->policy_supported ? "yes" : "no",
               yvex_qtype_support_storage_supported(row) ? "yes" : "no",
               row->emit_supported ? "yes" : "no",
               row->ggml_type == YVEX_GGUF_QTYPE_F32
                   ? "n/a"
                   : (row->quantize_supported ? "yes" : "no"),
               row->compute_supported ? "partial" : "no",
               row->notes ? row->notes : "");
    }
    yvex_cli_out_writef(stdout, "status: qtype-support\n");
    return 0;
}

static int command_convert(int arg_count, char **args)
{
    yvex_conversion_options options;
    yvex_conversion_summary summary;
    yvex_error err;
    const char *out_plan = NULL;
    int i;
    int rc;

    yvex_error_clear(&err);
    memset(&options, 0, sizeof(options));
    memset(&summary, 0, sizeof(summary));

    if (arg_count >= 3 && (strcmp(args[2], "--help") == 0 || strcmp(args[2], "-h") == 0)) {
        yvex_convert_help(stdout);
        return 0;
    }
    if (arg_count < 3 || (strcmp(args[2], "plan") != 0 && strcmp(args[2], "emit") != 0)) {
        yvex_cli_out_writef(stderr, "yvex: convert requires plan or emit\n");
        return 2;
    }

    for (i = 3; i < arg_count; ++i) {
        if (strcmp(args[i], "--overwrite") == 0) {
            options.overwrite = 1;
            continue;
        }
        if (strcmp(args[i], "--allow-unsupported-qtype") == 0) {
            options.allow_unsupported_qtype = 1;
            continue;
        }
        if (strcmp(args[i], "--require-all") == 0) {
            options.require_all = 1;
            continue;
        }
        if (i + 1 >= arg_count) {
            yvex_cli_out_writef(stderr, "yvex: convert option requires a value: %s\n", args[i]);
            return 2;
        }
        if (strcmp(args[i], "--arch") == 0) options.architecture = args[++i];
        else if (strcmp(args[i], "--source-manifest") == 0) options.source_manifest_path = args[++i];
        else if (strcmp(args[i], "--native-source") == 0) options.native_source_dir = args[++i];
        else if (strcmp(args[i], "--template") == 0) options.template_path = args[++i];
        else if (strcmp(args[i], "--quant-policy") == 0) options.quant_policy_path = args[++i];
        else if (strcmp(args[i], "--imatrix-manifest") == 0) options.imatrix_manifest_path = args[++i];
        else if (strcmp(args[i], "--out") == 0) options.out_path = args[++i];
        else if (strcmp(args[i], "--out-plan") == 0) out_plan = args[++i];
        else if (strcmp(args[i], "--tensor") == 0) options.tensor_name = args[++i];
        else if (strcmp(args[i], "--target-qtype") == 0) options.target_qtype = args[++i];
        else if (strcmp(args[i], "--limit") == 0) {
            char *end = NULL;
            options.limit_tensors = strtoull(args[++i], &end, 10);
            if (!end || *end != '\0') {
                yvex_cli_out_writef(stderr, "yvex: invalid convert limit\n");
                return 2;
            }
        } else {
            yvex_cli_out_writef(stderr, "yvex: unknown convert option: %s\n", args[i]);
            return 2;
        }
    }

    if (strcmp(args[2], "plan") == 0) {
        if (!options.architecture || !options.native_source_dir || !out_plan) {
            yvex_cli_out_writef(stderr, "yvex: convert plan requires --arch --native-source --out-plan\n");
            return 2;
        }
        rc = yvex_conversion_plan_write_json(&options, out_plan, &summary, &err);
        if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));
        yvex_cli_out_writef(stdout, "conversion plan: written\n");
        yvex_cli_out_writef(stdout, "architecture: %s\n", options.architecture);
        yvex_cli_out_writef(stdout, "native_tensors: %llu\n", summary.native_tensor_count);
        yvex_cli_out_writef(stdout, "planned_tensors: %llu\n", summary.planned_tensor_count);
        yvex_cli_out_writef(stdout, "unmapped_tensors: %llu\n", summary.unmapped_tensor_count);
        yvex_cli_out_writef(stdout, "unsupported_qtypes: %llu\n", summary.unsupported_qtype_count);
        yvex_cli_out_writef(stdout, "out: %s\n", out_plan);
        yvex_cli_out_writef(stdout, "status: conversion-plan-written\n");
        return 0;
    }

    if (!options.architecture || !options.native_source_dir || !options.tensor_name ||
        !options.target_qtype || !options.out_path) {
        yvex_cli_out_writef(stderr,
            "yvex: convert emit requires --arch --native-source --tensor --target-qtype --out\n");
        return 2;
    }
    rc = yvex_conversion_emit_gguf(&options, &summary, &err);
    if (rc != YVEX_OK) {
        yvex_cli_out_writef(stdout, "conversion emit: gguf\n");
        yvex_cli_out_writef(stdout, "architecture: %s\n", options.architecture);
        yvex_cli_out_writef(stdout, "source_tensor: %s\n", options.tensor_name);
        yvex_cli_out_writef(stdout, "target_qtype: %s\n", options.target_qtype);
        yvex_cli_out_writef(stdout, "status: conversion-failed\n");
        yvex_cli_out_writef(stderr, "reason: %s\n", yvex_error_message(&err));
        return exit_for_status(rc);
    }
    yvex_cli_out_writef(stdout, "conversion emit: gguf\n");
    yvex_cli_out_writef(stdout, "architecture: %s\n", options.architecture);
    yvex_cli_out_writef(stdout, "source_tensor: %s\n", options.tensor_name);
    yvex_cli_out_writef(stdout, "target_qtype: %s\n", options.target_qtype);
    yvex_cli_out_writef(stdout, "out: %s\n", options.out_path);
    yvex_cli_out_writef(stdout, "bytes_read: %llu\n", summary.bytes_read);
    yvex_cli_out_writef(stdout, "bytes_written: %llu\n", summary.bytes_written);
    yvex_cli_out_writef(stdout, "roundtrip_validated: %s\n", summary.roundtrip_validated ? "yes" : "no");
    yvex_cli_out_lines(stdout, literal_pair_8, sizeof(literal_pair_8) / sizeof(literal_pair_8[0]));
    return 0;
}

static int parse_imatrix_create_options(int arg_count, char **args,
                                        yvex_imatrix_manifest_options *options,
                                        const char **out_path)
{
    int i = 3;

    while (i < arg_count) {
        if (i + 1 >= arg_count) {
            yvex_cli_out_writef(stderr, "yvex: imatrix option requires a value: %s\n", args[i]);
            return 2;
        }
        if (strcmp(args[i], "--name") == 0) options->name = args[i + 1];
        else if (strcmp(args[i], "--arch") == 0) options->architecture = args[i + 1];
        else if (strcmp(args[i], "--source-manifest") == 0) options->source_manifest_path = args[i + 1];
        else if (strcmp(args[i], "--quant-policy") == 0) options->quant_policy_path = args[i + 1];
        else if (strcmp(args[i], "--imatrix") == 0) options->imatrix_path = args[i + 1];
        else if (strcmp(args[i], "--format") == 0) options->format = yvex_imatrix_format_from_name(args[i + 1]);
        else if (strcmp(args[i], "--status") == 0) options->status = yvex_imatrix_status_from_name(args[i + 1]);
        else if (strcmp(args[i], "--dataset") == 0) options->calibration_dataset = args[i + 1];
        else if (strcmp(args[i], "--command") == 0) options->calibration_command = args[i + 1];
        else if (strcmp(args[i], "--producer") == 0) options->producer = args[i + 1];
        else if (strcmp(args[i], "--out") == 0) *out_path = args[i + 1];
        else {
            yvex_cli_out_writef(stderr, "yvex: unknown imatrix option: %s\n", args[i]);
            return 2;
        }
        i += 2;
    }
    return 0;
}

static int parse_imatrix_manifest_option(int arg_count, char **args, const char **manifest_path)
{
    int i = 3;

    while (i < arg_count) {
        if (i + 1 >= arg_count) {
            yvex_cli_out_writef(stderr, "yvex: imatrix option requires a value: %s\n", args[i]);
            return 2;
        }
        if (strcmp(args[i], "--manifest") == 0) {
            *manifest_path = args[i + 1];
        } else {
            yvex_cli_out_writef(stderr, "yvex: unknown imatrix option: %s\n", args[i]);
            return 2;
        }
        i += 2;
    }
    return 0;
}

static void print_imatrix_summary(const char *mode,
                                  const char *manifest_path,
                                  const yvex_imatrix_summary *summary)
{
    yvex_cli_out_writef(stdout, "imatrix: %s\n", mode);
    if (manifest_path) yvex_cli_out_writef(stdout, "manifest: %s\n", manifest_path);
    yvex_cli_out_writef(stdout, "name: %s\n", summary->name ? summary->name : "");
    yvex_cli_out_writef(stdout, "architecture: %s\n", summary->architecture ? summary->architecture : "");
    yvex_cli_out_writef(stdout, "format: %s\n", yvex_imatrix_format_name(summary->format));
    yvex_cli_out_writef(stdout, "status: %s\n", yvex_imatrix_status_name(summary->status));
    yvex_cli_out_writef(stdout, "file_exists: %s\n", summary->file_exists ? "yes" : "no");
    yvex_cli_out_writef(stdout, "source_manifest: %s\n",
        summary->source_manifest_path ? summary->source_manifest_path : "");
    yvex_cli_out_writef(stdout, "quant_policy: %s\n", summary->quant_policy_path ? summary->quant_policy_path : "");
    yvex_cli_out_writef(stdout, "imatrix: %s\n", summary->imatrix_path ? summary->imatrix_path : "");
}

static int command_imatrix(int arg_count, char **args)
{
    yvex_error err;
    yvex_imatrix_manifest *manifest = NULL;
    yvex_imatrix_summary summary;
    int rc;

    yvex_error_clear(&err);
    if (arg_count >= 3 && (strcmp(args[2], "--help") == 0 || strcmp(args[2], "-h") == 0)) {
        yvex_imatrix_help(stdout);
        return 0;
    }
    if (arg_count < 3) {
        yvex_cli_out_writef(stderr, "yvex: imatrix requires create, inspect, or validate\n");
        return 2;
    }

    if (strcmp(args[2], "create") == 0) {
        yvex_imatrix_manifest_options options;
        const char *out_path = NULL;

        memset(&options, 0, sizeof(options));
        rc = parse_imatrix_create_options(arg_count, args, &options, &out_path);
        if (rc != 0) return rc;
        if (!options.name || !options.architecture || !options.imatrix_path || !out_path ||
            options.format == YVEX_IMATRIX_FORMAT_UNKNOWN ||
            options.status == YVEX_IMATRIX_STATUS_UNKNOWN) {
            yvex_cli_out_writef(stderr,
                "yvex: imatrix create requires --name --arch --imatrix --format --status --out\n");
            return 2;
        }
        rc = yvex_imatrix_manifest_create(&manifest, &options, &err);
        if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));
        rc = yvex_imatrix_manifest_validate(manifest, &err);
        if (rc == YVEX_OK) rc = yvex_imatrix_manifest_write_json(out_path, manifest, &err);
        if (rc == YVEX_OK) rc = yvex_imatrix_manifest_get_summary(manifest, &summary, &err);
        if (rc != YVEX_OK) {
            yvex_imatrix_manifest_close(manifest);
            return print_yvex_error(&err, exit_for_status(rc));
        }
        yvex_cli_out_writef(stdout, "imatrix manifest: written\n");
        yvex_cli_out_writef(stdout, "name: %s\n", summary.name);
        yvex_cli_out_writef(stdout, "architecture: %s\n", summary.architecture);
        yvex_cli_out_writef(stdout, "format: %s\n", yvex_imatrix_format_name(summary.format));
        yvex_cli_out_writef(stdout, "status: %s\n", yvex_imatrix_status_name(summary.status));
        yvex_cli_out_writef(stdout, "file_exists: %s\n", summary.file_exists ? "yes" : "no");
        yvex_cli_out_writef(stdout, "out: %s\n", out_path);
        yvex_cli_out_writef(stdout, "status: imatrix-manifest-written\n");
        yvex_imatrix_manifest_close(manifest);
        return 0;
    }

    if (strcmp(args[2], "inspect") == 0 || strcmp(args[2], "validate") == 0) {
        const char *manifest_path = NULL;

        rc = parse_imatrix_manifest_option(arg_count, args, &manifest_path);
        if (rc != 0) return rc;
        if (!manifest_path) {
            yvex_cli_out_writef(stderr, "yvex: imatrix %s requires --manifest FILE\n", args[2]);
            return 2;
        }
        rc = yvex_imatrix_manifest_open(&manifest, manifest_path, &err);
        if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));
        if (strcmp(args[2], "validate") == 0) {
            rc = yvex_imatrix_manifest_validate(manifest, &err);
            if (rc != YVEX_OK) {
                yvex_imatrix_manifest_close(manifest);
                return print_yvex_error(&err, exit_for_status(rc));
            }
        }
        rc = yvex_imatrix_manifest_get_summary(manifest, &summary, &err);
        if (rc != YVEX_OK) {
            yvex_imatrix_manifest_close(manifest);
            return print_yvex_error(&err, exit_for_status(rc));
        }
        print_imatrix_summary(args[2], manifest_path, &summary);
        if (strcmp(args[2], "validate") == 0) {
            yvex_cli_out_writef(stdout, "issues: %llu\n", summary.issue_count);
            yvex_cli_out_writef(stdout, "requires_imatrix_rules: %llu\n", summary.requires_imatrix_rule_count);
            yvex_cli_out_writef(stdout, "covered_rules: %llu\n", summary.covered_rule_count);
            yvex_cli_out_writef(stdout, "uncovered_rules: %llu\n", summary.uncovered_rule_count);
            yvex_cli_out_writef(stdout, "status: imatrix-%s\n",
                   summary.issue_count == 0 ? "valid" :
                   (summary.file_exists ? "partial" : "invalid"));
        } else {
            yvex_cli_out_writef(stdout, "status: imatrix-manifest\n");
        }
        yvex_imatrix_manifest_close(manifest);
        return 0;
    }

    yvex_cli_out_writef(stderr, "yvex: unknown imatrix subcommand: %s\n", args[2]);
    return 2;
}

static int command_native_weights(int arg_count, char **args)
{
    yvex_native_weight_options options;
    yvex_native_weight_table *table = NULL;
    yvex_native_weight_summary summary;
    yvex_error err;
    const char *tensor_name = NULL;
    unsigned long long limit = 20;
    int json = 0;
    int i;
    int rc;

    yvex_error_clear(&err);
    memset(&options, 0, sizeof(options));
    options.recursive = 1;

    if (arg_count == 3 && (strcmp(args[2], "--help") == 0 || strcmp(args[2], "-h") == 0)) {
        yvex_native_weights_help(stdout);
        return 0;
    }

    i = 2;
    while (i < arg_count) {
        if (strcmp(args[i], "--json") == 0) {
            json = 1;
            i++;
            continue;
        }
        if (i + 1 >= arg_count) {
            yvex_cli_out_writef(stderr, "yvex: native-weights option requires a value: %s\n", args[i]);
            return 2;
        }
        if (strcmp(args[i], "--source") == 0) {
            options.source_dir = args[i + 1];
        } else if (strcmp(args[i], "--limit") == 0) {
            char *end = NULL;
            limit = strtoull(args[i + 1], &end, 10);
            if (!end || *end != '\0') {
                yvex_cli_out_writef(stderr, "yvex: invalid native-weights limit: %s\n", args[i + 1]);
                return 2;
            }
        } else if (strcmp(args[i], "--tensor") == 0) {
            tensor_name = args[i + 1];
        } else {
            yvex_cli_out_writef(stderr, "yvex: unknown native-weights option: %s\n", args[i]);
            return 2;
        }
        i += 2;
    }

    if (!options.source_dir) {
        yvex_cli_out_writef(stderr, "yvex: native-weights requires --source DIR\n");
        return 2;
    }

    rc = yvex_native_weight_table_open(&table, &options, &err);
    if (rc != YVEX_OK) {
        return print_yvex_error(&err, exit_for_status(rc));
    }
    rc = yvex_native_weight_table_summary(table, &summary, &err);
    if (rc != YVEX_OK) {
        yvex_native_weight_table_close(table);
        return print_yvex_error(&err, exit_for_status(rc));
    }

    if (json) {
        yvex_cli_out_lines(stdout, literal_pair_7, sizeof(literal_pair_7) / sizeof(literal_pair_7[0]));
        yvex_cli_out_writef(stdout, "  \"source\": \"%s\",\n", options.source_dir);
        yvex_cli_out_writef(stdout, "  \"summary\": {\n");
        yvex_cli_out_writef(stdout, "    \"shard_count\": %llu,\n", summary.shard_count);
        yvex_cli_out_writef(stdout, "    \"tensor_count\": %llu,\n", summary.tensor_count);
        yvex_cli_out_writef(stdout, "    \"total_tensor_bytes\": %llu,\n", summary.total_tensor_bytes);
        yvex_cli_out_writef(stdout, "    \"unknown_dtype_count\": %llu,\n", summary.unknown_dtype_count);
        yvex_cli_out_writef(stdout, "    \"malformed_shard_count\": %llu\n", summary.malformed_shard_count);
        yvex_cli_out_lines(stdout, literal_pair_6, sizeof(literal_pair_6) / sizeof(literal_pair_6[0]));
        yvex_native_weight_table_close(table);
        return 0;
    }

    yvex_cli_out_writef(stdout, "native weights: safetensors\n");
    yvex_cli_out_writef(stdout, "source: %s\n", options.source_dir);
    yvex_cli_out_writef(stdout, "shards: %llu\n", summary.shard_count);
    yvex_cli_out_writef(stdout, "tensors: %llu\n", summary.tensor_count);
    yvex_cli_out_writef(stdout, "total_tensor_bytes: %llu\n", summary.total_tensor_bytes);
    yvex_cli_out_writef(stdout, "unknown_dtype_count: %llu\n", summary.unknown_dtype_count);
    yvex_cli_out_writef(stdout, "malformed_shard_count: %llu\n", summary.malformed_shard_count);
    yvex_cli_out_writef(stdout, "\n");

    if (tensor_name) {
        const yvex_native_weight_info *row = yvex_native_weight_table_find(table, tensor_name);
        if (!row) {
            yvex_native_weight_table_close(table);
            yvex_cli_out_writef(stderr, "yvex: native tensor not found: %s\n", tensor_name);
            return 4;
        }
        yvex_cli_out_writef(stdout, "0 %s shard=%s dtype=%s rank=%u shape=",
               row->name, row->shard_path, yvex_native_dtype_name(row->dtype), row->rank);
        print_native_dims(row->dims, row->rank);
        yvex_cli_out_writef(stdout, " bytes=%llu offsets=[%llu,%llu]\n", row->data_bytes, row->data_start,
            row->data_end);
    } else {
        unsigned long long count = yvex_native_weight_table_count(table);
        unsigned long long n = limit < count ? limit : count;
        unsigned long long idx;
        for (idx = 0; idx < n; ++idx) {
            const yvex_native_weight_info *row = yvex_native_weight_table_at(table, idx);
            yvex_cli_out_writef(stdout, "%llu %s shard=%s dtype=%s rank=%u shape=",
                   idx, row->name, row->shard_path, yvex_native_dtype_name(row->dtype), row->rank);
            print_native_dims(row->dims, row->rank);
            yvex_cli_out_writef(stdout, " bytes=%llu offsets=[%llu,%llu]\n", row->data_bytes,
                row->data_start, row->data_end);
        }
    }
    yvex_cli_out_writef(stdout, "status: %s\n", summary.shard_count == 0 ? "native-weights-empty" : "native-weights");
    yvex_native_weight_table_close(table);
    return 0;
}

static int command_tensor_map(int arg_count, char **args)
{
    yvex_weight_mapping_options options;
    yvex_weight_mapping_table *table = NULL;
    yvex_error err;
    const char *tensor_name = NULL;
    unsigned long long limit = 20;
    unsigned long long mapped = 0;
    unsigned long long unmapped = 0;
    unsigned long long shape_mismatch = 0;
    int json = 0;
    int i;
    int rc;

    yvex_error_clear(&err);
    memset(&options, 0, sizeof(options));

    if (arg_count == 3 && (strcmp(args[2], "--help") == 0 || strcmp(args[2], "-h") == 0)) {
        yvex_tensor_map_help(stdout);
        return 0;
    }

    i = 2;
    while (i < arg_count) {
        if (strcmp(args[i], "--json") == 0) {
            json = 1;
            i++;
            continue;
        }
        if (strcmp(args[i], "--require-all-native-mapped") == 0) {
            options.require_all_native_mapped = 1;
            i++;
            continue;
        }
        if (strcmp(args[i], "--require-all-template-matched") == 0) {
            options.require_all_template_matched = 1;
            i++;
            continue;
        }
        if (i + 1 >= arg_count) {
            yvex_cli_out_writef(stderr, "yvex: tensor-map option requires a value: %s\n", args[i]);
            return 2;
        }
        if (strcmp(args[i], "--arch") == 0) {
            options.architecture = args[i + 1];
        } else if (strcmp(args[i], "--native-source") == 0) {
            options.native_source_dir = args[i + 1];
        } else if (strcmp(args[i], "--template") == 0) {
            options.template_path = args[i + 1];
            options.compare_template = 1;
        } else if (strcmp(args[i], "--tensor") == 0) {
            tensor_name = args[i + 1];
        } else if (strcmp(args[i], "--limit") == 0) {
            char *end = NULL;
            limit = strtoull(args[i + 1], &end, 10);
            if (!end || *end != '\0') {
                yvex_cli_out_writef(stderr, "yvex: invalid tensor-map limit: %s\n", args[i + 1]);
                return 2;
            }
        } else {
            yvex_cli_out_writef(stderr, "yvex: unknown tensor-map option: %s\n", args[i]);
            return 2;
        }
        i += 2;
    }

    if (!options.architecture || !options.native_source_dir) {
        yvex_cli_out_writef(stderr, "yvex: tensor-map requires --arch NAME and --native-source DIR\n");
        return 2;
    }

    rc = yvex_weight_mapping_table_build(&table, &options, &err);
    if (rc != YVEX_OK) {
        return print_yvex_error(&err, exit_for_status(rc));
    }

    for (i = 0; (unsigned long long)i < yvex_weight_mapping_table_count(table); ++i) {
        const yvex_weight_mapping_info *row = yvex_weight_mapping_table_at(table, (unsigned long long)i);
        if (!row) continue;
        if (row->status == YVEX_WEIGHT_MAPPING_STATUS_MAPPED) mapped++;
        else if (row->status == YVEX_WEIGHT_MAPPING_STATUS_SHAPE_MISMATCH) shape_mismatch++;
        else unmapped++;
    }

    if (json) {
        yvex_cli_out_lines(stdout, literal_pair_5, sizeof(literal_pair_5) / sizeof(literal_pair_5[0]));
        yvex_cli_out_writef(stdout, "  \"architecture\": \"%s\",\n", options.architecture);
        yvex_cli_out_writef(stdout, "  \"native_source\": \"%s\",\n", options.native_source_dir);
        yvex_cli_out_writef(stdout, "  \"native_tensors\": %llu,\n", yvex_weight_mapping_table_count(table));
        yvex_cli_out_writef(stdout, "  \"mapped\": %llu,\n", mapped);
        yvex_cli_out_writef(stdout, "  \"unmapped\": %llu,\n", unmapped);
        yvex_cli_out_writef(stdout, "  \"shape_mismatch\": %llu\n", shape_mismatch);
        yvex_cli_out_writef(stdout, "}\n");
        yvex_weight_mapping_table_close(table);
        return 0;
    }

    yvex_cli_out_writef(stdout, "compile tensor map: %s\n", options.architecture);
    yvex_cli_out_writef(stdout, "native_source: %s\n", options.native_source_dir);
    if (options.template_path) {
        yvex_cli_out_writef(stdout, "template: %s\n", options.template_path);
    }
    yvex_cli_out_writef(stdout, "native_tensors: %llu\n", yvex_weight_mapping_table_count(table));
    yvex_cli_out_writef(stdout, "mapped: %llu\n", mapped);
    yvex_cli_out_writef(stdout, "unmapped: %llu\n", unmapped);
    yvex_cli_out_writef(stdout, "shape_mismatch: %llu\n", shape_mismatch);
    yvex_cli_out_writef(stdout, "\n");

    if (tensor_name) {
        const yvex_weight_mapping_info *row = yvex_weight_mapping_table_find_native(table, tensor_name);
        if (!row) {
            yvex_weight_mapping_table_close(table);
            yvex_cli_out_writef(stderr, "yvex: native tensor not found: %s\n", tensor_name);
            return 4;
        }
        yvex_cli_out_writef(stdout, "0 native=%s role=%s target=%s status=%s native_shape=",
               row->native_name, yvex_tensor_role_name(row->role), row->target_name,
               yvex_weight_mapping_status_name(row->status));
        print_native_dims(row->native_dims, row->native_rank);
        yvex_cli_out_writef(stdout, " target_shape=");
        if (row->target_rank > 0) print_native_dims(row->target_dims, row->target_rank);
        else yvex_cli_out_writef(stdout, "unknown");
        yvex_cli_out_writef(stdout, " transform=%s", row->requires_transpose ? "transpose" : "none");
        if (row->issue != YVEX_WEIGHT_MAPPING_ISSUE_NONE) {
            yvex_cli_out_writef(stdout, " issue=%s", yvex_weight_mapping_issue_kind_name(row->issue));
        }
        yvex_cli_out_writef(stdout, "\n");
    } else {
        unsigned long long count = yvex_weight_mapping_table_count(table);
        unsigned long long n = limit < count ? limit : count;
        unsigned long long idx;

        for (idx = 0; idx < n; ++idx) {
            const yvex_weight_mapping_info *row = yvex_weight_mapping_table_at(table, idx);
            yvex_cli_out_writef(stdout, "%llu native=%s role=%s target=%s status=%s native_shape=",
                   idx, row->native_name, yvex_tensor_role_name(row->role), row->target_name,
                   yvex_weight_mapping_status_name(row->status));
            print_native_dims(row->native_dims, row->native_rank);
            yvex_cli_out_writef(stdout, " target_shape=");
            if (row->target_rank > 0) print_native_dims(row->target_dims, row->target_rank);
            else yvex_cli_out_writef(stdout, "unknown");
            yvex_cli_out_writef(stdout, " transform=%s", row->requires_transpose ? "transpose" : "none");
            if (row->issue != YVEX_WEIGHT_MAPPING_ISSUE_NONE) {
                yvex_cli_out_writef(stdout, " issue=%s", yvex_weight_mapping_issue_kind_name(row->issue));
            }
            yvex_cli_out_writef(stdout, "\n");
        }
    }
    yvex_cli_out_writef(stdout, "status: tensor-map\n");
    yvex_weight_mapping_table_close(table);
    return 0;
}

static void print_quant_policy_rules(const yvex_quant_policy *policy)
{
    unsigned long long i;

    for (i = 0; i < yvex_quant_policy_rule_count(policy); ++i) {
        const yvex_quant_policy_rule *rule = yvex_quant_policy_rule_at(policy, i);
        if (!rule) continue;
        yvex_cli_out_writef(stdout,
            "%llu selector=%s:%s qtype=%s storage_supported=%s compute_supported=%s requires_imatrix=%s\n",
               i,
               yvex_quant_selector_kind_name(rule->selector_kind),
               rule->selector,
               yvex_quant_qtype_name(rule->qtype),
               rule->storage_supported ? "yes" : "no",
               rule->compute_supported ? "yes" : "no",
               rule->requires_imatrix ? "yes" : "no");
    }
}

static int parse_quant_policy_common(int arg_count, char **args, int start,
                                     const char **policy_path,
                                     const char **template_path,
                                     const char **arch,
                                     const char **out_path)
{
    int i = start;

    while (i < arg_count) {
        if (i + 1 >= arg_count) {
            yvex_cli_out_writef(stderr, "yvex: quant-policy option requires a value: %s\n", args[i]);
            return 2;
        }
        if (strcmp(args[i], "--policy") == 0) {
            *policy_path = args[i + 1];
        } else if (strcmp(args[i], "--template") == 0) {
            *template_path = args[i + 1];
        } else if (strcmp(args[i], "--arch") == 0) {
            *arch = args[i + 1];
        } else if (strcmp(args[i], "--out") == 0) {
            *out_path = args[i + 1];
        } else {
            yvex_cli_out_writef(stderr, "yvex: unknown quant-policy option: %s\n", args[i]);
            return 2;
        }
        i += 2;
    }
    return 0;
}

static int command_quant_policy(int arg_count, char **args)
{
    const char *policy_path = NULL;
    const char *template_path = NULL;
    const char *arch = NULL;
    const char *out_path = NULL;
    yvex_quant_policy *policy = NULL;
    yvex_quant_policy_summary summary;
    yvex_error err;
    int rc;

    yvex_error_clear(&err);
    if (arg_count >= 3 && (strcmp(args[2], "--help") == 0 || strcmp(args[2], "-h") == 0)) {
        yvex_quant_policy_help(stdout);
        return 0;
    }
    if (arg_count < 3) {
        yvex_cli_out_writef(stderr, "yvex: quant-policy requires inspect, validate, or derive\n");
        return 2;
    }
    rc = parse_quant_policy_common(arg_count, args, 3, &policy_path, &template_path, &arch, &out_path);
    if (rc != 0) return rc;

    if (strcmp(args[2], "derive") == 0) {
        if (!template_path || !arch || !out_path) {
            yvex_cli_out_writef(stderr, "yvex: quant-policy derive requires --template FILE --arch NAME --out FILE\n");
            return 2;
        }
        rc = yvex_quant_policy_create_from_template(&policy, template_path, arch, &err);
        if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));
        rc = yvex_quant_policy_write_json(out_path, policy, &err);
        if (rc != YVEX_OK) {
            yvex_quant_policy_close(policy);
            return print_yvex_error(&err, exit_for_status(rc));
        }
        rc = yvex_quant_policy_get_summary(policy, &summary, &err);
        if (rc != YVEX_OK) {
            yvex_quant_policy_close(policy);
            return print_yvex_error(&err, exit_for_status(rc));
        }
        yvex_cli_out_writef(stdout, "compile quant policy: derived\n");
        yvex_cli_out_writef(stdout, "architecture: %s\n", summary.architecture);
        yvex_cli_out_writef(stdout, "template: %s\n", template_path);
        yvex_cli_out_writef(stdout, "rules: %llu\n", summary.rule_count);
        yvex_cli_out_writef(stdout, "requires_imatrix: %llu\n", summary.requires_imatrix_count);
        yvex_cli_out_writef(stdout, "out: %s\n", out_path);
        yvex_cli_out_writef(stdout, "status: quant-policy-written\n");
        yvex_quant_policy_close(policy);
        return 0;
    }

    if (strcmp(args[2], "inspect") == 0 || strcmp(args[2], "validate") == 0) {
        if (!policy_path) {
            yvex_cli_out_writef(stderr, "yvex: quant-policy %s requires --policy FILE\n", args[2]);
            return 2;
        }
        rc = yvex_quant_policy_open(&policy, policy_path, &err);
        if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));
        if (strcmp(args[2], "validate") == 0) {
            rc = yvex_quant_policy_validate(policy, template_path, &err);
            if (rc != YVEX_OK) {
                yvex_quant_policy_close(policy);
                return print_yvex_error(&err, exit_for_status(rc));
            }
        }
        rc = yvex_quant_policy_get_summary(policy, &summary, &err);
        if (rc != YVEX_OK) {
            yvex_quant_policy_close(policy);
            return print_yvex_error(&err, exit_for_status(rc));
        }
        yvex_cli_out_writef(stdout, "compile quant policy: %s\n", args[2]);
        yvex_cli_out_writef(stdout, "policy: %s\n", policy_path);
        if (template_path) yvex_cli_out_writef(stdout, "template: %s\n", template_path);
        yvex_cli_out_writef(stdout, "name: %s\n", summary.name);
        yvex_cli_out_writef(stdout, "architecture: %s\n", summary.architecture);
        yvex_cli_out_writef(stdout, "rules: %llu\n", summary.rule_count);
        yvex_cli_out_writef(stdout, "issues: %llu\n", summary.issue_count);
        yvex_cli_out_writef(stdout, "requires_imatrix: %llu\n", summary.requires_imatrix_count);
        yvex_cli_out_writef(stdout, "storage_supported: %llu\n", summary.storage_supported_count);
        yvex_cli_out_writef(stdout, "compute_supported: %llu\n", summary.compute_supported_count);
        yvex_cli_out_writef(stdout, "\n");
        if (strcmp(args[2], "inspect") == 0) {
            print_quant_policy_rules(policy);
        }
        yvex_cli_out_writef(stdout, "status: %s\n", yvex_quant_policy_status_name(summary.status));
        yvex_quant_policy_close(policy);
        return 0;
    }

    yvex_cli_out_writef(stderr, "yvex: unknown quant-policy subcommand: %s\n", args[2]);
    return 2;
}

static int parse_quant_job_create_options(int arg_count, char **args,
                                          yvex_quant_job_options *options,
                                          const char **out_path)
{
    int i;

    memset(options, 0, sizeof(*options));
    *out_path = NULL;
    for (i = 3; i < arg_count; ++i) {
        if (i + 1 >= arg_count) {
            yvex_cli_out_writef(stderr, "yvex: quant-job option requires a value: %s\n", args[i]);
            return 2;
        }
        if (strcmp(args[i], "--name") == 0) options->name = args[++i];
        else if (strcmp(args[i], "--arch") == 0) options->architecture = args[++i];
        else if (strcmp(args[i], "--tool") == 0) options->tool = yvex_quant_job_tool_from_name(args[++i]);
        else if (strcmp(args[i], "--tool-path") == 0) options->tool_path = args[++i];
        else if (strcmp(args[i], "--source-manifest") == 0) options->source_manifest_path = args[++i];
        else if (strcmp(args[i], "--native-source") == 0) options->native_source_dir = args[++i];
        else if (strcmp(args[i], "--template") == 0) options->template_path = args[++i];
        else if (strcmp(args[i], "--quant-policy") == 0) options->quant_policy_path = args[++i];
        else if (strcmp(args[i], "--imatrix-manifest") == 0) options->imatrix_manifest_path = args[++i];
        else if (strcmp(args[i], "--imatrix") == 0) options->imatrix_path = args[++i];
        else if (strcmp(args[i], "--out-gguf") == 0) options->out_gguf_path = args[++i];
        else if (strcmp(args[i], "--log") == 0) options->log_path = args[++i];
        else if (strcmp(args[i], "--status") == 0) options->status = yvex_quant_job_status_from_name(args[++i]);
        else if (strcmp(args[i], "--command") == 0) options->command = args[++i];
        else if (strcmp(args[i], "--out") == 0) *out_path = args[++i];
        else {
            yvex_cli_out_writef(stderr, "yvex: unknown quant-job option: %s\n", args[i]);
            return 2;
        }
    }
    return 0;
}

static int parse_quant_job_manifest_option(int arg_count, char **args, const char **manifest_path)
{
    int i;

    *manifest_path = NULL;
    for (i = 3; i < arg_count; ++i) {
        if (i + 1 >= arg_count) {
            yvex_cli_out_writef(stderr, "yvex: quant-job option requires a value: %s\n", args[i]);
            return 2;
        }
        if (strcmp(args[i], "--manifest") == 0) *manifest_path = args[++i];
        else {
            yvex_cli_out_writef(stderr, "yvex: unknown quant-job option: %s\n", args[i]);
            return 2;
        }
    }
    return 0;
}

static void print_quant_job_summary(const char *mode,
                                    const char *path,
                                    const yvex_quant_job_summary *summary)
{
    yvex_cli_out_writef(stdout, "compile quant job: %s\n", mode);
    if (path) yvex_cli_out_writef(stdout, "manifest: %s\n", path);
    yvex_cli_out_writef(stdout, "name: %s\n", summary->name ? summary->name : "");
    yvex_cli_out_writef(stdout, "architecture: %s\n", summary->architecture ? summary->architecture : "");
    yvex_cli_out_writef(stdout, "tool: %s\n", yvex_quant_job_tool_name(summary->tool));
    yvex_cli_out_writef(stdout, "tool_path: %s\n", summary->tool_path ? summary->tool_path : "");
    yvex_cli_out_writef(stdout, "native_source: %s\n", summary->native_source_dir ? summary->native_source_dir : "");
    yvex_cli_out_writef(stdout, "template: %s\n", summary->template_path ? summary->template_path : "");
    yvex_cli_out_writef(stdout, "out_gguf: %s\n", summary->out_gguf_path ? summary->out_gguf_path : "");
    yvex_cli_out_writef(stdout, "log: %s\n", summary->log_path ? summary->log_path : "");
    yvex_cli_out_writef(stdout, "tool_exists: %s\n", summary->tool_exists ? "yes" : "no");
    yvex_cli_out_writef(stdout, "source_exists: %s\n", summary->source_exists ? "yes" : "no");
    yvex_cli_out_writef(stdout, "template_exists: %s\n", summary->template_exists ? "yes" : "no");
    yvex_cli_out_writef(stdout, "imatrix_exists: %s\n", summary->imatrix_exists ? "yes" : "no");
    yvex_cli_out_writef(stdout, "output_exists: %s\n", summary->output_exists ? "yes" : "no");
    yvex_cli_out_writef(stdout, "status: %s\n", yvex_quant_job_status_name(summary->status));
}

static int command_quant_job(int arg_count, char **args)
{
    yvex_quant_job_summary summary;
    yvex_error err;
    int rc;

    yvex_error_clear(&err);
    memset(&summary, 0, sizeof(summary));

    if (arg_count >= 3 && (strcmp(args[2], "--help") == 0 || strcmp(args[2], "-h") == 0)) {
        yvex_quant_job_help(stdout);
        return 0;
    }
    if (arg_count < 3 || (strcmp(args[2], "create") != 0 &&
                     strcmp(args[2], "inspect") != 0 &&
                     strcmp(args[2], "validate") != 0)) {
        yvex_cli_out_writef(stderr, "yvex: quant-job requires create, inspect, or validate\n");
        return 2;
    }

    if (strcmp(args[2], "create") == 0) {
        yvex_quant_job_options options;
        const char *out_path = NULL;

        rc = parse_quant_job_create_options(arg_count, args, &options, &out_path);
        if (rc != 0) return rc;
        if (!options.name || !options.architecture || !options.tool_path ||
            !options.native_source_dir || !options.template_path ||
            !options.out_gguf_path || !options.log_path || !options.command ||
            !out_path || options.status == YVEX_QUANT_JOB_STATUS_UNKNOWN) {
            yvex_cli_out_writef(stderr,
                "yvex: quant-job create requires --name --arch --tool --tool-path --native-source --"
                    "template --out-gguf --log --status --command --out\n");
            return 2;
        }
        rc = yvex_quant_job_write_json(out_path, &options, &summary, &err);
        if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));
        yvex_cli_out_writef(stdout, "compile quant job: written\n");
        yvex_cli_out_writef(stdout, "name: %s\n", summary.name);
        yvex_cli_out_writef(stdout, "architecture: %s\n", summary.architecture);
        yvex_cli_out_writef(stdout, "tool: %s\n", yvex_quant_job_tool_name(summary.tool));
        yvex_cli_out_writef(stdout, "tool_exists: %s\n", summary.tool_exists ? "yes" : "no");
        yvex_cli_out_writef(stdout, "source_exists: %s\n", summary.source_exists ? "yes" : "no");
        yvex_cli_out_writef(stdout, "template_exists: %s\n", summary.template_exists ? "yes" : "no");
        yvex_cli_out_writef(stdout, "imatrix_exists: %s\n", summary.imatrix_exists ? "yes" : "no");
        yvex_cli_out_writef(stdout, "output_exists: %s\n", summary.output_exists ? "yes" : "no");
        yvex_cli_out_writef(stdout, "status: %s\n", yvex_quant_job_status_name(summary.status));
        yvex_cli_out_writef(stdout, "out: %s\n", out_path);
        yvex_cli_out_writef(stdout, "status: quant-job-written\n");
        return 0;
    }

    {
        const char *manifest_path = NULL;

        rc = parse_quant_job_manifest_option(arg_count, args, &manifest_path);
        if (rc != 0) return rc;
        if (!manifest_path) {
            yvex_cli_out_writef(stderr, "yvex: quant-job %s requires --manifest FILE\n", args[2]);
            return 2;
        }
        rc = yvex_quant_job_validate(manifest_path, &summary, &err);
        if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));
        print_quant_job_summary(args[2], manifest_path, &summary);
        yvex_cli_out_writef(stdout, "status: quant-job-%s\n", strcmp(args[2], "validate") == 0 ? "valid" : "manifest");
        return 0;
    }
}

int yvex_convert_command(int arg_count, char **args)
{
    return command_convert(arg_count, args);
}

int yvex_gguf_template_command(int arg_count, char **args)
{
    return command_gguf_template(arg_count, args);
}

int yvex_gguf_emit_command(int arg_count, char **args)
{
    return command_gguf_emit(arg_count, args);
}

int yvex_imatrix_command(int arg_count, char **args)
{
    return command_imatrix(arg_count, args);
}

int yvex_native_weights_command(int arg_count, char **args)
{
    return command_native_weights(arg_count, args);
}

int yvex_tensor_map_command(int arg_count, char **args)
{
    return command_tensor_map(arg_count, args);
}

int yvex_quant_job_command(int arg_count, char **args)
{
    return command_quant_job(arg_count, args);
}

int yvex_quant_policy_command(int arg_count, char **args)
{
    return command_quant_policy(arg_count, args);
}

int yvex_qtype_support_command(int arg_count, char **args)
{
    return command_qtype_support(arg_count, args);
}

void yvex_convert_help(FILE *fp)
{
    yvex_cli_out_writef(fp, "usage: yvex compile quant convert plan --arch ARCH --native-source DIR --out-plan FILE\n");
    yvex_cli_out_lines(fp, literal_pair_4, sizeof(literal_pair_4) / sizeof(literal_pair_4[0]));
}

void yvex_gguf_template_help(FILE *fp)
{
    yvex_cli_out_writef(fp, "usage: yvex compile artifact template inspect|validate --template FILE\n");
    yvex_cli_out_lines(fp, literal_pair_3, sizeof(literal_pair_3) / sizeof(literal_pair_3[0]));
}

void yvex_gguf_emit_help(FILE *fp)
{
    yvex_cli_out_writef(fp,
        "usage: yvex compile artifact emit --out FILE [--template FILE] [--model-name NAME] [--arch ARCH] [-"
            "-target-qtype F32|F16] [--overwrite]\n\nGGUF emit writes a controlled YVEX-owned tensor artifact "
            "and validates the emitted file.\n");
}

void yvex_imatrix_help(FILE *fp)
{
    yvex_cli_out_writef(fp,
                        "usage: yvex compile quant imatrix create --name NAME --arch NAME "
                        "--imatrix FILE --format FORMAT --status STATUS --out FILE\n");
    yvex_cli_out_lines(fp, literal_pair_2, sizeof(literal_pair_2) / sizeof(literal_pair_2[0]));
}

void yvex_native_weights_help(FILE *fp)
{
    yvex_cli_out_writef(fp,
        "usage: yvex inspect source tensors --source DIR [--limit N] [--tensor NAME] [--json]\n\nNative weights "
            "reads safetensors headers and reports metadata only.\n");
}

void yvex_tensor_map_help(FILE *fp)
{
    yvex_cli_out_writef(fp,
        "usage: yvex compile tensor map --arch NAME --native-source DIR [--template FILE] [--tensor NAME] [--limit "
            "N] [--json]\n\nTensor map maps native safetensors names to canonical YVEX roles and proposed GGUF/"
            "template names.\n");
}

void yvex_quant_job_help(FILE *fp)
{
    yvex_cli_out_writef(fp,
        "usage: yvex compile quant job create --name NAME --arch ARCH --tool TOOL --tool-path FILE --native-source "
            "DIR --template FILE --out-gguf FILE --log FILE --status STATUS --command TEXT --out FILE\n");
    yvex_cli_out_lines(fp, literal_pair_1, sizeof(literal_pair_1) / sizeof(literal_pair_1[0]));
}

void yvex_quant_policy_help(FILE *fp)
{
    yvex_cli_out_writef(fp, "usage: yvex compile quant policy inspect|validate --policy FILE [--template FILE]\n");
    yvex_cli_out_lines(fp, literal_pair_0, sizeof(literal_pair_0) / sizeof(literal_pair_0[0]));
}

void yvex_qtype_support_help(FILE *fp)
{
    yvex_cli_out_writef(fp,
        "usage: yvex inspect qtype\n\nReports policy/storage/emit/quantize/compute support separately. "
            "Compute support is not implied by conversion support.\n");
}
