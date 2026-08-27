#include "src/cli/input/private.h"
#include "src/cli/io/private.h"
#include "src/cli/render/private.h"

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <yvex/artifact.h>
#include <yvex/backend.h>
#include <yvex/core.h>
#include <yvex/gguf.h>
#include <yvex/materialization.h>
#include <yvex/model.h>
#include <yvex/registry.h>

#define FIELD_COUNT(fields_) (sizeof(fields_) / sizeof((fields_)[0]))

static const char *const literal_pair_0[] = {
    "reason: current inspect artifact metadata could not be parsed", "status: models-metadata-drift"};

static const char *const literal_pair_1[] = {"format: gguf", "status: unsupported"};

static const char *const literal_pair_2[] = {"format: unknown", "status: unsupported"};

static const char *const literal_lines_0[] = {"metadata_status: fail", "readiness_status: fail",
                                              "metadata_issue_0_code: current-metadata-unavailable",
                                              "metadata_issue_0_registered: available"};

static const char inspect_help_text[] =
    "usage: yvex artifact show FILE_OR_ALIAS\n\nInspect parses a GGUF descriptor and prints a "
    "descriptor-only summary. It does not materialize weights or execute a graph.\n";
static const char integrity_help_text[] =
    "usage: yvex artifact verify FILE_OR_ALIAS [--expect-sha256 HASH] "
    "[--require-token-embedding] [--partial-token N]\n"
    "       yvex inspect artifact integrity FILE_OR_ALIAS [--backend cpu|cuda] "
    "[--expect-sha256 HASH] [--require-token-embedding] [--partial-token N] "
    "[--audit | --output normal|table|audit]\n"
    "\nIntegrity validates local GGUF structure, tensor accounting, digest identity when "
    "supplied, metadata drift, and selected embedding readiness. It is not a supply-chain "
    "security audit.\nDefault report output is compact. Use --audit for full diagnostic fields.\n";
static const char materialize_help_text[] =
    "usage: yvex artifact materialize --model FILE_OR_ALIAS --backend cpu|cuda [--require-all] "
    "[--allow-unsupported-dtype]\n\nMaterialize copies selected GGUF tensor bytes into "
    "backend-owned storage after integrity preflight. It does not execute prefill, decode, "
    "sampling, generation, or inference.\n";
static const char materialize_gate_help_text[] =
    "usage: yvex execute artifact materialize-gate check --model FILE_OR_ALIAS --label LABEL --family FAMILY "
    "--scope selected-tensor --expect-tensor NAME --expect-rank N --expect-dims D1[,D2,D3] "
    "--expect-dtype DTYPE --expect-bytes BYTES [--sha256 HASH] [--backend cpu] [--backend cuda] "
    "[--require-cpu] [--require-cuda] [--repeat N] [--check-cleanup] [--report-out FILE]\n\n"
    "The materialization gate validates identity, tensor facts, repeated backend "
    "materialization, cleanup, and failure classes.\n";
static const char metadata_help_text[] =
    "usage: yvex inspect artifact metadata FILE_OR_ALIAS\n\nMetadata prints parsed GGUF metadata key/value "
    "summaries. Arrays are summarized.\n";
static const char model_gate_help_text[] =
    "usage: yvex execute artifact model-gate check --model FILE_OR_ALIAS --label LABEL --family FAMILY "
    "--expect-tensor NAME --expect-rank N --expect-dims D1[,D2,D3] --expect-dtype DTYPE "
    "--expect-bytes BYTES [--sha256 HASH] [--backend cpu] [--backend cuda] [--require-cpu] "
    "[--require-cuda] [--report-out FILE]\n\nModel gate checks selected tensor identity, "
    "expected tensor specs, and requested CPU/CUDA materialization without claiming full-model "
    "support.\n";
static const char tensors_help_text[] =
    "usage: yvex inspect artifact tensors FILE_OR_ALIAS\n\nTensors prints YVEX tensor table rows with role, "
    "dtype, known storage bytes, and checked offsets.\n";

#define INTEGRITY_FIELD(key_, kind_, member_, fallback_)                                            \
    { key_, kind_, offsetof(yvex_artifact_integrity_report, member_), fallback_ }

static const yvex_render_field_spec integrity_counter_fields[] = {
    INTEGRITY_FIELD("tensor_count", YVEX_RENDER_FIELD_U64, tensor_count, NULL),
    INTEGRITY_FIELD("known_tensor_bytes", YVEX_RENDER_FIELD_U64, known_tensor_bytes, NULL),
    INTEGRITY_FIELD("tensor_ranges_checked", YVEX_RENDER_FIELD_U64, tensor_ranges_checked, NULL),
    INTEGRITY_FIELD("tensor_ranges_valid", YVEX_RENDER_FIELD_U64, tensor_ranges_valid, NULL),
    INTEGRITY_FIELD("tensor_ranges_invalid", YVEX_RENDER_FIELD_U64, tensor_ranges_invalid, NULL),
    INTEGRITY_FIELD("tensor_shapes_checked", YVEX_RENDER_FIELD_U64, tensor_shapes_checked, NULL),
    INTEGRITY_FIELD("tensor_shapes_valid", YVEX_RENDER_FIELD_U64, tensor_shapes_valid, NULL),
    INTEGRITY_FIELD("tensor_shapes_invalid", YVEX_RENDER_FIELD_U64, tensor_shapes_invalid, NULL),
    INTEGRITY_FIELD("tensor_dtypes_checked", YVEX_RENDER_FIELD_U64, tensor_dtypes_checked, NULL),
    INTEGRITY_FIELD("tensor_dtypes_valid", YVEX_RENDER_FIELD_U64, tensor_dtypes_valid, NULL),
    INTEGRITY_FIELD("tensor_dtypes_invalid", YVEX_RENDER_FIELD_U64, tensor_dtypes_invalid, NULL),
    INTEGRITY_FIELD("tensor_byte_counts_checked", YVEX_RENDER_FIELD_U64,
                    tensor_byte_counts_checked, NULL),
    INTEGRITY_FIELD("tensor_byte_counts_invalid", YVEX_RENDER_FIELD_U64,
                    tensor_byte_counts_invalid, NULL),
};
static const yvex_render_field_spec integrity_selected_fields[] = {
    INTEGRITY_FIELD("selected_embedding_shape", YVEX_RENDER_FIELD_TEXT_ARRAY,
                    selected_embedding_shape, ""),
    INTEGRITY_FIELD("selected_embedding_hidden_size", YVEX_RENDER_FIELD_U64,
                    selected_embedding_hidden_size, NULL),
    INTEGRITY_FIELD("selected_embedding_vocab_size", YVEX_RENDER_FIELD_U64,
                    selected_embedding_vocab_size, NULL),
    INTEGRITY_FIELD("selected_embedding_output_count", YVEX_RENDER_FIELD_U64,
                    selected_embedding_output_count, NULL),
    INTEGRITY_FIELD("selected_embedding_output_bytes", YVEX_RENDER_FIELD_U64,
                    selected_embedding_output_bytes, NULL),
    INTEGRITY_FIELD("selected_embedding_slice_bytes", YVEX_RENDER_FIELD_U64,
                    selected_embedding_slice_bytes, NULL),
};
static const yvex_render_field_spec integrity_identity_fields[] = {
    INTEGRITY_FIELD("identity_checked", YVEX_RENDER_FIELD_BOOL, identity_checked, NULL),
    INTEGRITY_FIELD("sha256", YVEX_RENDER_FIELD_TEXT_ARRAY, sha256, "unavailable"),
    INTEGRITY_FIELD("registered_sha256", YVEX_RENDER_FIELD_TEXT_ARRAY, registered_sha256,
                    "absent"),
};
static const yvex_render_field_spec integrity_result_fields[] = {
    INTEGRITY_FIELD("digest_status", YVEX_RENDER_FIELD_TEXT_ARRAY, digest_status, "unknown"),
};
static const yvex_render_field_spec integrity_issue_count_fields[] = {
    INTEGRITY_FIELD("integrity_errors", YVEX_RENDER_FIELD_U32, error_count, NULL),
    INTEGRITY_FIELD("integrity_warnings", YVEX_RENDER_FIELD_U32, warning_count, NULL),
};
static const yvex_render_field_spec integrity_audit_counter_fields[] = {
    INTEGRITY_FIELD("tensor_count", YVEX_RENDER_FIELD_U64, tensor_count, NULL),
    INTEGRITY_FIELD("known_tensor_bytes", YVEX_RENDER_FIELD_U64, known_tensor_bytes, NULL),
    INTEGRITY_FIELD("tensor_ranges_checked", YVEX_RENDER_FIELD_U64, tensor_ranges_checked, NULL),
    INTEGRITY_FIELD("tensor_ranges_invalid", YVEX_RENDER_FIELD_U64, tensor_ranges_invalid, NULL),
    INTEGRITY_FIELD("tensor_shapes_checked", YVEX_RENDER_FIELD_U64, tensor_shapes_checked, NULL),
    INTEGRITY_FIELD("tensor_shapes_invalid", YVEX_RENDER_FIELD_U64, tensor_shapes_invalid, NULL),
    INTEGRITY_FIELD("tensor_dtypes_checked", YVEX_RENDER_FIELD_U64, tensor_dtypes_checked, NULL),
    INTEGRITY_FIELD("tensor_dtypes_invalid", YVEX_RENDER_FIELD_U64, tensor_dtypes_invalid, NULL),
    INTEGRITY_FIELD("tensor_byte_counts_checked", YVEX_RENDER_FIELD_U64,
                    tensor_byte_counts_checked, NULL),
    INTEGRITY_FIELD("tensor_byte_counts_invalid", YVEX_RENDER_FIELD_U64,
                    tensor_byte_counts_invalid, NULL),
};
#undef INTEGRITY_FIELD

#define SUMMARY_FIELD(key_, member_)                                                               \
    { key_, YVEX_RENDER_FIELD_U64, offsetof(yvex_materialize_summary, member_), NULL }
static const yvex_render_field_spec materialize_summary_fields[] = {
    SUMMARY_FIELD("tensors_total", tensors_total),
    SUMMARY_FIELD("tensors_materialized", tensors_materialized),
    SUMMARY_FIELD("tensors_failed", tensors_failed),
    SUMMARY_FIELD("bytes_total", bytes_total),
    SUMMARY_FIELD("bytes_materialized", bytes_materialized),
    SUMMARY_FIELD("backend_allocated_bytes", backend_allocated_bytes),
};
static const yvex_render_field_spec materialize_gate_prefix_fields[] = {
    {"materialization_gate", YVEX_RENDER_FIELD_TEXT,
     offsetof(yvex_materialize_summary, materialization_gate), "fail"},
    {"materialization_phase", YVEX_RENDER_FIELD_TEXT,
     offsetof(yvex_materialize_summary, materialization_phase), "preflight"},
};
static const yvex_render_field_spec materialize_gate_detail_fields[] = {
    {"shape_status", YVEX_RENDER_FIELD_TEXT, offsetof(yvex_materialize_summary, shape_status),
     "unchecked"},
    {"range_status", YVEX_RENDER_FIELD_TEXT, offsetof(yvex_materialize_summary, range_status),
     "unchecked"},
    {"backend_status", YVEX_RENDER_FIELD_TEXT, offsetof(yvex_materialize_summary, backend_status),
     "not-opened"},
    {"allocation_attempted", YVEX_RENDER_FIELD_BOOL,
     offsetof(yvex_materialize_summary, allocation_attempted), NULL},
    {"transfer_attempted", YVEX_RENDER_FIELD_BOOL,
     offsetof(yvex_materialize_summary, transfer_attempted), NULL},
    {"cleanup_attempted", YVEX_RENDER_FIELD_BOOL,
     offsetof(yvex_materialize_summary, cleanup_attempted), NULL},
    {"cleanup_status", YVEX_RENDER_FIELD_TEXT, offsetof(yvex_materialize_summary, cleanup_status),
     "not-needed"},
    {"bytes_planned", YVEX_RENDER_FIELD_U64, offsetof(yvex_materialize_summary, bytes_planned),
     NULL},
    {"bytes_allocated", YVEX_RENDER_FIELD_U64, offsetof(yvex_materialize_summary, bytes_allocated),
     NULL},
    {"bytes_transferred", YVEX_RENDER_FIELD_U64,
     offsetof(yvex_materialize_summary, bytes_transferred), NULL},
};
#undef SUMMARY_FIELD

#define MATERIALIZE_FIELD(key_, kind_, member_, fallback_)                                         \
    { key_, kind_, offsetof(yvex_materialize_gate_summary, member_), fallback_ }

static const yvex_render_field_spec materialize_gate_heading_fields[] = {
    MATERIALIZE_FIELD("label", YVEX_RENDER_FIELD_TEXT, label, ""),
    MATERIALIZE_FIELD("family", YVEX_RENDER_FIELD_TEXT, family, ""),
};
static const yvex_render_field_spec materialize_gate_state_fields[] = {
    MATERIALIZE_FIELD("model", YVEX_RENDER_FIELD_TEXT, model_path, ""),
    MATERIALIZE_FIELD("expected_sha256", YVEX_RENDER_FIELD_TEXT, expected_sha256, ""),
    MATERIALIZE_FIELD("actual_sha256", YVEX_RENDER_FIELD_TEXT_ARRAY, actual_sha256, ""),
    MATERIALIZE_FIELD("digest_status", YVEX_RENDER_FIELD_TEXT, digest_status, "unrequested"),
    MATERIALIZE_FIELD("identity_status", YVEX_RENDER_FIELD_TEXT, identity_status, "unrequested"),
    MATERIALIZE_FIELD("metadata_status", YVEX_RENDER_FIELD_TEXT, metadata_status, "unregistered"),
    MATERIALIZE_FIELD("materialization_gate", YVEX_RENDER_FIELD_TEXT, materialization_gate, "fail"),
    MATERIALIZE_FIELD("materialization_phase", YVEX_RENDER_FIELD_TEXT, materialization_phase,
                      "preflight"),
    MATERIALIZE_FIELD("integrity_status", YVEX_RENDER_FIELD_TEXT, integrity_status, "unchecked"),
    MATERIALIZE_FIELD("shape_status", YVEX_RENDER_FIELD_TEXT, shape_status, "unchecked"),
    MATERIALIZE_FIELD("range_status", YVEX_RENDER_FIELD_TEXT, range_status, "unchecked"),
    MATERIALIZE_FIELD("backend_status", YVEX_RENDER_FIELD_TEXT, backend_status, "not-opened"),
    MATERIALIZE_FIELD("allocation_attempted", YVEX_RENDER_FIELD_BOOL, allocation_attempted, NULL),
    MATERIALIZE_FIELD("transfer_attempted", YVEX_RENDER_FIELD_BOOL, transfer_attempted, NULL),
    MATERIALIZE_FIELD("cleanup_attempted", YVEX_RENDER_FIELD_BOOL, cleanup_attempted, NULL),
    MATERIALIZE_FIELD("cleanup_status", YVEX_RENDER_FIELD_TEXT, cleanup_status, "not-needed"),
    MATERIALIZE_FIELD("bytes_planned", YVEX_RENDER_FIELD_U64, bytes_planned, NULL),
    MATERIALIZE_FIELD("bytes_allocated", YVEX_RENDER_FIELD_U64, bytes_allocated, NULL),
    MATERIALIZE_FIELD("bytes_transferred", YVEX_RENDER_FIELD_U64, bytes_transferred, NULL),
    MATERIALIZE_FIELD("file_bytes", YVEX_RENDER_FIELD_U64, file_bytes, NULL),
    MATERIALIZE_FIELD("tensor_count", YVEX_RENDER_FIELD_U64, tensor_count, NULL),
    MATERIALIZE_FIELD("expected_tensor_matches", YVEX_RENDER_FIELD_U64, expected_tensor_matches,
                      NULL),
    MATERIALIZE_FIELD("expected_tensor_mismatches", YVEX_RENDER_FIELD_U64,
                      expected_tensor_mismatches, NULL),
    MATERIALIZE_FIELD("bytes_materialized_cpu", YVEX_RENDER_FIELD_U64, bytes_materialized_cpu,
                      NULL),
    MATERIALIZE_FIELD("bytes_materialized_cuda", YVEX_RENDER_FIELD_U64, bytes_materialized_cuda,
                      NULL),
};
#undef MATERIALIZE_FIELD
#define MODEL_GATE_FIELD(key_, kind_, member_, fallback_)                                          \
    { key_, kind_, offsetof(yvex_model_gate_summary, member_), fallback_ }

static const yvex_render_field_spec model_gate_state_fields[] = {
    MODEL_GATE_FIELD("label", YVEX_RENDER_FIELD_TEXT, model_label, ""),
    MODEL_GATE_FIELD("family", YVEX_RENDER_FIELD_TEXT, family, ""),
    MODEL_GATE_FIELD("model", YVEX_RENDER_FIELD_TEXT, model_path, ""),
    MODEL_GATE_FIELD("expected_sha256", YVEX_RENDER_FIELD_TEXT, expected_sha256, ""),
    MODEL_GATE_FIELD("actual_sha256", YVEX_RENDER_FIELD_TEXT_ARRAY, actual_sha256, ""),
    MODEL_GATE_FIELD("digest_status", YVEX_RENDER_FIELD_TEXT, digest_status, "unrequested"),
    MODEL_GATE_FIELD("identity_status", YVEX_RENDER_FIELD_TEXT, identity_status, "unrequested"),
    MODEL_GATE_FIELD("file_bytes", YVEX_RENDER_FIELD_U64, file_bytes, NULL),
    MODEL_GATE_FIELD("tensor_count", YVEX_RENDER_FIELD_U64, tensor_count, NULL),
    MODEL_GATE_FIELD("expected_tensor_matches", YVEX_RENDER_FIELD_U64, expected_tensor_matches,
                     NULL),
    MODEL_GATE_FIELD("expected_tensor_mismatches", YVEX_RENDER_FIELD_U64,
                     expected_tensor_mismatches, NULL),
};
#undef MODEL_GATE_FIELD

static int command_integrity_report(int arg_count, char **args);

typedef struct {
    yvex_artifact *artifact;
    yvex_gguf *gguf;
    yvex_tensor_table *tensors;
    yvex_model_descriptor *model;
} artifact_view;

static void artifact_view_close(artifact_view *view) {
    if (!view)
        return;
    yvex_model_descriptor_close(view->model);
    yvex_tensor_table_close(view->tensors);
    yvex_gguf_close(view->gguf);
    yvex_artifact_close(view->artifact);
    memset(view, 0, sizeof(*view));
}

static int artifact_view_open(artifact_view *view, const char *path, int require_tensors,
                              int require_model, yvex_error *err) {
    int rc;
    memset(view, 0, sizeof(*view));
    rc = open_artifact_for_gguf(path, &view->artifact, err);
    if (rc == YVEX_OK)
        rc = yvex_gguf_open(&view->gguf, view->artifact, err);
    if (rc == YVEX_OK && (require_tensors || require_model))
        rc = yvex_tensor_table_from_gguf(&view->tensors, view->gguf, err);
    if (rc == YVEX_OK && require_model)
        rc = yvex_model_descriptor_from_gguf(&view->model, view->gguf, view->tensors, err);
    if (rc != YVEX_OK)
        artifact_view_close(view);
    return rc;
}

static void print_integrity_report(const yvex_artifact_integrity_report *report,
                                   const char *model_label) {
    unsigned int i;
    yvex_cli_out_writef(stdout, "artifact_integrity: check\nmodel: %s\nformat: %s\n",
                        model_label ? model_label : report->path,
                        report->format[0] ? report->format : "unknown");
    if (report->version)
        yvex_cli_out_writef(stdout, "version: %u\n", report->version);
    yvex_cli_out_writef(stdout, "file_size: %llu\n", report->file_size);
    if (report->architecture[0])
        yvex_cli_out_writef(stdout, "architecture: %s\n", report->architecture);
    render_object_fields(stdout, report, integrity_counter_fields, FIELD_COUNT(integrity_counter_fields));
    if (report->selected_embedding_shape[0])
        render_object_fields(stdout, report, integrity_selected_fields,
                             FIELD_COUNT(integrity_selected_fields));
    render_object_fields(stdout, report, integrity_identity_fields, FIELD_COUNT(integrity_identity_fields));
    if (report->expected_sha256[0])
        yvex_cli_out_writef(stdout, "expected_sha256: %s\nactual_sha256: %s\n",
                            report->expected_sha256,
                            report->sha256[0] ? report->sha256 : "unavailable");
    render_object_fields(stdout, report, integrity_result_fields, FIELD_COUNT(integrity_result_fields));
    yvex_cli_out_writef(stdout, "integrity_status: %s\n", report->passed ? "pass" : "fail");
    render_object_fields(stdout, report, integrity_issue_count_fields,
                         FIELD_COUNT(integrity_issue_count_fields));
    for (i = 0; i < report->issue_count; ++i) {
        const yvex_integrity_issue *issue = yvex_artifact_integrity_issue_at(report, i);
        const char *prefix;
        if (!issue)
            continue;
        prefix = issue->severity == YVEX_INTEGRITY_SEVERITY_WARNING ? "warning" : "error";
        yvex_cli_out_writef(stdout, "%s_%u_code: %s\n", prefix, i, issue->code);
        if (issue->tensor[0])
            yvex_cli_out_writef(stdout, "%s_%u_tensor: %s\n", prefix, i, issue->tensor);
        if (issue->has_range) {
            yvex_cli_out_writef(stdout,
                                "%s_%u_relative_offset: %llu\n%s_%u_absolute_offset: %llu\n"
                                "%s_%u_tensor_bytes: %llu\n%s_%u_file_size: %llu\n",
                                prefix, i, issue->relative_offset, prefix, i,
                                issue->absolute_offset, prefix, i, issue->tensor_bytes, prefix, i,
                                issue->file_size);
        }
        yvex_cli_out_writef(stdout, "%s_%u_reason: %s\n", prefix, i, issue->reason);
    }
    yvex_cli_out_writef(stdout, "status: %s\n",
                        report->passed ? "artifact-integrity-pass" : "artifact-integrity-fail");
}

static void print_metadata_value(const yvex_gguf_value *value) {
    unsigned long long u64;
    long long i64;
    double f64;
    int bool_value;
    const char *string_data;
    unsigned long long string_len;
    yvex_gguf_array_info array;
    switch (yvex_gguf_value_type_of(value)) {
    case YVEX_GGUF_VALUE_UINT8:
    case YVEX_GGUF_VALUE_UINT16:
    case YVEX_GGUF_VALUE_UINT32:
    case YVEX_GGUF_VALUE_UINT64:
        if (yvex_gguf_value_as_u64(value, &u64) == YVEX_OK) {
            yvex_cli_out_writef(stdout, "%llu", u64);
        }
        break;
    case YVEX_GGUF_VALUE_INT8:
    case YVEX_GGUF_VALUE_INT16:
    case YVEX_GGUF_VALUE_INT32:
    case YVEX_GGUF_VALUE_INT64:
        if (yvex_gguf_value_as_i64(value, &i64) == YVEX_OK) {
            yvex_cli_out_writef(stdout, "%lld", i64);
        }
        break;
    case YVEX_GGUF_VALUE_FLOAT32:
    case YVEX_GGUF_VALUE_FLOAT64:
        if (yvex_gguf_value_as_f64(value, &f64) == YVEX_OK) {
            yvex_cli_out_writef(stdout, "%g", f64);
        }
        break;
    case YVEX_GGUF_VALUE_BOOL:
        if (yvex_gguf_value_as_bool(value, &bool_value) == YVEX_OK) {
            yvex_cli_out_writef(stdout, "%s", bool_value ? "true" : "false");
        }
        break;
    case YVEX_GGUF_VALUE_STRING:
        if (yvex_gguf_value_as_string(value, &string_data, &string_len) == YVEX_OK) {
            print_quoted_bytes(string_data, string_len);
        }
        break;
    case YVEX_GGUF_VALUE_ARRAY:
        if (yvex_gguf_value_array_info(value, &array) == YVEX_OK) {
            yvex_cli_out_writef(stdout, "array<%s>[%llu]",
                                yvex_gguf_value_type_name(array.element_type), array.count);
        }
        break;
    case YVEX_GGUF_VALUE_INVALID:
        yvex_cli_out_writef(stdout, "%s", "<invalid>");
        break;
    }
}

typedef struct {
    yvex_artifact_integrity_options integrity;
    const char *model;
    const char *backend;
    int audit;
} integrity_cli_options;

static int integrity_options_parse(int arg_count, char **args, int report_mode,
                                   integrity_cli_options *options) {
    const char *surface = report_mode ? "integrity report" : "integrity";
    int i;
    memset(options, 0, sizeof(*options));
    for (i = 3; i < arg_count; ++i) {
        const char *option = args[i];
        if (strcmp(option, "--require-token-embedding") == 0) {
            options->integrity.require_token_embedding = 1;
            continue;
        }
        if (report_mode && strcmp(option, "--audit") == 0) {
            options->audit = 1;
            continue;
        }
        if (i + 1 >= arg_count) {
            const char *requirement = strcmp(option, "--model") == 0           ? "FILE_OR_ALIAS"
                                      : strcmp(option, "--expect-sha256") == 0 ? "HASH"
                                      : strcmp(option, "--backend") == 0       ? "cpu|cuda"
                                      : strcmp(option, "--partial-token") == 0
                                          ? "a non-negative integer"
                                      : strcmp(option, "--output") == 0 ? "normal, table, or audit"
                                                                        : NULL;
            if (requirement)
                yvex_cli_out_writef(stderr, "yvex: %s requires %s\n", option, requirement);
            else
                yvex_cli_out_writef(stderr, "yvex: unknown %s option: %s\n", surface, option);
            return 2;
        }
        if (strcmp(option, "--model") == 0)
            options->model = args[++i];
        else if (strcmp(option, "--expect-sha256") == 0)
            options->integrity.expect_sha256 = args[++i];
        else if (strcmp(option, "--partial-token") == 0) {
            if (!parse_uint_allow_zero(args[++i], &options->integrity.token_id)) {
                yvex_cli_out_writef(stderr,
                                    "yvex: --partial-token requires a non-negative integer\n");
                return 2;
            }
            options->integrity.require_token_embedding = 1;
        } else if (report_mode && strcmp(option, "--backend") == 0) {
            options->backend = args[++i];
        } else if (report_mode && strcmp(option, "--output") == 0) {
            const char *mode = args[++i];
            if (strcmp(mode, "normal") == 0 || strcmp(mode, "table") == 0)
                options->audit = 0;
            else if (strcmp(mode, "audit") == 0)
                options->audit = 1;
            else {
                yvex_cli_out_writef(stderr, "yvex: integrity report unsupported output mode: %s\n",
                                    mode);
                return 2;
            }
        } else {
            yvex_cli_out_writef(stderr, "yvex: unknown %s option: %s\n", surface, option);
            yvex_cli_out_writef(stderr, "Try 'yvex help integrity' for usage.\n");
            return 2;
        }
    }
    if (!options->model) {
        yvex_cli_out_writef(stderr, "yvex: %s requires --model FILE_OR_ALIAS\n",
                            report_mode ? "integrity report" : "integrity check");
        return 2;
    }
    if (options->backend && !cli_backend_name_valid(options->backend)) {
        yvex_cli_out_writef(stderr, "yvex: unknown backend kind: %s\n", options->backend);
        return 2;
    }
    return 0;
}

static int command_integrity(int arg_count, char **args) {
    integrity_cli_options parsed;
    yvex_artifact_integrity_report report;
    yvex_model_ref ref;
    yvex_error err;
    int rc;
    yvex_error_clear(&err);
    memset(&report, 0, sizeof(report));
    memset(&ref, 0, sizeof(ref));
    if (arg_count >= 3 && (strcmp(args[2], "--help") == 0 || strcmp(args[2], "-h") == 0)) {
        yvex_integrity_help(stdout);
        return 0;
    }
    if (arg_count >= 3 && strcmp(args[2], "report") == 0) {
        return command_integrity_report(arg_count, args);
    }
    if (arg_count < 3 || strcmp(args[2], "check") != 0) {
        yvex_cli_out_writef(stderr, "yvex: integrity requires check or report\n");
        yvex_cli_out_writef(
            stderr,
            "usage: yvex artifact verify FILE_OR_ALIAS [--expect-sha256 HASH] "
            "[--require-token-"
            "embedding] [--partial-token N] | yvex inspect artifact integrity FILE_OR_ALIAS "
            "[--backend cpu|"
            "cuda] [--expect-sha256 HASH] [--require-token-embedding] [--partial-token N]\n");
        return 2;
    }
    rc = integrity_options_parse(arg_count, args, 0, &parsed);
    if (rc != 0)
        return rc;
    rc = yvex_model_ref_resolve(&ref, parsed.model, NULL, &err);
    if (rc != YVEX_OK) {
        return print_yvex_error(&err, exit_for_status(rc));
    }
    if (ref.kind == YVEX_MODEL_REF_ALIAS && ref.sha256 && ref.sha256[0]) {
        parsed.integrity.registered_sha256 = ref.sha256;
    }
    rc = yvex_artifact_integrity_check_path(ref.path, &parsed.integrity, &report, &err);
    print_integrity_report(&report, ref.path);
    yvex_model_ref_clear(&ref);
    if (rc != YVEX_OK) {
        return exit_for_status(rc);
    }
    return 0;
}

static int command_inspect(int arg_count, char **args) {
    artifact_view view = {0};
    yvex_gguf_probe probe;
    const yvex_gguf_header *header;
    yvex_artifact_integrity_options integrity_options;
    yvex_artifact_integrity_report integrity_report;
    yvex_error err;
    int rc;
    yvex_error_clear(&err);
    memset(&integrity_options, 0, sizeof(integrity_options));
    memset(&integrity_report, 0, sizeof(integrity_report));
    if (arg_count != 3 || strcmp(args[2], "--help") == 0 || strcmp(args[2], "-h") == 0) {
        if (arg_count == 3) {
            yvex_inspect_help(stdout);
            return 0;
        }
        yvex_cli_out_writef(stderr, "yvex: inspect requires exactly one FILE_OR_ALIAS\n");
        yvex_cli_out_writef(stderr, "usage: yvex artifact show FILE_OR_ALIAS\n");
        return 2;
    }
    rc = open_artifact_for_gguf(args[2], &view.artifact, &err);
    if (rc != YVEX_OK) {
        return print_yvex_error(&err, exit_for_status(rc));
    }
    rc = yvex_gguf_probe_file(view.artifact, &probe, &err);
    if (rc == YVEX_OK && !probe.is_gguf) {
        yvex_cli_out_lines(stdout, literal_pair_2,
                           sizeof(literal_pair_2) / sizeof(literal_pair_2[0]));
        artifact_view_close(&view);
        return 5;
    }
    if (rc != YVEX_OK) {
        if (rc == YVEX_ERR_UNSUPPORTED) {
            yvex_cli_out_lines(stdout, literal_pair_1,
                               sizeof(literal_pair_1) / sizeof(literal_pair_1[0]));
        }
        artifact_view_close(&view);
        return print_yvex_error(&err, exit_for_status(rc));
    }
    rc = yvex_gguf_open(&view.gguf, view.artifact, &err);
    if (rc == YVEX_OK)
        rc = yvex_tensor_table_from_gguf(&view.tensors, view.gguf, &err);
    if (rc == YVEX_OK)
        rc = yvex_model_descriptor_from_gguf(&view.model, view.gguf, view.tensors, &err);
    if (rc == YVEX_OK)
        rc = yvex_artifact_integrity_validate(view.artifact, view.gguf, view.tensors,
                                              &integrity_options, &integrity_report, &err);
    if (rc == YVEX_OK) {
        header = yvex_gguf_header_view(view.gguf);
        yvex_cli_out_writef(stdout, "format: gguf\n");
        yvex_cli_out_writef(stdout, "version: %u\n", header->version);
        yvex_cli_out_writef(stdout, "metadata_count: %llu\n", header->metadata_count);
        yvex_cli_out_writef(stdout, "tensor_count: %llu\n", header->tensor_count);
        yvex_cli_out_writef(stdout, "tensor_data_offset: %llu\n",
                            yvex_gguf_tensor_data_offset(view.gguf));
        yvex_cli_out_writef(stdout, "alignment: %u\n", yvex_gguf_alignment(view.gguf));
        yvex_cli_out_writef(stdout, "architecture: %s\n",
                            yvex_arch_name(yvex_model_arch(view.model)));
        yvex_cli_out_writef(stdout, "model_name: %s\n", yvex_model_name(view.model));
        yvex_cli_out_writef(stdout, "known_tensor_bytes: %llu\n",
                            yvex_model_total_storage_bytes(view.model));
        yvex_cli_out_writef(stdout, "unsupported_tensor_accounting: %llu\n",
                            yvex_model_unsupported_tensor_accounting_count(view.model));
        yvex_cli_out_writef(stdout, "status: descriptor-only\n");
        artifact_view_close(&view);
        return 0;
    }
    artifact_view_close(&view);
    return print_yvex_error(&err, exit_for_status(rc));
}

static int command_metadata(int arg_count, char **args) {
    artifact_view view;
    const yvex_gguf_header *header;
    yvex_error err;
    unsigned long long i;
    int rc;
    yvex_error_clear(&err);
    if (arg_count != 3 || strcmp(args[2], "--help") == 0 || strcmp(args[2], "-h") == 0) {
        if (arg_count == 3) {
            yvex_metadata_help(stdout);
            return 0;
        }
        yvex_cli_out_writef(stderr, "yvex: metadata requires exactly one FILE_OR_ALIAS\n");
        yvex_cli_out_writef(stderr, "usage: yvex inspect artifact metadata FILE_OR_ALIAS\n");
        return 2;
    }
    rc = artifact_view_open(&view, args[2], 0, 0, &err);
    if (rc != YVEX_OK)
        return print_yvex_error(&err, exit_for_status(rc));
    header = yvex_gguf_header_view(view.gguf);
    yvex_cli_out_writef(stdout, "format: gguf\n");
    yvex_cli_out_writef(stdout, "version: %u\n", header->version);
    yvex_cli_out_writef(stdout, "metadata_count: %llu\n", yvex_gguf_metadata_count(view.gguf));
    yvex_cli_out_writef(stdout, "\n");
    for (i = 0; i < yvex_gguf_metadata_count(view.gguf); ++i) {
        const char *key = yvex_gguf_metadata_key(view.gguf, i);
        const yvex_gguf_value *value = yvex_gguf_metadata_value(view.gguf, i);
        yvex_cli_out_writef(stdout, "%s = ", key ? key : "");
        print_metadata_value(value);
        yvex_cli_out_writef(stdout, "\n");
    }
    artifact_view_close(&view);
    return 0;
}

static void print_registered_identity(const yvex_model_ref *ref,
                                      const yvex_artifact_file_identity *identity,
                                      const char *surface, const char *digest_status,
                                      const char *identity_status) {
    yvex_cli_out_writef(
        stdout,
        "artifact_identity: check\nsurface: %s\nalias: %s\npath: %s\nregistered_sha256: %s\n"
        "current_sha256: %s\nregistered_file_size: %llu\ncurrent_file_size: %llu\n"
        "digest_status: %s\nidentity_status: %s\n",
        surface ? surface : "unknown", ref->alias ? ref->alias : "", ref->path ? ref->path : "",
        ref->sha256 && ref->sha256[0] ? ref->sha256 : "absent",
        identity->sha256[0] ? identity->sha256 : "unavailable", ref->registered_file_size,
        identity->file_size, digest_status, identity_status);
}

int enforce_registered_identity_cli(const yvex_model_ref *ref, const char *surface) {
    yvex_artifact_file_identity identity;
    yvex_model_metadata_snapshot current_metadata;
    yvex_model_registry_entry registered_metadata;
    yvex_model_metadata_drift_report metadata_report;
    yvex_error err;
    const char *identity_status = "pass";
    const char *digest_status = "pass";
    const char *reason = "current file identity matches registered alias";
    int pass = 1;
    int rc;
    if (!ref || ref->kind != YVEX_MODEL_REF_ALIAS) {
        return YVEX_OK;
    }
    yvex_error_clear(&err);
    memset(&identity, 0, sizeof(identity));
    rc = yvex_artifact_identity_read(ref->path, &identity, &err);
    if (rc != YVEX_OK) {
        identity_status = "fail";
        digest_status = "fail";
        reason = yvex_error_message(&err);
        pass = 0;
    } else if (!ref->sha256 || !ref->sha256[0] || !yvex_sha256_hex_is_valid(ref->sha256)) {
        identity_status = "missing";
        digest_status = "missing";
        reason = "registered alias lacks digest identity; re-add model";
        pass = 0;
        rc = YVEX_ERR_STATE;
    } else if (strcmp(ref->sha256, identity.sha256) != 0 ||
               (ref->registered_file_size != 0ull &&
                ref->registered_file_size != identity.file_size)) {
        identity_status = "fail";
        digest_status = "fail";
        reason = "digest mismatch for registered alias";
        pass = 0;
        rc = YVEX_ERR_STATE;
    } else {
        rc = YVEX_OK;
    }
    if (pass) {
        memset(&current_metadata, 0, sizeof(current_metadata));
        memset(&registered_metadata, 0, sizeof(registered_metadata));
        memset(&metadata_report, 0, sizeof(metadata_report));
        rc = yvex_model_metadata_snapshot_read(&current_metadata, ref->path, &err);
        if (rc != YVEX_OK) {
            print_registered_identity(ref, &identity, surface, digest_status, identity_status);
            yvex_cli_out_lines(stdout, literal_lines_0,
                               sizeof(literal_lines_0) / sizeof(literal_lines_0[0]));
            yvex_cli_out_writef(stdout, "metadata_issue_0_current: %s\n", yvex_error_message(&err));
            yvex_cli_out_lines(stdout, literal_pair_0,
                               sizeof(literal_pair_0) / sizeof(literal_pair_0[0]));
            return exit_for_status(YVEX_ERR_STATE);
        }
        yvex_model_ref_registry_entry_view(ref, &registered_metadata);
        rc = yvex_model_registry_compare_metadata(&registered_metadata, &current_metadata.entry,
                                                  &metadata_report, &err);
        if (rc != YVEX_OK || strcmp(metadata_report.metadata_status, "pass") != 0 ||
            strcmp(metadata_report.readiness_status, "pass") != 0) {
            const char *status = strcmp(metadata_report.metadata_status, "missing") == 0
                                     ? "models-metadata-missing"
                                     : "models-metadata-drift";
            print_registered_identity(ref, &identity, surface, digest_status, identity_status);
            print_metadata_drift_cli(&metadata_report);
            yvex_cli_out_writef(
                stdout,
                "reason: registered alias metadata does not match current artifact facts\n");
            yvex_cli_out_writef(stdout, "status: %s\n", status);
            return exit_for_status(YVEX_ERR_STATE);
        }
    }
    if (!pass) {
        print_registered_identity(ref, &identity, surface, digest_status, identity_status);
        yvex_cli_out_writef(stdout, "reason: %s\n", reason);
        yvex_cli_out_writef(stdout, "status: %s\n",
                            strcmp(identity_status, "missing") == 0 ? "models-identity-missing"
                                                                    : "models-identity-fail");
    }
    return rc;
}

static void print_materialization_summary(const yvex_model_context *ctx,
                                          const yvex_model_ref *model_ref, const char *backend_name,
                                          const yvex_materialize_summary *summary) {
    yvex_cli_out_writef(stdout, "materialization status: %s\n",
                        yvex_weight_status_name(summary->status));
    yvex_cli_out_writef(stdout, "model: %s\n",
                        yvex_model_name(ctx->model)[0] ? yvex_model_name(ctx->model) : "unknown");
    yvex_cli_out_writef(stdout, "backend: %s\n", backend_name);
    render_object_fields(stdout, summary, materialize_gate_prefix_fields,
                         FIELD_COUNT(materialize_gate_prefix_fields));
    yvex_cli_out_writef(stdout, "integrity_status: pass\nidentity_status: %s\nmetadata_status: %s\n",
                        model_ref->kind == YVEX_MODEL_REF_ALIAS ? "pass" : "unregistered",
                        model_ref->kind == YVEX_MODEL_REF_ALIAS ? "pass" : "unregistered");
    render_object_fields(stdout, summary, materialize_gate_detail_fields,
                         FIELD_COUNT(materialize_gate_detail_fields));
    render_object_fields(stdout, summary, materialize_summary_fields,
                         FIELD_COUNT(materialize_summary_fields));
    yvex_cli_out_writef(
        stdout,
        "execution_ready: false\nreason: graph partial; materialized weights do not imply "
        "executable inference\nstatus: %s\n",
        summary->status == YVEX_WEIGHT_STATUS_MATERIALIZED ? "weights-materialized"
                                                           : "weights-partial");
}

typedef struct {
    yvex_model_context context;
    yvex_backend *backend;
    yvex_weight_table *weights;
    yvex_model_ref reference;
} materialize_cli_state;

typedef enum {
    MATERIALIZE_REFUSAL_IDENTITY,
    MATERIALIZE_REFUSAL_CONTEXT,
    MATERIALIZE_REFUSAL_BACKEND,
    MATERIALIZE_REFUSAL_DTYPE,
    MATERIALIZE_REFUSAL_TRANSFER,
    MATERIALIZE_REFUSAL_ALLOCATION,
    MATERIALIZE_REFUSAL_PREFLIGHT
} materialize_refusal_kind;

enum {
    MATERIALIZE_REFUSAL_HEADER = 1u << 0u,
    MATERIALIZE_REFUSAL_REASON = 1u << 1u,
    MATERIALIZE_REFUSAL_ALLOCATED = 1u << 2u,
    MATERIALIZE_REFUSAL_TRANSFERRED = 1u << 3u,
    MATERIALIZE_REFUSAL_CLEANED = 1u << 4u
};

typedef struct {
    const char *phase;
    const char *integrity;
    const char *identity;
    const char *metadata;
    const char *shape;
    const char *range;
    const char *backend;
    const char *status;
    unsigned int flags;
} materialize_refusal_spec;

static const materialize_refusal_spec materialize_refusals[] = {
    {"preflight", "not-checked", "fail", "fail", "not-checked", "not-checked", "not-opened",
     "materialization-integrity-fail", 0u},
    {"preflight", "fail", NULL, NULL, "unchecked", "unchecked", "not-opened",
     "materialization-integrity-fail", 0u},
    {"preflight", "pass", NULL, NULL, "pass", "pass", "unavailable", "weights-unsupported",
     MATERIALIZE_REFUSAL_HEADER | MATERIALIZE_REFUSAL_REASON},
    {"preflight", "pass", NULL, NULL, "fail", "fail", "ready", "weights-unsupported",
     MATERIALIZE_REFUSAL_HEADER | MATERIALIZE_REFUSAL_REASON},
    {"transfer", "pass", NULL, NULL, "pass", "pass", "ready",
     "materialization-failed-cleaned", MATERIALIZE_REFUSAL_ALLOCATED |
                                                MATERIALIZE_REFUSAL_TRANSFERRED |
                                                MATERIALIZE_REFUSAL_CLEANED},
    {"allocation", "pass", NULL, NULL, "pass", "pass", "ready",
     "materialization-failed-cleaned", MATERIALIZE_REFUSAL_ALLOCATED |
                                                MATERIALIZE_REFUSAL_CLEANED},
    {"preflight", "fail", NULL, NULL, "fail", "fail", "ready",
     "materialization-integrity-fail", 0u},
};

static void print_materialize_refusal(const yvex_model_ref *ref, const char *backend_name,
                                      materialize_refusal_kind kind, const char *reason) {
    const materialize_refusal_spec *spec = &materialize_refusals[kind];
    const char *registered = ref->kind == YVEX_MODEL_REF_ALIAS ? "pass" : "unregistered";
    if (spec->flags & MATERIALIZE_REFUSAL_HEADER) {
        yvex_cli_out_writef(stdout, "materialization status: unsupported\n");
        yvex_cli_out_writef(stdout, "backend: %s\n", backend_name);
    }
    yvex_cli_out_writef(
        stdout,
        "materialization_gate: fail\nmaterialization_phase: %s\nintegrity_status: %s\n"
        "identity_status: %s\nmetadata_status: %s\nshape_status: %s\nrange_status: %s\n"
        "backend_status: %s\nallocation_attempted: %s\ntransfer_attempted: %s\n"
        "cleanup_attempted: %s\ncleanup_status: %s\nbytes_planned: 0\nbytes_allocated: 0\n"
        "bytes_transferred: 0\n",
        spec->phase, spec->integrity, spec->identity ? spec->identity : registered,
        spec->metadata ? spec->metadata : registered, spec->shape, spec->range, spec->backend,
        spec->flags & MATERIALIZE_REFUSAL_ALLOCATED ? "true" : "false",
        spec->flags & MATERIALIZE_REFUSAL_TRANSFERRED ? "true" : "false",
        spec->flags & MATERIALIZE_REFUSAL_CLEANED ? "true" : "false",
        spec->flags & MATERIALIZE_REFUSAL_CLEANED ? "pass" : "not-needed");
    if (spec->flags & MATERIALIZE_REFUSAL_REASON)
        yvex_cli_out_writef(stdout, "reason: %s\n", reason ? reason : "");
    yvex_cli_out_writef(stdout, "status: %s\n", spec->status);
}

static void materialize_cli_close(materialize_cli_state *state) {
    if (!state)
        return;
    yvex_weight_table_close(state->weights);
    yvex_backend_close(state->backend);
    yvex_model_context_close(&state->context);
    yvex_model_ref_clear(&state->reference);
    memset(state, 0, sizeof(*state));
}

static int command_materialize(int arg_count, char **args) {
    materialize_cli_state state;
    yvex_backend_options backend_options;
    yvex_materialize_options materialize_options;
    yvex_materialize_summary summary;
    yvex_error err;
    const char *model_path = NULL;
    const char *backend_name = NULL;
    int i;
    int rc;
    yvex_error_clear(&err);
    memset(&state, 0, sizeof(state));
    memset(&backend_options, 0, sizeof(backend_options));
    memset(&materialize_options, 0, sizeof(materialize_options));
    if (arg_count == 3 && (strcmp(args[2], "--help") == 0 || strcmp(args[2], "-h") == 0)) {
        yvex_materialize_help(stdout);
        return 0;
    }
    for (i = 2; i < arg_count; ++i) {
        if (strcmp(args[i], "--model") == 0) {
            if (i + 1 >= arg_count) {
                yvex_cli_out_writef(stderr, "yvex: --model requires FILE_OR_ALIAS\n");
                return 2;
            }
            model_path = args[++i];
        } else if (strcmp(args[i], "--backend") == 0) {
            if (i + 1 >= arg_count) {
                yvex_cli_out_writef(stderr, "yvex: --backend requires cpu or cuda\n");
                return 2;
            }
            backend_name = args[++i];
        } else if (strcmp(args[i], "--require-all") == 0) {
            materialize_options.require_all_tensors = 1;
        } else if (strcmp(args[i], "--allow-unsupported-dtype") == 0) {
            materialize_options.allow_unsupported_dtype = 1;
        } else {
            yvex_cli_out_writef(stderr, "yvex: unknown materialize option: %s\n", args[i]);
            yvex_cli_out_writef(stderr, "Try 'yvex help materialize' for usage.\n");
            return 2;
        }
    }
    if (!model_path || !backend_name) {
        yvex_cli_out_writef(
            stderr, "yvex: materialize requires --model FILE_OR_ALIAS and --backend cpu|cuda\n");
        yvex_cli_out_writef(stderr, "usage: yvex artifact materialize --model FILE_OR_ALIAS --backend "
                                    "cpu|cuda [--require-all] [--allow-"
                                    "unsupported-dtype]\n");
        return 2;
    }
    if (strcmp(backend_name, "cpu") == 0) {
        backend_options.kind = YVEX_BACKEND_KIND_CPU;
    } else if (strcmp(backend_name, "cuda") == 0) {
        backend_options.kind = YVEX_BACKEND_KIND_CUDA;
    } else {
        yvex_cli_out_writef(stderr, "yvex: unknown backend kind: %s\n", backend_name);
        return 2;
    }
    rc = yvex_model_ref_resolve(&state.reference, model_path, NULL, &err);
    if (rc != YVEX_OK)
        return print_yvex_error(&err, exit_for_status(rc));
    rc = enforce_registered_identity_cli(&state.reference, "materialize");
    if (rc != YVEX_OK) {
        print_materialize_refusal(&state.reference, backend_name, MATERIALIZE_REFUSAL_IDENTITY,
                                  NULL);
        rc = exit_for_status(rc);
        goto done;
    }
    rc = yvex_model_context_open(state.reference.path, &state.context, &err);
    if (rc != YVEX_OK) {
        print_materialize_refusal(&state.reference, backend_name, MATERIALIZE_REFUSAL_CONTEXT,
                                  NULL);
        rc = print_yvex_error(&err, exit_for_status(rc));
        goto done;
    }
    rc = yvex_backend_open(&state.backend, &backend_options, &err);
    if (rc == YVEX_ERR_UNSUPPORTED) {
        print_materialize_refusal(&state.reference, backend_name, MATERIALIZE_REFUSAL_BACKEND,
                                  yvex_error_message(&err));
        rc = 5;
        goto done;
    }
    if (rc != YVEX_OK) {
        rc = print_yvex_error(&err, exit_for_status(rc));
        goto done;
    }
    materialize_options.backend_name = backend_name;
    rc = yvex_weight_table_materialize(&state.weights, state.context.artifact, state.context.gguf,
                                       state.context.table, state.backend, &materialize_options,
                                       &err);
    if (rc == YVEX_ERR_UNSUPPORTED) {
        print_materialize_refusal(&state.reference, backend_name, MATERIALIZE_REFUSAL_DTYPE,
                                  yvex_error_message(&err));
        rc = 5;
        goto done;
    }
    if (rc != YVEX_OK) {
        if (getenv("YVEX_TEST_FAIL_MATERIALIZE_AFTER_TRANSFER")) {
            print_materialize_refusal(&state.reference, backend_name,
                                      MATERIALIZE_REFUSAL_TRANSFER, NULL);
        } else if (getenv("YVEX_TEST_FAIL_MATERIALIZE_AFTER_ALLOC")) {
            print_materialize_refusal(&state.reference, backend_name,
                                      MATERIALIZE_REFUSAL_ALLOCATION, NULL);
        } else {
            print_materialize_refusal(&state.reference, backend_name,
                                      MATERIALIZE_REFUSAL_PREFLIGHT, NULL);
        }
        rc = print_yvex_error(&err, exit_for_status(rc));
        goto done;
    }
    rc = yvex_weight_table_get_summary(state.weights, &summary, &err);
    if (rc != YVEX_OK) {
        rc = print_yvex_error(&err, exit_for_status(rc));
        goto done;
    }
    print_materialization_summary(&state.context, &state.reference, backend_name, &summary);
    rc = 0;
done:
    materialize_cli_close(&state);
    return rc;
}

static yvex_materialize_scope parse_materialize_scope_name(const char *name) {
    if (!name)
        return YVEX_MATERIALIZE_SCOPE_UNKNOWN;
    if (strcmp(name, "selected-tensor") == 0)
        return YVEX_MATERIALIZE_SCOPE_SELECTED_TENSOR;
    if (strcmp(name, "partial-model") == 0)
        return YVEX_MATERIALIZE_SCOPE_PARTIAL_MODEL;
    if (strcmp(name, "full-model") == 0)
        return YVEX_MATERIALIZE_SCOPE_FULL_MODEL;
    return YVEX_MATERIALIZE_SCOPE_UNKNOWN;
}

typedef enum gate_cli_kind { GATE_CLI_MODEL, GATE_CLI_MATERIALIZE } gate_cli_kind;

typedef struct gate_cli_request {
    const char *model;
    const char *label;
    const char *family;
    const char *sha256;
    const char *tensor_name;
    const char *tensor_dtype;
    const char *report_out;
    unsigned long long tensor_dims[4];
    unsigned long long tensor_bytes;
    unsigned int tensor_rank;
    unsigned int tensor_seen;
    unsigned int repeat_count;
    yvex_materialize_scope scope;
    int require_cpu;
    int require_cuda;
    int check_cpu;
    int check_cuda;
    int check_cleanup;
    int json;
} gate_cli_request;

enum {
    GATE_TENSOR_NAME = 1u << 0u,
    GATE_TENSOR_RANK = 1u << 1u,
    GATE_TENSOR_DIMS = 1u << 2u,
    GATE_TENSOR_DTYPE = 1u << 3u,
    GATE_TENSOR_BYTES = 1u << 4u,
    GATE_TENSOR_COMPLETE = (1u << 5u) - 1u
};

static int gate_cli_parse(int arg_count, char **args, gate_cli_kind kind,
                          gate_cli_request *request) {
    const char *surface = kind == GATE_CLI_MATERIALIZE ? "materialize-gate" : "model-gate";
    int i;
    memset(request, 0, sizeof(*request));
    request->scope = YVEX_MATERIALIZE_SCOPE_UNKNOWN;
    request->repeat_count = 1u;
    for (i = 3; i < arg_count; ++i) {
        const char *option = args[i];
        const char *value;
        unsigned long long parsed;
        if (strcmp(option, "--require-cpu") == 0) {
            request->require_cpu = request->check_cpu = 1;
            continue;
        }
        if (strcmp(option, "--require-cuda") == 0) {
            request->require_cuda = request->check_cuda = 1;
            continue;
        }
        if (kind == GATE_CLI_MATERIALIZE && strcmp(option, "--check-cleanup") == 0) {
            request->check_cleanup = 1;
            continue;
        }
        if (kind == GATE_CLI_MATERIALIZE && strcmp(option, "--json") == 0) {
            request->json = 1;
            continue;
        }
        if (i + 1 >= arg_count) {
            yvex_cli_out_writef(stderr, "yvex: %s option requires a value: %s\n", surface, option);
            return 2;
        }
        value = args[++i];
        if (strcmp(option, "--model") == 0)
            request->model = value;
        else if (strcmp(option, "--label") == 0)
            request->label = value;
        else if (strcmp(option, "--family") == 0)
            request->family = value;
        else if (strcmp(option, "--sha256") == 0)
            request->sha256 = value;
        else if (strcmp(option, "--expect-tensor") == 0) {
            request->tensor_name = value;
            request->tensor_seen |= GATE_TENSOR_NAME;
        } else if (strcmp(option, "--expect-rank") == 0) {
            if (!parse_positive_ull(value, &parsed) || parsed > 4ull) {
                yvex_cli_out_writef(stderr, "yvex: invalid --expect-rank\n");
                return 2;
            }
            request->tensor_rank = (unsigned int)parsed;
            request->tensor_seen |= GATE_TENSOR_RANK;
        } else if (strcmp(option, "--expect-dims") == 0) {
            if (!(request->tensor_seen & GATE_TENSOR_RANK) ||
                !parse_dims_csv(value, request->tensor_rank, request->tensor_dims)) {
                yvex_cli_out_writef(
                    stderr,
                    "yvex: invalid --expect-dims; pass --expect-rank before --expect-dims\n");
                return 2;
            }
            request->tensor_seen |= GATE_TENSOR_DIMS;
        } else if (strcmp(option, "--expect-dtype") == 0) {
            request->tensor_dtype = value;
            request->tensor_seen |= GATE_TENSOR_DTYPE;
        } else if (strcmp(option, "--expect-bytes") == 0) {
            if (!parse_positive_ull(value, &request->tensor_bytes)) {
                yvex_cli_out_writef(stderr, "yvex: invalid --expect-bytes\n");
                return 2;
            }
            request->tensor_seen |= GATE_TENSOR_BYTES;
        } else if (strcmp(option, "--backend") == 0) {
            if (strcmp(value, "cpu") == 0)
                request->check_cpu = 1;
            else if (strcmp(value, "cuda") == 0)
                request->check_cuda = 1;
            else {
                yvex_cli_out_writef(stderr, "yvex: %s backend must be cpu or cuda\n", surface);
                return 2;
            }
        } else if (strcmp(option, "--report-out") == 0) {
            request->report_out = value;
        } else if (kind == GATE_CLI_MATERIALIZE && strcmp(option, "--scope") == 0) {
            request->scope = parse_materialize_scope_name(value);
            if (request->scope == YVEX_MATERIALIZE_SCOPE_UNKNOWN) {
                yvex_cli_out_writef(stderr, "yvex: invalid materialize-gate scope\n");
                return 2;
            }
        } else if (kind == GATE_CLI_MATERIALIZE && strcmp(option, "--repeat") == 0) {
            if (!parse_positive_ull(value, &parsed) || parsed > 1000ull) {
                yvex_cli_out_writef(stderr, "yvex: invalid --repeat\n");
                return 2;
            }
            request->repeat_count = (unsigned int)parsed;
        } else {
            yvex_cli_out_writef(stderr, "yvex: unknown %s option: %s\n", surface, option);
            yvex_cli_out_writef(stderr, "Try 'yvex help %s' for usage.\n", surface);
            return 2;
        }
    }
    return 0;
}

static int gate_ref_open(const gate_cli_request *request, gate_cli_kind kind, yvex_model_ref *ref,
                         const char **path, const char **sha256, const char **metadata_status,
                         yvex_error *err) {
    const char *surface = kind == GATE_CLI_MATERIALIZE ? "materialize-gate" : "model-gate";
    int rc = yvex_model_ref_resolve(ref, request->model, NULL, err);
    if (rc != YVEX_OK)
        return print_yvex_error(err, exit_for_status(rc));
    rc = enforce_registered_identity_cli(ref, surface);
    if (rc != YVEX_OK) {
        yvex_model_ref_clear(ref);
        return exit_for_status(rc);
    }
    *path = ref->path;
    *sha256 = request->sha256;
    if ((!*sha256 || !(*sha256)[0]) && ref->kind == YVEX_MODEL_REF_ALIAS && ref->sha256 &&
        ref->sha256[0])
        *sha256 = ref->sha256;
    if (metadata_status)
        *metadata_status = ref->kind == YVEX_MODEL_REF_ALIAS ? "pass" : "unregistered";
    return 0;
}

static int gate_report_file(const char *path, gate_cli_kind kind, const void *options,
                            const void *summary, const char *reason);

static void print_expected_tensor(FILE *fp, const char *name, unsigned int rank,
                                  const unsigned long long dims[4], const char *dtype,
                                  unsigned long long bytes) {
    unsigned int axis;
    yvex_cli_out_writef(
        fp, "expected_tensor: %s\nexpected_rank: %u\nexpected_dims:", name ? name : "", rank);
    for (axis = 0; axis < rank; ++axis)
        yvex_cli_out_writef(fp, axis == 0 ? " %llu" : ",%llu", dims[axis]);
    yvex_cli_out_writef(fp, "\nexpected_dtype: %s\nexpected_bytes: %llu\n", dtype ? dtype : "",
                        bytes);
}

static void print_materialize_gate_report(FILE *fp, const yvex_materialize_gate_options *options,
                                          const yvex_materialize_gate_summary *summary,
                                          const char *reason) {
    yvex_cli_out_writef(fp, "materialize gate: check\n");
    render_object_fields(fp, summary, materialize_gate_heading_fields,
                         FIELD_COUNT(materialize_gate_heading_fields));
    yvex_cli_out_writef(fp, "scope: %s\n", yvex_materialize_scope_name(summary->scope));
    render_object_fields(fp, summary, materialize_gate_state_fields,
                         FIELD_COUNT(materialize_gate_state_fields));
    yvex_cli_out_writef(fp, "cpu: %s\n", yvex_materialize_backend_status_name(summary->cpu_status));
    yvex_cli_out_writef(fp, "cuda: %s\n",
                        yvex_materialize_backend_status_name(summary->cuda_status));
    yvex_cli_out_writef(fp, "repeat_count: %u\n", summary->repeat_count);
    yvex_cli_out_writef(fp, "cleanup_verified: %s\n", summary->cleanup_verified ? "yes" : "no");
    yvex_cli_out_writef(fp, "execution_ready: false\n");
    if (options && options->expected_tensor_count == 1 && options->expected_tensors) {
        const yvex_materialize_expected_tensor *t = &options->expected_tensors[0];
        print_expected_tensor(fp, t->name, t->rank, t->dims, t->dtype, t->bytes);
    }
    yvex_cli_out_writef(fp, "failure_class: %s\n",
                        yvex_materialize_failure_class_name(summary->failure_class));
    yvex_cli_out_writef(fp, "reason: %s\n",
                        reason && reason[0] ? reason : "materialization hardening gate");
    yvex_cli_out_writef(fp, "status: %s\n", yvex_materialize_gate_status_name(summary->status));
}

static int command_materialize_gate(int arg_count, char **args) {
    yvex_materialize_gate_options options;
    yvex_materialize_expected_tensor expected;
    yvex_materialize_gate_summary summary;
    gate_cli_request request;
    yvex_model_ref model_ref;
    yvex_error err;
    int report_rc;
    int rc;
    yvex_error_clear(&err);
    memset(&options, 0, sizeof(options));
    memset(&expected, 0, sizeof(expected));
    memset(&summary, 0, sizeof(summary));
    memset(&model_ref, 0, sizeof(model_ref));
    if (arg_count >= 3 && (strcmp(args[2], "--help") == 0 || strcmp(args[2], "-h") == 0)) {
        yvex_materialize_gate_help(stdout);
        return 0;
    }
    if (arg_count < 3 || strcmp(args[2], "check") != 0) {
        yvex_cli_out_writef(stderr, "yvex: materialize-gate requires subcommand check\n");
        return 2;
    }
    rc = gate_cli_parse(arg_count, args, GATE_CLI_MATERIALIZE, &request);
    if (rc != 0)
        return rc;
    options.model_path = request.model;
    options.label = request.label;
    options.family = request.family;
    options.sha256 = request.sha256;
    options.scope = request.scope;
    options.require_cpu = request.require_cpu;
    options.require_cuda = request.require_cuda;
    options.check_cpu = request.check_cpu;
    options.check_cuda = request.check_cuda;
    options.check_cleanup = request.check_cleanup;
    options.json = request.json;
    options.repeat_count = request.repeat_count;
    if (!request.model || !request.label || !request.family ||
        request.scope == YVEX_MATERIALIZE_SCOPE_UNKNOWN) {
        yvex_cli_out_writef(
            stderr, "yvex: materialize-gate check requires --model --label --family --scope\n");
        return 2;
    }
    if (request.tensor_seen != 0u) {
        if (request.tensor_seen != GATE_TENSOR_COMPLETE) {
            yvex_cli_out_writef(stderr,
                                "yvex: materialize-gate expected tensor spec must be complete\n");
            return 2;
        }
        expected.name = request.tensor_name;
        expected.rank = request.tensor_rank;
        memcpy(expected.dims, request.tensor_dims, sizeof(expected.dims));
        expected.dtype = request.tensor_dtype;
        expected.bytes = request.tensor_bytes;
        options.expected_tensors = &expected;
        options.expected_tensor_count = 1;
    }
    rc = gate_ref_open(&request, GATE_CLI_MATERIALIZE, &model_ref, &options.model_path,
                       &options.sha256, &options.metadata_status, &err);
    if (rc != 0)
        return rc;
    rc = yvex_materialize_gate_check(&options, &summary, &err);
    print_materialize_gate_report(stdout, &options, &summary,
                                  rc == YVEX_OK ? NULL : yvex_error_message(&err));
    report_rc = gate_report_file(request.report_out, GATE_CLI_MATERIALIZE, &options, &summary,
                                 rc == YVEX_OK ? NULL : yvex_error_message(&err));
    yvex_model_ref_clear(&model_ref);
    return report_rc != 0 ? report_rc : rc == YVEX_OK ? 0 : exit_for_status(rc);
}

static void print_model_gate_report(FILE *fp, const yvex_model_gate_options *options,
                                    const yvex_model_gate_summary *summary, const char *reason) {
    yvex_cli_out_writef(fp, "model gate: check\n");
    render_object_fields(fp, summary, model_gate_state_fields, FIELD_COUNT(model_gate_state_fields));
    yvex_cli_out_writef(fp, "cpu: %s\n", yvex_model_gate_backend_status_name(summary->cpu_status));
    yvex_cli_out_writef(fp, "cuda: %s\n",
                        yvex_model_gate_backend_status_name(summary->cuda_status));
    yvex_cli_out_writef(fp, "support_level: %s\n",
                        yvex_model_support_level_name(summary->support_level));
    yvex_cli_out_writef(fp, "execution_ready: false\n");
    if (options && options->expected_tensor_count == 1 && options->expected_tensors) {
        const yvex_model_gate_expected_tensor *t = &options->expected_tensors[0];
        print_expected_tensor(fp, t->name, t->rank, t->dims, t->dtype, t->bytes);
    }
    yvex_cli_out_writef(fp, "reason: %s\n",
                        reason && reason[0] ? reason : "selected tensor materialization gate");
    yvex_cli_out_writef(fp, "status: %s\n", yvex_model_gate_status_name(summary->status));
}

static int gate_report_file(const char *path, gate_cli_kind kind, const void *options,
                            const void *summary, const char *reason) {
    FILE *fp;
    if (!path)
        return 0;
    fp = fopen(path, "wb");
    if (!fp) {
        yvex_cli_out_writef(stderr, "yvex: cannot write report: %s\n", path);
        return 1;
    }
    if (kind == GATE_CLI_MATERIALIZE)
        print_materialize_gate_report(fp, options, summary, reason);
    else
        print_model_gate_report(fp, options, summary, reason);
    fclose(fp);
    return 0;
}

static int command_model_gate(int arg_count, char **args) {
    yvex_model_gate_options options;
    yvex_model_gate_expected_tensor expected;
    yvex_model_gate_summary summary;
    gate_cli_request request;
    yvex_model_ref model_ref;
    yvex_error err;
    int report_rc;
    int rc;
    yvex_error_clear(&err);
    memset(&options, 0, sizeof(options));
    memset(&expected, 0, sizeof(expected));
    memset(&summary, 0, sizeof(summary));
    memset(&model_ref, 0, sizeof(model_ref));
    if (arg_count >= 3 && (strcmp(args[2], "--help") == 0 || strcmp(args[2], "-h") == 0)) {
        yvex_model_gate_help(stdout);
        return 0;
    }
    if (arg_count < 3 || strcmp(args[2], "check") != 0) {
        yvex_cli_out_writef(stderr, "yvex: model-gate requires subcommand check\n");
        return 2;
    }
    rc = gate_cli_parse(arg_count, args, GATE_CLI_MODEL, &request);
    if (rc != 0)
        return rc;
    if (!request.model || !request.label || !request.family ||
        request.tensor_seen != GATE_TENSOR_COMPLETE) {
        yvex_cli_out_writef(stderr, "yvex: model-gate check requires --model --label --family and "
                                    "one complete --expect-* tensor spec\n");
        return 2;
    }
    options.model_path = request.model;
    options.model_label = request.label;
    options.family = request.family;
    options.artifact_sha256 = request.sha256;
    options.require_cpu = request.require_cpu;
    options.require_cuda = request.require_cuda;
    options.check_cpu = request.check_cpu;
    options.check_cuda = request.check_cuda;
    expected.name = request.tensor_name;
    expected.rank = request.tensor_rank;
    memcpy(expected.dims, request.tensor_dims, sizeof(expected.dims));
    expected.dtype = request.tensor_dtype;
    expected.bytes = request.tensor_bytes;
    options.expected_tensors = &expected;
    options.expected_tensor_count = 1;
    rc = gate_ref_open(&request, GATE_CLI_MODEL, &model_ref, &options.model_path,
                       &options.artifact_sha256, NULL, &err);
    if (rc != 0)
        return rc;
    rc = yvex_model_gate_check(&options, &summary, &err);
    print_model_gate_report(stdout, &options, &summary,
                            rc == YVEX_OK ? NULL : yvex_error_message(&err));
    report_rc = gate_report_file(request.report_out, GATE_CLI_MODEL, &options, &summary,
                                 rc == YVEX_OK ? NULL : yvex_error_message(&err));
    yvex_model_ref_clear(&model_ref);
    return report_rc != 0 ? report_rc : rc == YVEX_OK ? 0 : exit_for_status(rc);
}

typedef struct {
    yvex_artifact_integrity_options integrity_options;
    yvex_artifact_integrity_report integrity_report;
    yvex_model_metadata_snapshot current_metadata;
    yvex_model_registry_entry registered_metadata;
    yvex_model_metadata_drift_report metadata_report;
    yvex_backend_options backend_options;
    yvex_model_ref ref;
    const char *model_arg;
    const char *backend_name;
    const char *identity_status;
    const char *metadata_status;
    const char *readiness_status;
    const char *support_level;
    const char *materialization_preflight;
    const char *materialization_gate;
    const char *materialization_backend;
    const char *backend_status;
    const char *report_status;
    const char *status;
    const char *model_input_kind;
    unsigned long long selected_hidden_size;
    unsigned long long selected_vocab_size;
    unsigned long long selected_output_count;
    unsigned long long selected_output_bytes;
    unsigned long long selected_slice_bytes;
    int audit_output;
    int metadata_checked;
    int selected_ready;
    int hard_fail;
} integrity_cli_state;

#define STATE_FIELD(key_, kind_, member_, fallback_)                                                \
    { key_, kind_, offsetof(integrity_cli_state, member_), fallback_ }
#define STATE_NESTED_FIELD(key_, kind_, outer_, type_, member_, fallback_)                          \
    { key_, kind_, offsetof(integrity_cli_state, outer_) + offsetof(type_, member_), fallback_ }
#define STATE_METADATA_FIELD(key_, kind_, member_, fallback_)                                       \
    {                                                                                               \
        key_, kind_,                                                                                \
            offsetof(integrity_cli_state, current_metadata) +                                      \
                offsetof(yvex_model_metadata_snapshot, entry) +                                    \
                offsetof(yvex_model_registry_entry, member_),                                      \
            fallback_                                                                              \
    }

static const yvex_render_field_spec integrity_audit_request_fields[] = {
    STATE_FIELD("model", YVEX_RENDER_FIELD_TEXT, model_arg, ""),
    STATE_NESTED_FIELD("resolved_path", YVEX_RENDER_FIELD_TEXT, ref, yvex_model_ref, path, ""),
    STATE_FIELD("model_input_kind", YVEX_RENDER_FIELD_TEXT, model_input_kind, ""),
};
static const yvex_render_field_spec integrity_audit_identity_fields[] = {
    STATE_NESTED_FIELD("architecture", YVEX_RENDER_FIELD_TEXT_ARRAY, integrity_report,
                       yvex_artifact_integrity_report, architecture, "unknown"),
    STATE_FIELD("identity_status", YVEX_RENDER_FIELD_TEXT, identity_status, "unknown"),
    STATE_NESTED_FIELD("digest_status", YVEX_RENDER_FIELD_TEXT_ARRAY, integrity_report,
                       yvex_artifact_integrity_report, digest_status, "unknown"),
    STATE_NESTED_FIELD("sha256", YVEX_RENDER_FIELD_TEXT_ARRAY, integrity_report,
                       yvex_artifact_integrity_report, sha256, "unavailable"),
    STATE_NESTED_FIELD("registered_sha256", YVEX_RENDER_FIELD_TEXT_ARRAY, integrity_report,
                       yvex_artifact_integrity_report, registered_sha256, "absent"),
};
static const yvex_render_field_spec integrity_audit_metadata_fields[] = {
    STATE_FIELD("metadata_status", YVEX_RENDER_FIELD_TEXT, metadata_status, "unknown"),
    STATE_FIELD("readiness_status", YVEX_RENDER_FIELD_TEXT, readiness_status, "unknown"),
    STATE_FIELD("support_level", YVEX_RENDER_FIELD_TEXT, support_level, "unknown"),
};
static const yvex_render_field_spec integrity_audit_primary_fields[] = {
    STATE_METADATA_FIELD("primary_tensor", YVEX_RENDER_FIELD_TEXT, primary_tensor_name, ""),
    STATE_METADATA_FIELD("primary_tensor_role", YVEX_RENDER_FIELD_TEXT, primary_tensor_role, ""),
    STATE_METADATA_FIELD("primary_tensor_dtype", YVEX_RENDER_FIELD_TEXT, primary_tensor_dtype, ""),
    STATE_METADATA_FIELD("primary_tensor_rank", YVEX_RENDER_FIELD_U32, primary_tensor_rank, NULL),
    STATE_METADATA_FIELD("primary_tensor_dims", YVEX_RENDER_FIELD_TEXT, primary_tensor_dims, ""),
    STATE_METADATA_FIELD("primary_tensor_bytes", YVEX_RENDER_FIELD_U64, primary_tensor_bytes, NULL),
};
static const yvex_render_field_spec integrity_audit_selection_fields[] = {
    STATE_FIELD("selected_embedding_ready", YVEX_RENDER_FIELD_BOOL, selected_ready, NULL),
    STATE_FIELD("selected_embedding_hidden_size", YVEX_RENDER_FIELD_U64, selected_hidden_size,
                NULL),
    STATE_FIELD("selected_embedding_vocab_size", YVEX_RENDER_FIELD_U64, selected_vocab_size,
                NULL),
    STATE_FIELD("selected_embedding_output_count", YVEX_RENDER_FIELD_U64, selected_output_count,
                NULL),
    STATE_FIELD("selected_embedding_output_bytes", YVEX_RENDER_FIELD_U64, selected_output_bytes,
                NULL),
    STATE_FIELD("selected_embedding_slice_bytes", YVEX_RENDER_FIELD_U64, selected_slice_bytes,
                NULL),
};
static const yvex_render_field_spec integrity_audit_backend_fields[] = {
    STATE_FIELD("backend_status", YVEX_RENDER_FIELD_TEXT, backend_status, "unknown"),
    STATE_FIELD("materialization_preflight", YVEX_RENDER_FIELD_TEXT, materialization_preflight,
                "unknown"),
    STATE_FIELD("materialization_backend", YVEX_RENDER_FIELD_TEXT, materialization_backend,
                "unknown"),
    STATE_FIELD("materialization_gate", YVEX_RENDER_FIELD_TEXT, materialization_gate, "unknown"),
    STATE_NESTED_FIELD("allocation_required_bytes", YVEX_RENDER_FIELD_U64, integrity_report,
                       yvex_artifact_integrity_report, known_tensor_bytes, NULL),
};
#undef STATE_METADATA_FIELD
#undef STATE_NESTED_FIELD
#undef STATE_FIELD

static void integrity_cli_state_init(integrity_cli_state *state) {
    memset(state, 0, sizeof(*state));
    state->identity_status = "unregistered";
    state->metadata_status = "unregistered";
    state->readiness_status = "not-checked";
    state->support_level = "not-checked";
    state->materialization_preflight = "not-checked";
    state->materialization_gate = "not-checked";
    state->materialization_backend = "not-checked";
    state->backend_status = "not-checked";
    state->report_status = "pass";
    state->status = "integrity-report-pass";
}

static int integrity_cli_parse(int arg_count, char **args, integrity_cli_state *state) {
    integrity_cli_options parsed;
    int rc = integrity_options_parse(arg_count, args, 1, &parsed);
    if (rc != 0)
        return rc;
    state->integrity_options = parsed.integrity;
    state->model_arg = parsed.model;
    state->backend_name = parsed.backend;
    state->audit_output = parsed.audit;
    return 0;
}

static void integrity_cli_metadata(integrity_cli_state *state) {
    yvex_error metadata_err;
    state->model_input_kind = state->ref.kind == YVEX_MODEL_REF_ALIAS ? "alias" : "path";
    if (state->ref.kind == YVEX_MODEL_REF_ALIAS) {
        if (!state->ref.sha256 || !state->ref.sha256[0] ||
            !yvex_sha256_hex_is_valid(state->ref.sha256)) {
            state->identity_status = "missing";
            state->hard_fail = 1;
        } else if (strcmp(state->integrity_report.digest_status, "pass") == 0 &&
                   (state->ref.registered_file_size == 0ull ||
                    state->ref.registered_file_size == state->integrity_report.file_size)) {
            state->identity_status = "pass";
        } else {
            state->identity_status = "fail";
            state->hard_fail = 1;
        }
    }
    if (!state->integrity_report.passed)
        return;
    yvex_error_clear(&metadata_err);
    if (yvex_model_metadata_snapshot_read(&state->current_metadata, state->ref.path,
                                          &metadata_err) != YVEX_OK) {
        if (state->ref.kind == YVEX_MODEL_REF_ALIAS &&
            strcmp(state->identity_status, "pass") == 0) {
            state->metadata_status = "fail";
            state->readiness_status = "fail";
            state->hard_fail = 1;
        }
        yvex_error_clear(&metadata_err);
        return;
    }
    state->metadata_checked = 1;
    state->support_level = state->current_metadata.entry.support_level &&
                                   state->current_metadata.entry.support_level[0]
                               ? state->current_metadata.entry.support_level
                               : "not-checked";
    state->selected_ready = state->current_metadata.entry.selected_embedding_ready;
    state->selected_hidden_size = state->current_metadata.entry.selected_embedding_hidden_size;
    state->selected_vocab_size = state->current_metadata.entry.selected_embedding_vocab_size;
    state->selected_output_count = state->current_metadata.entry.selected_embedding_output_count;
    state->selected_slice_bytes = state->current_metadata.entry.selected_embedding_slice_bytes;
    state->selected_output_bytes = state->selected_output_count * sizeof(float);
    if (state->ref.kind == YVEX_MODEL_REF_ALIAS && strcmp(state->identity_status, "pass") == 0) {
        yvex_model_ref_registry_entry_view(&state->ref, &state->registered_metadata);
        if (yvex_model_registry_compare_metadata(
                &state->registered_metadata, &state->current_metadata.entry,
                &state->metadata_report, &metadata_err) == YVEX_OK) {
            state->metadata_status = state->metadata_report.metadata_status[0]
                                         ? state->metadata_report.metadata_status
                                         : "unknown";
            state->readiness_status = state->metadata_report.readiness_status[0]
                                          ? state->metadata_report.readiness_status
                                          : "unknown";
            state->support_level = state->ref.support_level && state->ref.support_level[0]
                                       ? state->ref.support_level
                                       : state->support_level;
            if (strcmp(state->metadata_status, "pass") != 0 ||
                strcmp(state->readiness_status, "pass") != 0)
                state->hard_fail = 1;
        } else {
            state->metadata_status = "fail";
            state->readiness_status = "fail";
            state->hard_fail = 1;
        }
    } else if (state->ref.kind == YVEX_MODEL_REF_ALIAS) {
        state->metadata_status = "not-checked";
        state->readiness_status = "not-checked";
    } else {
        state->metadata_status = "unregistered";
        state->readiness_status = state->selected_ready ? "pass" : "not-checked";
    }
}

static void integrity_cli_selected_shape(integrity_cli_state *state) {
    if (state->integrity_report.selected_embedding_shape[0]) {
        state->selected_ready =
            strcmp(state->integrity_report.selected_embedding_shape, "valid") == 0;
        state->selected_hidden_size = state->integrity_report.selected_embedding_hidden_size;
        state->selected_vocab_size = state->integrity_report.selected_embedding_vocab_size;
        state->selected_output_count = state->integrity_report.selected_embedding_output_count;
        state->selected_output_bytes = state->integrity_report.selected_embedding_output_bytes;
        state->selected_slice_bytes = state->integrity_report.selected_embedding_slice_bytes;
        state->readiness_status = state->selected_ready ? "pass" : "fail";
    } else if (state->integrity_options.require_token_embedding) {
        state->selected_ready = 0;
        state->readiness_status = "fail";
        state->hard_fail = 1;
    }
}

static void integrity_cli_backend(integrity_cli_state *state, yvex_error *err) {
    yvex_backend *backend = NULL;
    int rc;
    if (!state->backend_name)
        return;
    state->materialization_backend = state->backend_name;
    if (!state->hard_fail) {
        state->backend_options.kind = strcmp(state->backend_name, "cuda") == 0
                                          ? YVEX_BACKEND_KIND_CUDA
                                          : YVEX_BACKEND_KIND_CPU;
        rc = yvex_backend_open(&backend, &state->backend_options, err);
        if (rc == YVEX_OK) {
            state->backend_status = yvex_backend_status_name(yvex_backend_status_of(backend));
            if (yvex_backend_supports(backend, YVEX_BACKEND_CAP_TENSOR_ALLOC) &&
                yvex_backend_supports(backend, YVEX_BACKEND_CAP_TENSOR_READ_WRITE)) {
                state->materialization_preflight = "pass";
                state->materialization_gate = "pass";
            } else {
                state->materialization_preflight = "fail";
                state->materialization_gate = "fail";
                state->hard_fail = 1;
            }
            yvex_backend_close(backend);
        } else {
            state->backend_status = rc == YVEX_ERR_UNSUPPORTED ? "unavailable" : "fail";
            state->materialization_preflight = "fail";
            state->materialization_gate = "fail";
            state->hard_fail = 1;
            yvex_error_clear(err);
        }
    } else {
        state->backend_status = "not-opened";
        state->materialization_preflight = "fail";
        state->materialization_gate = "fail";
    }
}

static int integrity_cli_render_normal(integrity_cli_state *state) {
    const char *top_blocker = "none";
    if (state->hard_fail) {
        if (state->integrity_report.error_count > 0u && state->integrity_report.issues[0].code[0]) {
            top_blocker = state->integrity_report.issues[0].code;
        } else if (strcmp(state->identity_status, "pass") != 0) {
            top_blocker = "identity";
        } else if (strcmp(state->metadata_status, "pass") != 0 &&
                   strcmp(state->metadata_status, "unregistered") != 0) {
            top_blocker = "metadata";
        } else if (strcmp(state->materialization_preflight, "pass") != 0 &&
                   strcmp(state->materialization_preflight, "not-checked") != 0) {
            top_blocker = "materialization-preflight";
        } else {
            top_blocker = "integrity";
        }
    }
    yvex_cli_out_writef(stdout, "integrity: %s model=%s backend=%s\n",
                        state->hard_fail ? "fail" : "pass", state->model_arg,
                        state->backend_name ? state->backend_name : "not-requested");
    yvex_cli_out_writef(
        stdout, "identity: %s digest: %s tensors=%llu invalid_ranges=%llu\n",
        state->identity_status,
        state->integrity_report.digest_status[0] ? state->integrity_report.digest_status
                                                 : "unknown",
        state->integrity_report.tensor_count, state->integrity_report.tensor_ranges_invalid);
    yvex_cli_out_writef(stdout, "materialization_preflight: %s\n",
                        state->materialization_preflight);
    yvex_cli_out_writef(stdout, "top_blocker: %s\n", top_blocker);
    yvex_cli_out_writef(stdout, "boundary: integrity gate only, generation unsupported\n");
    yvex_cli_out_writef(stdout, "status: %s\n", state->status);
    return state->hard_fail ? exit_for_status(YVEX_ERR_STATE) : 0;
}

static int integrity_cli_render_audit(integrity_cli_state *state) {
    const yvex_artifact_integrity_report *report = &state->integrity_report;
    int i;
    yvex_cli_out_writef(stdout, "artifact_integrity_report: summary\n");
    render_object_fields(stdout, state, integrity_audit_request_fields,
                         FIELD_COUNT(integrity_audit_request_fields));
    yvex_cli_out_writef(stdout, "format: %s\n", report->format[0] ? report->format : "unknown");
    if (report->version)
        yvex_cli_out_writef(stdout, "version: %u\n", report->version);
    render_object_fields(stdout, state, integrity_audit_identity_fields,
                         FIELD_COUNT(integrity_audit_identity_fields));
    if (report->expected_sha256[0]) {
        yvex_cli_out_writef(stdout, "expected_sha256: %s\nactual_sha256: %s\n",
                            report->expected_sha256,
                            report->sha256[0] ? report->sha256 : "unavailable");
    }
    render_object_fields(stdout, state, integrity_audit_metadata_fields,
                         FIELD_COUNT(integrity_audit_metadata_fields));
    yvex_cli_out_writef(stdout, "integrity_status: %s\n", report->passed ? "pass" : "fail");
    render_object_fields(stdout, report, integrity_issue_count_fields,
                         FIELD_COUNT(integrity_issue_count_fields));
    render_object_fields(stdout, report, integrity_audit_counter_fields,
                         FIELD_COUNT(integrity_audit_counter_fields));
    render_object_fields(stdout, state, integrity_audit_primary_fields,
                         FIELD_COUNT(integrity_audit_primary_fields));
    render_object_fields(stdout, state, integrity_audit_selection_fields,
                         FIELD_COUNT(integrity_audit_selection_fields));
    render_object_fields(stdout, state, integrity_audit_backend_fields,
                         FIELD_COUNT(integrity_audit_backend_fields));
    if (state->metadata_checked && state->ref.kind == YVEX_MODEL_REF_ALIAS) {
        for (i = 0; i < (int)state->metadata_report.issue_count; ++i) {
            yvex_cli_out_writef(stdout, "metadata_issue_%u_code: %s\n", (unsigned int)i,
                                state->metadata_report.issues[i].code);
            yvex_cli_out_writef(stdout, "metadata_issue_%u_registered: %s\n", (unsigned int)i,
                                state->metadata_report.issues[i].registered_value);
            yvex_cli_out_writef(stdout, "metadata_issue_%u_current: %s\n", (unsigned int)i,
                                state->metadata_report.issues[i].current_value);
        }
    }
    yvex_cli_out_writef(stdout,
                        "execution_ready: false\nprefill_ready: false\nlogits_ready: "
                        "false\ngeneration: unsupported\n"
                        "report_status: %s\nstatus: %s\n",
                        state->report_status, state->status);
    return state->hard_fail ? exit_for_status(YVEX_ERR_STATE) : 0;
}

static int command_integrity_report(int arg_count, char **args) {
    integrity_cli_state state;
    yvex_error err;
    int rc;
    integrity_cli_state_init(&state);
    yvex_error_clear(&err);
    rc = integrity_cli_parse(arg_count, args, &state);
    if (rc != 0)
        return rc;
    rc = yvex_model_ref_resolve(&state.ref, state.model_arg, NULL, &err);
    if (rc != YVEX_OK) {
        return print_yvex_error(&err, exit_for_status(rc));
    }
    if (state.ref.kind == YVEX_MODEL_REF_ALIAS && state.ref.sha256 && state.ref.sha256[0]) {
        state.integrity_options.registered_sha256 = state.ref.sha256;
    }
    rc = yvex_artifact_integrity_check_path(state.ref.path, &state.integrity_options,
                                            &state.integrity_report, &err);
    (void)rc;
    if (!state.integrity_report.passed)
        state.hard_fail = 1;
    integrity_cli_metadata(&state);
    integrity_cli_selected_shape(&state);
    integrity_cli_backend(&state, &err);
    if (state.hard_fail) {
        state.report_status = "fail";
        state.status = "integrity-report-fail";
    }
    rc = state.audit_output ? integrity_cli_render_audit(&state)
                            : integrity_cli_render_normal(&state);
    yvex_model_ref_clear(&state.ref);
    return rc;
}

static int command_tensors(int arg_count, char **args) {
    artifact_view view;
    const yvex_gguf_header *header;
    yvex_artifact_integrity_options integrity_options;
    yvex_artifact_integrity_report integrity_report;
    yvex_error err;
    unsigned long long i;
    int rc;
    yvex_error_clear(&err);
    memset(&integrity_options, 0, sizeof(integrity_options));
    memset(&integrity_report, 0, sizeof(integrity_report));
    if (arg_count != 3 || strcmp(args[2], "--help") == 0 || strcmp(args[2], "-h") == 0) {
        if (arg_count == 3) {
            yvex_tensors_help(stdout);
            return 0;
        }
        yvex_cli_out_writef(stderr, "yvex: tensors requires exactly one FILE_OR_ALIAS\n");
        yvex_cli_out_writef(stderr, "usage: yvex inspect artifact tensors FILE_OR_ALIAS\n");
        return 2;
    }
    rc = artifact_view_open(&view, args[2], 1, 0, &err);
    if (rc == YVEX_OK)
        rc = yvex_artifact_integrity_validate(view.artifact, view.gguf, view.tensors,
                                              &integrity_options, &integrity_report, &err);
    if (rc != YVEX_OK) {
        artifact_view_close(&view);
        return print_yvex_error(&err, exit_for_status(rc));
    }
    header = yvex_gguf_header_view(view.gguf);
    yvex_cli_out_writef(stdout, "format: gguf\n");
    yvex_cli_out_writef(stdout, "version: %u\n", header->version);
    yvex_cli_out_writef(stdout, "tensor_count: %llu\n", yvex_gguf_tensor_count(view.gguf));
    yvex_cli_out_writef(stdout, "tensor_data_offset: %llu\n",
                        yvex_gguf_tensor_data_offset(view.gguf));
    yvex_cli_out_writef(stdout, "alignment: %u\n", yvex_gguf_alignment(view.gguf));
    yvex_cli_out_writef(stdout, "\n");
    for (i = 0; i < yvex_tensor_table_count(view.tensors); ++i) {
        const yvex_tensor_info *tensor = yvex_tensor_table_at(view.tensors, i);
        yvex_tensor_range range;
        int range_rc;
        memset(&range, 0, sizeof(range));
        range_rc = yvex_tensor_range_validate(view.artifact, view.gguf, tensor, &range, &err);
        yvex_cli_out_writef(stdout, "%llu %s role=%s rank=%u dims=", i, tensor->name,
                            yvex_tensor_role_name(tensor->role), tensor->rank);
        print_tensor_dims(tensor->dims, tensor->rank);
        yvex_cli_out_writef(stdout, " dtype=%s bytes=%llu offset=%llu absolute=%llu",
                            yvex_dtype_name(tensor->dtype), tensor->storage_bytes,
                            tensor->relative_offset, tensor->absolute_offset);
        if (range_rc == YVEX_OK) {
            yvex_cli_out_writef(stdout,
                                " range=%llu..%llu range_status=valid alignment_status=%s\n",
                                range.tensor_absolute_offset, range.tensor_end_offset,
                                range.aligned ? "valid" : "invalid");
        } else {
            yvex_cli_out_writef(stdout, " range_status=invalid alignment_status=unknown\n");
            yvex_error_clear(&err);
        }
    }
    artifact_view_close(&view);
    return 0;
}

int yvex_inspect_command(int arg_count, char **args) {
    return command_inspect(arg_count, args);
}

int yvex_integrity_command(int arg_count, char **args) {
    return command_integrity(arg_count, args);
}

int yvex_materialize_command(int arg_count, char **args) {
    return command_materialize(arg_count, args);
}

int yvex_materialize_gate_command(int arg_count, char **args) {
    return command_materialize_gate(arg_count, args);
}

int yvex_metadata_command(int arg_count, char **args) {
    return command_metadata(arg_count, args);
}

int yvex_model_gate_command(int arg_count, char **args) {
    return command_model_gate(arg_count, args);
}

int yvex_tensors_command(int arg_count, char **args) {
    return command_tensors(arg_count, args);
}

void yvex_inspect_help(FILE *fp) {
    yvex_cli_out_fputs(inspect_help_text, fp);
}

void yvex_integrity_help(FILE *fp) {
    yvex_cli_out_fputs(integrity_help_text, fp);
}

void yvex_materialize_help(FILE *fp) {
    yvex_cli_out_fputs(materialize_help_text, fp);
}

void yvex_materialize_gate_help(FILE *fp) {
    yvex_cli_out_fputs(materialize_gate_help_text, fp);
}

void yvex_metadata_help(FILE *fp) {
    yvex_cli_out_fputs(metadata_help_text, fp);
}

void yvex_model_gate_help(FILE *fp) {
    yvex_cli_out_fputs(model_gate_help_text, fp);
}

void yvex_tensors_help(FILE *fp) {
    yvex_cli_out_fputs(tensors_help_text, fp);
}
