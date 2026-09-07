/*
 * Project layout admission into typed integrity evidence.
 *
 * Global layout acceptance precedes every subordinate proof. Integrity acceptance does not promote
 * complete-model support.
 */

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yvex/artifact.h>
#include <yvex/model.h>
#include <yvex/qtype.h>
#include <yvex/internal/core.h>
#include <yvex/internal/artifact_storage.h>

static int integrity_refuse(yvex_error *err,
                            yvex_status status,
                            const char *where,
                            const char *message) {
    yvex_error_set(err, status, where, message);
    return status;
}

static int integrity_refusef(yvex_error *err,
                             yvex_status status,
                             const char *where,
                             const char *format,
                             ...) {
    char message[YVEX_ERROR_MESSAGE_CAP];
    va_list args;

    va_start(args, format);
    (void)vsnprintf(message, sizeof(message), format, args);
    va_end(args);
    message[sizeof(message) - 1u] = '\0';
    yvex_error_set(err, status, where, message);
    return status;
}

static int add_error(yvex_artifact_integrity_report *report,
                     const char *code,
                     const char *tensor,
                     const char *reason);

static int add_range_error(yvex_artifact_integrity_report *report,
                           const char *code,
                           const char *tensor,
                           const char *reason,
                           const yvex_tensor_range *range);

static const char *const parse_issue_codes[] = {
    [YVEX_GGUF_PARSE_INVALID_MAGIC] = "bad-magic",
    [YVEX_GGUF_PARSE_UNSUPPORTED_VERSION] = "unsupported-version",
    [YVEX_GGUF_PARSE_INVALID_COUNT] = "parser-resource-refused",
    [YVEX_GGUF_PARSE_RESOURCE_LIMIT] = "parser-resource-refused",
    [YVEX_GGUF_PARSE_MALFORMED_KEY] = "metadata-parse-failed",
    [YVEX_GGUF_PARSE_DUPLICATE_METADATA_KEY] = "metadata-parse-failed",
    [YVEX_GGUF_PARSE_UNSUPPORTED_METADATA_TYPE] = "metadata-parse-failed",
    [YVEX_GGUF_PARSE_MALFORMED_VALUE] = "metadata-parse-failed",
    [YVEX_GGUF_PARSE_MALFORMED_STRING] = "malformed-string",
    [YVEX_GGUF_PARSE_MALFORMED_ARRAY] = "metadata-parse-failed",
    [YVEX_GGUF_PARSE_INVALID_ALIGNMENT] = "tensor-alignment-invalid",
    [YVEX_GGUF_PARSE_MALFORMED_TENSOR_NAME] = "tensor-directory-parse-failed",
    [YVEX_GGUF_PARSE_DUPLICATE_TENSOR_NAME] = "duplicate-tensor-name",
    [YVEX_GGUF_PARSE_INVALID_RANK] = "rank-out-of-range",
    [YVEX_GGUF_PARSE_INVALID_DIMENSION] = "zero-dimension",
    [YVEX_GGUF_PARSE_REFUSED_QTYPE] = "unknown-dtype",
    [YVEX_GGUF_PARSE_OFFSET_OVERFLOW] = "tensor-absolute-offset-overflow",
    [YVEX_GGUF_PARSE_INCOMPLETE_DIRECTORY] = "tensor-range-out-of-file",
    [YVEX_GGUF_PARSE_ALLOCATION_FAILURE] = "parser-resource-refused",
    [YVEX_GGUF_PARSE_EMPTY_METADATA_KEY] = "metadata-parse-failed",
    [YVEX_GGUF_PARSE_EMPTY_TENSOR_NAME] = "empty-tensor-name",
    [YVEX_GGUF_PARSE_ELEMENT_COUNT_OVERFLOW] = "element-count-overflow",
    [YVEX_GGUF_PARSE_ROW_COUNT_OVERFLOW] = "element-count-overflow",
    [YVEX_GGUF_PARSE_ROW_BYTE_OVERFLOW] = "row-byte-overflow",
    [YVEX_GGUF_PARSE_TOTAL_BYTE_OVERFLOW] = "tensor-byte-count-overflow",
};

static int digest_requested(const yvex_artifact_integrity_options *options) {
    return options && ((options->expect_sha256 && options->expect_sha256[0]) ||
                       (options->registered_sha256 && options->registered_sha256[0]));
}

/* Append canonical artifact integrity fields to a deterministic identity stream. */
static void apply_identity_digest(const yvex_artifact_file_identity *identity,
                                  const yvex_artifact_integrity_options *options,
                                  yvex_artifact_integrity_report *report) {
    const char *expected = NULL;
    const char *registered = NULL;

    if (!identity || !report) {
        return;
    }

    report->identity_checked = 1;
    report->file_size = identity->file_size;
    yvex_core_text_copy(report->sha256, YVEX_INTEGRITY_SHA256_CAP, identity->sha256);
    yvex_core_text_copy(report->digest_status, YVEX_INTEGRITY_DIGEST_STATUS_CAP, "unregistered");

    if (options) {
        expected = options->expect_sha256;
        registered = options->registered_sha256;
    }
    if (registered && registered[0]) {
        yvex_core_text_copy(report->registered_sha256, YVEX_INTEGRITY_SHA256_CAP, registered);
    }

    if (expected && expected[0]) {
        yvex_core_text_copy(report->expected_sha256, YVEX_INTEGRITY_SHA256_CAP, expected);
        if (!yvex_sha256_hex_is_valid(expected)) {
            yvex_core_text_copy(report->digest_status, YVEX_INTEGRITY_DIGEST_STATUS_CAP, "fail");
            (void)add_error(report,
                            "digest-invalid",
                            "",
                            "expected SHA-256 must be 64 lowercase or uppercase hex characters");
        } else if (strcmp(identity->sha256, expected) == 0) {
            yvex_core_text_copy(report->digest_status, YVEX_INTEGRITY_DIGEST_STATUS_CAP, "pass");
        } else {
            yvex_core_text_copy(report->digest_status, YVEX_INTEGRITY_DIGEST_STATUS_CAP, "fail");
            (void)add_error(report,
                            "digest-mismatch",
                            "",
                            "expected SHA-256 does not match current artifact bytes");
        }
        return;
    }

    if (registered && registered[0]) {
        if (!yvex_sha256_hex_is_valid(registered)) {
            yvex_core_text_copy(report->digest_status, YVEX_INTEGRITY_DIGEST_STATUS_CAP, "missing");
            (void)add_error(
                report, "digest-missing", "", "registered alias lacks valid SHA-256 identity");
        } else if (strcmp(identity->sha256, registered) == 0) {
            yvex_core_text_copy(report->digest_status, YVEX_INTEGRITY_DIGEST_STATUS_CAP, "pass");
        } else {
            yvex_core_text_copy(report->digest_status, YVEX_INTEGRITY_DIGEST_STATUS_CAP, "fail");
            (void)add_error(report,
                            "digest-mismatch",
                            "",
                            "registered SHA-256 does not match current artifact bytes");
        }
    }
}

static int add_issue(yvex_artifact_integrity_report *report,
                     yvex_integrity_severity severity,
                     const char *code,
                     const char *tensor,
                     const char *reason) {
    yvex_integrity_issue *issue;

    if (!report) {
        return YVEX_ERR_INVALID_ARG;
    }

    if (severity == YVEX_INTEGRITY_SEVERITY_WARNING) {
        report->warning_count += 1u;
    } else {
        report->error_count += 1u;
    }

    if (report->issue_count >= YVEX_INTEGRITY_MAX_ISSUES) {
        return YVEX_OK;
    }

    issue = &report->issues[report->issue_count++];
    issue->severity = severity;
    yvex_core_text_copy(issue->code, YVEX_INTEGRITY_CODE_CAP, code);
    yvex_core_text_copy(issue->tensor, YVEX_INTEGRITY_TENSOR_CAP, tensor);
    yvex_core_text_copy(issue->reason, YVEX_INTEGRITY_REASON_CAP, reason);
    return YVEX_OK;
}

static int add_error(yvex_artifact_integrity_report *report,
                     const char *code,
                     const char *tensor,
                     const char *reason) {
    return add_issue(report, YVEX_INTEGRITY_SEVERITY_ERROR, code, tensor, reason);
}

static int add_range_error(yvex_artifact_integrity_report *report,
                           const char *code,
                           const char *tensor,
                           const char *reason,
                           const yvex_tensor_range *range) {
    unsigned int before;
    int rc;

    if (!report) {
        return YVEX_ERR_INVALID_ARG;
    }
    before = report->issue_count;
    rc = add_error(report, code, tensor, reason);
    if (rc == YVEX_OK && range && report->issue_count > before) {
        yvex_integrity_issue *issue = &report->issues[before];
        issue->relative_offset = range->tensor_relative_offset;
        issue->absolute_offset = range->tensor_absolute_offset;
        issue->tensor_bytes = range->tensor_bytes;
        issue->file_size = range->file_size;
        issue->has_range = 1;
    }
    return rc;
}

static int add_layout_error(yvex_artifact_integrity_report *report,
                            const yvex_gguf_layout_result *layout) {
    unsigned int before;
    int rc;
    if (!report || !layout)
        return YVEX_ERR_INVALID_ARG;
    before = report->issue_count;
    rc = add_error(
        report, yvex_gguf_layout_code_name(layout->code), layout->tensor_name, layout->reason);
    if (rc == YVEX_OK && report->issue_count > before) {
        yvex_integrity_issue *issue = &report->issues[before];
        issue->relative_offset = layout->declared_relative_offset;
        issue->absolute_offset = layout->failure_absolute_offset;
        issue->tensor_bytes = layout->tensor_raw_size;
        issue->file_size = layout->actual_file_size;
        issue->has_range = 1;
    }
    return rc;
}

static int apply_requested_digest(const yvex_artifact *artifact,
                                  const yvex_artifact_integrity_options *options,
                                  yvex_artifact_integrity_report *report,
                                  yvex_error *err) {
    yvex_artifact_file_identity identity;
    int rc;

    if (!digest_requested(options)) {
        yvex_core_text_copy(report->digest_status, YVEX_INTEGRITY_DIGEST_STATUS_CAP, "not-requested");
        return YVEX_OK;
    }
    if (options->expect_sha256 && options->expect_sha256[0] &&
        !yvex_sha256_hex_is_valid(options->expect_sha256)) {
        yvex_core_text_copy(report->expected_sha256, YVEX_INTEGRITY_SHA256_CAP, options->expect_sha256);
        yvex_core_text_copy(report->digest_status, YVEX_INTEGRITY_DIGEST_STATUS_CAP, "fail");
        return add_error(
            report, "digest-invalid", "", "expected SHA-256 must be 64 hex characters");
    }
    if ((!options->expect_sha256 || !options->expect_sha256[0]) && options->registered_sha256 &&
        options->registered_sha256[0] && !yvex_sha256_hex_is_valid(options->registered_sha256)) {
        yvex_core_text_copy(
            report->registered_sha256, YVEX_INTEGRITY_SHA256_CAP, options->registered_sha256);
        yvex_core_text_copy(report->digest_status, YVEX_INTEGRITY_DIGEST_STATUS_CAP, "missing");
        return add_error(
            report, "digest-missing", "", "registered alias lacks valid SHA-256 identity");
    }

    rc = yvex_artifact_identity_read_open(artifact, &identity, err);
    if (rc != YVEX_OK) {
        yvex_error drift_error;
        yvex_error_clear(&drift_error);
        yvex_core_text_copy(report->digest_status, YVEX_INTEGRITY_DIGEST_STATUS_CAP, "fail");
        if (yvex_artifact_snapshot_validate(artifact, NULL, &drift_error) != YVEX_OK) {
            return add_error(report,
                             "file-identity-drift",
                             "",
                             "artifact identity changed during digest validation");
        }
        return add_error(report, "digest-read-failed", "", yvex_error_message(err));
    }
    apply_identity_digest(&identity, options, report);
    if (!strcmp(report->digest_status, "pass")) {
        yvex_paths paths;
        yvex_artifact_reopen_lease lease;
        yvex_error cache_error;
        /* Byte verification is independent of model admission. Cache only the stable byte proof. */
        yvex_error_clear(&cache_error);
        if (yvex_paths_default(&paths, &cache_error) == YVEX_OK)
            (void)yvex_artifact_reopen_lease_publish(artifact, identity.sha256,
                                                    paths.cache_dir, &lease, &cache_error);
    }
    return YVEX_OK;
}

static int tensor_element_count(const yvex_tensor_info *tensor, unsigned long long *out) {
    unsigned long long product = 1ull;
    unsigned int i;

    if (!tensor || !out) {
        return 0;
    }
    if (tensor->rank == 0u || tensor->rank > YVEX_TENSOR_MAX_DIMS) {
        return 0;
    }
    for (i = 0; i < tensor->rank; ++i) {
        if (tensor->dims[i] == 0ull) {
            return 0;
        }
        if (!yvex_core_u64_mul(product, tensor->dims[i], &product)) {
            return 0;
        }
    }
    *out = product;
    return 1;
}

int yvex_tensor_shape_accounting_validate(const yvex_tensor_info *tensor,
                                          yvex_tensor_shape_accounting *out,
                                          yvex_error *err) {
    yvex_tensor_shape_accounting local;
    yvex_tensor_shape_accounting *accounting = out ? out : &local;
    const yvex_dtype_info *dtype_info;
    const yvex_gguf_qtype_geometry *geometry;
    unsigned long long element_count = 0ull;
    unsigned long long storage_byte_count = 0ull;
    unsigned int i;
    int rc;

    if (!tensor) {
        return integrity_refuse(err, YVEX_ERR_INVALID_ARG, "yvex_tensor_shape_accounting_validate",
            "tensor is required");
    }

    memset(accounting, 0, sizeof(*accounting));
    yvex_core_text_copy(accounting->tensor_name, YVEX_INTEGRITY_TENSOR_CAP, tensor->name);
    accounting->dtype = tensor->dtype;
    accounting->rank = tensor->rank;
    for (i = 0; i < tensor->rank && i < YVEX_TENSOR_MAX_DIMS; ++i) {
        accounting->dims[i] = tensor->dims[i];
    }

    if (!tensor->name || tensor->name[0] == '\0') {
        yvex_error_set(
            err, YVEX_ERR_FORMAT, "yvex_tensor_shape_accounting_validate", "tensor name is empty");
        return YVEX_ERR_FORMAT;
    }
    if (tensor->rank == 0u || tensor->rank > YVEX_TENSOR_MAX_DIMS) {
        return integrity_refusef(err, YVEX_ERR_FORMAT, "yvex_tensor_shape_accounting_validate",
            "tensor-rank-invalid: %s rank %u", tensor->name, tensor->rank);
    }
    for (i = 0; i < tensor->rank; ++i) {
        if (tensor->dims[i] == 0ull) {
            return integrity_refusef(err, YVEX_ERR_FORMAT, "yvex_tensor_shape_accounting_validate",
                "tensor-dim-zero: %s", tensor->name);
        }
    }
    if (!tensor_element_count(tensor, &element_count)) {
        return integrity_refusef(err, YVEX_ERR_BOUNDS, "yvex_tensor_shape_accounting_validate",
            "tensor-element-count-overflow: %s", tensor->name);
    }
    accounting->element_count = element_count;
    accounting->shape_valid = 1;

    dtype_info = yvex_dtype_get_info(tensor->dtype);
    if (!dtype_info || dtype_info->dtype == YVEX_DTYPE_UNKNOWN) {
        return integrity_refusef(err, YVEX_ERR_UNSUPPORTED, "yvex_tensor_shape_accounting_validate",
            "tensor-dtype-unknown: %s", tensor->name);
    }
    accounting->dtype_known = 1;
    accounting->dtype_valid = 1;
    geometry = yvex_gguf_qtype_geometry_find(dtype_info->ggml_type);
    accounting->storage_supported = yvex_dtype_storage_supported(tensor->dtype);
    if (geometry) {
        accounting->storage_unit_bytes = geometry->scalar_width > 0u
                                             ? (unsigned long long)geometry->scalar_width
                                             : (unsigned long long)geometry->bytes_per_block;
    }
    accounting->compute_supported_for_selected_embedding = tensor->dtype == YVEX_DTYPE_F16 ? 1 : 0;
    accounting->compute_supported_for_fixture_embedding = tensor->dtype == YVEX_DTYPE_F32 ? 1 : 0;

    rc = yvex_dtype_tensor_storage_bytes(
        tensor->dtype, tensor->dims, tensor->rank, &storage_byte_count, err);
    if (rc == YVEX_ERR_UNSUPPORTED) {
        yvex_error_setf(err,
                        YVEX_ERR_UNSUPPORTED,
                        "yvex_tensor_shape_accounting_validate",
                        "tensor-dtype-size-unknown: %s",
                        tensor->name);
        return rc;
    }
    if (rc != YVEX_OK)
        return rc;
    if (storage_byte_count == 0ull) {
        return integrity_refusef(err, YVEX_ERR_BOUNDS, "yvex_tensor_shape_accounting_validate",
            "tensor-byte-count-overflow: %s tensor byte count is zero", tensor->name);
    }
    if (tensor->storage_bytes != 0ull && tensor->storage_bytes != storage_byte_count) {
        return integrity_refusef(err, YVEX_ERR_BOUNDS, "yvex_tensor_shape_accounting_validate",
            "tensor-byte-count-overflow: %s tensor table byte count mismatch", tensor->name);
    }

    accounting->byte_count_computable = 1;
    accounting->storage_byte_count = storage_byte_count;
    accounting->byte_count_valid = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_selected_embedding_shape_validate(const yvex_tensor_info *tensor,
                                           unsigned int token_id,
                                           yvex_selected_embedding_shape *out,
                                           yvex_error *err) {
    yvex_selected_embedding_shape local;
    yvex_selected_embedding_shape *shape = out ? out : &local;
    yvex_tensor_shape_accounting accounting;
    unsigned long long hidden_size;
    unsigned long long vocab_size;
    unsigned long long output_bytes = 0ull;
    unsigned long long slice_bytes = 0ull;
    int rc;

    if (!tensor) {
        return integrity_refuse(err, YVEX_ERR_INVALID_ARG, "yvex_selected_embedding_shape_validate",
            "tensor is required");
    }

    memset(shape, 0, sizeof(*shape));
    yvex_core_text_copy(shape->tensor_name, YVEX_INTEGRITY_TENSOR_CAP, tensor->name);
    shape->token_id = token_id;

    memset(&accounting, 0, sizeof(accounting));
    rc = yvex_tensor_shape_accounting_validate(tensor, &accounting, err);
    if (rc != YVEX_OK) {
        return rc;
    }
    if (tensor->rank != 2u) {
        return integrity_refusef(err, YVEX_ERR_FORMAT, "yvex_selected_embedding_shape_validate",
            "selected-embedding-rank-invalid: %s rank %u", tensor->name, tensor->rank);
    }
    if (tensor->dtype != YVEX_DTYPE_F16) {
        return integrity_refusef(err, YVEX_ERR_UNSUPPORTED,
            "yvex_selected_embedding_shape_validate",
            "selected-embedding-dtype-invalid: %s requires F16", tensor->name);
    }

    hidden_size = tensor->dims[0];
    vocab_size = tensor->dims[1];
    shape->hidden_size = hidden_size;
    shape->vocab_size = vocab_size;
    if ((unsigned long long)token_id >= vocab_size) {
        return integrity_refusef(err, YVEX_ERR_BOUNDS, "yvex_selected_embedding_shape_validate",
            "token-out-of-range: %u >= %llu", token_id, vocab_size);
    }
    if (!yvex_core_u64_mul(hidden_size, (unsigned long long)sizeof(float), &output_bytes)) {
        return integrity_refusef(err, YVEX_ERR_BOUNDS, "yvex_selected_embedding_shape_validate",
            "selected-embedding-output-byte-count-overflow: %s", tensor->name);
    }
    if (!yvex_core_u64_mul(hidden_size, 2ull, &slice_bytes)) {
        return integrity_refusef(err, YVEX_ERR_BOUNDS, "yvex_selected_embedding_shape_validate",
            "selected-embedding-slice-byte-count-overflow: %s", tensor->name);
    }

    shape->output_count = hidden_size;
    shape->output_bytes = output_bytes;
    shape->slice_bytes = slice_bytes;
    shape->shape_valid = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_tensor_range_validate(const yvex_artifact *artifact,
                               const yvex_gguf *gguf,
                               const yvex_tensor_info *tensor,
                               yvex_tensor_range *out,
                               yvex_error *err) {
    yvex_tensor_range local;
    yvex_tensor_range *range = out ? out : &local;
    yvex_tensor_shape_accounting accounting;
    unsigned long long absolute_offset = 0ull;
    unsigned long long end_offset = 0ull;
    unsigned int i;
    int rc;

    if (!artifact || !gguf || !tensor) {
        return integrity_refuse(err, YVEX_ERR_INVALID_ARG, "yvex_tensor_range_validate",
            "artifact, gguf, and tensor are required");
    }

    memset(range, 0, sizeof(*range));
    yvex_core_text_copy(range->tensor_name, YVEX_INTEGRITY_TENSOR_CAP, tensor->name);
    range->dtype = tensor->dtype;
    range->rank = tensor->rank;
    for (i = 0; i < tensor->rank && i < YVEX_TENSOR_MAX_DIMS; ++i) {
        range->dims[i] = tensor->dims[i];
    }
    range->tensor_relative_offset = tensor->relative_offset;
    range->tensor_data_offset = yvex_gguf_tensor_data_offset(gguf);
    range->file_size = yvex_artifact_size(artifact);
    range->alignment = (unsigned long long)yvex_gguf_alignment(gguf);

    memset(&accounting, 0, sizeof(accounting));
    rc = yvex_tensor_shape_accounting_validate(tensor, &accounting, err);
    if (rc != YVEX_OK) {
        return rc;
    }
    range->element_count = accounting.element_count;
    range->dtype_size = accounting.storage_unit_bytes;
    range->tensor_bytes = accounting.storage_byte_count;
    if (range->alignment > 0ull && (tensor->relative_offset % range->alignment) != 0ull) {
        return integrity_refusef(err, YVEX_ERR_FORMAT, "yvex_tensor_range_validate",
            "tensor-alignment-invalid: %s relative offset %llu is not aligned to %llu",
            tensor->name, tensor->relative_offset, range->alignment);
    }
    range->aligned = 1;

    if (!yvex_core_u64_add(range->tensor_data_offset, tensor->relative_offset, &absolute_offset)) {
        return integrity_refusef(err, YVEX_ERR_BOUNDS, "yvex_tensor_range_validate",
            "tensor-absolute-offset-overflow: %s", tensor->name);
    }
    range->tensor_absolute_offset = absolute_offset;
    if (tensor->absolute_offset != 0ull && tensor->absolute_offset != absolute_offset) {
        return integrity_refusef(err, YVEX_ERR_BOUNDS, "yvex_tensor_range_validate",
            "tensor-absolute-offset-overflow: %s tensor table absolute offset mismatch", tensor->name);
    }
    if (absolute_offset < range->tensor_data_offset) {
        return integrity_refusef(err, YVEX_ERR_BOUNDS, "yvex_tensor_range_validate",
            "tensor-offset-before-data: %s", tensor->name);
    }
    if (absolute_offset > range->file_size) {
        return integrity_refusef(err, YVEX_ERR_BOUNDS, "yvex_tensor_range_validate",
            "tensor-offset-out-of-file: %s absolute offset %llu exceeds file size %llu",
            tensor->name, absolute_offset, range->file_size);
    }
    if (!yvex_core_u64_add(absolute_offset, range->tensor_bytes, &end_offset)) {
        return integrity_refusef(err, YVEX_ERR_BOUNDS, "yvex_tensor_range_validate",
            "tensor-end-offset-overflow: %s", tensor->name);
    }
    range->tensor_end_offset = end_offset;
    if (end_offset > range->file_size) {
        return integrity_refusef(err, YVEX_ERR_BOUNDS, "yvex_tensor_range_validate",
            "tensor-range-out-of-file: %s byte range %llu..%llu exceeds file size %llu",
            tensor->name, absolute_offset, end_offset, range->file_size);
    }

    range->range_valid = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_tensor_embedding_slice_range_validate(const yvex_tensor_range *range,
                                               unsigned int token_id,
                                               yvex_tensor_slice_range *out,
                                               yvex_error *err) {
    yvex_tensor_slice_range local;
    yvex_tensor_slice_range *slice = out ? out : &local;
    unsigned long long hidden_size;
    unsigned long long vocab_size;
    unsigned long long token_offset_bytes = 0ull;
    unsigned long long slice_bytes = 0ull;
    unsigned long long slice_absolute_offset = 0ull;
    unsigned long long slice_end_offset = 0ull;

    if (!range) {
        return integrity_refuse(err, YVEX_ERR_INVALID_ARG,
            "yvex_tensor_embedding_slice_range_validate", "range is required");
    }
    memset(slice, 0, sizeof(*slice));
    slice->token_id = token_id;

    if (!range->range_valid || range->rank != 2u) {
        return integrity_refuse(err, YVEX_ERR_FORMAT, "yvex_tensor_embedding_slice_range_validate",
            "selected embedding slice requires a valid rank-2 tensor range");
    }
    if (range->dtype != YVEX_DTYPE_F16) {
        return integrity_refusef(err, YVEX_ERR_UNSUPPORTED,
            "yvex_tensor_embedding_slice_range_validate",
            "selected-embedding-dtype-invalid: %s requires F16", range->tensor_name);
    }
    hidden_size = range->dims[0];
    vocab_size = range->dims[1];
    if (hidden_size == 0ull || vocab_size == 0ull) {
        return integrity_refuse(err, YVEX_ERR_FORMAT, "yvex_tensor_embedding_slice_range_validate",
            "selected embedding dimensions must be non-zero");
    }
    if ((unsigned long long)token_id >= vocab_size) {
        return integrity_refusef(err, YVEX_ERR_BOUNDS, "yvex_tensor_embedding_slice_range_validate",
            "token-out-of-range: %u >= %llu", token_id, vocab_size);
    }
    if (range->dtype_size == 0ull) {
        return integrity_refuse(err, YVEX_ERR_UNSUPPORTED,
            "yvex_tensor_embedding_slice_range_validate", "tensor-dtype-size-unknown");
    }
    if (!yvex_core_u64_mul(hidden_size, range->dtype_size, &slice_bytes)) {
        return integrity_refuse(err, YVEX_ERR_BOUNDS, "yvex_tensor_embedding_slice_range_validate",
            "token-slice-byte-count-overflow");
    }
    if (!yvex_core_u64_mul((unsigned long long)token_id, slice_bytes, &token_offset_bytes)) {
        return integrity_refuse(err, YVEX_ERR_BOUNDS, "yvex_tensor_embedding_slice_range_validate",
            "token-slice-offset-overflow");
    }
    if (!yvex_core_u64_add(
            range->tensor_absolute_offset, token_offset_bytes, &slice_absolute_offset)) {
        return integrity_refuse(err, YVEX_ERR_BOUNDS, "yvex_tensor_embedding_slice_range_validate",
            "token-slice-offset-overflow");
    }
    if (!yvex_core_u64_add(slice_absolute_offset, slice_bytes, &slice_end_offset)) {
        return integrity_refuse(err, YVEX_ERR_BOUNDS, "yvex_tensor_embedding_slice_range_validate",
            "token-slice-end-overflow");
    }
    if (slice_absolute_offset < range->tensor_absolute_offset ||
        slice_end_offset > range->tensor_end_offset) {
        return integrity_refuse(err, YVEX_ERR_BOUNDS, "yvex_tensor_embedding_slice_range_validate",
            "token-slice-range-out-of-tensor");
    }

    slice->slice_bytes = slice_bytes;
    slice->slice_relative_offset = token_offset_bytes;
    slice->slice_absolute_offset = slice_absolute_offset;
    slice->slice_end_offset = slice_end_offset;
    slice->range_valid = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}

static void set_basic_report(const yvex_artifact *artifact,
                             const yvex_gguf *gguf,
                             const yvex_tensor_table *tensors,
                             yvex_artifact_integrity_report *report) {
    const yvex_gguf_header *header;
    const yvex_gguf_value *arch_value;
    const char *arch_text = NULL;
    unsigned long long arch_len = 0ull;

    memset(report, 0, sizeof(*report));
    report->checked = 1;
    yvex_core_text_copy(report->format, YVEX_INTEGRITY_FORMAT_CAP, "gguf");
    yvex_core_text_copy(report->digest_status, YVEX_INTEGRITY_DIGEST_STATUS_CAP, "not-requested");

    if (artifact) {
        yvex_core_text_copy(report->path, YVEX_ARTIFACT_PATH_CAP, yvex_artifact_path(artifact));
        report->file_size = yvex_artifact_size(artifact);
    }
    if (gguf) {
        header = yvex_gguf_header_view(gguf);
        if (header) {
            report->version = header->version;
            report->tensor_count = header->tensor_count;
        }
        arch_value = yvex_gguf_metadata_find(gguf, "general.architecture");
        if (arch_value && yvex_gguf_value_as_string(arch_value, &arch_text, &arch_len) == YVEX_OK &&
            arch_text) {
            unsigned long long copy_len = arch_len;
            if (copy_len >= (unsigned long long)YVEX_INTEGRITY_ARCH_CAP) {
                copy_len = (unsigned long long)YVEX_INTEGRITY_ARCH_CAP - 1ull;
            }
            memcpy(report->architecture, arch_text, (size_t)copy_len);
            report->architecture[copy_len] = '\0';
        }
    }
    if (tensors) {
        report->tensor_count = yvex_tensor_table_count(tensors);
    }
}

static void map_parse_result_to_report(const yvex_gguf_parse_result *result,
                                       const yvex_error *source,
                                       yvex_artifact_integrity_report *report) {
    const char *message = yvex_error_message(source);
    const char *code = "tensor-directory-parse-failed";

    if (!result) {
        code = yvex_error_code(source) == YVEX_ERR_IO ? "file-open-failed" : code;
    } else if (result->code == YVEX_GGUF_PARSE_SHORT_READ) {
        code = result->section == YVEX_GGUF_PARSE_SECTION_CONTAINER ? "file-too-small"
               : result->section == YVEX_GGUF_PARSE_SECTION_METADATA
                   ? "metadata-parse-failed"
                   : code;
    } else {
        unsigned int parsed_code = (unsigned int)result->code;

        if (parsed_code < sizeof(parse_issue_codes) / sizeof(parse_issue_codes[0]) &&
            parse_issue_codes[parsed_code])
            code = parse_issue_codes[parsed_code];
    }

    (void)add_error(report, code, "", message);
}

static void set_integrity_error(yvex_error *err, const yvex_artifact_integrity_report *report) {
    const yvex_integrity_issue *issue = NULL;

    if (report && report->issue_count > 0u) {
        issue = &report->issues[0];
    }
    if (issue && issue->reason[0]) {
        yvex_error_setf(
            err, YVEX_ERR_FORMAT, "yvex_artifact_integrity", "%s: %s", issue->code, issue->reason);
    } else {
        yvex_error_set(err,
                       YVEX_ERR_FORMAT,
                       "yvex_artifact_integrity",
                       "artifact integrity validation failed");
    }
}

int yvex_artifact_integrity_validate(const yvex_artifact *artifact,
                                     const yvex_gguf *gguf,
                                     const yvex_tensor_table *tensors,
                                     const yvex_artifact_integrity_options *options,
                                     yvex_artifact_integrity_report *out,
                                     yvex_error *err) {
    yvex_artifact_integrity_report local;
    yvex_artifact_integrity_report *report = out ? out : &local;
    unsigned long long tensor_count;
    int layout_rc;

    if (!artifact || !gguf || !tensors) {
        return integrity_refuse(err, YVEX_ERR_INVALID_ARG, "yvex_artifact_integrity_validate",
            "artifact, gguf, and tensors are required");
    }

    set_basic_report(artifact, gguf, tensors, report);
    if (digest_requested(options)) {
        yvex_core_text_copy(report->digest_status, YVEX_INTEGRITY_DIGEST_STATUS_CAP, "not-checked");
    }

    report->layout_checked = 1;
    layout_rc = yvex_gguf_layout_validate(artifact, gguf, &report->layout, err);
    report->tensor_ranges_checked = report->layout.tensor_count;
    report->tensor_ranges_valid = report->layout.tensors_validated;
    if (layout_rc != YVEX_OK) {
        report->tensor_ranges_invalid = 1ull;
        (void)add_layout_error(report, &report->layout);
        report->passed = 0;
        set_integrity_error(err, report);
        return YVEX_ERR_FORMAT;
    }

    tensor_count = yvex_tensor_table_count(tensors);
    if (tensor_count != report->layout.tensor_count) {
        (void)add_error(report,
                        "tensor-table-count-mismatch",
                        "",
                        "tensor table count differs from the canonical parsed layout");
    }
    report->known_tensor_bytes = report->layout.raw_tensor_bytes;
    report->tensor_shapes_checked = tensor_count;
    report->tensor_shapes_valid = tensor_count;
    report->tensor_dtypes_checked = tensor_count;
    report->tensor_dtypes_valid = tensor_count;
    report->tensor_byte_counts_checked = tensor_count;

    if (options && options->require_token_embedding) {
        const yvex_tensor_info *tensor = yvex_tensor_table_find(tensors, "token_embd.weight");

        if (!tensor) {
            (void)add_error(report,
                            "required-tensor-missing",
                            "token_embd.weight",
                            "required tensor not found: token_embd.weight");
        } else {
            yvex_selected_embedding_shape selected_shape;
            int shape_rc = YVEX_OK;

            memset(&selected_shape, 0, sizeof(selected_shape));
            if (tensor->rank != 2u) {
                shape_rc = YVEX_ERR_FORMAT;
                (void)add_error(report,
                                "required-tensor-rank-invalid",
                                tensor->name,
                                "selected embedding requires a rank-2 tensor");
            } else if (tensor->dtype != YVEX_DTYPE_F16) {
                shape_rc = YVEX_ERR_UNSUPPORTED;
                (void)add_error(report,
                                "required-tensor-dtype-invalid",
                                tensor->name,
                                "selected embedding requires F16 storage");
            } else if ((unsigned long long)options->token_id >= tensor->dims[1]) {
                shape_rc = YVEX_ERR_BOUNDS;
                (void)add_error(report,
                                "token-out-of-range",
                                tensor->name,
                                "partial token id is outside embedding range");
            } else {
                shape_rc = yvex_selected_embedding_shape_validate(
                    tensor, options->token_id, &selected_shape, err);
                if (shape_rc != YVEX_OK) {
                    (void)add_error(report,
                                    "selected-embedding-shape-invalid",
                                    tensor->name,
                                    yvex_error_message(err));
                    yvex_error_clear(err);
                }
            }
            if (shape_rc == YVEX_OK) {
                yvex_tensor_range range;
                yvex_tensor_slice_range slice;
                int rc;

                yvex_core_text_copy(
                    report->selected_embedding_shape, YVEX_INTEGRITY_DIGEST_STATUS_CAP, "valid");
                report->selected_embedding_hidden_size = selected_shape.hidden_size;
                report->selected_embedding_vocab_size = selected_shape.vocab_size;
                report->selected_embedding_output_count = selected_shape.output_count;
                report->selected_embedding_output_bytes = selected_shape.output_bytes;
                report->selected_embedding_slice_bytes = selected_shape.slice_bytes;

                memset(&range, 0, sizeof(range));
                memset(&slice, 0, sizeof(slice));
                rc = yvex_tensor_range_validate(artifact, gguf, tensor, &range, err);
                if (rc == YVEX_OK) {
                    rc = yvex_tensor_embedding_slice_range_validate(
                        &range, options->token_id, &slice, err);
                }
                if (rc != YVEX_OK) {
                    (void)add_range_error(report,
                                          "selected-embedding-range-invalid",
                                          tensor->name,
                                          yvex_error_message(err),
                                          &range);
                    yvex_error_clear(err);
                }
            }
        }
    }

    (void)apply_requested_digest(artifact, options, report, err);

    report->passed = report->error_count == 0u ? 1 : 0;
    if (!report->passed) {
        set_integrity_error(err, report);
        return YVEX_ERR_FORMAT;
    }

    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_artifact_integrity_check_path(const char *path,
                                       const yvex_artifact_integrity_options *options,
                                       yvex_artifact_integrity_report *out,
                                       yvex_error *err) {
    yvex_artifact_options artifact_options;
    yvex_artifact *artifact = NULL;
    yvex_gguf *gguf = NULL;
    yvex_tensor_table *tensors = NULL;
    yvex_gguf_parse_result parse_result;
    yvex_artifact_integrity_report local;
    yvex_artifact_integrity_report *report = out ? out : &local;
    int rc;

    memset(&artifact_options, 0, sizeof(artifact_options));
    memset(report, 0, sizeof(*report));
    report->checked = 1;
    yvex_core_text_copy(report->path, YVEX_ARTIFACT_PATH_CAP, path);
    yvex_core_text_copy(report->format, YVEX_INTEGRITY_FORMAT_CAP, "unknown");
    yvex_core_text_copy(report->digest_status,
                   YVEX_INTEGRITY_DIGEST_STATUS_CAP,
                   digest_requested(options) ? "not-checked" : "not-requested");

    if (!path || path[0] == '\0') {
        (void)add_error(report, "file-open-failed", "", "model path is required");
        set_integrity_error(err, report);
        return YVEX_ERR_INVALID_ARG;
    }

    artifact_options.path = path;
    artifact_options.readonly = 1;
    rc = yvex_artifact_open(&artifact, &artifact_options, err);
    if (rc != YVEX_OK) {
        map_parse_result_to_report(NULL, err, report);
        set_integrity_error(err, report);
        return rc;
    }
    report->file_size = yvex_artifact_size(artifact);

    rc = yvex_gguf_open_ex(&gguf, artifact, NULL, &parse_result, err);
    if (rc == YVEX_OK) {
        rc = yvex_tensor_table_from_gguf(&tensors, gguf, err);
    }
    if (rc != YVEX_OK) {
        if (gguf) {
            set_basic_report(artifact, gguf, tensors, report);
        }
        map_parse_result_to_report(&parse_result, err, report);
        yvex_core_text_copy(report->digest_status,
                       YVEX_INTEGRITY_DIGEST_STATUS_CAP,
                       digest_requested(options) ? "not-checked" : "not-requested");
        yvex_tensor_table_close(tensors);
        yvex_gguf_close(gguf);
        yvex_artifact_close(artifact);
        set_integrity_error(err, report);
        return rc;
    }

    rc = yvex_artifact_integrity_validate(artifact, gguf, tensors, options, report, err);
    report->passed = report->error_count == 0u ? 1 : 0;
    yvex_tensor_table_close(tensors);
    yvex_gguf_close(gguf);
    yvex_artifact_close(artifact);
    if (!report->passed) {
        set_integrity_error(err, report);
        return rc != YVEX_OK ? rc : YVEX_ERR_FORMAT;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

const yvex_integrity_issue *
yvex_artifact_integrity_issue_at(const yvex_artifact_integrity_report *report, unsigned int index) {
    if (!report || index >= report->issue_count) {
        return NULL;
    }
    return &report->issues[index];
}
