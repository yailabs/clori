/*
 * Admit one canonical GGUF byte layout through checked interval and padding validation.
 *
 * Validation is one linear tensor pass; it reads only directory and tensor padding and reports
 * zero tensor payload bytes read. Accepted layout is structural container admission only.
 */
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <yvex/artifact.h>
#include <yvex/gguf.h>
#include <yvex/internal/core.h>
#include <yvex/qtype.h>

#define YVEX_GGUF_LAYOUT_PADDING_CHUNK 4096u

static const char *const layout_code_names[] = {
    "ok", "invalid-argument", "invalid-alignment-type", "invalid-alignment-value",
    "alignment-not-power-of-two", "alignment-overflow", "tensor-storage-refused",
    "first-offset-not-zero", "tensor-offset-reversed", "unexpected-layout-gap",
    "duplicate-tensor-start", "complete-tensor-overlap", "partial-tensor-overlap",
    "padded-tensor-overlap", "tensor-raw-end-overflow", "tensor-padded-end-overflow",
    "layout-aggregate-overflow", "tensor-payload-truncated", "layout-padding-truncated",
    "directory-padding-nonzero", "tensor-padding-nonzero", "noncanonical-trailing-bytes",
    "file-identity-drift", "layout-io-failure", "parsed-view-file-mismatch",
    "tensor-info-missing", "tensor-range-mismatch",
};

static void layout_result_init(yvex_gguf_layout_result *result) {
    memset(result, 0, sizeof(*result));
    result->code = YVEX_GGUF_LAYOUT_INVALID_ARGUMENT;
    result->tensor_index = ULLONG_MAX;
    yvex_core_text_copy(result->reason, sizeof(result->reason), "layout not evaluated");
}

/*
 * Add one layout counter without unsigned overflow.
 *
 * Null output or overflow returns a typed layout code. This arithmetic helper owns no file or
 * layout state.
 */
yvex_gguf_layout_code yvex_gguf_layout_sum_checked(unsigned long long current,
                                                   unsigned long long addition,
                                                   unsigned long long *out) {
    if (!out)
        return YVEX_GGUF_LAYOUT_INVALID_ARGUMENT;
    if (addition > ULLONG_MAX - current)
        return YVEX_GGUF_LAYOUT_AGGREGATE_OVERFLOW;
    *out = current + addition;
    return YVEX_GGUF_LAYOUT_OK;
}

static int align_up_power_of_two(unsigned long long value, unsigned int alignment,
                                 unsigned long long *out) {
    unsigned long long mask;
    unsigned long long sum;
    if (!out || alignment == 0u || (alignment & (alignment - 1u)) != 0u)
        return 0;
    mask = (unsigned long long)alignment - 1ull;
    if (yvex_gguf_layout_sum_checked(value, mask, &sum) != YVEX_GGUF_LAYOUT_OK)
        return 0;
    *out = sum & ~mask;
    return 1;
}

/*
 * Derive exact raw and padded ends for one relative tensor interval.
 *
 * Relative start, raw bytes, power-of-two alignment, and writable end values. Invalid alignment,
 * null outputs, or overflow return distinct layout codes.
 */
yvex_gguf_layout_code yvex_gguf_layout_interval_measure(unsigned long long relative_offset,
                                                        unsigned long long raw_size,
                                                        unsigned int alignment,
                                                        unsigned long long *raw_end,
                                                        unsigned long long *padded_end) {
    if (!raw_end || !padded_end)
        return YVEX_GGUF_LAYOUT_INVALID_ARGUMENT;
    if (alignment == 0u)
        return YVEX_GGUF_LAYOUT_INVALID_ALIGNMENT_VALUE;
    if ((alignment & (alignment - 1u)) != 0u) {
        return YVEX_GGUF_LAYOUT_ALIGNMENT_NOT_POWER_OF_TWO;
    }
    if (raw_size > ULLONG_MAX - relative_offset) {
        return YVEX_GGUF_LAYOUT_RAW_END_OVERFLOW;
    }
    *raw_end = relative_offset + raw_size;
    if (!align_up_power_of_two(*raw_end, alignment, padded_end)) {
        return YVEX_GGUF_LAYOUT_PADDED_END_OVERFLOW;
    }
    return YVEX_GGUF_LAYOUT_OK;
}

static int layout_fail(yvex_gguf_layout_result *result, yvex_gguf_layout_code code,
                       const char *reason, yvex_error *err) {
    result->accepted = 0;
    result->code = code;
    yvex_core_text_copy(result->reason, sizeof(result->reason), reason);
    yvex_error_setf(err, YVEX_ERR_FORMAT, "yvex_gguf_layout_validate", "%s: %s",
                    yvex_gguf_layout_code_name(code), reason);
    return YVEX_ERR_FORMAT;
}

/*
 * Prove that one exact required padding interval contains only zero bytes.
 *
 * Performs bounded positioned reads and advances padding-byte/call counters.
 */
static int read_zero_padding(const yvex_artifact *artifact, unsigned long long offset,
                             unsigned long long length, yvex_gguf_layout_code nonzero_code,
                             yvex_gguf_layout_result *result, yvex_error *err) {
    unsigned char bytes[YVEX_GGUF_LAYOUT_PADDING_CHUNK];
    unsigned long long done = 0ull;

    while (done < length) {
        unsigned long long remaining = length - done;
        size_t take =
            remaining > (unsigned long long)sizeof(bytes) ? sizeof(bytes) : (size_t)remaining;
        size_t i;
        int rc = yvex_artifact_read_at(artifact, offset + done, bytes, take, err);
        if (rc != YVEX_OK) {
            result->failure_absolute_offset = offset + done;
            return layout_fail(result, YVEX_GGUF_LAYOUT_IO_FAILURE,
                               "required GGUF padding could not be read", err);
        }
        result->padding_bytes_read += (unsigned long long)take;
        result->padding_read_calls += 1ull;
        for (i = 0u; i < take; ++i) {
            if (bytes[i] != 0u) {
                result->failure_absolute_offset = offset + done + (unsigned long long)i;
                return layout_fail(result, nonzero_code,
                                   nonzero_code == YVEX_GGUF_LAYOUT_DIRECTORY_PADDING_NONZERO
                                       ? "GGUF directory-to-data padding contains a nonzero byte"
                                       : "GGUF tensor padding contains a nonzero byte",
                                   err);
            }
        }
        done += (unsigned long long)take;
    }
    return YVEX_OK;
}

const char *yvex_gguf_layout_code_name(yvex_gguf_layout_code code) {
    return code >= YVEX_GGUF_LAYOUT_OK &&
                   (size_t)code < sizeof(layout_code_names) / sizeof(layout_code_names[0])
               ? layout_code_names[code]
               : "unknown-layout-failure";
}

typedef struct {
    unsigned long long expected;
    unsigned long long previous_start;
    unsigned long long previous_raw_end;
} layout_cursor;

/*
 * Validate one directory tensor and advance the canonical physical cursor.
 *
 * Checks geometry/ranges, reads only its padding, and advances counters on success. Overlap, gap,
 * overflow, truncation, nonzero padding, or mismatch leaves typed refusal.
 */
static int layout_validate_tensor(const yvex_artifact *artifact, const yvex_gguf *gguf,
                                  unsigned long long index, layout_cursor *cursor,
                                  yvex_gguf_layout_result *out, yvex_error *err) {
    const yvex_gguf_tensor_info *tensor = yvex_gguf_tensor_at(gguf, index);
    yvex_gguf_qtype_storage_result storage;
    unsigned long long raw_end;
    unsigned long long padded_end;
    unsigned long long raw_absolute;
    unsigned long long raw_absolute_end;
    unsigned long long padded_absolute;
    unsigned long long padding_bytes;
    yvex_gguf_layout_code interval_code;
    int status;

    out->tensor_index = index;
    if (!tensor)
        return layout_fail(out, YVEX_GGUF_LAYOUT_TENSOR_INFO_MISSING,
                           "parsed tensor directory row is unavailable", err);
    yvex_core_text_copy(out->tensor_name, sizeof(out->tensor_name), tensor->name);
    out->expected_relative_offset = cursor->expected;
    out->declared_relative_offset = tensor->relative_offset;
    if (yvex_gguf_qtype_tensor_storage(tensor->ggml_type, tensor->dims, tensor->rank, &storage) !=
        YVEX_GGUF_QTYPE_STORAGE_OK)
        return layout_fail(out, YVEX_GGUF_LAYOUT_STORAGE_REFUSED,
                           storage.reason ? storage.reason : "tensor qtype storage is refused",
                           err);
    out->tensor_raw_size = storage.total_bytes;
    if (tensor->storage_bytes != storage.total_bytes)
        return layout_fail(out, YVEX_GGUF_LAYOUT_STORAGE_REFUSED,
                           "parsed tensor storage differs from canonical qtype geometry", err);
    interval_code = yvex_gguf_layout_interval_measure(tensor->relative_offset, storage.total_bytes,
                                                      out->alignment, &raw_end, &padded_end);
    if (interval_code != YVEX_GGUF_LAYOUT_OK)
        return layout_fail(out, interval_code,
                           interval_code == YVEX_GGUF_LAYOUT_RAW_END_OVERFLOW
                               ? "tensor relative raw end overflows"
                               : "tensor padded end overflows",
                           err);
    out->tensor_raw_end = raw_end;
    out->tensor_padded_end = padded_end;
    out->tensor_padded_size = padded_end - tensor->relative_offset;
    if (index == 0ull && tensor->relative_offset != 0ull)
        return layout_fail(out, YVEX_GGUF_LAYOUT_FIRST_OFFSET_NOT_ZERO,
                           "the first tensor must begin at relative offset zero", err);
    if (index > 0ull && tensor->relative_offset != cursor->expected) {
        if (tensor->relative_offset > cursor->expected)
            return layout_fail(out, YVEX_GGUF_LAYOUT_UNEXPECTED_GAP,
                               "tensor offset leaves a gap after the previous padded interval",
                               err);
        if (tensor->relative_offset == cursor->previous_start)
            return layout_fail(out, YVEX_GGUF_LAYOUT_DUPLICATE_START,
                               "two directory entries declare the same tensor start", err);
        if (tensor->relative_offset < cursor->previous_start)
            return layout_fail(out, YVEX_GGUF_LAYOUT_OFFSET_REVERSED,
                               "tensor offset reverses directory order", err);
        if (tensor->relative_offset < cursor->previous_raw_end)
            return layout_fail(
                out,
                raw_end <= cursor->previous_raw_end ? YVEX_GGUF_LAYOUT_COMPLETE_OVERLAP
                                                    : YVEX_GGUF_LAYOUT_PARTIAL_OVERLAP,
                raw_end <= cursor->previous_raw_end
                    ? "tensor interval is contained in the previous tensor payload"
                    : "tensor interval partially overlaps the previous tensor payload",
                err);
        return layout_fail(out, YVEX_GGUF_LAYOUT_PADDED_OVERLAP,
                           "tensor starts inside required padding", err);
    }
    if (yvex_gguf_layout_sum_checked(out->tensor_data_offset, tensor->relative_offset,
                                     &raw_absolute) != YVEX_GGUF_LAYOUT_OK ||
        yvex_gguf_layout_sum_checked(raw_absolute, storage.total_bytes, &raw_absolute_end) !=
            YVEX_GGUF_LAYOUT_OK ||
        yvex_gguf_layout_sum_checked(out->tensor_data_offset, padded_end, &padded_absolute) !=
            YVEX_GGUF_LAYOUT_OK)
        return layout_fail(out, YVEX_GGUF_LAYOUT_AGGREGATE_OVERFLOW,
                           "absolute tensor layout arithmetic overflows", err);
    if (tensor->absolute_offset != raw_absolute || tensor->absolute_end_offset != raw_absolute_end)
        return layout_fail(out, YVEX_GGUF_LAYOUT_TENSOR_RANGE_MISMATCH,
                           "parsed tensor absolute range differs from canonical layout", err);
    if (tensor->absolute_end_offset > out->actual_file_size) {
        out->failure_absolute_offset = tensor->absolute_end_offset;
        return layout_fail(out, YVEX_GGUF_LAYOUT_TENSOR_PAYLOAD_TRUNCATED,
                           "tensor payload interval exceeds the opened file", err);
    }
    if (padded_absolute > out->actual_file_size) {
        out->failure_absolute_offset = tensor->absolute_end_offset;
        return layout_fail(out, YVEX_GGUF_LAYOUT_PADDING_TRUNCATED,
                           "required tensor padding exceeds the opened file", err);
    }
    if (yvex_gguf_layout_sum_checked(out->raw_tensor_bytes, storage.total_bytes,
                                     &out->raw_tensor_bytes) != YVEX_GGUF_LAYOUT_OK)
        return layout_fail(out, YVEX_GGUF_LAYOUT_AGGREGATE_OVERFLOW,
                           "aggregate raw tensor bytes overflow", err);
    padding_bytes = padded_end - raw_end;
    if (index + 1ull < out->tensor_count) {
        if (yvex_gguf_layout_sum_checked(out->inter_tensor_padding_bytes, padding_bytes,
                                         &out->inter_tensor_padding_bytes) != YVEX_GGUF_LAYOUT_OK)
            return layout_fail(out, YVEX_GGUF_LAYOUT_AGGREGATE_OVERFLOW,
                               "aggregate inter-tensor padding overflows", err);
    } else {
        out->final_tensor_padding_bytes = padding_bytes;
    }
    status = read_zero_padding(artifact, tensor->absolute_end_offset, padding_bytes,
                               YVEX_GGUF_LAYOUT_TENSOR_PADDING_NONZERO, out, err);
    if (status != YVEX_OK)
        return status;
    cursor->previous_start = tensor->relative_offset;
    cursor->previous_raw_end = raw_end;
    cursor->expected = padded_end;
    out->tensors_validated++;
    return YVEX_OK;
}

/*
 * Validate the complete canonical directory order and padded file span.
 *
 * Performs one linear padding-only pass and two snapshot-drift validations. Any geometry, range,
 * padding, tail, I/O, or drift violation refuses the whole layout.
 */
int yvex_gguf_layout_validate(const yvex_artifact *artifact, const yvex_gguf *gguf,
                              yvex_gguf_layout_result *out, yvex_error *err) {
    const yvex_gguf_reader_stats *stats;
    const yvex_gguf_value *alignment_value;
    unsigned long long alignment_metadata = 0ull;
    layout_cursor cursor;
    unsigned long long i;
    int rc;

    if (!out) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "yvex_gguf_layout_validate",
                       "layout output is required");
        return YVEX_ERR_INVALID_ARG;
    }
    layout_result_init(out);
    memset(&cursor, 0, sizeof(cursor));
    if (!artifact || !gguf) {
        return layout_fail(out, YVEX_GGUF_LAYOUT_INVALID_ARGUMENT,
                           "artifact and parsed GGUF view are required", err);
    }

    out->alignment = yvex_gguf_alignment(gguf);
    out->tensor_count = yvex_gguf_tensor_count(gguf);
    out->tensor_data_offset = yvex_gguf_tensor_data_offset(gguf);
    out->actual_file_size = yvex_artifact_size(artifact);
    stats = yvex_gguf_reader_stats_view(gguf);
    if (!stats || stats->file_size != out->actual_file_size) {
        return layout_fail(out, YVEX_GGUF_LAYOUT_VIEW_FILE_MISMATCH,
                           "parsed GGUF view does not describe the opened artifact size", err);
    }
    out->structural_bytes_read = stats->structural_bytes_read;
    out->directory_end = stats->directory_end_offset;
    out->tensor_payload_bytes_read = 0ull;

    rc = yvex_artifact_snapshot_validate(artifact, NULL, err);
    if (rc != YVEX_OK) {
        return layout_fail(out, YVEX_GGUF_LAYOUT_FILE_IDENTITY_DRIFT,
                           "artifact identity changed before layout validation", err);
    }

    alignment_value = yvex_gguf_metadata_find(gguf, "general.alignment");
    if (alignment_value) {
        if (yvex_gguf_value_type_of(alignment_value) != YVEX_GGUF_VALUE_UINT32 ||
            yvex_gguf_value_as_u64(alignment_value, &alignment_metadata) != YVEX_OK) {
            return layout_fail(out, YVEX_GGUF_LAYOUT_INVALID_ALIGNMENT_TYPE,
                               "general.alignment must have GGUF uint32 type", err);
        }
        if (alignment_metadata != (unsigned long long)out->alignment) {
            return layout_fail(out, YVEX_GGUF_LAYOUT_VIEW_FILE_MISMATCH,
                               "parsed alignment differs from metadata", err);
        }
    }
    if (out->alignment == 0u) {
        return layout_fail(out, YVEX_GGUF_LAYOUT_INVALID_ALIGNMENT_VALUE,
                           "general.alignment must be nonzero", err);
    }
    if ((out->alignment & (out->alignment - 1u)) != 0u) {
        return layout_fail(out, YVEX_GGUF_LAYOUT_ALIGNMENT_NOT_POWER_OF_TWO,
                           "general.alignment must be a power of two", err);
    }
    if (!align_up_power_of_two(out->directory_end, out->alignment, &out->tensor_data_offset)) {
        return layout_fail(out, YVEX_GGUF_LAYOUT_ALIGNMENT_OVERFLOW,
                           "directory-to-data alignment overflows", err);
    }
    if (out->tensor_data_offset != yvex_gguf_tensor_data_offset(gguf)) {
        return layout_fail(out, YVEX_GGUF_LAYOUT_VIEW_FILE_MISMATCH,
                           "parsed tensor-data offset is not the canonical aligned directory end",
                           err);
    }
    out->directory_padding_bytes = out->tensor_data_offset - out->directory_end;
    if (out->tensor_data_offset > out->actual_file_size) {
        out->failure_absolute_offset = out->directory_end;
        return layout_fail(out, YVEX_GGUF_LAYOUT_PADDING_TRUNCATED,
                           "directory-to-data padding is truncated", err);
    }
    rc = read_zero_padding(artifact, out->directory_end, out->directory_padding_bytes,
                           YVEX_GGUF_LAYOUT_DIRECTORY_PADDING_NONZERO, out, err);
    if (rc != YVEX_OK)
        return rc;

    for (i = 0ull; i < out->tensor_count; ++i) {
        rc = layout_validate_tensor(artifact, gguf, i, &cursor, out, err);
        if (rc != YVEX_OK)
            return rc;
    }

    out->data_section_span = cursor.expected;
    if (yvex_gguf_layout_sum_checked(out->directory_padding_bytes, out->inter_tensor_padding_bytes,
                                     &out->total_padding_bytes) != YVEX_GGUF_LAYOUT_OK ||
        yvex_gguf_layout_sum_checked(out->total_padding_bytes, out->final_tensor_padding_bytes,
                                     &out->total_padding_bytes) != YVEX_GGUF_LAYOUT_OK ||
        yvex_gguf_layout_sum_checked(out->tensor_data_offset, out->data_section_span,
                                     &out->required_file_end) != YVEX_GGUF_LAYOUT_OK) {
        return layout_fail(out, YVEX_GGUF_LAYOUT_AGGREGATE_OVERFLOW,
                           "aggregate GGUF layout span overflows", err);
    }
    if (out->required_file_end > out->actual_file_size) {
        return layout_fail(out, YVEX_GGUF_LAYOUT_PADDING_TRUNCATED,
                           "GGUF file ends before the required padded data span", err);
    }
    out->trailing_bytes = out->actual_file_size - out->required_file_end;
    if (out->trailing_bytes != 0ull) {
        out->failure_absolute_offset = out->required_file_end;
        return layout_fail(out, YVEX_GGUF_LAYOUT_NONCANONICAL_TAIL,
                           "bytes after the canonical padded data span are not admitted", err);
    }
    rc = yvex_artifact_snapshot_validate(artifact, NULL, err);
    if (rc != YVEX_OK) {
        return layout_fail(out, YVEX_GGUF_LAYOUT_FILE_IDENTITY_DRIFT,
                           "artifact identity changed during layout validation", err);
    }

    out->tensor_index = ULLONG_MAX;
    out->tensor_name[0] = '\0';
    out->code = YVEX_GGUF_LAYOUT_OK;
    out->accepted = 1;
    yvex_core_text_copy(out->reason, sizeof(out->reason),
                "canonical GGUF directory order, padded spans, and zero padding accepted");
    yvex_error_clear(err);
    return YVEX_OK;
}
