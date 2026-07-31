/*
 * Define bounded reader policy and typed projection of structural parse failures.
 *
 * Parse classification is carried by typed facts and never reconstructed from human-readable error
 * strings. Structural reader acceptance is not complete artifact integrity, writer roundtrip,
 * materialization, or runtime support.
 */
#include <limits.h>
#include <stdio.h>
#include <yvex/internal/gguf.h>

void yvex_gguf_reader_options_default(yvex_gguf_reader_options *options) {
    if (!options)
        return;
    options->max_metadata_entries = 1048576ull;
    options->max_tensor_entries = 1048576ull;
    options->max_array_entries = 16777216ull;
    options->max_total_array_entries = 33554432ull;
    options->max_string_bytes = 67108864ull;
    options->max_total_string_bytes = 1073741824ull;
    options->max_owned_bytes = 2147483648ull;
    options->max_structural_bytes = 2147483648ull;
    options->max_array_depth = 16u;
}

void yvex_gguf_parse_result_reset(yvex_gguf_parse_result *result) {
    if (!result)
        return;
    result->code = YVEX_GGUF_PARSE_OK;
    result->section = YVEX_GGUF_PARSE_SECTION_NONE;
    result->byte_offset = 0ull;
    result->record_index = ULLONG_MAX;
    result->reason = "GGUF structural reader accepted input";
}

static int parse_code_error(yvex_gguf_parse_code code) {
    switch (code) {
    case YVEX_GGUF_PARSE_OK:
        return YVEX_OK;
    case YVEX_GGUF_PARSE_INVALID_ARGUMENT:
        return YVEX_ERR_INVALID_ARG;
    case YVEX_GGUF_PARSE_FILE_UNREADABLE:
        return YVEX_ERR_IO;
    case YVEX_GGUF_PARSE_SHORT_READ:
        return YVEX_ERR_BOUNDS;
    case YVEX_GGUF_PARSE_UNSUPPORTED_VERSION:
    case YVEX_GGUF_PARSE_UNSUPPORTED_METADATA_TYPE:
    case YVEX_GGUF_PARSE_REFUSED_QTYPE:
        return YVEX_ERR_UNSUPPORTED;
    case YVEX_GGUF_PARSE_RESOURCE_LIMIT:
    case YVEX_GGUF_PARSE_OFFSET_OVERFLOW:
    case YVEX_GGUF_PARSE_INCOMPLETE_DIRECTORY:
    case YVEX_GGUF_PARSE_ELEMENT_COUNT_OVERFLOW:
    case YVEX_GGUF_PARSE_ROW_COUNT_OVERFLOW:
    case YVEX_GGUF_PARSE_ROW_BYTE_OVERFLOW:
    case YVEX_GGUF_PARSE_TOTAL_BYTE_OVERFLOW:
        return YVEX_ERR_BOUNDS;
    case YVEX_GGUF_PARSE_ALLOCATION_FAILURE:
        return YVEX_ERR_NOMEM;
    default:
        return YVEX_ERR_FORMAT;
    }
}

int yvex_gguf_reader_fail(yvex_gguf_parse_result *result, yvex_gguf_parse_code code,
                          yvex_gguf_parse_section section, unsigned long long byte_offset,
                          unsigned long long record_index, yvex_error *err, const char *where,
                          const char *reason) {
    int rc = parse_code_error(code);
    if (result) {
        result->code = code;
        result->section = section;
        result->byte_offset = byte_offset;
        result->record_index = record_index;
        result->reason = reason ? reason : "GGUF structural reader refused input";
    }
    yvex_error_set(err, rc, where ? where : "yvex_gguf_open_ex",
                   reason ? reason : "GGUF structural reader refused input");
    return rc;
}
