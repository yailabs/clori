/*
 * Provide approved direct JSON text output for CLI plumbing surfaces.
 *
 * Helpers serialize only caller-provided fields and do not claim uniform JSON. JSON writer
 * primitives are not command-level JSON support by themselves.
 */
#include "src/cli/io/private.h"

#include <math.h>

static void json_string(FILE *fp, const char *text) {
    const unsigned char *p = (const unsigned char *)(text ? text : "");

    (void)yvex_cli_out_char(fp, '"');
    while (*p) {
        if (*p == '"' || *p == '\\') {
            (void)yvex_cli_out_char(fp, '\\');
            (void)yvex_cli_out_char(fp, *p);
        } else if (*p == '\n') {
            (void)yvex_cli_out_puts(fp, "\\n");
        } else if (*p == '\r') {
            (void)yvex_cli_out_puts(fp, "\\r");
        } else if (*p == '\t') {
            (void)yvex_cli_out_puts(fp, "\\t");
        } else if (*p < 0x20u) {
            (void)yvex_cli_out_writef(fp, "\\u%04x", (unsigned int)*p);
        } else {
            (void)yvex_cli_out_char(fp, *p);
        }
        ++p;
    }
    (void)yvex_cli_out_char(fp, '"');
}

void yvex_cli_json_begin(FILE *fp) {
    yvex_cli_out_line(fp, "{");
}

void yvex_cli_json_end(FILE *fp) {
    yvex_cli_out_line(fp, "}");
}

static int json_field(FILE *fp, const char *key, yvex_cli_field_kind kind, const void *value,
                      int comma) {
    (void)yvex_cli_out_puts(fp, "  ");
    json_string(fp, key);
    (void)yvex_cli_out_puts(fp, ": ");
    switch (kind) {
    case YVEX_CLI_FIELD_TEXT:
    case YVEX_CLI_FIELD_TEXT_ARRAY:
        json_string(fp, value);
        break;
    case YVEX_CLI_FIELD_U64:
        (void)yvex_cli_out_writef(fp, "%llu", *(const unsigned long long *)value);
        break;
    case YVEX_CLI_FIELD_U32:
        (void)yvex_cli_out_writef(fp, "%u", *(const unsigned int *)value);
        break;
    case YVEX_CLI_FIELD_I32:
        (void)yvex_cli_out_writef(fp, "%d", *(const int *)value);
        break;
    case YVEX_CLI_FIELD_BOOL:
        (void)yvex_cli_out_puts(fp, *(const int *)value ? "true" : "false");
        break;
    case YVEX_CLI_FIELD_DOUBLE:
        if (isfinite(*(const double *)value))
            (void)yvex_cli_out_writef(fp, "%.17g", *(const double *)value);
        else
            (void)yvex_cli_out_puts(fp, "null");
        break;
    default:
        return YVEX_ERR_UNSUPPORTED;
    }
    (void)yvex_cli_out_writef(fp, "%s\n", comma ? "," : "");
    return ferror(fp) ? YVEX_ERR_IO : YVEX_OK;
}

void yvex_cli_json_field_str(FILE *fp, const char *key, const char *value, int comma) {
    (void)json_field(fp, key, YVEX_CLI_FIELD_TEXT_ARRAY, value ? value : "", comma);
}

void yvex_cli_json_field_u64(FILE *fp, const char *key, unsigned long long value, int comma) {
    (void)json_field(fp, key, YVEX_CLI_FIELD_U64, &value, comma);
}

void yvex_cli_json_field_bool(FILE *fp, const char *key, int value, int comma) {
    (void)json_field(fp, key, YVEX_CLI_FIELD_BOOL, &value, comma);
}

int yvex_cli_json_fields(FILE *fp, const void *object, const yvex_cli_field_spec *fields,
                         size_t field_count, int comma) {
    const unsigned char *base = object;
    size_t index;

    if (!fp || !object || (!fields && field_count))
        return YVEX_ERR_INVALID_ARG;
    for (index = 0; index < field_count; ++index) {
        const yvex_cli_field_spec *field = &fields[index];
        const void *value = base + field->offset;
        int separator = comma || index + 1u < field_count;
        if (field->kind == YVEX_CLI_FIELD_TEXT)
            value = *(const char *const *)value;
        if (json_field(fp, field->key, field->kind, value, separator) != YVEX_OK)
            return ferror(fp) ? YVEX_ERR_IO : YVEX_ERR_UNSUPPORTED;
    }
    return ferror(fp) ? YVEX_ERR_IO : YVEX_OK;
}
