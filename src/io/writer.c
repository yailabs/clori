/*
 * Centralize JSON string escaping for file-backed serializers.
 *
 * Writes only to caller-provided FILE handles and never chooses standard streams. JSON
 * serialization is not source verification or runtime readiness.
 */
#include <yvex/internal/io.h>

void yvex_file_json_write_string(FILE *fp, const char *s)
{
    const unsigned char *p = (const unsigned char *)(s ? s : "");

    fputc('"', fp);
    while (*p) {
        unsigned char ch = *p++;
        switch (ch) {
        case '"':
            fputs("\\\"", fp);
            break;
        case '\\':
            fputs("\\\\", fp);
            break;
        case '\b':
            fputs("\\b", fp);
            break;
        case '\f':
            fputs("\\f", fp);
            break;
        case '\n':
            fputs("\\n", fp);
            break;
        case '\r':
            fputs("\\r", fp);
            break;
        case '\t':
            fputs("\\t", fp);
            break;
        default:
            if (ch < 32) {
                fprintf(fp, "\\u%04x", (unsigned int)ch);
            } else {
                fputc((int)ch, fp);
            }
            break;
        }
    }
    fputc('"', fp);
}

void yvex_file_json_write_field(FILE *fp,
                                const char *indent,
                                const char *name,
                                const char *value,
                                int comma)
{
    fputs(indent ? indent : "", fp);
    yvex_file_json_write_string(fp, name);
    fputs(": ", fp);
    yvex_file_json_write_string(fp, value);
    fputs(comma ? ",\n" : "\n", fp);
}
