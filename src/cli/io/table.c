/* Render bounded, line-oriented CLI tables without terminal ownership. */
#define _POSIX_C_SOURCE 200809L
#include "src/cli/io/private.h"
#include <yvex/internal/cli_table.h>

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/ioctl.h>
#include <unistd.h>

void yvex_cli_precision_format(char *out, size_t capacity, const char *raw)
{
    static const struct { const char *raw, *display; } names[] = {
        {"iq2xxs-q2k-mxfp4", "IQ2_XXS/Q2_K/MXFP4"},
        {"iq2xxs", "IQ2_XXS"},
        {"iq2_xxs", "IQ2_XXS"},
        {"q2k", "Q2_K"},
        {"q2_k", "Q2_K"},
        {"mxfp4", "MXFP4"},
        {"bf16", "BF16"},
        {"f16", "FP16"},
        {"fp16", "FP16"},
        {"f32", "FP32"},
        {"fp32", "FP32"}
    };
    size_t index;
    if (!out || !capacity) return;
    for (index = 0u; index < sizeof(names) / sizeof(names[0]); ++index)
        if (raw && !strcasecmp(raw, names[index].raw)) {
            snprintf(out, capacity, "%s", names[index].display);
            return;
        }
    snprintf(out, capacity, "%s", raw && raw[0] ? raw : "not recorded");
}

#define CLI_TABLE_COLUMN_CAP 16u
#define CLI_TABLE_CELL_CAP 1024u

static size_t table_text_width(const char *text)
{
    const unsigned char *cursor = (const unsigned char *)(text ? text : "");
    size_t width = 0u;
    while (*cursor) {
        if ((*cursor & 0xc0u) != 0x80u) width++;
        cursor++;
    }
    return width;
}

unsigned int yvex_cli_terminal_columns(FILE *fp)
{
    struct winsize window;
    const char *configured = getenv("COLUMNS");
    char *end = NULL;
    unsigned long value;
    int fd = fp ? fileno(fp) : -1;

    if (configured && configured[0]) {
        errno = 0;
        value = strtoul(configured, &end, 10);
        if (!errno && end && !*end && value >= 20u && value <= 1000u)
            return (unsigned int)value;
    }
    memset(&window, 0, sizeof(window));
    if (fd >= 0 && isatty(fd) && ioctl(fd, TIOCGWINSZ, &window) == 0 &&
        window.ws_col >= 20u)
        return window.ws_col;
    return 120u;
}

static const char *table_tone(const yvex_cli_terminal_style *style,
                              yvex_cli_table_tone tone)
{
    switch (tone) {
    case YVEX_CLI_TABLE_ACCENT: return style->accent;
    case YVEX_CLI_TABLE_SUCCESS: return style->success;
    case YVEX_CLI_TABLE_WARNING: return style->warning;
    case YVEX_CLI_TABLE_ERROR: return style->error;
    case YVEX_CLI_TABLE_DIM: return style->dim;
    case YVEX_CLI_TABLE_PLAIN: break;
    }
    return "";
}

static void table_elide(char out[CLI_TABLE_CELL_CAP], const char *text,
                        size_t width, int middle)
{
    static const char ellipsis[] = "\xe2\x80\xa6";
    size_t bytes = strlen(text ? text : "");
    size_t visible = table_text_width(text);
    size_t left, right;

    if (visible <= width && bytes < CLI_TABLE_CELL_CAP) {
        memcpy(out, text ? text : "", bytes + 1u);
        return;
    }
    if (width < 2u) {
        if (width) memcpy(out, ellipsis, sizeof(ellipsis));
        else out[0] = '\0';
        return;
    }
    /* Product selectors and paths are ASCII by contract.  A non-ASCII cell that
     * needs truncation is conservatively byte-clipped at a UTF-8 boundary. */
    if (!middle) {
        size_t count = width - 1u;
        if (count >= CLI_TABLE_CELL_CAP - sizeof(ellipsis))
            count = CLI_TABLE_CELL_CAP - sizeof(ellipsis);
        while (count && (((unsigned char)text[count]) & 0xc0u) == 0x80u) count--;
        memcpy(out, text, count);
        memcpy(out + count, ellipsis, sizeof(ellipsis));
        return;
    }
    left = (width - 1u) / 2u;
    right = width - 1u - left;
    if (left + right + sizeof(ellipsis) > CLI_TABLE_CELL_CAP) {
        right = CLI_TABLE_CELL_CAP - left - sizeof(ellipsis);
    }
    memcpy(out, text, left);
    memcpy(out + left, ellipsis, sizeof(ellipsis) - 1u);
    memcpy(out + left + sizeof(ellipsis) - 1u, text + bytes - right, right);
    out[left + sizeof(ellipsis) - 1u + right] = '\0';
}

static void table_widths(const yvex_cli_table_column *columns,
                         size_t column_count,
                         const yvex_cli_table_row *rows,
                         size_t row_count,
                         unsigned int available,
                         unsigned int widths[CLI_TABLE_COLUMN_CAP])
{
    size_t column, row;
    unsigned int total = column_count ? (unsigned int)(2u * (column_count - 1u)) : 0u;

    for (column = 0u; column < column_count; ++column) {
        size_t natural = table_text_width(columns[column].heading);
        for (row = 0u; row < row_count; ++row) {
            size_t value = table_text_width(rows[row].cells[column].text);
            if (value > natural) natural = value;
        }
        if (natural < columns[column].minimum_width)
            natural = columns[column].minimum_width;
        if (columns[column].maximum_width && natural > columns[column].maximum_width)
            natural = columns[column].maximum_width;
        widths[column] = (unsigned int)natural;
        total += widths[column];
    }
    while (total > available) {
        size_t selected = column_count;
        unsigned int slack = 0u;
        for (column = 0u; column < column_count; ++column) {
            unsigned int minimum = columns[column].minimum_width;
            if (widths[column] > minimum && widths[column] - minimum > slack) {
                selected = column;
                slack = widths[column] - minimum;
            }
        }
        if (selected == column_count) break;
        widths[selected]--;
        total--;
    }
    while (total > available) {
        size_t selected = column_count;
        unsigned int widest = 0u;
        for (column = 0u; column < column_count; ++column)
            if (widths[column] > 1u && widths[column] > widest) {
                selected = column;
                widest = widths[column];
            }
        if (selected == column_count) break;
        widths[selected]--;
        total--;
    }
}

static int table_cell_write(FILE *fp, const yvex_cli_table_cell *cell,
                            const yvex_cli_table_column *column,
                            unsigned int width,
                            const yvex_cli_terminal_style *style)
{
    char rendered[CLI_TABLE_CELL_CAP];
    const char *tone = table_tone(style, cell->tone);
    size_t visible;

    table_elide(rendered, cell->text ? cell->text : "", width,
                column->middle_elide);
    visible = table_text_width(rendered);
    if (column->alignment == YVEX_CLI_TABLE_RIGHT && visible < width &&
        yvex_cli_out_writef(fp, "%*s", (int)(width - visible), "") < 0)
        return -1;
    if (tone[0] && yvex_cli_out_fputs(tone, fp) < 0) return -1;
    if (yvex_cli_out_fputs(rendered, fp) < 0) return -1;
    if (tone[0] && yvex_cli_out_fputs(style->reset, fp) < 0) return -1;
    if (column->alignment == YVEX_CLI_TABLE_LEFT && visible < width &&
        yvex_cli_out_writef(fp, "%*s", (int)(width - visible), "") < 0)
        return -1;
    return 0;
}

static int table_render_at_width(FILE *fp,
                                 const yvex_cli_table_column *columns,
                                 size_t column_count,
                                 const yvex_cli_table_row *rows,
                                 size_t row_count,
                                 unsigned int available)
{
    unsigned int widths[CLI_TABLE_COLUMN_CAP] = {0};
    yvex_cli_terminal_style style;
    size_t column, row;
    int headings = 0;

    if (!fp || !columns || !column_count || column_count > CLI_TABLE_COLUMN_CAP ||
        (row_count && !rows))
        return YVEX_ERR_INVALID_ARG;
    yvex_cli_terminal_style_get(fp, &style);
    table_widths(columns, column_count, rows, row_count, available, widths);
    for (column = 0u; column < column_count; ++column)
        headings |= columns[column].heading && columns[column].heading[0];
    if (headings) {
        for (column = 0u; column < column_count; ++column) {
            yvex_cli_table_cell heading = {columns[column].heading,
                                           YVEX_CLI_TABLE_PLAIN};
            if (style.strong[0] && yvex_cli_out_fputs(style.strong, fp) < 0)
                return YVEX_ERR_IO;
            if (table_cell_write(fp, &heading, &columns[column], widths[column],
                                 &style) < 0)
                return YVEX_ERR_IO;
            if (style.strong[0] && yvex_cli_out_fputs(style.reset, fp) < 0)
                return YVEX_ERR_IO;
            if (column + 1u < column_count &&
                yvex_cli_out_fputs("  ", fp) < 0)
                return YVEX_ERR_IO;
        }
        if (yvex_cli_out_char(fp, '\n') < 0) return YVEX_ERR_IO;
    }
    for (row = 0u; row < row_count; ++row) {
        for (column = 0u; column < column_count; ++column) {
            if (table_cell_write(fp, &rows[row].cells[column], &columns[column],
                                 widths[column], &style) < 0)
                return YVEX_ERR_IO;
            if (column + 1u < column_count && yvex_cli_out_fputs("  ", fp) < 0)
                return YVEX_ERR_IO;
        }
        if (yvex_cli_out_char(fp, '\n') < 0) return YVEX_ERR_IO;
        if (rows[row].secondary && rows[row].secondary[0]) {
            char rendered[CLI_TABLE_CELL_CAP];
            const char *secondary = rows[row].secondary;
            const char *tone = table_tone(&style, rows[row].secondary_tone);
            if (isatty(fileno(fp))) {
                table_elide(rendered, secondary, available > 2u ? available - 2u : 1u,
                            0);
                secondary = rendered;
            }
            if (tone[0] && yvex_cli_out_fputs(tone, fp) < 0) return YVEX_ERR_IO;
            if (yvex_cli_out_writef(fp, "  %s", secondary) < 0)
                return YVEX_ERR_IO;
            if (tone[0] && yvex_cli_out_fputs(style.reset, fp) < 0) return YVEX_ERR_IO;
            if (yvex_cli_out_char(fp, '\n') < 0) return YVEX_ERR_IO;
        }
    }
    return YVEX_OK;
}

int yvex_cli_table_render(FILE *fp,
                          const yvex_cli_table_column *columns,
                          size_t column_count,
                          const yvex_cli_table_row *rows,
                          size_t row_count)
{
    return table_render_at_width(fp, columns, column_count, rows, row_count,
                                 yvex_cli_terminal_columns(fp));
}

int yvex_cli_table_render_width(FILE *fp,
                                const yvex_cli_table_column *columns,
                                size_t column_count,
                                const yvex_cli_table_row *rows,
                                size_t row_count,
                                unsigned int width)
{
    if (width < 20u || width > 1000u) return YVEX_ERR_INVALID_ARG;
    return table_render_at_width(fp, columns, column_count, rows, row_count,
                                 width);
}
