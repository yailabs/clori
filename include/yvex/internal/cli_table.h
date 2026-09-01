/* Bounded scrollback-safe table projection shared by CLI adapters and renderers. */
#ifndef INCLUDE_YVEX_INTERNAL_CLI_TABLE_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_CLI_TABLE_H_INCLUDED

#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    YVEX_CLI_TABLE_LEFT = 0,
    YVEX_CLI_TABLE_RIGHT
} yvex_cli_table_alignment;

typedef enum {
    YVEX_CLI_TABLE_PLAIN = 0,
    YVEX_CLI_TABLE_ACCENT,
    YVEX_CLI_TABLE_SUCCESS,
    YVEX_CLI_TABLE_WARNING,
    YVEX_CLI_TABLE_ERROR,
    YVEX_CLI_TABLE_DIM
} yvex_cli_table_tone;

typedef struct {
    const char *heading;
    unsigned int minimum_width;
    unsigned int maximum_width;
    yvex_cli_table_alignment alignment;
    int middle_elide;
} yvex_cli_table_column;

typedef struct {
    const char *text;
    yvex_cli_table_tone tone;
} yvex_cli_table_cell;

typedef struct {
    const yvex_cli_table_cell *cells;
    const char *secondary;
    yvex_cli_table_tone secondary_tone;
} yvex_cli_table_row;

unsigned int yvex_cli_terminal_columns(FILE *fp);
void yvex_cli_precision_format(char *out, size_t capacity, const char *raw);
int yvex_cli_table_render(FILE *fp,
                          const yvex_cli_table_column *columns,
                          size_t column_count,
                          const yvex_cli_table_row *rows,
                          size_t row_count);
int yvex_cli_table_render_width(FILE *fp,
                                const yvex_cli_table_column *columns,
                                size_t column_count,
                                const yvex_cli_table_row *rows,
                                size_t row_count,
                                unsigned int width);

#ifdef __cplusplus
}
#endif

#endif /* INCLUDE_YVEX_INTERNAL_CLI_TABLE_H_INCLUDED */
