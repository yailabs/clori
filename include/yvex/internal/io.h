/* File serializers share escaping helpers here without acquiring stream ownership. */
#ifndef INCLUDE_YVEX_INTERNAL_IO_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_IO_H_INCLUDED

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Escaped JSON fields. */
void yvex_file_json_write_string(FILE *fp, const char *s);
void yvex_file_json_write_field(FILE *fp,
                                const char *indent,
                                const char *name,
                                const char *value,
                                int comma);

#ifdef __cplusplus
}
#endif

#endif /* INCLUDE_YVEX_INTERNAL_IO_H_INCLUDED */
