/*
 * Provide approved direct text output calls for operator normal/table/audit text.
 *
 * Direct stdio output stays in this file; callers provide target streams; wrappers preserve stdio
 * return behavior where legacy code checks it. Writer calls serialize existing facts only and do
 * not create capability.
 */
#include "src/cli/io/private.h"

#include <ctype.h>
#include <errno.h>
#include <operator/registry.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum { COMPLETION_CANDIDATE_CAP = 256, COMPLETION_TEXT_CAP = 128 };

typedef struct {
    char text[COMPLETION_CANDIDATE_CAP][COMPLETION_TEXT_CAP];
    size_t count;
} completion_candidates;

static int completion_visible(const yvex_operator_descriptor *descriptor)
{
    return descriptor->cli_projection &&
           descriptor->visibility != YVEX_OPERATOR_VISIBILITY_REMOVED &&
           descriptor->visibility != YVEX_OPERATOR_VISIBILITY_API_ONLY &&
           descriptor->visibility != YVEX_OPERATOR_VISIBILITY_TEST_ONLY;
}

static int completion_prefix_matches(const yvex_operator_descriptor *descriptor,
                                     size_t count, const char *const *words)
{
    size_t index;
    if (count > descriptor->command_word_count) return 0;
    for (index = 0u; index < count; ++index)
        if (strcmp(descriptor->command_words[index], words[index])) return 0;
    return 1;
}

static void completion_add(completion_candidates *candidates, const char *value)
{
    const unsigned char *cursor = (const unsigned char *)value;
    size_t index;
    if (!value[0] || strlen(value) >= COMPLETION_TEXT_CAP) return;
    while (*cursor) {
        if (!(isalnum(*cursor) || strchr("._:/@+-", *cursor))) return;
        cursor++;
    }
    for (index = 0u; index < candidates->count; ++index)
        if (!strcmp(candidates->text[index], value)) return;
    if (candidates->count >= COMPLETION_CANDIDATE_CAP) return;
    (void)snprintf(candidates->text[candidates->count], COMPLETION_TEXT_CAP, "%s", value);
    candidates->count++;
}

static void completion_add_metadata(completion_candidates *candidates, const char *values)
{
    const char *cursor = values;
    if (!strcmp(values, "none")) return;
    while (*cursor) {
        const char *end = strchr(cursor, '|');
        size_t extent = end ? (size_t)(end - cursor) : strlen(cursor);
        char item[COMPLETION_TEXT_CAP];
        if (extent < sizeof(item)) {
            memcpy(item, cursor, extent);
            item[extent] = '\0';
            completion_add(candidates, item);
        }
        if (!end) break;
        cursor = end + 1;
    }
}

static completion_candidates completion_collect(size_t prefix_count,
                                                const char *const *prefix)
{
    completion_candidates candidates = {{{0}}, 0u};
    size_t descriptor_index;
    for (descriptor_index = 0u; descriptor_index < yvex_operator_descriptor_count;
         ++descriptor_index) {
        const yvex_operator_descriptor *descriptor =
            &yvex_operator_descriptors[descriptor_index];
        size_t index;
        if (!completion_visible(descriptor) ||
            !completion_prefix_matches(descriptor, prefix_count, prefix))
            continue;
        if (descriptor->command_word_count > prefix_count) {
            completion_add(&candidates, descriptor->command_words[prefix_count]);
            continue;
        }
        for (index = 0u; index < descriptor->flag_count; ++index) {
            completion_add(&candidates, descriptor->flags[index].name);
            completion_add_metadata(&candidates, descriptor->flags[index].aliases);
        }
        for (index = 0u; index < descriptor->argument_count; ++index)
            completion_add_metadata(&candidates, descriptor->arguments[index].enum_values);
    }
    return candidates;
}

static void completion_emit_case(FILE *output, const char *shell,
                                 size_t prefix_count, const char *const *prefix)
{
    completion_candidates candidates = completion_collect(prefix_count, prefix);
    size_t index;
    if (!candidates.count) return;
    if (!strcmp(shell, "fish")) fputs("    case '", output);
    else fputs("    '", output);
    for (index = 0u; index < prefix_count; ++index)
        fprintf(output, "%s%s", index ? " " : "", prefix[index]);
    if (!strcmp(shell, "fish")) fputs("'\n      set candidates", output);
    else fputs("') candidates='", output);
    for (index = 0u; index < candidates.count; ++index)
        fprintf(output, " %s", candidates.text[index]);
    if (!strcmp(shell, "fish")) fputc('\n', output);
    else fputs(" ' ;;\n", output);
}

static void completion_emit_cases(FILE *output, const char *shell)
{
    size_t descriptor_index, prefix_count, prior;
    for (descriptor_index = 0u; descriptor_index < yvex_operator_descriptor_count;
         ++descriptor_index) {
        const yvex_operator_descriptor *descriptor =
            &yvex_operator_descriptors[descriptor_index];
        if (!completion_visible(descriptor)) continue;
        for (prefix_count = 0u; prefix_count <= descriptor->command_word_count;
             ++prefix_count) {
            int seen = 0;
            for (prior = 0u; prior < descriptor_index && !seen; ++prior) {
                const yvex_operator_descriptor *candidate =
                    &yvex_operator_descriptors[prior];
                if (completion_visible(candidate) &&
                    candidate->command_word_count >= prefix_count &&
                    completion_prefix_matches(candidate, prefix_count,
                                              descriptor->command_words))
                    seen = 1;
            }
            if (!seen)
                completion_emit_case(output, shell, prefix_count,
                                     descriptor->command_words);
        }
    }
}

int yvex_cli_completion_command(int argc, char **argv, size_t consumed)
{
    const char *shell = consumed + 1u < (size_t)argc ? argv[consumed + 1u] : NULL;
    if (!shell) return 2;
    if (!strcmp(shell, "bash")) {
        fputs("_yvex_complete() {\n"
              "  local cur=${COMP_WORDS[COMP_CWORD]} path='' candidates=''\n"
              "  if (( COMP_CWORD > 1 )); then "
              "path=${COMP_WORDS[*]:1:$((COMP_CWORD-1))}; fi\n"
              "  case \"$path\" in\n", stdout);
        completion_emit_cases(stdout, shell);
        fputs("  esac\n  COMPREPLY=( $(compgen -W \"$candidates\" -- \"$cur\") )\n"
              "}\ncomplete -F _yvex_complete yvex\n", stdout);
        return 0;
    }
    if (!strcmp(shell, "zsh")) {
        fputs("#compdef yvex\n_yvex_complete() {\n"
              "  local path='' candidates=''\n"
              "  if (( CURRENT > 2 )); then path=${(j: :)words[2,$((CURRENT-1))]}; fi\n"
              "  case \"$path\" in\n", stdout);
        completion_emit_cases(stdout, shell);
        fputs("  esac\n  compadd -- ${(z)candidates}\n}\ncompdef _yvex_complete yvex\n", stdout);
        return 0;
    }
    if (!strcmp(shell, "fish")) {
        fputs("function __yvex_candidates\n"
              "  set -l tokens (commandline -opc)\n"
              "  set -e tokens[1]\n"
              "  set -l path (string join ' ' $tokens)\n"
              "  set -l candidates\n"
              "  switch $path\n", stdout);
        completion_emit_cases(stdout, shell);
        fputs("  end\n  printf '%s\\n' $candidates\nend\n"
              "complete -c yvex -f -a '(__yvex_candidates)'\n", stdout);
        return 0;
    }
    fprintf(stderr, "yvex: completion shell must be bash, zsh, or fish\n");
    return 2;
}

int yvex_cli_out_vwritef(FILE *fp, const char *fmt, va_list ap)
{
    return vfprintf(fp ? fp : stdout, fmt ? fmt : "", ap);
}

int yvex_cli_out_writef(FILE *fp, const char *fmt, ...)
{
    va_list ap;
    int rc;

    va_start(ap, fmt);
    rc = yvex_cli_out_vwritef(fp, fmt, ap);
    va_end(ap);
    return rc;
}

int yvex_cli_out_puts(FILE *fp, const char *text)
{
    return fputs(text ? text : "", fp ? fp : stdout);
}

int yvex_cli_out_fputs(const char *text, FILE *fp)
{
    return yvex_cli_out_puts(fp, text);
}

int yvex_cli_out_char(FILE *fp, int ch)
{
    return fputc(ch, fp ? fp : stdout);
}

int yvex_cli_out_flush(FILE *fp)
{
    FILE *stream = fp ? fp : stdout;

    return fflush(stream) == 0 && !ferror(stream) ? YVEX_OK : YVEX_ERR_IO;
}

FILE *yvex_cli_out_stdout(void)
{
    return stdout;
}

FILE *yvex_cli_out_stderr(void)
{
    return stderr;
}

void yvex_cli_out_line(FILE *fp, const char *text)
{
    (void)yvex_cli_out_puts(fp, text);
    (void)yvex_cli_out_char(fp, '\n');
}

void yvex_cli_out_lines(FILE *fp,
                        const char *const *lines,
                        size_t line_count)
{
    size_t i;

    if (!lines) {
        return;
    }
    for (i = 0; i < line_count; ++i) {
        yvex_cli_out_line(fp, lines[i]);
    }
}

void yvex_cli_out_kv_str(FILE *fp, const char *key, const char *value)
{
    (void)yvex_cli_out_writef(fp, "%s: %s\n", key ? key : "", value ? value : "");
}

void yvex_cli_out_kv_bool(FILE *fp, const char *key, int value)
{
    yvex_cli_out_kv_str(fp, key, value ? "true" : "false");
}

int yvex_cli_out_fields(FILE *fp,
                        const void *object,
                        const yvex_cli_field_spec *fields,
                        size_t field_count)
{
    const unsigned char *base = object;
    size_t i;

    if (!object || (!fields && field_count != 0u)) {
        return -1;
    }
    for (i = 0; i < field_count; ++i) {
        const yvex_cli_field_spec *field = &fields[i];
        const void *value = base + field->offset;
        const char *text;
        int rc;

        switch (field->kind) {
        case YVEX_CLI_FIELD_TEXT:
            text = *(const char *const *)value;
            rc = yvex_cli_out_writef(fp, "%s: %s\n", field->key,
                                     text && text[0]
                                         ? text
                                         : (field->fallback ? field->fallback : "unknown"));
            break;
        case YVEX_CLI_FIELD_TEXT_ARRAY:
            text = value;
            rc = yvex_cli_out_writef(fp, "%s: %s\n", field->key,
                                     text[0]
                                         ? text
                                         : (field->fallback ? field->fallback : "unknown"));
            break;
        case YVEX_CLI_FIELD_U64:
            rc = yvex_cli_out_writef(fp, "%s: %llu\n", field->key,
                                     *(const unsigned long long *)value);
            break;
        case YVEX_CLI_FIELD_U32:
            rc = yvex_cli_out_writef(fp, "%s: %u\n", field->key,
                                     *(const unsigned int *)value);
            break;
        case YVEX_CLI_FIELD_I32:
            rc = yvex_cli_out_writef(fp, "%s: %d\n", field->key, *(const int *)value);
            break;
        case YVEX_CLI_FIELD_BOOL:
            rc = yvex_cli_out_writef(fp, "%s: %s\n", field->key,
                                     *(const int *)value ? "true" : "false");
            break;
        case YVEX_CLI_FIELD_DOUBLE:
            rc = yvex_cli_out_writef(fp, "%s: %.17g\n", field->key,
                                     *(const double *)value);
            break;
        case YVEX_CLI_FIELD_FLOAT9:
            rc = yvex_cli_out_writef(fp, "%s: %.9g\n", field->key,
                                     *(const double *)value);
            break;
        case YVEX_CLI_FIELD_HEX64:
            rc = yvex_cli_out_writef(fp, "%s: %016llx\n", field->key,
                                     *(const unsigned long long *)value);
            break;
        default:
            return -1;
        }
        if (rc < 0) {
            return -1;
        }
    }
    return 0;
}

int print_yvex_error(const yvex_error *err, int exit_code)
{
    yvex_cli_out_writef(stderr, "yvex: %s: %s\n", yvex_error_where(err),
                        yvex_error_message(err));
    return exit_code;
}

int exit_for_status(int status)
{
    switch (status) {
    case YVEX_ERR_INVALID_ARG:
        return 2;
    case YVEX_ERR_IO:
        return 3;
    case YVEX_ERR_FORMAT:
    case YVEX_ERR_BOUNDS:
        return 4;
    case YVEX_ERR_UNSUPPORTED:
        return 5;
    default:
        return 1;
    }
}

/*
 * Parse one complete unsigned integer without accepting signs or suffixes.
 *
 * Returns false and leaves result ownership with the caller.
 */
int parse_ull_allow_zero(const char *text, unsigned long long *out)
{
    char *end = NULL;
    unsigned long long value;

    if (!text || !out || text[0] == '\0' || text[0] == '-') return 0;
    errno = 0;
    value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') return 0;
    *out = value;
    return 1;
}

int parse_positive_ull(const char *text, unsigned long long *out)
{
    return parse_ull_allow_zero(text, out) && *out != 0;
}

int parse_uint_allow_zero(const char *text, unsigned int *out)
{
    unsigned long long value;

    if (!out || !parse_ull_allow_zero(text, &value) || value > UINT32_MAX) return 0;
    *out = (unsigned int)value;
    return 1;
}

void print_quoted_bytes(const char *data, unsigned long long len)
{
    unsigned long long i;

    yvex_cli_out_writef(stdout, "\"");
    for (i = 0; i < len; ++i) {
        unsigned char ch = (unsigned char)data[i];
        if (ch == '"' || ch == '\\') {
            yvex_cli_out_writef(stdout, "\\%c", (int)ch);
        } else if (ch == '\n') {
            yvex_cli_out_writef(stdout, "\\n");
        } else if (ch == '\r') {
            yvex_cli_out_writef(stdout, "\\r");
        } else if (ch == '\t') {
            yvex_cli_out_writef(stdout, "\\t");
        } else if (ch < 32 || ch > 126) {
            yvex_cli_out_writef(stdout, "\\x%02x", (unsigned int)ch);
        } else {
            yvex_cli_out_writef(stdout, "%c", (int)ch);
        }
    }
    yvex_cli_out_writef(stdout, "\"");
}

int open_artifact_for_gguf(const char *path, yvex_artifact **artifact, yvex_error *err)
{
    yvex_artifact_options options;
    yvex_model_ref ref;
    int rc;

    memset(&options, 0, sizeof(options));
    memset(&ref, 0, sizeof(ref));
    rc = yvex_model_ref_resolve(&ref, path, NULL, err);
    if (rc != YVEX_OK) return rc;
    options.path = ref.path;
    options.readonly = 1;
    rc = yvex_artifact_open(artifact, &options, err);
    yvex_model_ref_clear(&ref);
    return rc;
}

void print_tensor_dims(const unsigned long long *dims, unsigned int rank)
{
    unsigned int i;

    yvex_cli_out_writef(stdout, "[");
    for (i = 0; i < rank; ++i) {
        if (i > 0) yvex_cli_out_writef(stdout, ",");
        yvex_cli_out_writef(stdout, "%llu", dims[i]);
    }
    yvex_cli_out_writef(stdout, "]");
}

void print_native_dims(const unsigned long long *dims, unsigned int rank)
{
    print_tensor_dims(dims, rank);
}

void print_token_ids(const yvex_tokens *tokens)
{
    unsigned long long i;

    yvex_cli_out_writef(stdout, "ids:");
    for (i = 0; i < tokens->len; ++i) yvex_cli_out_writef(stdout, " %u", tokens->ids[i]);
    yvex_cli_out_writef(stdout, "\n");
}

int parse_id_list(const char *text, unsigned int **out_ids, unsigned long long *out_len)
{
    unsigned int *ids = NULL;
    unsigned long long len = 0;
    unsigned long long capacity = 0;
    const char *cursor = text;

    if (!text || !out_ids || !out_len) return 0;
    *out_ids = NULL;
    *out_len = 0;
    while (*cursor) {
        char *end = NULL;
        unsigned long value = strtoul(cursor, &end, 10);
        unsigned int *next;

        if (end == cursor || value > UINT32_MAX) goto fail;
        if (len == capacity) {
            unsigned long long next_capacity = capacity == 0 ? 8 : capacity * 2u;
            if (next_capacity > (unsigned long long)(SIZE_MAX / sizeof(*ids))) goto fail;
            next = realloc(ids, (size_t)next_capacity * sizeof(*ids));
            if (!next) goto fail;
            ids = next;
            capacity = next_capacity;
        }
        ids[len++] = (unsigned int)value;
        if (*end == ',') {
            cursor = end + 1;
        } else if (*end == '\0') {
            cursor = end;
        } else {
            goto fail;
        }
    }
    if (len == 0) goto fail;
    *out_ids = ids;
    *out_len = len;
    return 1;

fail:
    free(ids);
    return 0;
}

/*
 * Parse an exact-rank comma-separated tensor shape.
 *
 * Rejects invalid rank, malformed fields, zero dimensions, and overflow.
 */
int parse_dims_csv(const char *text, unsigned int rank, unsigned long long dims[4])
{
    const char *cursor = text;
    char *end = NULL;
    unsigned int i;

    if (!text || !dims || rank == 0 || rank > 4u) return 0;
    memset(dims, 0, 4u * sizeof(*dims));
    for (i = 0; i < rank; ++i) {
        errno = 0;
        dims[i] = strtoull(cursor, &end, 10);
        if (errno != 0 || end == cursor || dims[i] == 0) return 0;
        if (i + 1u < rank) {
            if (*end != ',') return 0;
            cursor = end + 1;
        } else if (*end != '\0') {
            return 0;
        }
    }
    return 1;
}
