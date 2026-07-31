/*
 * Remove independent command and REPL syntax parsers.
 *
 * One immutable descriptor supplies every admitted spelling, arity, and relation. Syntax is
 * admitted before one typed runtime or offline adapter is selected.
 */

#include "src/cli/input/private.h"

#include <operator/registry.h>

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int refuse(yvex_cli_operator_invocation *out, const char *format, ...)
{
    va_list arguments;
    va_start(arguments, format);
    (void)vsnprintf(out->message, sizeof(out->message), format, arguments);
    va_end(arguments);
    return 2;
}

static int metadata_contains(const char *values, const char *spelling)
{
    size_t extent = strlen(spelling);
    const char *cursor = values;
    if (!strcmp(values, "none")) return 0;
    while (*cursor) {
        const char *end = strchr(cursor, '|');
        size_t count = end ? (size_t)(end - cursor) : strlen(cursor);
        if (count == extent && !memcmp(cursor, spelling, count)) return 1;
        if (!end) break;
        cursor = end + 1;
    }
    return 0;
}

static int value_valid(const char *value, const char *type, const char *range,
                       const char *enum_values)
{
    char *end = NULL;
    if (!strcmp(type, "enum")) return metadata_contains(enum_values, value);
    if (!strcmp(type, "u64")) {
        unsigned long long parsed, minimum = 0u, maximum = ULLONG_MAX;
        const char *separator;
        char lower[32], upper[32];
        size_t lower_count;
        if (*value == '-') return 0;
        errno = 0;
        parsed = strtoull(value, &end, 10);
        if (errno || !end || *end) return 0;
        separator = strcmp(range, "delegated") ? strstr(range, "..") : NULL;
        if (!separator) return 1;
        lower_count = (size_t)(separator - range);
        if (!lower_count || lower_count >= sizeof(lower) ||
            strlen(separator + 2u) >= sizeof(upper))
            return 0;
        memcpy(lower, range, lower_count);
        lower[lower_count] = '\0';
        (void)snprintf(upper, sizeof(upper), "%s", separator + 2u);
        errno = 0;
        minimum = strtoull(lower, &end, 10);
        if (errno || !end || *end) return 0;
        errno = 0;
        maximum = strtoull(upper, &end, 10);
        return !errno && end && !*end && minimum <= maximum &&
               parsed >= minimum && parsed <= maximum;
    }
    if (!strcmp(type, "number")) {
        double parsed, minimum, maximum;
        const char *separator = strcmp(range, "delegated") ? strstr(range, "..") : NULL;
        errno = 0;
        parsed = strtod(value, &end);
        if (errno || !end || *end || !isfinite(parsed)) return 0;
        if (!separator) return 1;
        errno = 0;
        minimum = strtod(range, &end);
        if (errno || end != separator) return 0;
        errno = 0;
        maximum = strtod(separator + 2u, &end);
        return !errno && end && !*end && isfinite(minimum) && isfinite(maximum) &&
               minimum <= maximum && parsed >= minimum && parsed <= maximum;
    }
    return 1;
}

static const yvex_operator_argument_descriptor *argument_at(
    const yvex_operator_argument_descriptor *arguments, size_t count,
    size_t position)
{
    size_t index, offset = 0u;
    for (index = 0u; index < count; ++index) {
        if (!strcmp(arguments[index].multiplicity, "many") || offset == position)
            return &arguments[index];
        offset++;
    }
    return NULL;
}

static void argument_bounds(const yvex_operator_argument_descriptor *arguments,
                            size_t count, size_t *minimum, size_t *maximum,
                            int *unlimited)
{
    size_t index;
    *minimum = 0u;
    *maximum = 0u;
    *unlimited = 0;
    for (index = 0u; index < count; ++index) {
        if (arguments[index].required) (*minimum)++;
        if (!strcmp(arguments[index].multiplicity, "many")) *unlimited = 1;
        else (*maximum)++;
    }
}

static const yvex_operator_flag_descriptor *flag_find(
    const yvex_operator_descriptor *operation, const char *spelling,
    size_t *position)
{
    size_t index;
    for (index = 0u; index < operation->flag_count; ++index) {
        const yvex_operator_flag_descriptor *flag = &operation->flags[index];
        if (!strcmp(flag->name, spelling) || metadata_contains(flag->aliases, spelling)) {
            *position = index;
            return flag;
        }
    }
    return NULL;
}

int yvex_cli_operator_argv_parse(const yvex_operator_descriptor *operation,
                                 int argc, char **argv, size_t consumed,
                                 yvex_cli_operator_invocation *out)
{
    unsigned char *seen;
    size_t minimum, maximum, positionals = 0u, index;
    int unlimited, status = 0;
    if (!operation || !argv || !out) return 2;
    memset(out, 0, sizeof(*out));
    seen = calloc(operation->flag_count ? operation->flag_count : 1u, 1u);
    if (!seen) return refuse(out, "parser allocation failed");
    argument_bounds(operation->arguments, operation->argument_count,
                    &minimum, &maximum, &unlimited);
    for (index = consumed + 1u; index < (size_t)argc; ++index) {
        const char *value = argv[index];
        size_t flag_index = 0u;
        const yvex_operator_flag_descriptor *flag;
        if (value[0] != '-') {
            const yvex_operator_argument_descriptor *argument =
                argument_at(operation->arguments, operation->argument_count, positionals);
            if (argument && !value_valid(value, argument->value_type,
                                         argument->range, argument->enum_values)) {
                status = refuse(out, "invalid %s: %s", argument->name, value);
                break;
            }
            positionals++;
            continue;
        }
        flag = flag_find(operation, value, &flag_index);
        if (!flag) {
            status = refuse(out, "unknown flag: %s", value);
            break;
        }
        if (!strcmp(flag->name, "--help")) out->help_requested = 1;
        if (seen[flag_index] && strcmp(flag->multiplicity, "repeatable")) {
            status = refuse(out, "duplicate flag: %s", flag->name);
            break;
        }
        seen[flag_index] = 1u;
        if (flag->takes_value) {
            if (++index >= (size_t)argc) {
                status = refuse(out, "%s requires a value", flag->name);
                break;
            }
            if (!value_valid(argv[index], flag->value_type, flag->range,
                             flag->enum_values)) {
                status = refuse(out, "invalid value for %s: %s", flag->name,
                                argv[index]);
                break;
            }
        }
    }
    if (!status && !out->help_requested &&
        (positionals < minimum || (!unlimited && positionals > maximum))) {
        if (unlimited)
            status = refuse(out, "expected %zu or more positional arguments, received %zu",
                            minimum, positionals);
        else if (minimum == maximum)
            status = refuse(out, "expected %zu positional argument%s, received %zu",
                            minimum, minimum == 1u ? "" : "s", positionals);
        else
            status = refuse(out, "expected %zu to %zu positional arguments, received %zu",
                            minimum, maximum, positionals);
    }
    if (!status && !out->help_requested) {
        for (index = 0u; index < operation->flag_count && !status; ++index) {
            const yvex_operator_flag_descriptor *flag = &operation->flags[index];
            size_t relation;
            if (flag->required && !seen[index]) {
                status = refuse(out, "required flag missing: %s", flag->name);
                break;
            }
            if (!seen[index]) continue;
            for (relation = 0u; relation < operation->flag_count; ++relation) {
                const char *other = operation->flags[relation].name;
                if (seen[relation] && metadata_contains(flag->conflicts, other)) {
                    status = refuse(out, "%s conflicts with %s", flag->name, other);
                    break;
                }
                if (!seen[relation] && metadata_contains(flag->dependencies, other)) {
                    status = refuse(out, "%s requires %s", flag->name, other);
                    break;
                }
            }
        }
    }
    free(seen);
    return status;
}

int yvex_cli_operator_slash_parse(const yvex_operator_descriptor *operation,
                                  const char *text,
                                  yvex_cli_operator_invocation *out)
{
    char *save = NULL, *token;
    size_t minimum, maximum, index;
    int unlimited;
    if (!operation || !out) return 2;
    memset(out, 0, sizeof(*out));
    if (text && text[0]) {
        out->argument_storage = strdup(text);
        if (!out->argument_storage) return refuse(out, "parser allocation failed");
        for (token = strtok_r(out->argument_storage, " \t", &save); token;
             token = strtok_r(NULL, " \t", &save)) {
            if (out->argument_count == YVEX_OPERATOR_ARGUMENT_MAX)
                return refuse(out, "too many slash arguments");
            out->arguments[out->argument_count++] = token;
        }
    }
    argument_bounds(operation->slash_arguments, operation->slash_argument_count,
                    &minimum, &maximum, &unlimited);
    if (out->argument_count < minimum ||
        (!unlimited && out->argument_count > maximum)) {
        if (unlimited)
            return refuse(out, "expected %zu or more slash arguments, received %zu",
                          minimum, out->argument_count);
        if (minimum == maximum)
            return refuse(out, "expected %zu slash argument%s, received %zu",
                          minimum, minimum == 1u ? "" : "s", out->argument_count);
        return refuse(out, "expected %zu to %zu slash arguments, received %zu",
                      minimum, maximum, out->argument_count);
    }
    for (index = 0u; index < out->argument_count; ++index) {
        const yvex_operator_argument_descriptor *argument =
            argument_at(operation->slash_arguments,
                        operation->slash_argument_count, index);
        if (!argument || !value_valid(out->arguments[index], argument->value_type,
                                      argument->range, argument->enum_values))
            return refuse(out, "invalid slash argument: %s", out->arguments[index]);
    }
    return 0;
}

void yvex_cli_operator_invocation_close(yvex_cli_operator_invocation *invocation)
{
    if (!invocation) return;
    free(invocation->argument_storage);
    memset(invocation, 0, sizeof(*invocation));
}
