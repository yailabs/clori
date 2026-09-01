/* Resolve memorable model selectors to exact deployment profiles without weakening lineage. */
#define _POSIX_C_SOURCE 200809L
#include "src/cli/model_artifacts/private.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    unsigned long long model_index;
    const yvex_model_library_entry *model;
    unsigned long long profile_count;
    char ordinal[16], format[32], precision[96], size[32], variants[24];
} product_model_candidate;

static void selection_size(char out[32], unsigned long long bytes)
{
    static const char *const units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double value = (double)bytes;
    unsigned int unit = 0u;
    if (!bytes) {
        snprintf(out, 32u, "--");
        return;
    }
    while (value >= 1024.0 && unit + 1u < sizeof(units) / sizeof(units[0])) {
        value /= 1024.0;
        unit++;
    }
    if (unit) snprintf(out, 32u, "%.2f %s", value, units[unit]);
    else snprintf(out, 32u, "%llu B", bytes);
}

static const yvex_model_artifact_fact *selection_artifact(
    const yvex_model_library *library, unsigned long long model_index,
    const yvex_model_runtime_profile_fact *profile)
{
    unsigned long long index;
    for (index = 0u; index < yvex_model_library_artifact_count(library, model_index);
         ++index) {
        const yvex_model_artifact_fact *artifact =
            yvex_model_library_artifact_at(library, model_index, index);
        if (!strcmp(artifact->identity, profile->artifact_identity)) return artifact;
    }
    return NULL;
}

/* One porcelain choice is one physical representation and execution class.
 * Re-registered deployment bindings for that choice are historical revisions,
 * not additional models or variants.  Registry order is the durable revision
 * order, so the last launchable row is the current deployment projection while
 * every exact profile remains available through the advanced profile surface. */
static int profile_product_choice_equal(
    const yvex_model_runtime_profile_fact *left,
    const yvex_model_runtime_profile_fact *right)
{
    if (strcmp(left->artifact_identity, right->artifact_identity) ||
        strcmp(left->backend, right->backend) ||
        strcmp(left->engine_kind, right->engine_kind) ||
        strcmp(left->execution_strategy, right->execution_strategy))
        return 0;
    if (left->artifact_identity[0]) return 1;
    return !strcmp(left->profile, right->profile) &&
           !strcmp(left->installation, right->installation) &&
           !strcmp(left->runtime_target, right->runtime_target);
}

static void selection_variant(char out[YVEX_REMOTE_PRECISION_CAP + 24u],
                              const yvex_model_runtime_profile_fact *profile,
                              const yvex_model_artifact_fact *artifact)
{
    const char *base = strrchr(profile->runtime_binding, '/');
    const char *kind = artifact && artifact->physical_variant[0]
                           ? artifact->physical_variant
                           : artifact && artifact->artifact_class[0]
                                 ? artifact->artifact_class
                                 : profile->artifact_class[0]
                                       ? profile->artifact_class
                                       : profile->profile[0] ? profile->profile : "default";
    base = artifact && artifact->identity[0] ? artifact->identity : NULL;
    if (base)
        snprintf(out, YVEX_REMOTE_PRECISION_CAP + 24u, "%.96s@%.8s", kind,
                 base);
    else {
        base = strrchr(profile->runtime_binding, '/');
        base = base ? base + 1 : profile->runtime_binding;
        if (base[0])
            snprintf(out, YVEX_REMOTE_PRECISION_CAP + 24u, "%.96s@%.8s", kind,
                     base);
        else snprintf(out, YVEX_REMOTE_PRECISION_CAP + 24u, "%.119s", kind);
    }
}

static void profile_candidate_set(
    yvex_cli_model_profile_candidate *candidate,
    const yvex_model_library *library,
    unsigned long long model_index,
    unsigned long long profile_index,
    const yvex_model_runtime_profile_fact *profile)
{
    candidate->model_index = model_index;
    candidate->profile_index = profile_index;
    candidate->model = yvex_model_library_at(library, model_index);
    candidate->profile = profile;
    candidate->artifact = selection_artifact(library, model_index, profile);
    selection_variant(candidate->variant, profile, candidate->artifact);
    if (!strcmp(profile->profile, "composite")) {
        snprintf(candidate->format, sizeof(candidate->format), "%s", "composite");
        snprintf(candidate->precision, sizeof(candidate->precision), "%s",
                 "component-defined");
        snprintf(candidate->size, sizeof(candidate->size), "%s", "varies");
        return;
    }
    snprintf(candidate->format, sizeof(candidate->format), "%s",
             candidate->artifact && candidate->artifact->format[0]
                 ? candidate->artifact->format : "package");
    yvex_cli_precision_format(
        candidate->precision, sizeof(candidate->precision),
        candidate->artifact && candidate->artifact->physical_variant[0]
            ? candidate->artifact->physical_variant
            : candidate->artifact ? candidate->artifact->artifact_class
                                  : profile->artifact_class);
    selection_size(candidate->size,
                   candidate->artifact ? candidate->artifact->file_size : 0u);
}

unsigned long long yvex_cli_model_profile_candidates(
    const yvex_model_library *library, unsigned long long model_index, int text_only,
    yvex_cli_model_profile_candidate out[YVEX_MODELS_ARTIFACT_ROWS_CAP])
{
    unsigned long long index, count = 0u, selected = 0u;
    for (index = 0u; index < yvex_model_library_profile_count(library, model_index);
         ++index) {
        const yvex_model_runtime_profile_fact *profile =
            yvex_model_library_profile_at(library, model_index, index);
        yvex_cli_model_profile_candidate *candidate = NULL;
        unsigned long long choice;
        if (!profile->launchable || (text_only && strcmp(profile->engine_kind, "text")))
            continue;
        for (choice = 0u; choice < count; ++choice) {
            if (profile_product_choice_equal(out[choice].profile, profile)) {
                candidate = &out[choice];
                break;
            }
        }
        if (!candidate) {
            if (count == YVEX_MODELS_ARTIFACT_ROWS_CAP) break;
            candidate = &out[count];
            memset(candidate, 0, sizeof(*candidate));
            snprintf(candidate->ordinal, sizeof(candidate->ordinal), "%u",
                     (unsigned int)(count + 1u));
            count++;
        }
        candidate->revision_count++;
        profile_candidate_set(candidate, library, model_index, index, profile);
    }
    for (index = 1u; index < count; ++index)
        if (out[index].profile_index > out[selected].profile_index) selected = index;
    if (count) out[selected].selected = 1;
    return count;
}

static unsigned long long selectable_profile_count(
    const yvex_model_library *library, unsigned long long model_index, int text_only)
{
    yvex_cli_model_profile_candidate candidates[YVEX_MODELS_ARTIFACT_ROWS_CAP];
    return yvex_cli_model_profile_candidates(library, model_index, text_only, candidates);
}

static unsigned long long model_candidates_build(
    const yvex_model_library *library, int text_only,
    product_model_candidate out[YVEX_MODELS_ARTIFACT_ROWS_CAP])
{
    unsigned long long index, count = 0u;
    for (index = 0u; index < yvex_model_library_count(library); ++index) {
        unsigned long long profiles = selectable_profile_count(library, index, text_only);
        product_model_candidate *candidate;
        if (!profiles) continue;
        if (count == YVEX_MODELS_ARTIFACT_ROWS_CAP) break;
        candidate = &out[count];
        memset(candidate, 0, sizeof(*candidate));
        candidate->model_index = index;
        candidate->model = yvex_model_library_at(library, index);
        candidate->profile_count = profiles;
        snprintf(candidate->ordinal, sizeof(candidate->ordinal), "%llu", count + 1u);
        snprintf(candidate->variants, sizeof(candidate->variants), "%llu", profiles);
        {
            yvex_cli_model_profile_candidate choices[YVEX_MODELS_ARTIFACT_ROWS_CAP];
            unsigned long long choice, choice_count = yvex_cli_model_profile_candidates(
                library, index, text_only, choices);
            for (choice = 0u; choice < choice_count; ++choice)
                if (choices[choice].selected) break;
            if (choice == choice_count) choice = 0u;
            snprintf(candidate->format, sizeof(candidate->format), "%s",
                     choices[choice].format);
            snprintf(candidate->precision, sizeof(candidate->precision), "%s",
                     choices[choice].precision);
            snprintf(candidate->size, sizeof(candidate->size), "%s",
                     choices[choice].size);
        }
        count++;
    }
    return count;
}

static int selection_read(char line[YVEX_MODEL_LIBRARY_ID_CAP], const char *label,
                          const char *object)
{
    char *newline;
    yvex_cli_out_writef(stdout, "%s (number or exact %s, q to cancel) > ",
                        label, object);
    yvex_cli_out_flush(stdout);
    if (!fgets(line, YVEX_MODEL_LIBRARY_ID_CAP, stdin)) return 0;
    newline = strpbrk(line, "\r\n");
    if (newline) *newline = '\0';
    return line[0] && strcmp(line, "q") && strcmp(line, "Q");
}

static int model_choice_parse(const yvex_model_library *library,
                              const product_model_candidate *candidates,
                              unsigned long long count, const char *text,
                              unsigned long long *model_index)
{
    char *end = NULL;
    unsigned long long ordinal;
    size_t index;
    errno = 0;
    ordinal = strtoull(text, &end, 10);
    if (!errno && end != text && !*end && ordinal && ordinal <= count) {
        *model_index = candidates[ordinal - 1u].model_index;
        return 1;
    }
    if (yvex_cli_model_find(library, text, model_index) == 1) {
        for (index = 0u; index < count; ++index)
            if (candidates[index].model_index == *model_index) return 1;
    }
    return 0;
}

static int model_selector_render(const yvex_model_library *library, int text_only,
                                 unsigned long long *model_index)
{
    static const yvex_cli_table_column columns[] = {
        {"#", 1u, 3u, YVEX_CLI_TABLE_RIGHT, 0},
        {"MODEL", 12u, 36u, YVEX_CLI_TABLE_LEFT, 0},
        {"FORMAT", 6u, 12u, YVEX_CLI_TABLE_LEFT, 0},
        {"QUANT/PRECISION", 8u, 22u, YVEX_CLI_TABLE_LEFT, 0},
        {"SIZE", 6u, 12u, YVEX_CLI_TABLE_RIGHT, 0},
        {"CHOICES", 3u, 7u, YVEX_CLI_TABLE_RIGHT, 0}
    };
    product_model_candidate candidates[YVEX_MODELS_ARTIFACT_ROWS_CAP];
    yvex_cli_table_cell cells[YVEX_MODELS_ARTIFACT_ROWS_CAP][6];
    yvex_cli_table_row rows[YVEX_MODELS_ARTIFACT_ROWS_CAP];
    char line[YVEX_MODEL_LIBRARY_ID_CAP];
    unsigned long long count, index;
    count = model_candidates_build(library, text_only, candidates);
    if (!count) {
        yvex_cli_out_fputs("yvex: no launchable models are known locally\n", stderr);
        return 1;
    }
    for (index = 0u; index < count; ++index) {
        cells[index][0] = (yvex_cli_table_cell){candidates[index].ordinal,
                                                YVEX_CLI_TABLE_DIM};
        cells[index][1] = (yvex_cli_table_cell){yvex_cli_model_selector(
                                                    candidates[index].model),
                                                YVEX_CLI_TABLE_ACCENT};
        cells[index][2] = (yvex_cli_table_cell){candidates[index].format,
                                                YVEX_CLI_TABLE_PLAIN};
        cells[index][3] = (yvex_cli_table_cell){candidates[index].precision,
                                                YVEX_CLI_TABLE_PLAIN};
        cells[index][4] = (yvex_cli_table_cell){candidates[index].size,
                                                YVEX_CLI_TABLE_PLAIN};
        cells[index][5] = (yvex_cli_table_cell){candidates[index].variants,
                                                YVEX_CLI_TABLE_PLAIN};
        rows[index] = (yvex_cli_table_row){cells[index], NULL, YVEX_CLI_TABLE_PLAIN};
    }
    yvex_cli_out_fputs("Select model\n\n", stdout);
    (void)yvex_cli_table_render(stdout, columns, 6u, rows, (size_t)count);
    for (;;) {
        if (!selection_read(line, "Model", "model")) return 2;
        if (model_choice_parse(library, candidates, count, line, model_index)) return 0;
        yvex_cli_out_fputs("Unknown or unavailable model; choose a listed number or exact name.\n",
                           stdout);
    }
}

static void profile_selector_render(const yvex_cli_model_profile_candidate *candidates,
                                    unsigned long long count)
{
    static const yvex_cli_table_column columns[] = {
        {"#", 1u, 3u, YVEX_CLI_TABLE_RIGHT, 0},
        {"ROLE", 8u, 9u, YVEX_CLI_TABLE_LEFT, 0},
        {"VARIANT", 18u, 44u, YVEX_CLI_TABLE_LEFT, 0},
        {"FORMAT", 5u, 10u, YVEX_CLI_TABLE_LEFT, 0},
        {"QUANT/PRECISION", 8u, 22u, YVEX_CLI_TABLE_LEFT, 0},
        {"SIZE", 6u, 12u, YVEX_CLI_TABLE_RIGHT, 0},
        {"BACKEND", 5u, 10u, YVEX_CLI_TABLE_LEFT, 0},
        {"MODE", 4u, 11u, YVEX_CLI_TABLE_LEFT, 0}
    };
    yvex_cli_table_cell cells[YVEX_MODELS_ARTIFACT_ROWS_CAP][8];
    yvex_cli_table_row rows[YVEX_MODELS_ARTIFACT_ROWS_CAP];
    unsigned long long index;
    for (index = 0u; index < count; ++index) {
        const yvex_cli_model_profile_candidate *candidate = &candidates[index];
        cells[index][0] = (yvex_cli_table_cell){candidate->ordinal, YVEX_CLI_TABLE_DIM};
        cells[index][1] = (yvex_cli_table_cell){
            candidate->selected ? "selected" : "alternate",
            candidate->selected ? YVEX_CLI_TABLE_SUCCESS : YVEX_CLI_TABLE_PLAIN};
        cells[index][2] = (yvex_cli_table_cell){candidate->variant,
                                                YVEX_CLI_TABLE_ACCENT};
        cells[index][3] = (yvex_cli_table_cell){candidate->format,
                                                YVEX_CLI_TABLE_PLAIN};
        cells[index][4] = (yvex_cli_table_cell){candidate->precision,
                                                YVEX_CLI_TABLE_PLAIN};
        cells[index][5] = (yvex_cli_table_cell){candidate->size,
                                                YVEX_CLI_TABLE_PLAIN};
        cells[index][6] = (yvex_cli_table_cell){candidate->profile->backend,
                                                YVEX_CLI_TABLE_PLAIN};
        cells[index][7] = (yvex_cli_table_cell){candidate->profile->engine_kind,
                                                YVEX_CLI_TABLE_PLAIN};
        rows[index] = (yvex_cli_table_row){cells[index], NULL, YVEX_CLI_TABLE_PLAIN};
    }
    (void)yvex_cli_table_render(stdout, columns, 8u, rows, (size_t)count);
}

static int profile_matches(const yvex_cli_model_profile_candidate *candidate,
                           const char *variant)
{
    return !strcmp(candidate->variant, variant) ||
           !strcmp(candidate->profile->artifact_identity, variant) ||
           (candidate->profile->artifact_class[0] &&
            !strcmp(candidate->profile->artifact_class, variant)) ||
           (candidate->artifact && candidate->artifact->physical_variant[0] &&
            !strcmp(candidate->artifact->physical_variant, variant));
}

static int profile_choice_parse(const yvex_cli_model_profile_candidate *candidates,
                                unsigned long long count, const char *text,
                                unsigned long long *choice)
{
    char *end = NULL;
    unsigned long long index, ordinal, matches = 0u, selected = 0u;
    errno = 0;
    ordinal = strtoull(text, &end, 10);
    if (!errno && end != text && !*end && ordinal && ordinal <= count) {
        *choice = ordinal - 1u;
        return 1;
    }
    for (index = 0u; index < count; ++index)
        if (profile_matches(&candidates[index], text)) {
            selected = index;
            matches++;
        }
    if (matches == 1u) {
        *choice = selected;
        return 1;
    }
    return 0;
}

static int profile_select(const yvex_cli_model_profile_candidate *candidates,
                          unsigned long long count, const char *variant,
                          int use_selected, unsigned long long *choice)
{
    char line[YVEX_MODEL_LIBRARY_ID_CAP];
    if (variant) {
        if (profile_choice_parse(candidates, count, variant, choice)) return 0;
        yvex_cli_out_writef(stderr,
                            "yvex: model variant is unknown or ambiguous: %s\n",
                            variant);
        return 2;
    }
    if (count == 1u) {
        *choice = 0u;
        return 0;
    }
    if (use_selected) {
        unsigned long long index;
        for (index = 0u; index < count; ++index)
            if (candidates[index].selected) {
                *choice = index;
                return 0;
            }
    }
    if (!isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO)) {
        yvex_cli_out_fputs(
            "yvex: this model has multiple launchable variants\n"
            "hint: use `yvex model show MODEL` and pass one exact `--variant`\n",
            stderr);
        return 2;
    }
    yvex_cli_out_fputs("\nSelect representation and deployment\n\n", stdout);
    profile_selector_render(candidates, count);
    for (;;) {
        if (!selection_read(line, "Variant", "variant")) return 2;
        if (profile_choice_parse(candidates, count, line, choice)) return 0;
        yvex_cli_out_fputs("Unknown or ambiguous variant; choose a listed number or exact variant.\n",
                           stdout);
    }
}

static void selection_copy(yvex_cli_model_profile_selection *out,
                           const yvex_cli_model_profile_candidate *candidate)
{
    const yvex_model_artifact_fact *artifact = candidate->artifact;
    memset(out, 0, sizeof(*out));
    snprintf(out->model_selector, sizeof(out->model_selector), "%s",
             yvex_cli_model_selector(candidate->model));
    snprintf(out->model_name, sizeof(out->model_name), "%s",
             candidate->model->display_name);
    snprintf(out->family, sizeof(out->family), "%s", candidate->model->family);
    snprintf(out->profile_alias, sizeof(out->profile_alias), "%s",
             candidate->profile->alias);
    snprintf(out->variant, sizeof(out->variant), "%s", candidate->variant);
    snprintf(out->artifact_identity, sizeof(out->artifact_identity), "%s",
             candidate->profile->artifact_identity);
    snprintf(out->format, sizeof(out->format), "%s", candidate->format);
    snprintf(out->quant_precision, sizeof(out->quant_precision), "%s",
             candidate->precision);
    snprintf(out->backend, sizeof(out->backend), "%s", candidate->profile->backend);
    snprintf(out->engine_kind, sizeof(out->engine_kind), "%s",
             candidate->profile->engine_kind);
    snprintf(out->execution_strategy, sizeof(out->execution_strategy), "%s",
             candidate->profile->execution_strategy);
    snprintf(out->runtime_target, sizeof(out->runtime_target), "%s",
             candidate->profile->runtime_target);
    out->representation_bytes = artifact ? artifact->file_size : 0u;
    out->context_capacity = candidate->profile->context_capacity;
}

int yvex_cli_model_profile_select(const char *model, const char *variant,
                                  int text_only,
                                  yvex_cli_model_profile_selection *selection)
{
    yvex_model_library *library = NULL;
    yvex_local_catalog_options options = {0};
    yvex_error err;
    yvex_cli_model_profile_candidate candidates[YVEX_MODELS_ARTIFACT_ROWS_CAP];
    unsigned long long model_index, count, choice = 0u;
    int rc;
    if (!selection) return 2;
    yvex_error_clear(&err);
    rc = yvex_model_library_open(&library, &options, &err);
    if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));
    if (model) {
        int matches = yvex_cli_model_find(library, model, &model_index);
        if (matches != 1) {
            yvex_cli_out_writef(stderr, "yvex: model selector %s: %s\n", model,
                                matches ? "ambiguous; use the exact identity"
                                        : "not found");
            yvex_model_library_close(library);
            return 2;
        }
    } else {
        if (!isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO)) {
            yvex_cli_out_fputs(
                "yvex: model load requires MODEL when input is not a terminal\n",
                stderr);
            yvex_model_library_close(library);
            return 2;
        }
        rc = model_selector_render(library, text_only, &model_index);
        if (rc) {
            yvex_model_library_close(library);
            return rc;
        }
    }
    count = yvex_cli_model_profile_candidates(library, model_index, text_only,
                                               candidates);
    if (!count) {
        yvex_cli_out_writef(stderr, "yvex: model is not launchable%s: %s\n",
                            text_only ? " as text" : "",
                            yvex_cli_model_selector(yvex_model_library_at(library,
                                                                          model_index)));
        yvex_model_library_close(library);
        return 1;
    }
    rc = profile_select(candidates, count, variant, model != NULL, &choice);
    if (!rc) selection_copy(selection, &candidates[choice]);
    yvex_model_library_close(library);
    return rc;
}

int yvex_cli_model_profile_resolve_alias(
    const char *alias, yvex_cli_model_profile_selection *selection)
{
    yvex_model_library *library = NULL;
    yvex_local_catalog_options options = {0};
    yvex_error err;
    unsigned long long model_index, profile_index;
    int rc;
    if (!alias || !selection) return 2;
    yvex_error_clear(&err);
    rc = yvex_model_library_open(&library, &options, &err);
    if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));
    for (model_index = 0u; model_index < yvex_model_library_count(library);
         ++model_index) {
        for (profile_index = 0u;
             profile_index < yvex_model_library_profile_count(library, model_index);
             ++profile_index) {
            const yvex_model_runtime_profile_fact *profile =
                yvex_model_library_profile_at(library, model_index, profile_index);
            yvex_cli_model_profile_candidate candidate;
            if (strcmp(profile->alias, alias)) continue;
            memset(&candidate, 0, sizeof(candidate));
            candidate.model = yvex_model_library_at(library, model_index);
            candidate.profile = profile;
            candidate.artifact = selection_artifact(library, model_index, profile);
            selection_variant(candidate.variant, profile, candidate.artifact);
            if (!strcmp(profile->profile, "composite")) {
                snprintf(candidate.format, sizeof(candidate.format), "%s", "composite");
                snprintf(candidate.precision, sizeof(candidate.precision), "%s",
                         "component-defined");
            } else {
                snprintf(candidate.format, sizeof(candidate.format), "%s",
                         candidate.artifact && candidate.artifact->format[0]
                             ? candidate.artifact->format : "package");
                yvex_cli_precision_format(
                    candidate.precision, sizeof(candidate.precision),
                    candidate.artifact && candidate.artifact->physical_variant[0]
                        ? candidate.artifact->physical_variant
                        : candidate.artifact ? candidate.artifact->artifact_class
                                             : profile->artifact_class);
            }
            selection_copy(selection, &candidate);
            yvex_model_library_close(library);
            return 0;
        }
    }
    yvex_model_library_close(library);
    return 2;
}
