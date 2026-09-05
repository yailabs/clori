/* Derive the human model workflow from exact source, artifact, profile, and model facts. */
#define _POSIX_C_SOURCE 200809L
#include "src/cli/model_artifacts/private.h"

#include <sys/stat.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <yvex/internal/source_distribution.h>

#define PRODUCT_COLUMN_CAP 10u

typedef struct {
    const yvex_model_library_entry *model;
    const yvex_local_source_record *source;
    const yvex_model_artifact_fact *artifact;
    const yvex_model_runtime_profile_fact *profile;
    unsigned long long model_index;
    char selector[YVEX_MODEL_LIBRARY_ID_CAP];
    char origin[32];
    char format[YVEX_REMOTE_FORMAT_CAP];
    char precision[YVEX_REMOTE_PRECISION_CAP];
    char size[32];
    char state[24];
    char execution[32];
    char variants[24];
    char location[YVEX_PATH_CAP];
} product_model_fact;

typedef struct {
    char value[PRODUCT_COLUMN_CAP][YVEX_PATH_CAP];
    char secondary[YVEX_PATH_CAP + 256u];
    yvex_cli_table_cell cells[PRODUCT_COLUMN_CAP];
    yvex_cli_table_row row;
} product_table_row;

typedef struct {
    yvex_cli_loaded_model_fact models[YVEX_SERVER_IMPLEMENTATION_MAXIMUM_ENGINES];
    unsigned long long count;
} product_runtime_view;

typedef struct {
    const char *role;
    char path[YVEX_PATH_CAP];
    char format[16];
    char precision[96];
    char size[32];
    char state[24];
    unsigned long long bytes;
    int present;
} product_component_fact;

typedef struct {
    product_component_fact items[4];
    unsigned int count;
    unsigned long long bytes;
    int complete;
} product_composite_fact;

static void product_runtime_open(product_runtime_view *view)
{
    yvex_error err;
    memset(view, 0, sizeof(*view));
    yvex_error_clear(&err);
    if (yvex_cli_loaded_models_snapshot(
            view->models, YVEX_SERVER_IMPLEMENTATION_MAXIMUM_ENGINES,
            &view->count, &err) != YVEX_OK)
        view->count = 0u;
}

static const yvex_cli_loaded_model_fact *product_loaded(
    const product_runtime_view *view, const char *model,
    const char *artifact_identity, const char *profile_alias)
{
    unsigned long long index;
    if (!view) return NULL;
    for (index = 0u; index < view->count; ++index) {
        const yvex_cli_loaded_model_fact *loaded = &view->models[index];
        if (strcmp(loaded->model.model_selector, model)) continue;
        if (artifact_identity && artifact_identity[0] &&
            strcmp(loaded->model.artifact_identity, artifact_identity)) continue;
        if (profile_alias && profile_alias[0] &&
            strcmp(loaded->model.profile_alias, profile_alias)) continue;
        return loaded;
    }
    return NULL;
}

const char *yvex_cli_model_selector(const yvex_model_library_entry *model)
{
    if (model->model[0]) return model->model;
    if (model->display_name[0]) return model->display_name;
    return model->identity;
}

static const yvex_local_source_record *product_source(
    const yvex_model_library *library, unsigned long long model_index)
{
    const yvex_local_source_record *first = NULL;
    unsigned long long index;
    for (index = 0u; index < yvex_model_library_source_count(library, model_index);
         ++index) {
        const yvex_local_source_record *source =
            yvex_model_library_source_at(library, model_index, index);
        if (!first) first = source;
        if (source->path[0] && strcmp(source->acquisition_state, "source-missing"))
            return source;
    }
    return first;
}

static const yvex_model_runtime_profile_fact *product_profile(
    const yvex_model_library *library, unsigned long long model_index,
    const yvex_model_artifact_fact *artifact)
{
    const yvex_model_runtime_profile_fact *selected = NULL;
    unsigned long long index;
    for (index = 0u; index < yvex_model_library_profile_count(library, model_index);
         ++index) {
        const yvex_model_runtime_profile_fact *profile =
            yvex_model_library_profile_at(library, model_index, index);
        if (!profile->launchable) continue;
        if (artifact && strcmp(profile->artifact_identity, artifact->identity)) continue;
        if (selected && (!artifact || strcmp(selected->alias, profile->alias))) return NULL;
        selected = profile;
    }
    return selected;
}

static const char *product_origin(const yvex_local_source_record *source,
                                  const yvex_model_library_entry *model)
{
    const char *provider = source && source->provider[0] ? source->provider : model->provider;
    if (!strcmp(provider, "huggingface")) return "Hugging Face";
    if (provider[0]) return provider;
    if (source && !strcmp(source->storage_kind, "external")) return "external";
    return model->artifact_count ? "local" : "unknown";
}

static const char *product_source_state(const yvex_model_library_entry *model,
                                        const yvex_local_source_record *source)
{
    if (!source) return "UNBOUND";
    if (!strcmp(source->acquisition_state, "source-remote")) return "REMOTE";
    if (!strcmp(source->acquisition_state, "source-missing"))
        return !strcmp(source->storage_kind, "external") ? "BLOCKED" : "REMOTE";
    if (!strcmp(source->acquisition_state, "source-partial")) return "BLOCKED";
    if (!strcmp(source->storage_kind, "external")) return "EXTERNAL";
    if (!strcmp(source->storage_kind, "remote")) return "REMOTE";
    if (!source->family[0] || !strcmp(source->family, "unknown"))
        return "UNSUPPORTED";
    if (model && model->identity_kind ==
                     YVEX_MODEL_IDENTITY_PROVIDER_REPOSITORY_REVISION)
        return "UNBOUND";
    if (!strcmp(source->verification_state, "payload-verified")) return "VERIFIED";
    return source->path[0] ? "SOURCE" : "UNBOUND";
}

static const char *product_model_state(const yvex_model_library *library,
                                       unsigned long long model_index,
                                       const yvex_model_library_entry *model)
{
    unsigned long long index;
    int verified = 0, external = 0, acquired = 0, remote = 0, blocked = 0;
    if (model->profile_launchable) return "READY";
    if (model->profile_count) return "BLOCKED";
    if (model->artifact_count) return "PREPARING";
    for (index = 0u; index < yvex_model_library_source_count(library, model_index);
         ++index) {
        const yvex_local_source_record *source =
            yvex_model_library_source_at(library, model_index, index);
        if (!source) continue;
        if (!strcmp(source->acquisition_state, "source-remote") ||
            (!strcmp(source->acquisition_state, "source-missing") &&
             strcmp(source->storage_kind, "external"))) {
            remote = 1;
            continue;
        }
        if (!strcmp(source->acquisition_state, "source-partial") ||
            (!strcmp(source->acquisition_state, "source-missing") &&
             !strcmp(source->storage_kind, "external"))) {
            blocked = 1;
            continue;
        }
        if (!strcmp(source->storage_kind, "external")) external = 1;
        else if (!strcmp(source->storage_kind, "remote")) remote = 1;
        else if (!strcmp(source->verification_state, "payload-verified"))
            verified = 1;
        else if (source->path[0]) acquired = 1;
    }
    if ((verified || acquired || external) &&
        (!model->family[0] || !strcmp(model->family, "unknown")))
        return "UNSUPPORTED";
    if ((verified || acquired || external) &&
        model->identity_kind == YVEX_MODEL_IDENTITY_PROVIDER_REPOSITORY_REVISION)
        return "UNBOUND";
    if (verified) return "VERIFIED";
    if (external) return "EXTERNAL";
    if (acquired) return "SOURCE";
    if (remote || model->remote_available) return "REMOTE";
    if (blocked) return "BLOCKED";
    return "UNBOUND";
}

static unsigned long long product_representation_count(
    const yvex_model_library_entry *model)
{
    if (!model) return 0u;
    return model->artifact_count ? model->artifact_count : model->source_count;
}

static yvex_cli_table_tone product_state_tone(const char *state)
{
    if (!strcmp(state, "READY") || !strcmp(state, "LOADED") ||
        !strcmp(state, "VERIFIED")) return YVEX_CLI_TABLE_SUCCESS;
    if (!strcmp(state, "BLOCKED") || !strcmp(state, "FAILED"))
        return YVEX_CLI_TABLE_ERROR;
    return YVEX_CLI_TABLE_WARNING;
}

static void product_size(char out[32], unsigned long long bytes, int known)
{
    static const char *const units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double value = (double)bytes;
    unsigned int unit = 0u;
    if (!known) {
        snprintf(out, 32u, "%s", "--");
        return;
    }
    while (value >= 1024.0 && unit + 1u < sizeof(units) / sizeof(units[0])) {
        value /= 1024.0;
        unit++;
    }
    if (!unit) snprintf(out, 32u, "%llu %s", bytes, units[unit]);
    else snprintf(out, 32u, "%.2f %s", value, units[unit]);
}

static void product_component_precision(product_component_fact *component)
{
    yvex_artifact_options options = {component->path, 1, 0};
    yvex_artifact *artifact = NULL;
    yvex_gguf *gguf = NULL;
    yvex_error err;
    const char *types[8];
    unsigned int type_count = 0u;
    unsigned long long index;
    size_t used = 0u;
    yvex_error_clear(&err);
    if (yvex_artifact_open(&artifact, &options, &err) != YVEX_OK ||
        yvex_gguf_open(&gguf, artifact, &err) != YVEX_OK) {
        snprintf(component->precision, sizeof(component->precision), "%s",
                 "metadata unavailable");
        yvex_gguf_close(gguf);
        yvex_artifact_close(artifact);
        return;
    }
    for (index = 0u; index < yvex_gguf_tensor_count(gguf); ++index) {
        const yvex_gguf_tensor_info *tensor = yvex_gguf_tensor_at(gguf, index);
        unsigned int type;
        if (!tensor || !tensor->ggml_type_name) continue;
        for (type = 0u; type < type_count; ++type)
            if (!strcmp(types[type], tensor->ggml_type_name)) break;
        if (type == type_count && type_count < 8u)
            types[type_count++] = tensor->ggml_type_name;
    }
    component->precision[0] = '\0';
    for (index = 0u; index < type_count; ++index) {
        int written = snprintf(component->precision + used,
                               sizeof(component->precision) - used,
                               "%s%s", index ? "/" : "", types[index]);
        if (written < 0 || (size_t)written >= sizeof(component->precision) - used)
            break;
        used += (size_t)written;
    }
    if (!component->precision[0])
        snprintf(component->precision, sizeof(component->precision), "%s", "unknown");
    yvex_gguf_close(gguf);
    yvex_artifact_close(artifact);
}

static int product_composite_build(product_composite_fact *out,
                                   const yvex_model_runtime_profile_fact *profile,
                                   int inspect_precision)
{
    const yvex_component_variant_adapter *adapter;
    yvex_media_target_profile target = {0};
    const char *roles[] = {"text encoder", "transformer", "video VAE", "audio VAE"};
    const char *paths[4];
    yvex_error err;
    unsigned int index;
    if (!out) return 0;
    memset(out, 0, sizeof(*out));
    if (!profile || strcmp(profile->profile, "composite") ||
        !profile->installation[0] || !profile->runtime_target[0])
        return 0;
    adapter = yvex_graph_component_variant_find(profile->runtime_target);
    if (!adapter || !adapter->media_target_profile) return 0;
    yvex_error_clear(&err);
    if (adapter->media_target_profile(&target, &err) != YVEX_OK) return 0;
    paths[0] = target.text_artifact;
    paths[1] = target.transformer_artifact;
    paths[2] = target.video_artifact;
    paths[3] = target.audio_artifact;
    out->count = 4u;
    out->complete = 1;
    for (index = 0u; index < out->count; ++index) {
        product_component_fact *component = &out->items[index];
        struct stat status;
        component->role = roles[index];
        snprintf(component->format, sizeof(component->format), "%s", "GGUF");
        if (!paths[index] || !paths[index][0] ||
            snprintf(component->path, sizeof(component->path), "%s/%s",
                     profile->installation, paths[index]) >= (int)sizeof(component->path) ||
            stat(component->path, &status) != 0 || !S_ISREG(status.st_mode)) {
            snprintf(component->state, sizeof(component->state), "%s", "MISSING");
            snprintf(component->precision, sizeof(component->precision), "%s", "--");
            out->complete = 0;
            continue;
        }
        component->present = 1;
        component->bytes = (unsigned long long)status.st_size;
        out->bytes += component->bytes;
        product_size(component->size, component->bytes, 1);
        snprintf(component->state, sizeof(component->state), "%s", "PRESENT");
        snprintf(component->precision, sizeof(component->precision), "%s",
                 "metadata-defined");
        if (inspect_precision) product_component_precision(component);
    }
    return 1;
}

static void product_source_location(char out[YVEX_PATH_CAP],
                                    const yvex_local_source_record *source)
{
    if (!source) {
        out[0] = '\0';
        return;
    }
    if (!strcmp(source->acquisition_state, "source-missing") && source->origin_uri[0])
        snprintf(out, YVEX_PATH_CAP, "%s", source->origin_uri);
    else if (source->path[0]) snprintf(out, YVEX_PATH_CAP, "%s", source->path);
    else if (source->origin_uri[0])
        snprintf(out, YVEX_PATH_CAP, "%s", source->origin_uri);
    else if (source->repository[0])
        snprintf(out, YVEX_PATH_CAP, "hf://%s%s%s", source->repository,
                 source->revision[0] ? "@" : "", source->revision);
}

static void product_fact_build(product_model_fact *fact,
                               const yvex_model_library *library,
                               unsigned long long model_index,
                               const yvex_local_source_record *source,
                               const yvex_model_artifact_fact *artifact,
                               const product_runtime_view *runtime)
{
    const yvex_cli_loaded_model_fact *loaded;
    const yvex_model_runtime_profile_fact *selected_profile = NULL;
    unsigned long long variants;
    memset(fact, 0, sizeof(*fact));
    fact->model_index = model_index;
    fact->model = yvex_model_library_at(library, model_index);
    snprintf(fact->selector, sizeof(fact->selector), "%s",
             yvex_cli_model_selector(fact->model));
    loaded = source ? NULL : product_loaded(runtime, fact->selector,
                                            artifact ? artifact->identity : NULL, NULL);
    fact->source = source ? source : product_source(library, model_index);
    fact->artifact = artifact;
    if (!fact->artifact && loaded)
        for (variants = 0u;
             variants < yvex_model_library_artifact_count(library, model_index);
             ++variants) {
            const yvex_model_artifact_fact *candidate =
                yvex_model_library_artifact_at(library, model_index, variants);
            if (!strcmp(candidate->identity, loaded->model.artifact_identity)) {
                fact->artifact = candidate;
                break;
            }
    }
    if (!source && !fact->artifact && !loaded && fact->model->profile_launchable) {
        yvex_cli_model_profile_candidate candidates[YVEX_MODELS_ARTIFACT_ROWS_CAP];
        unsigned long long candidate_count, candidate_index;
        candidate_count = yvex_cli_model_profile_candidates(
            library, model_index, 0, candidates);
        for (candidate_index = 0u; candidate_index < candidate_count;
             ++candidate_index)
            if (candidates[candidate_index].selected) {
                fact->artifact = candidates[candidate_index].artifact;
                selected_profile = candidates[candidate_index].profile;
                break;
            }
    }
    if (!source && !fact->artifact && fact->model->artifact_count == 1u)
        fact->artifact = yvex_model_library_artifact_at(library, model_index, 0u);
    fact->profile = source ? NULL
                           : selected_profile ? selected_profile
                                              : product_profile(library, model_index,
                                                                fact->artifact);
    if (!fact->profile && fact->artifact) {
        yvex_cli_model_profile_candidate candidates[YVEX_MODELS_ARTIFACT_ROWS_CAP];
        unsigned long long candidate_count, candidate_index;
        candidate_count = yvex_cli_model_profile_candidates(
            library, model_index, 0, candidates);
        for (candidate_index = 0u; candidate_index < candidate_count;
             ++candidate_index)
            if (candidates[candidate_index].artifact &&
                !strcmp(candidates[candidate_index].artifact->identity,
                        fact->artifact->identity)) {
                fact->profile = candidates[candidate_index].profile;
                break;
            }
    }
    snprintf(fact->origin, sizeof(fact->origin), "%s",
             product_origin(fact->source, fact->model));
    if (fact->profile && !strcmp(fact->profile->profile, "composite")) {
        product_composite_fact composite;
        snprintf(fact->format, sizeof(fact->format), "%s", "composite");
        snprintf(fact->precision, sizeof(fact->precision), "%s", "mixed components");
        if (product_composite_build(&composite, fact->profile, 0) && composite.complete)
            product_size(fact->size, composite.bytes, 1);
        else snprintf(fact->size, sizeof(fact->size), "%s", "composite");
        snprintf(fact->location, sizeof(fact->location), "%s",
                 fact->profile->installation[0] ? fact->profile->installation
                                                : fact->artifact
                                                      ? fact->artifact->path : "");
    } else if (fact->artifact) {
        snprintf(fact->format, sizeof(fact->format), "%s",
                 fact->artifact->format[0] ? fact->artifact->format : "package");
        yvex_cli_precision_format(
            fact->precision, sizeof(fact->precision),
            fact->artifact->physical_variant[0] ? fact->artifact->physical_variant
                                                : fact->artifact->artifact_class);
        product_size(fact->size, fact->artifact->file_size, 1);
        snprintf(fact->location, sizeof(fact->location), "%s", fact->artifact->path);
    } else if (fact->model->artifact_count > 1u) {
        snprintf(fact->format, sizeof(fact->format), "%s", "multiple");
        snprintf(fact->precision, sizeof(fact->precision), "%s", "select variant");
        snprintf(fact->size, sizeof(fact->size), "%s", "varies");
        if (fact->source) product_source_location(fact->location, fact->source);
        else snprintf(fact->location, sizeof(fact->location), "%llu local representations",
                      fact->model->artifact_count);
    } else if (!source && fact->model->source_count > 1u) {
        snprintf(fact->format, sizeof(fact->format), "%s", "multiple");
        snprintf(fact->precision, sizeof(fact->precision), "%s", "select variant");
        snprintf(fact->size, sizeof(fact->size), "%s", "varies");
        snprintf(fact->location, sizeof(fact->location),
                 "%llu local source representations", fact->model->source_count);
    } else if (fact->source) {
        snprintf(fact->format, sizeof(fact->format), "%s",
                 fact->source->format[0] ? fact->source->format : "source");
        yvex_cli_precision_format(fact->precision, sizeof(fact->precision),
                                  fact->source->precision);
        product_size(fact->size, fact->source->size_bytes, fact->source->size_known);
        product_source_location(fact->location, fact->source);
    } else {
        snprintf(fact->format, sizeof(fact->format), "%s", "--");
        snprintf(fact->precision, sizeof(fact->precision), "%s", "--");
        snprintf(fact->size, sizeof(fact->size), "%s", "--");
    }
    snprintf(fact->state, sizeof(fact->state), "%s",
             source ? product_source_state(fact->model, source)
                    : product_model_state(library, model_index, fact->model));
    if (loaded) {
        snprintf(fact->state, sizeof(fact->state), "%s", "LOADED");
        snprintf(fact->execution, sizeof(fact->execution), "%s",
                 loaded->model.backend);
    } else if (fact->profile)
        snprintf(fact->execution, sizeof(fact->execution), "%s", fact->profile->backend);
    else if (fact->model->profile_launchable)
        snprintf(fact->execution, sizeof(fact->execution), "%s", "select variant");
    else if (fact->model->identity_kind ==
             YVEX_MODEL_IDENTITY_PROVIDER_REPOSITORY_REVISION)
        snprintf(fact->execution, sizeof(fact->execution), "%s", "unbound source");
    else if (!strcmp(fact->state, "REMOTE"))
        snprintf(fact->execution, sizeof(fact->execution), "%s", "not acquired");
    else if (!strcmp(fact->state, "UNBOUND"))
        snprintf(fact->execution, sizeof(fact->execution), "%s", "not linked");
    else if (!strcmp(fact->state, "UNSUPPORTED"))
        snprintf(fact->execution, sizeof(fact->execution), "%s", "unsupported");
    else if (fact->model->profile_count)
        snprintf(fact->execution, sizeof(fact->execution), "%s", "not current");
    else snprintf(fact->execution, sizeof(fact->execution), "%s", "not prepared");
    variants = product_representation_count(fact->model);
    snprintf(fact->variants, sizeof(fact->variants), "%llu", variants);
}

static void product_cell(product_table_row *row, unsigned int column,
                         const char *text, yvex_cli_table_tone tone)
{
    snprintf(row->value[column], sizeof(row->value[column]), "%s",
             text && text[0] ? text : "--");
    row->cells[column].text = row->value[column];
    row->cells[column].tone = tone;
}

static void product_default_row(product_table_row *row,
                                const product_model_fact *fact, int wide)
{
    memset(row, 0, sizeof(*row));
    product_cell(row, 0u, fact->selector, YVEX_CLI_TABLE_ACCENT);
    if (wide) {
        product_cell(row, 1u, fact->model->family, YVEX_CLI_TABLE_PLAIN);
        product_cell(row, 2u, fact->origin, YVEX_CLI_TABLE_PLAIN);
        product_cell(row, 3u, fact->format, YVEX_CLI_TABLE_PLAIN);
        product_cell(row, 4u, fact->precision, YVEX_CLI_TABLE_PLAIN);
        product_cell(row, 5u, fact->size, YVEX_CLI_TABLE_PLAIN);
        product_cell(row, 6u, fact->state, product_state_tone(fact->state));
        product_cell(row, 7u, fact->execution, YVEX_CLI_TABLE_PLAIN);
        product_cell(row, 8u, fact->variants, YVEX_CLI_TABLE_PLAIN);
        product_cell(row, 9u, fact->location, YVEX_CLI_TABLE_DIM);
    } else {
        product_cell(row, 1u, fact->state, product_state_tone(fact->state));
        product_cell(row, 2u, fact->format, YVEX_CLI_TABLE_PLAIN);
        product_cell(row, 3u, fact->size, YVEX_CLI_TABLE_PLAIN);
        product_cell(row, 4u, fact->execution, YVEX_CLI_TABLE_PLAIN);
        snprintf(row->secondary, sizeof(row->secondary),
                 "family %s · origin %s · %s representation%s",
                 fact->model->family, fact->origin,
                 fact->variants, !strcmp(fact->variants, "1") ? "" : "s");
        row->row.secondary = row->secondary;
        row->row.secondary_tone = YVEX_CLI_TABLE_DIM;
    }
    row->row.cells = row->cells;
}

static int product_text_contains(const char *text, const char *query)
{
    size_t extent;
    if (!query || !query[0]) return 1;
    if (!text) return 0;
    extent = strlen(query);
    while (*text) {
        if (!strncasecmp(text, query, extent)) return 1;
        text++;
    }
    return 0;
}

static int product_query_match(const yvex_model_library_entry *model,
                               const char *query)
{
    return product_text_contains(model->display_name, query) ||
           product_text_contains(model->model, query) ||
           product_text_contains(model->family, query) ||
           product_text_contains(model->runtime_target, query) ||
           product_text_contains(model->identity, query) ||
           product_text_contains(model->repository, query);
}

static int product_table_default(const yvex_model_library *library, int force_wide,
                                 const product_runtime_view *runtime,
                                 const char *query, unsigned long long offset,
                                 unsigned long long limit)
{
    static const yvex_cli_table_column wide_columns[] = {
        {"MODEL", 12u, 28u, YVEX_CLI_TABLE_LEFT, 0},
        {"FAMILY", 8u, 15u, YVEX_CLI_TABLE_LEFT, 0},
        {"ORIGIN", 8u, 14u, YVEX_CLI_TABLE_LEFT, 0},
        {"FORMAT", 6u, 12u, YVEX_CLI_TABLE_LEFT, 0},
        {"QUANT/PRECISION", 10u, 28u, YVEX_CLI_TABLE_LEFT, 0},
        {"SIZE", 8u, 12u, YVEX_CLI_TABLE_RIGHT, 0},
        {"STATE", 7u, 10u, YVEX_CLI_TABLE_LEFT, 0},
        {"EXEC", 6u, 14u, YVEX_CLI_TABLE_LEFT, 0},
        {"VARIANTS", 8u, 8u, YVEX_CLI_TABLE_RIGHT, 0},
        {"LOCATION", 18u, 0u, YVEX_CLI_TABLE_LEFT, 1}
    };
    static const yvex_cli_table_column narrow_columns[] = {
        {"MODEL", 12u, 28u, YVEX_CLI_TABLE_LEFT, 0},
        {"STATE", 7u, 11u, YVEX_CLI_TABLE_LEFT, 0},
        {"FORMAT", 6u, 12u, YVEX_CLI_TABLE_LEFT, 0},
        {"SIZE", 8u, 12u, YVEX_CLI_TABLE_RIGHT, 0},
        {"EXECUTION", 8u, 16u, YVEX_CLI_TABLE_LEFT, 0}
    };
    product_table_row *storage;
    yvex_cli_table_row *rows;
    unsigned long long index, matched = 0u, count = 0u, cursor = 0u;
    int wide = force_wide || yvex_cli_terminal_columns(stdout) >= 160u;
    int rc;
    for (index = 0u; index < yvex_model_library_count(library); ++index) {
        if (!product_query_match(yvex_model_library_at(library, index), query))
            continue;
        if (matched++ < offset) continue;
        if (count == limit) break;
        count++;
    }
    storage = calloc((size_t)(count ? count : 1u), sizeof(*storage));
    rows = calloc((size_t)(count ? count : 1u), sizeof(*rows));
    if (!storage || !rows) {
        free(storage); free(rows);
        yvex_cli_out_fputs("yvex: model catalog table allocation failed\n", stderr);
        return 1;
    }
    matched = 0u;
    for (index = 0u; index < yvex_model_library_count(library) && cursor < count;
         ++index) {
        product_model_fact fact;
        if (!product_query_match(yvex_model_library_at(library, index), query))
            continue;
        if (matched++ < offset) continue;
        product_fact_build(&fact, library, index, NULL, NULL, runtime);
        product_default_row(&storage[cursor], &fact, wide);
        rows[cursor] = storage[cursor].row;
        cursor++;
    }
    rc = force_wide
             ? yvex_cli_table_render_width(stdout, wide_columns, 10u, rows,
                                           (size_t)count, 1000u)
             : yvex_cli_table_render(stdout,
                                     wide ? wide_columns : narrow_columns,
                                     wide ? 10u : 5u, rows, (size_t)count);
    free(rows); free(storage);
    return rc == YVEX_OK ? 0 : 1;
}

static int product_table_all(const yvex_model_library *library,
                             const product_runtime_view *runtime,
                             int force_wide)
{
    static const yvex_cli_table_column columns[] = {
        {"MODEL", 12u, 28u, YVEX_CLI_TABLE_LEFT, 0},
        {"ROLE", 8u, 14u, YVEX_CLI_TABLE_LEFT, 0},
        {"FORMAT", 6u, 12u, YVEX_CLI_TABLE_LEFT, 0},
        {"QUANT/PRECISION", 10u, 28u, YVEX_CLI_TABLE_LEFT, 0},
        {"SIZE", 8u, 12u, YVEX_CLI_TABLE_RIGHT, 0},
        {"STATE", 7u, 10u, YVEX_CLI_TABLE_LEFT, 0},
        {"EXEC", 6u, 14u, YVEX_CLI_TABLE_LEFT, 0},
        {"LOCATION", 18u, 0u, YVEX_CLI_TABLE_LEFT, 1}
    };
    unsigned long long model_index, representation_index, row_count = 0u, cursor = 0u;
    product_table_row *storage;
    yvex_cli_table_row *rows;
    int rc;
    for (model_index = 0u; model_index < yvex_model_library_count(library); ++model_index) {
        unsigned long long count =
            yvex_model_library_source_count(library, model_index) +
            yvex_model_library_artifact_count(library, model_index);
        row_count += count ? count : 1u;
    }
    storage = calloc((size_t)(row_count ? row_count : 1u), sizeof(*storage));
    rows = calloc((size_t)(row_count ? row_count : 1u), sizeof(*rows));
    if (!storage || !rows) { free(storage); free(rows); return 1; }
    for (model_index = 0u; model_index < yvex_model_library_count(library); ++model_index) {
        unsigned long long first_row = cursor;
        yvex_cli_model_profile_candidate candidates[YVEX_MODELS_ARTIFACT_ROWS_CAP];
        const char *selected_artifact = NULL;
        unsigned long long candidate_count = yvex_cli_model_profile_candidates(
            library, model_index, 0, candidates);
        unsigned long long source_count =
            yvex_model_library_source_count(library, model_index);
        unsigned long long artifact_count =
            yvex_model_library_artifact_count(library, model_index);
        for (representation_index = 0u; representation_index < candidate_count;
             ++representation_index)
            if (candidates[representation_index].selected &&
                candidates[representation_index].artifact)
                selected_artifact = candidates[representation_index].artifact->identity;
        for (representation_index = 0u; representation_index < source_count;
             ++representation_index) {
            product_model_fact fact;
            product_table_row *row = &storage[cursor];
            const yvex_local_source_record *source =
                yvex_model_library_source_at(library, model_index,
                                             representation_index);
            product_fact_build(&fact, library, model_index, source, NULL, runtime);
            memset(row, 0, sizeof(*row));
            product_cell(row, 0u, cursor == first_row ? fact.selector : "",
                         YVEX_CLI_TABLE_ACCENT);
            product_cell(row, 1u, "source", YVEX_CLI_TABLE_PLAIN);
            product_cell(row, 2u, fact.format, YVEX_CLI_TABLE_PLAIN);
            product_cell(row, 3u, fact.precision, YVEX_CLI_TABLE_PLAIN);
            product_cell(row, 4u, fact.size, YVEX_CLI_TABLE_PLAIN);
            product_cell(row, 5u, fact.state, product_state_tone(fact.state));
            product_cell(row, 6u, fact.execution, YVEX_CLI_TABLE_PLAIN);
            product_cell(row, 7u, fact.location, YVEX_CLI_TABLE_DIM);
            row->row.cells = row->cells;
            if (!force_wide) {
                snprintf(row->secondary, sizeof(row->secondary), "%s",
                         fact.location);
                row->row.secondary = row->secondary;
                row->row.secondary_tone = YVEX_CLI_TABLE_DIM;
            }
            rows[cursor++] = row->row;
        }
        for (representation_index = 0u; representation_index < artifact_count;
             ++representation_index) {
            product_model_fact fact;
            product_table_row *row = &storage[cursor];
            const yvex_model_artifact_fact *artifact =
                yvex_model_library_artifact_at(library, model_index,
                                               representation_index);
            product_fact_build(&fact, library, model_index, NULL, artifact, runtime);
            memset(row, 0, sizeof(*row));
            product_cell(row, 0u, cursor == first_row ? fact.selector : "",
                         YVEX_CLI_TABLE_ACCENT);
            product_cell(row, 1u,
                         selected_artifact &&
                                 !strcmp(selected_artifact, artifact->identity)
                             ? "selected"
                             : "alternate",
                         selected_artifact &&
                                 !strcmp(selected_artifact, artifact->identity)
                             ? YVEX_CLI_TABLE_SUCCESS
                             : YVEX_CLI_TABLE_PLAIN);
            product_cell(row, 2u, fact.format, YVEX_CLI_TABLE_PLAIN);
            product_cell(row, 3u, fact.precision, YVEX_CLI_TABLE_PLAIN);
            product_cell(row, 4u, fact.size, YVEX_CLI_TABLE_PLAIN);
            product_cell(row, 5u, fact.state, product_state_tone(fact.state));
            product_cell(row, 6u, fact.execution, YVEX_CLI_TABLE_PLAIN);
            product_cell(row, 7u, fact.location, YVEX_CLI_TABLE_DIM);
            row->row.cells = row->cells;
            if (!force_wide) {
                snprintf(row->secondary, sizeof(row->secondary), "%s",
                         fact.location);
                row->row.secondary = row->secondary;
                row->row.secondary_tone = YVEX_CLI_TABLE_DIM;
            }
            rows[cursor++] = row->row;
        }
        if (!source_count && !artifact_count) {
            product_model_fact fact;
            product_table_row *row = &storage[cursor];
            product_fact_build(&fact, library, model_index, NULL, NULL, runtime);
            memset(row, 0, sizeof(*row));
            product_cell(row, 0u, fact.selector, YVEX_CLI_TABLE_ACCENT);
            product_cell(row, 1u, "catalog", YVEX_CLI_TABLE_PLAIN);
            product_cell(row, 2u, fact.format, YVEX_CLI_TABLE_PLAIN);
            product_cell(row, 3u, fact.precision, YVEX_CLI_TABLE_PLAIN);
            product_cell(row, 4u, fact.size, YVEX_CLI_TABLE_PLAIN);
            product_cell(row, 5u, fact.state, product_state_tone(fact.state));
            product_cell(row, 6u, fact.execution, YVEX_CLI_TABLE_PLAIN);
            product_cell(row, 7u, fact.location, YVEX_CLI_TABLE_DIM);
            row->row.cells = row->cells;
            if (!force_wide) {
                snprintf(row->secondary, sizeof(row->secondary), "%s",
                         fact.location);
                row->row.secondary = row->secondary;
                row->row.secondary_tone = YVEX_CLI_TABLE_DIM;
            }
            rows[cursor++] = row->row;
        }
    }
    rc = force_wide
             ? yvex_cli_table_render_width(stdout, columns, 8u, rows,
                                           (size_t)row_count, 1000u)
             : yvex_cli_table_render(stdout, columns, 8u, rows,
                                     (size_t)row_count);
    free(rows); free(storage);
    return rc == YVEX_OK ? 0 : 1;
}

static void product_json_source(const yvex_local_source_record *source)
{
    yvex_cli_out_fputs("{\"provider\":", stdout);
    yvex_cli_out_json_string(stdout, source->provider);
    yvex_cli_out_fputs(",\"repository\":", stdout);
    yvex_cli_out_json_string(stdout, source->repository);
    yvex_cli_out_fputs(",\"revision\":", stdout);
    yvex_cli_out_json_string(stdout, source->revision);
    yvex_cli_out_fputs(",\"origin_uri\":", stdout);
    yvex_cli_out_json_string(stdout, source->origin_uri);
    yvex_cli_out_fputs(",\"storage\":", stdout);
    yvex_cli_out_json_string(stdout, source->storage_kind);
    yvex_cli_out_fputs(",\"path\":", stdout);
    yvex_cli_out_json_string(stdout, source->path);
    yvex_cli_out_fputs(",\"format\":", stdout);
    yvex_cli_out_json_string(stdout, source->format);
    yvex_cli_out_fputs(",\"precision\":", stdout);
    yvex_cli_out_json_string(stdout, source->precision);
    yvex_cli_out_fputs(",\"verification\":", stdout);
    yvex_cli_out_json_string(stdout, source->verification_state);
    yvex_cli_out_fputs(",\"state\":", stdout);
    yvex_cli_out_json_string(stdout, source->acquisition_state);
    yvex_cli_out_fputs(",\"digest\":", stdout);
    yvex_cli_out_json_string(stdout, source->digest);
    yvex_cli_out_writef(stdout, ",\"size_bytes\":%llu,\"size_known\":%s}",
                        source->size_bytes, source->size_known ? "true" : "false");
}

static void product_json_artifact(const yvex_model_artifact_fact *artifact)
{
    yvex_cli_out_fputs("{\"identity\":", stdout);
    yvex_cli_out_json_string(stdout, artifact->identity);
    yvex_cli_out_fputs(",\"path\":", stdout);
    yvex_cli_out_json_string(stdout, artifact->path);
    yvex_cli_out_fputs(",\"format\":", stdout);
    yvex_cli_out_json_string(stdout, artifact->format);
    yvex_cli_out_fputs(",\"quant_precision\":", stdout);
    yvex_cli_out_json_string(stdout, artifact->physical_variant[0]
                                     ? artifact->physical_variant : artifact->artifact_class);
    yvex_cli_out_writef(stdout, ",\"size_bytes\":%llu,\"tensor_count\":%llu}",
                        artifact->file_size, artifact->tensor_count);
}

static void product_json_components(
    const yvex_model_runtime_profile_fact *profile)
{
    product_composite_fact composite;
    unsigned int index;
    yvex_cli_out_fputs("[", stdout);
    if (product_composite_build(&composite, profile, 1))
        for (index = 0u; index < composite.count; ++index) {
            const product_component_fact *component = &composite.items[index];
            if (index) yvex_cli_out_char(stdout, ',');
            yvex_cli_out_fputs("{\"role\":", stdout);
            yvex_cli_out_json_string(stdout, component->role);
            yvex_cli_out_fputs(",\"path\":", stdout);
            yvex_cli_out_json_string(stdout, component->path);
            yvex_cli_out_fputs(",\"format\":", stdout);
            yvex_cli_out_json_string(stdout, component->format);
            yvex_cli_out_fputs(",\"quant_precision\":", stdout);
            yvex_cli_out_json_string(stdout, component->precision);
            yvex_cli_out_fputs(",\"state\":", stdout);
            yvex_cli_out_json_string(stdout, component->state);
            yvex_cli_out_writef(stdout, ",\"size_bytes\":%llu,\"present\":%s}",
                                component->bytes,
                                component->present ? "true" : "false");
        }
    yvex_cli_out_char(stdout, ']');
}

static void product_json_profile(const yvex_model_runtime_profile_fact *profile)
{
    yvex_cli_out_fputs("{\"identity\":", stdout);
    yvex_cli_out_json_string(stdout, profile->alias);
    yvex_cli_out_fputs(",\"artifact_identity\":", stdout);
    yvex_cli_out_json_string(stdout, profile->artifact_identity);
    yvex_cli_out_fputs(",\"backend\":", stdout);
    yvex_cli_out_json_string(stdout, profile->backend);
    yvex_cli_out_fputs(",\"engine_kind\":", stdout);
    yvex_cli_out_json_string(stdout, profile->engine_kind);
    yvex_cli_out_fputs(",\"strategy\":", stdout);
    yvex_cli_out_json_string(stdout, profile->execution_strategy);
    yvex_cli_out_writef(stdout, ",\"context\":%llu,\"launchable\":%s,\"blocker\":",
                        profile->context_capacity, profile->launchable ? "true" : "false");
    yvex_cli_out_json_string(stdout, profile->blocker);
    yvex_cli_out_writef(
        stdout,
        ",\"capabilities\":{\"input_mask\":%llu,\"output_mask\":%llu,"
        "\"properties\":%llu,\"maximum_input_parts\":%llu}",
        profile->capabilities.input_kinds,
        profile->capabilities.output_kinds,
        profile->capabilities.execution_properties,
        profile->capabilities.maximum_input_parts);
    yvex_cli_out_char(stdout, '}');
}

static void product_json_model(const yvex_model_library *library,
                               unsigned long long model_index,
                               const product_runtime_view *runtime)
{
    product_model_fact fact;
    unsigned long long index;
    product_fact_build(&fact, library, model_index, NULL, NULL, runtime);
    yvex_cli_out_fputs("{\"selector\":", stdout);
    yvex_cli_out_json_string(stdout, fact.selector);
    yvex_cli_out_fputs(",\"identity\":", stdout);
    yvex_cli_out_json_string(stdout, fact.model->identity);
    yvex_cli_out_writef(stdout, ",\"working_set\":%s",
                        yvex_model_library_is_working_set(library, model_index)
                            ? "true" : "false");
    yvex_cli_out_fputs(",\"name\":", stdout);
    yvex_cli_out_json_string(stdout, fact.model->display_name);
    yvex_cli_out_fputs(",\"family\":", stdout);
    yvex_cli_out_json_string(stdout, fact.model->family);
    yvex_cli_out_fputs(",\"origin\":", stdout);
    yvex_cli_out_json_string(stdout, fact.origin);
    yvex_cli_out_fputs(",\"format\":", stdout);
    yvex_cli_out_json_string(stdout, fact.format);
    yvex_cli_out_fputs(",\"quant_precision\":", stdout);
    yvex_cli_out_json_string(stdout, fact.precision);
    yvex_cli_out_fputs(",\"state\":", stdout);
    yvex_cli_out_json_string(stdout, fact.state);
    yvex_cli_out_fputs(",\"execution\":", stdout);
    yvex_cli_out_json_string(stdout, fact.execution);
    yvex_cli_out_fputs(",\"location\":", stdout);
    yvex_cli_out_json_string(stdout, fact.location);
    yvex_cli_out_fputs(",\"selected_profile\":", stdout);
    if (fact.profile) yvex_cli_out_json_string(stdout, fact.profile->alias);
    else yvex_cli_out_fputs("null", stdout);
    yvex_cli_out_fputs(",\"recommendation\":null", stdout);
    yvex_cli_out_writef(stdout, ",\"representation_count\":%llu",
                        product_representation_count(fact.model));
    yvex_cli_out_fputs(",\"sources\":[", stdout);
    for (index = 0u; index < yvex_model_library_source_count(library, model_index); ++index) {
        if (index) yvex_cli_out_char(stdout, ',');
        product_json_source(yvex_model_library_source_at(library, model_index, index));
    }
    yvex_cli_out_fputs("],\"representations\":[", stdout);
    for (index = 0u; index < yvex_model_library_artifact_count(library, model_index); ++index) {
        if (index) yvex_cli_out_char(stdout, ',');
        product_json_artifact(yvex_model_library_artifact_at(library, model_index, index));
    }
    yvex_cli_out_fputs("],\"profiles\":[", stdout);
    for (index = 0u; index < yvex_model_library_profile_count(library, model_index); ++index) {
        if (index) yvex_cli_out_char(stdout, ',');
        product_json_profile(yvex_model_library_profile_at(library, model_index, index));
    }
    yvex_cli_out_fputs("],\"components\":", stdout);
    product_json_components(fact.profile);
    yvex_cli_out_fputs(",\"loaded_engines\":[", stdout);
    {
        unsigned long long emitted = 0u;
        for (index = 0u; runtime && index < runtime->count; ++index) {
            const yvex_cli_loaded_model_fact *loaded = &runtime->models[index];
            if (strcmp(loaded->model.model_selector, fact.selector)) continue;
            if (emitted++) yvex_cli_out_char(stdout, ',');
            yvex_cli_out_fputs("{\"profile_identity\":", stdout);
            yvex_cli_out_json_string(stdout, loaded->model.profile_alias);
            yvex_cli_out_fputs(",\"variant\":", stdout);
            yvex_cli_out_json_string(stdout, loaded->model.variant);
            yvex_cli_out_writef(stdout,
                                ",\"generation\":%llu,\"sessions\":%llu}",
                                loaded->engine.generation, loaded->session_count);
        }
    }
    yvex_cli_out_fputs("]}", stdout);
}

static int product_match(const yvex_model_library_entry *model, const char *selector)
{
    return !strcmp(model->identity, selector) ||
           (model->display_name[0] && !strcmp(model->display_name, selector)) ||
           (model->model[0] && !strcmp(model->model, selector)) ||
           (model->runtime_target[0] && !strcmp(model->runtime_target, selector));
}

int yvex_cli_model_find(const yvex_model_library *library, const char *selector,
                        unsigned long long *model_index)
{
    unsigned long long index;
    int matches = 0;
    for (index = 0u; index < yvex_model_library_count(library); ++index) {
        unsigned long long source_index;
        int matched = product_match(yvex_model_library_at(library, index), selector);
        for (source_index = 0u; !matched &&
             source_index < yvex_model_library_source_count(library, index);
             ++source_index) {
            const yvex_local_source_record *source =
                yvex_model_library_source_at(library, index, source_index);
            matched = source->name[0] && !strcmp(source->name, selector);
        }
        if (!matched) continue;
        *model_index = index;
        matches++;
    }
    return matches;
}

static void show_key_table(const product_model_fact *fact)
{
    static const yvex_cli_table_column columns[] = {
        {"", 10u, 22u, YVEX_CLI_TABLE_LEFT, 0},
        {"", 20u, 100u, YVEX_CLI_TABLE_LEFT, 1}
    };
    const char *lineage =
        fact->model->identity_kind == YVEX_MODEL_IDENTITY_PROVIDER_REPOSITORY_REVISION
            ? "provider source; no authenticated target lineage"
            : fact->model->identity_kind == YVEX_MODEL_IDENTITY_TARGET
                  ? "authenticated runtime target"
                  : "catalog identity";
    const char *keys[] = {"Name", "Identity", "Family", "State", "Execution",
                          "Lineage", "Recommendation"};
    const char *values[] = {fact->selector, fact->model->identity,
                            fact->model->family, fact->state, fact->execution,
                            lineage, "not recorded"};
    yvex_cli_table_cell cells[7][2];
    yvex_cli_table_row rows[7];
    size_t index;
    for (index = 0u; index < 7u; ++index) {
        cells[index][0] = (yvex_cli_table_cell){keys[index], YVEX_CLI_TABLE_DIM};
        cells[index][1] = (yvex_cli_table_cell){values[index],
            index == 0u ? YVEX_CLI_TABLE_ACCENT
            : index == 3u ? product_state_tone(fact->state) : YVEX_CLI_TABLE_PLAIN};
        rows[index] = (yvex_cli_table_row){cells[index], NULL,
                                           YVEX_CLI_TABLE_DIM};
    }
    (void)yvex_cli_table_render(stdout, columns, 2u, rows, 7u);
}

static void show_sources(const yvex_model_library *library, unsigned long long model_index)
{
    static const yvex_cli_table_column columns[] = {
        {"SOURCE", 14u, 30u, YVEX_CLI_TABLE_LEFT, 1},
        {"FORMAT", 6u, 14u, YVEX_CLI_TABLE_LEFT, 0},
        {"PRECISION", 8u, 20u, YVEX_CLI_TABLE_LEFT, 0},
        {"SIZE", 8u, 12u, YVEX_CLI_TABLE_RIGHT, 0},
        {"VERIFY", 10u, 22u, YVEX_CLI_TABLE_LEFT, 0}
    };
    const yvex_model_library_entry *model =
        yvex_model_library_at(library, model_index);
    unsigned long long index, count = yvex_model_library_source_count(library, model_index);
    product_table_row *storage = calloc((size_t)(count ? count : 1u), sizeof(*storage));
    yvex_cli_table_row *rows = calloc((size_t)(count ? count : 1u), sizeof(*rows));
    if (!storage || !rows) { free(storage); free(rows); return; }
    for (index = 0u; index < count; ++index) {
        const yvex_local_source_record *source =
            yvex_model_library_source_at(library, model_index, index);
        char size[32], location[YVEX_PATH_CAP], precision[YVEX_REMOTE_PRECISION_CAP];
        product_size(size, source->size_bytes, source->size_known);
        product_source_location(location, source);
        product_cell(&storage[index], 0u, source->repository,
                     YVEX_CLI_TABLE_ACCENT);
        product_cell(&storage[index], 1u,
                     source->format[0] ? source->format
                                       : source->representation,
                     YVEX_CLI_TABLE_PLAIN);
        yvex_cli_precision_format(precision, sizeof(precision), source->precision);
        product_cell(&storage[index], 2u, precision, YVEX_CLI_TABLE_PLAIN);
        product_cell(&storage[index], 3u, size, YVEX_CLI_TABLE_PLAIN);
        product_cell(&storage[index], 4u, source->verification_state,
                     strstr(source->verification_state, "verified")
                         ? YVEX_CLI_TABLE_SUCCESS : YVEX_CLI_TABLE_WARNING);
        storage[index].row.cells = storage[index].cells;
        snprintf(storage[index].secondary, sizeof(storage[index].secondary),
                 "%s · revision %s · %s · %s", product_origin(source, model),
                 source->revision, source->storage_kind, location);
        storage[index].row.secondary = storage[index].secondary;
        storage[index].row.secondary_tone = YVEX_CLI_TABLE_DIM;
        rows[index] = storage[index].row;
    }
    if (count) {
        (void)yvex_cli_table_render(stdout, columns, 5u, rows, (size_t)count);
    }
    else yvex_cli_out_fputs("  no proven source lineage\n", stdout);
    free(rows); free(storage);
}

static void show_artifacts(const yvex_model_library *library, unsigned long long model_index)
{
    static const yvex_cli_table_column columns[] = {
        {"FORMAT", 8u, 14u, YVEX_CLI_TABLE_LEFT, 0},
        {"QUANT/PRECISION", 12u, 30u, YVEX_CLI_TABLE_LEFT, 0},
        {"SIZE", 8u, 12u, YVEX_CLI_TABLE_RIGHT, 0},
        {"ARTIFACT", 12u, 24u, YVEX_CLI_TABLE_LEFT, 1}
    };
    unsigned long long index, count = yvex_model_library_artifact_count(library, model_index);
    product_table_row *storage = calloc((size_t)(count ? count : 1u), sizeof(*storage));
    yvex_cli_table_row *rows = calloc((size_t)(count ? count : 1u), sizeof(*rows));
    if (!storage || !rows) { free(storage); free(rows); return; }
    for (index = 0u; index < count; ++index) {
        const yvex_model_artifact_fact *artifact =
            yvex_model_library_artifact_at(library, model_index, index);
        char size[32], precision[YVEX_REMOTE_PRECISION_CAP];
        product_size(size, artifact->file_size, 1);
        yvex_cli_precision_format(precision, sizeof(precision),
                                  artifact->physical_variant[0]
                                      ? artifact->physical_variant
                                      : artifact->artifact_class);
        product_cell(&storage[index], 0u, artifact->format, YVEX_CLI_TABLE_PLAIN);
        product_cell(&storage[index], 1u, precision, YVEX_CLI_TABLE_PLAIN);
        product_cell(&storage[index], 2u, size, YVEX_CLI_TABLE_PLAIN);
        product_cell(&storage[index], 3u, artifact->identity, YVEX_CLI_TABLE_DIM);
        storage[index].row.cells = storage[index].cells;
        snprintf(storage[index].secondary, sizeof(storage[index].secondary),
                 "location %s", artifact->path);
        storage[index].row.secondary = storage[index].secondary;
        storage[index].row.secondary_tone = YVEX_CLI_TABLE_DIM;
        rows[index] = storage[index].row;
    }
    if (count) (void)yvex_cli_table_render(stdout, columns, 4u, rows, (size_t)count);
    else yvex_cli_out_fputs("  no prepared representation\n", stdout);
    free(rows); free(storage);
}

static void show_components(const yvex_model_runtime_profile_fact *profile)
{
    static const yvex_cli_table_column columns[] = {
        {"COMPONENT", 10u, 18u, YVEX_CLI_TABLE_LEFT, 0},
        {"FORMAT", 6u, 10u, YVEX_CLI_TABLE_LEFT, 0},
        {"QUANT/PRECISION", 12u, 32u, YVEX_CLI_TABLE_LEFT, 0},
        {"SIZE", 8u, 12u, YVEX_CLI_TABLE_RIGHT, 0},
        {"STATE", 7u, 10u, YVEX_CLI_TABLE_LEFT, 0}
    };
    product_composite_fact composite;
    product_table_row storage[4];
    yvex_cli_table_row rows[4];
    unsigned int index;
    if (!product_composite_build(&composite, profile, 1)) {
        yvex_cli_out_fputs("  composite component profile unavailable\n", stdout);
        return;
    }
    memset(storage, 0, sizeof(storage));
    for (index = 0u; index < composite.count; ++index) {
        const product_component_fact *component = &composite.items[index];
        product_cell(&storage[index], 0u, component->role, YVEX_CLI_TABLE_ACCENT);
        product_cell(&storage[index], 1u, component->format, YVEX_CLI_TABLE_PLAIN);
        product_cell(&storage[index], 2u, component->precision, YVEX_CLI_TABLE_PLAIN);
        product_cell(&storage[index], 3u, component->size, YVEX_CLI_TABLE_PLAIN);
        product_cell(&storage[index], 4u, component->state,
                     component->present ? YVEX_CLI_TABLE_SUCCESS
                                        : YVEX_CLI_TABLE_ERROR);
        storage[index].row.cells = storage[index].cells;
        snprintf(storage[index].secondary, sizeof(storage[index].secondary),
                 "location %s", component->path);
        storage[index].row.secondary = storage[index].secondary;
        storage[index].row.secondary_tone = YVEX_CLI_TABLE_DIM;
        rows[index] = storage[index].row;
    }
    (void)yvex_cli_table_render(stdout, columns, 5u, rows, composite.count);
    if (composite.complete) {
        char total[32];
        product_size(total, composite.bytes, 1);
        yvex_cli_out_writef(stdout,
                            "\n  composite total %s across %u runtime components\n",
                            total, composite.count);
    } else
        yvex_cli_out_fputs("\n  composite incomplete: one or more components are missing\n",
                           stdout);
}

static void show_runtime(const yvex_model_library *library,
                         unsigned long long model_index,
                         const product_runtime_view *runtime)
{
    static const yvex_cli_table_column columns[] = {
        {"ROLE", 8u, 9u, YVEX_CLI_TABLE_LEFT, 0},
        {"REPRESENTATION", 14u, 36u, YVEX_CLI_TABLE_LEFT, 1},
        {"QUANT/PRECISION", 10u, 24u, YVEX_CLI_TABLE_LEFT, 0},
        {"BACKEND", 7u, 10u, YVEX_CLI_TABLE_LEFT, 0},
        {"CONTEXT", 7u, 10u, YVEX_CLI_TABLE_RIGHT, 0},
        {"STATE", 8u, 12u, YVEX_CLI_TABLE_LEFT, 0}
    };
    yvex_cli_model_profile_candidate candidates[YVEX_MODELS_ARTIFACT_ROWS_CAP];
    const yvex_model_library_entry *model = yvex_model_library_at(library, model_index);
    unsigned long long index, count = yvex_cli_model_profile_candidates(
        library, model_index, 0, candidates);
    product_table_row *storage = calloc((size_t)(count ? count : 1u), sizeof(*storage));
    yvex_cli_table_row *rows = calloc((size_t)(count ? count : 1u), sizeof(*rows));
    if (!storage || !rows) { free(storage); free(rows); return; }
    for (index = 0u; index < count; ++index) {
        const yvex_cli_model_profile_candidate *candidate = &candidates[index];
        const yvex_model_runtime_profile_fact *profile = candidate->profile;
        const yvex_cli_loaded_model_fact *loaded = product_loaded(
            runtime, yvex_cli_model_selector(model), profile->artifact_identity, NULL);
        char context[32];
        snprintf(context, sizeof(context), "%llu", profile->context_capacity);
        product_cell(&storage[index], 0u,
                     candidate->selected ? "selected" : "alternate",
                     candidate->selected ? YVEX_CLI_TABLE_SUCCESS
                                         : YVEX_CLI_TABLE_PLAIN);
        product_cell(&storage[index], 1u, candidate->variant, YVEX_CLI_TABLE_ACCENT);
        product_cell(&storage[index], 2u, candidate->precision, YVEX_CLI_TABLE_PLAIN);
        product_cell(&storage[index], 3u, profile->backend, YVEX_CLI_TABLE_PLAIN);
        product_cell(&storage[index], 4u, context, YVEX_CLI_TABLE_PLAIN);
        product_cell(&storage[index], 5u,
                     loaded ? "LOADED" : profile->launchable ? "READY" : "BLOCKED",
                     loaded || profile->launchable ? YVEX_CLI_TABLE_SUCCESS
                                                   : YVEX_CLI_TABLE_ERROR);
        if (loaded)
            snprintf(storage[index].secondary, sizeof(storage[index].secondary),
                     "%s · %s · %s · generation %llu · %llu session%s · %llu revision%s",
                     candidate->format, candidate->size, profile->execution_strategy,
                     loaded->engine.generation, loaded->session_count,
                     loaded->session_count == 1ull ? "" : "s",
                     candidate->revision_count,
                     candidate->revision_count == 1ull ? "" : "s");
        else
            snprintf(storage[index].secondary, sizeof(storage[index].secondary),
                     "profile %s · %s · %s · %s · %llu revision%s",
                     profile->alias, candidate->format, candidate->size,
                     profile->execution_strategy, candidate->revision_count,
                     candidate->revision_count == 1ull ? "" : "s");
        storage[index].row.cells = storage[index].cells;
        storage[index].row.secondary = storage[index].secondary;
        storage[index].row.secondary_tone = YVEX_CLI_TABLE_DIM;
        rows[index] = storage[index].row;
    }
    if (count) {
        (void)yvex_cli_table_render(stdout, columns, 6u, rows, (size_t)count);
    }
    else yvex_cli_out_fputs("  not launchable; run `yvex model prepare MODEL`\n", stdout);
    free(rows); free(storage);
}

static int product_library_open(int arg_count, char **args, int start,
                                const char *command, unsigned int allowed,
                                yvex_model_library **library,
                                yvex_cli_model_list_options *cli)
{
    yvex_local_catalog_options options = {0};
    yvex_error err;
    int rc = model_local_list_options_parse(arg_count, args, start, command,
                                            allowed, cli);
    if (rc) return rc;
    options.models_root = cli->models_root;
    options.registry_path = cli->registry_path;
    yvex_error_clear(&err);
    rc = yvex_model_library_open(library, &options, &err);
    return rc == YVEX_OK ? 0 : print_yvex_error(&err, exit_for_status(rc));
}

int yvex_model_catalog_list_command(int arg_count, char **args)
{
    yvex_model_library *library = NULL;
    yvex_cli_model_list_options cli;
    product_runtime_view runtime;
    unsigned long long index;
    int rc = product_library_open(arg_count, args, 3, "model list",
                                  YVEX_MODEL_LOCAL_OPTIONS_DETAIL,
                                  &library, &cli);
    if (rc) return rc;
    product_runtime_open(&runtime);
    if (cli.output_mode == YVEX_MODEL_CATALOG_OUTPUT_JSON) {
        yvex_cli_out_fputs("{\"schema\":\"yvex.model.list.v3\",\"models\":[", stdout);
        for (index = 0u; index < yvex_model_library_count(library); ++index) {
            if (index) yvex_cli_out_char(stdout, ',');
            product_json_model(library, index, &runtime);
        }
        yvex_cli_out_fputs("]}\n", stdout);
    } else {
        yvex_cli_out_fputs("MODELS\n\n", stdout);
        if (cli.all_representations)
            rc = product_table_all(library, &runtime, cli.wide);
        else rc = product_table_default(library, cli.wide, &runtime, NULL, 0u,
                                        ULLONG_MAX);
    }
    yvex_model_library_close(library);
    return rc;
}

int yvex_model_catalog_show_command(int arg_count, char **args)
{
    yvex_model_library *library = NULL;
    yvex_cli_model_list_options cli;
    product_model_fact fact;
    product_runtime_view runtime;
    unsigned long long model_index = 0u;
    int matches, rc;
    if (arg_count < 4) {
        yvex_cli_out_fputs("yvex: model show requires MODEL\n", stderr);
        return 2;
    }
    if (strstr(args[3], "://")) {
        yvex_source_locator locator;
        yvex_remote_inspect_options inspect = {0};
        yvex_local_catalog_options local_options = {0};
        yvex_remote_catalog *remote = NULL;
        yvex_local_catalog *local = NULL;
        yvex_error err;
        rc = model_local_list_options_parse(arg_count, args, 4, "model show", 0u,
                                            &cli);
        if (rc) return rc;
        yvex_error_clear(&err);
        rc = yvex_source_locator_parse(args[3], &locator, &err);
        if (rc != YVEX_OK)
            return print_yvex_error(&err, exit_for_status(rc));
        if (locator.kind != YVEX_SOURCE_LOCATOR_HUGGINGFACE) {
            yvex_cli_out_writef(stderr,
                                "yvex: model show remote transport unavailable: %s\n",
                                yvex_source_locator_kind_name(locator.kind));
            return 3;
        }
        inspect.provider = YVEX_ACCOUNT_PROVIDER_HUGGINGFACE;
        inspect.repository = locator.repository;
        inspect.revision = locator.revision_present ? locator.revision : NULL;
        rc = yvex_remote_model_inspect(&remote, &inspect, &err);
        local_options.models_root = cli.models_root;
        if (rc == YVEX_OK) rc = yvex_local_catalog_open(&local, &local_options, &err);
        if (rc != YVEX_OK) {
            if (!strcmp(yvex_error_message(&err), "provider authentication is required"))
                yvex_cli_out_fputs(
                    "yvex: provider authentication required\n"
                    "run:\n  yvex source accounts login huggingface\n",
                    stderr);
            else print_yvex_error(&err, exit_for_status(rc));
            yvex_local_catalog_close(local);
            yvex_remote_catalog_close(remote);
            return exit_for_status(rc);
        }
        rc = yvex_remote_catalog_render(stdout, remote, local,
                                        cli.output_mode, 1);
        yvex_local_catalog_close(local);
        yvex_remote_catalog_close(remote);
        return rc == YVEX_OK ? 0 : 1;
    }
    rc = product_library_open(arg_count, args, 4, "model show", 0u,
                              &library, &cli);
    if (rc) return rc;
    matches = yvex_cli_model_find(library, args[3], &model_index);
    if (matches != 1) {
        yvex_cli_out_writef(stderr, "yvex: model selector %s: %s\n", args[3],
                            matches ? "ambiguous; use the exact identity" : "not found");
        yvex_model_library_close(library);
        return 2;
    }
    product_runtime_open(&runtime);
    if (cli.output_mode == YVEX_MODEL_CATALOG_OUTPUT_JSON) {
        yvex_cli_out_fputs("{\"schema\":\"yvex.model.v3\",\"model\":", stdout);
        product_json_model(library, model_index, &runtime);
        yvex_cli_out_fputs("}\n", stdout);
        yvex_model_library_close(library);
        return 0;
    }
    product_fact_build(&fact, library, model_index, NULL, NULL, &runtime);
    yvex_cli_out_fputs("MODEL\n", stdout);
    show_key_table(&fact);
    yvex_cli_out_fputs("\nORIGIN / SOURCE\n", stdout);
    show_sources(library, model_index);
    yvex_cli_out_fputs("\nREPRESENTATIONS\n", stdout);
    if (fact.profile && !strcmp(fact.profile->profile, "composite")) {
        show_components(fact.profile);
        yvex_cli_out_fputs("\nCATALOG ARTIFACT RECORDS\n", stdout);
    }
    show_artifacts(library, model_index);
    yvex_cli_out_fputs("\nRUNTIME\n", stdout);
    show_runtime(library, model_index, &runtime);
    yvex_model_library_close(library);
    return 0;
}

int yvex_model_catalog_search_local(
    const yvex_cli_model_search_options *options)
{
    yvex_local_catalog_options open = {0};
    yvex_model_library *library = NULL;
    product_runtime_view runtime;
    yvex_error err;
    unsigned long long index, emitted = 0u;
    unsigned long long offset, limit;
    int rc;
    if (!options) return 2;
    open.models_root = options->models_root;
    yvex_error_clear(&err);
    rc = yvex_model_library_open(&library, &open, &err);
    if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));
    product_runtime_open(&runtime);
    limit = options->page_size ? options->page_size : ULLONG_MAX;
    offset = options->page > 1u
                 ? (unsigned long long)(options->page - 1u) * limit : 0u;
    if (options->output_mode == YVEX_MODEL_CATALOG_OUTPUT_JSON) {
        yvex_cli_out_fputs(
            "{\"schema\":\"yvex.model.search.v1\",\"provider\":\"local\",\"query\":",
            stdout);
        yvex_cli_out_json_string(stdout, options->query ? options->query : "");
        yvex_cli_out_fputs(",\"models\":[", stdout);
        for (index = 0u; index < yvex_model_library_count(library); ++index) {
            if (!product_query_match(yvex_model_library_at(library, index),
                                     options->query))
                continue;
            if (offset) { offset--; continue; }
            if (emitted == limit) break;
            if (emitted++) yvex_cli_out_char(stdout, ',');
            product_json_model(library, index, &runtime);
        }
        yvex_cli_out_fputs("]}\n", stdout);
    } else {
        yvex_cli_out_writef(stdout, "LOCAL MODELS · \"%s\"\n\n",
                            options->query ? options->query : "");
        rc = product_table_default(library, 0, &runtime, options->query,
                                   offset, limit);
    }
    yvex_model_library_close(library);
    return rc;
}
