/*
 * Join remote provider truth with local package and live-engine observations for operator
 * presentation. The joined record is a projection and never becomes a model-domain authority.
 */
#include "src/cli/model_artifacts/private.h"
#include "src/cli/render/private.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yvex/internal/io.h>

static void remote_parameter_text(char *out, size_t capacity, const yvex_remote_model *model)
{
    double billions;
    if (!model->parameter_count_known) {
        snprintf(out, capacity, "unknown");
        return;
    }
    billions = (double)model->parameter_count / 1000000000.0;
    if (billions >= 1.0)
        snprintf(out, capacity, "%.1fB", billions);
    else
        snprintf(out, capacity, "%.1fM", (double)model->parameter_count / 1000000.0);
}

typedef struct {
    char source_revision[YVEX_REMOTE_REVISION_CAP];
    char package_revision[YVEX_REMOTE_REVISION_CAP];
    char source_representation[YVEX_REMOTE_PRECISION_CAP];
    char package_representation[YVEX_REMOTE_PRECISION_CAP];
    int source, package, related_revision;
} remote_local_projection;

static void remote_local_project(remote_local_projection *projection,
                                 const yvex_remote_model *remote,
                                 const yvex_local_catalog *local)
{
    unsigned long long index;

    memset(projection, 0, sizeof(*projection));
    if (!remote || !local) return;
    for (index = 0ull; index < yvex_local_catalog_source_count(local); ++index) {
        const yvex_local_source_record *source = yvex_local_catalog_source_at(local, index);

        if (!source || !source->repository[0] ||
            strcmp(source->repository, remote->repository) != 0)
            continue;
        snprintf(projection->source_revision, sizeof(projection->source_revision), "%s",
                 source->revision);
        snprintf(projection->source_representation,
                 sizeof(projection->source_representation), "%s", source->representation);
        if (!source->revision[0] || !remote->resolved_revision[0] ||
            strcmp(source->revision, remote->resolved_revision) != 0) {
            projection->related_revision = 1;
        } else {
            projection->source = 1;
        }
    }
    for (index = 0ull; index < yvex_local_catalog_package_count(local); ++index) {
        const yvex_local_package_record *package =
            yvex_local_catalog_package_at(local, index);

        if (!package || !package->repository[0] ||
            strcmp(package->repository, remote->repository) != 0)
            continue;
        snprintf(projection->package_revision, sizeof(projection->package_revision), "%s",
                 package->revision);
        snprintf(projection->package_representation,
                 sizeof(projection->package_representation), "%s", package->representation);
        if (!package->revision[0] || !remote->resolved_revision[0] ||
            strcmp(package->revision, remote->resolved_revision) != 0) {
            projection->related_revision = 1;
        } else {
            projection->package = 1;
        }
    }
}

static int remote_representation_local(
    const remote_local_projection *projection,
    const yvex_model_representation *representation)
{
    const char *source = projection->source_representation;
    const char *package = projection->package_representation;

    if ((!projection->source && !projection->package) || !representation)
        return 0;
    if (representation->kind == YVEX_MODEL_REPRESENTATION_SAFETENSORS)
        return (projection->source && strstr(source, "safetensors")) ||
               (projection->package && strstr(package, "safetensors"));
    if (representation->kind == YVEX_MODEL_REPRESENTATION_GGUF)
        return (projection->source && strstr(source, "gguf")) ||
               (projection->package && strstr(package, "gguf"));
    return 0;
}

static const char *remote_product_status(const yvex_remote_model *model)
{
    if (model->kind == YVEX_REMOTE_MODEL_ADAPTER ||
        model->kind == YVEX_REMOTE_MODEL_COMPONENT ||
        model->kind == YVEX_REMOTE_MODEL_DELTA ||
        model->kind == YVEX_REMOTE_MODEL_DERIVATIVE)
        return "related";
    if (model->support_stage >= YVEX_MODEL_SUPPORT_PACKAGE_PREPARATION) return "supported";
    if (model->support_stage == YVEX_MODEL_SUPPORT_PHYSICAL_INSPECTION) return "inspect";
    if (model->support_stage == YVEX_MODEL_SUPPORT_SOURCE_INGEST) return "acquirable";
    if (model->support_stage == YVEX_MODEL_SUPPORT_ARCHITECTURE_RECOGNIZED)
        return "recognized";
    return "unknown";
}

typedef struct {
    char model[YVEX_REMOTE_NAME_CAP];
    char provider[32];
    char repository[YVEX_REMOTE_REPOSITORY_CAP];
    char architecture[YVEX_REMOTE_NAME_CAP];
    char formats[64];
    char size[32];
    char state[24];
    char support[24];
    char location[YVEX_PATH_CAP];
    yvex_cli_table_cell cells[9];
    yvex_cli_table_row row;
} remote_search_table_row;

static void remote_search_formats(const yvex_remote_catalog *catalog,
                                  unsigned long long row,
                                  const yvex_remote_model *model,
                                  char out[64])
{
    unsigned int index;
    out[0] = '\0';
    for (index = 0u; index < model->representation_count; ++index) {
        const yvex_model_representation *representation =
            yvex_remote_catalog_representation_at(catalog, row, index);
        size_t used = strlen(out);
        if (!representation || strstr(out, representation->format)) continue;
        snprintf(out + used, 64u - used, "%s%s", used ? "," : "",
                 representation->format);
    }
    if (!out[0]) snprintf(out, 64u, "%s", "unknown");
}

static void remote_search_size(const yvex_remote_catalog *catalog,
                               unsigned long long row,
                               const yvex_remote_model *model,
                               char out[32])
{
    const yvex_model_representation *representation;
    if (model->representation_count != 1u) {
        snprintf(out, 32u, "%s",
                 model->representation_count ? "varies" : "unknown");
        return;
    }
    representation = yvex_remote_catalog_representation_at(catalog, row, 0u);
    if (representation && representation->size_known)
        model_download_format_bytes(out, 32u, representation->size_bytes);
    else snprintf(out, 32u, "%s", "unknown");
}

static void remote_search_row_build(remote_search_table_row *row,
                                    const yvex_remote_catalog *catalog,
                                    const yvex_local_catalog *local_catalog,
                                    unsigned long long index)
{
    const yvex_remote_model *model = yvex_remote_catalog_at(catalog, index);
    const char *name = strrchr(model->repository, '/');
    remote_local_projection local;
    remote_local_project(&local, model, local_catalog);
    memset(row, 0, sizeof(*row));
    snprintf(row->model, sizeof(row->model), "%.127s",
             name ? name + 1 : model->repository);
    snprintf(row->provider, sizeof(row->provider), "%s",
             !strcmp(model->provider, "huggingface") ? "Hugging Face"
                                                     : model->provider);
    snprintf(row->repository, sizeof(row->repository), "%s", model->repository);
    snprintf(row->architecture, sizeof(row->architecture), "%s",
             model->architecture[0] ? model->architecture
                                    : model->family[0] ? model->family : "unknown");
    remote_search_formats(catalog, index, model, row->formats);
    remote_search_size(catalog, index, model, row->size);
    snprintf(row->state, sizeof(row->state), "%s",
             local.package ? "package" : local.source ? "source"
             : local.related_revision ? "other revision" : "remote");
    snprintf(row->support, sizeof(row->support), "%s", remote_product_status(model));
    snprintf(row->location, sizeof(row->location), "hf://%s%s%s",
             model->repository, model->resolved_revision[0] ? "@" : "",
             model->resolved_revision);
    row->cells[0] = (yvex_cli_table_cell){row->model, YVEX_CLI_TABLE_ACCENT};
    row->cells[1] = (yvex_cli_table_cell){row->provider, YVEX_CLI_TABLE_PLAIN};
    row->cells[2] = (yvex_cli_table_cell){row->repository, YVEX_CLI_TABLE_PLAIN};
    row->cells[3] = (yvex_cli_table_cell){row->architecture, YVEX_CLI_TABLE_PLAIN};
    row->cells[4] = (yvex_cli_table_cell){row->formats, YVEX_CLI_TABLE_PLAIN};
    row->cells[5] = (yvex_cli_table_cell){row->size, YVEX_CLI_TABLE_PLAIN};
    row->cells[6] = (yvex_cli_table_cell){row->state,
        local.package ? YVEX_CLI_TABLE_SUCCESS : YVEX_CLI_TABLE_WARNING};
    row->cells[7] = (yvex_cli_table_cell){row->support,
        model->support_stage >= YVEX_MODEL_SUPPORT_PACKAGE_PREPARATION
            ? YVEX_CLI_TABLE_SUCCESS : YVEX_CLI_TABLE_WARNING};
    row->cells[8] = (yvex_cli_table_cell){row->location, YVEX_CLI_TABLE_DIM};
    row->row.cells = row->cells;
}

static int remote_catalog_render_table(FILE *fp, const yvex_remote_catalog *catalog,
                                       const yvex_local_catalog *local_catalog,
                                       int representations)
{
    unsigned long long count = yvex_remote_catalog_count(catalog);
    unsigned long long row;
    if (representations && count == 1u) {
        const yvex_remote_model *model = yvex_remote_catalog_at(catalog, 0u);
        remote_local_projection local;
        char parameters[32];
        unsigned int index;
        remote_local_project(&local, model, local_catalog);
        remote_parameter_text(parameters, sizeof(parameters), model);
        yvex_cli_out_writef(fp, "MODEL\n  repository  %s\n  provider    %s\n", model->repository,
                            model->provider);
        yvex_cli_out_writef(fp, "  kind        %s%s\n  family      %s\n  parameters  %s\n",
                            yvex_remote_model_kind_name(model->kind),
                            model->kind_provisional ? " (provisional)" : "",
                            model->family[0] ? model->family : "unknown", parameters);
        yvex_cli_out_writef(fp, "  access      %s\n\nREVISION\n  requested   %s\n  resolved    %s\n",
                            model->gated_known ? (model->gated ? "gated" : "public") : "unknown",
                            model->revision_reference[0] ? model->revision_reference : "default",
                            model->resolved_revision[0] ? model->resolved_revision : "unavailable");
        yvex_cli_out_writef(fp, "\nLOCAL LIFECYCLE\n  source      %s\n  package     %s\n  engine      %s\n\n",
                            local.source ? "acquired"
                                         : (local.source_revision[0]
                                                       ? "available at another revision"
                                                       : "no"),
                            local.package ? "available"
                                          : (local.package_revision[0]
                                                        ? "available at another revision"
                                                        : "no"),
                            "not-observed");
        if (local.source_revision[0] || local.package_revision[0])
            yvex_cli_out_writef(fp, "  local revision  %s\n\n",
                                local.package_revision[0]
                                    ? local.package_revision
                                    : local.source_revision);
        yvex_cli_out_fputs("REPRESENTATIONS\n", fp);
        yvex_cli_out_writef(fp, "%-23s %-13s %-18s %10s %6s %-9s %s\n",
                            "REPRESENTATION", "FORMAT", "PRECISION/QTYPE", "SIZE", "FILES",
                            "LOCAL", "YVEX COMPATIBILITY");
        for (index = 0u; index < model->representation_count; ++index) {
            const yvex_model_representation *representation =
                yvex_remote_catalog_representation_at(catalog, 0u, index);
            char size[32];
            char precision[YVEX_REMOTE_PRECISION_CAP + 16u];
            if (!representation) continue;
            if (representation->size_known)
                model_download_format_bytes(size, sizeof(size), representation->size_bytes);
            else
                snprintf(size, sizeof(size), "unknown");
            snprintf(precision, sizeof(precision), "%s%s",
                     representation->precision[0] ? representation->precision : "unknown",
                     strcmp(representation->precision_evidence, "filename-hint") == 0
                         ? " (filename)"
                         : (representation->provisional ? " ?" : ""));
            yvex_cli_out_writef(fp, "%-23.23s %-13.13s %-18.18s %10.10s %6llu %-9s %s\n",
                                representation->identity, representation->format,
                                precision,
                                size, representation->file_count,
                                remote_representation_local(&local, representation)
                                    ? "yes" : "no",
                                representation->compatibility);
        }
        yvex_cli_out_writef(fp, "\nYVEX  %s\n", remote_product_status(model));
        yvex_cli_out_writef(fp, "provider_files: %u (use --audit or --json for exact paths)\n",
                            model->available_file_count);
        return ferror(fp) ? YVEX_ERR_IO : YVEX_OK;
    }
    {
        static const yvex_cli_table_column columns[] = {
            {"MODEL", 12u, 28u, YVEX_CLI_TABLE_LEFT, 0},
            {"PROVIDER", 8u, 14u, YVEX_CLI_TABLE_LEFT, 0},
            {"REPOSITORY", 14u, 34u, YVEX_CLI_TABLE_LEFT, 1},
            {"ARCH", 8u, 22u, YVEX_CLI_TABLE_LEFT, 0},
            {"FORMATS", 7u, 16u, YVEX_CLI_TABLE_LEFT, 0},
            {"SIZE", 7u, 12u, YVEX_CLI_TABLE_RIGHT, 0},
            {"STATE", 6u, 14u, YVEX_CLI_TABLE_LEFT, 0},
            {"YVEX", 7u, 12u, YVEX_CLI_TABLE_LEFT, 0},
            {"LOCATION", 16u, 60u, YVEX_CLI_TABLE_LEFT, 1}
        };
        remote_search_table_row *storage =
            calloc((size_t)(count ? count : 1u), sizeof(*storage));
        yvex_cli_table_row *rows =
            calloc((size_t)(count ? count : 1u), sizeof(*rows));
        int rc;
        if (!storage || !rows) {
            free(storage); free(rows);
            return YVEX_ERR_NOMEM;
        }
        yvex_cli_out_writef(fp, "REMOTE MODELS%s%s%s\n\n",
                            yvex_remote_catalog_query(catalog)[0] ? " · \"" : "",
                            yvex_remote_catalog_query(catalog),
                            yvex_remote_catalog_query(catalog)[0] ? "\"" : "");
        for (row = 0u; row < count; ++row) {
            remote_search_row_build(&storage[row], catalog, local_catalog, row);
            rows[row] = storage[row].row;
        }
        rc = yvex_cli_table_render(fp, columns, 9u, rows, (size_t)count);
        free(rows); free(storage);
        if (rc != YVEX_OK) return rc;
    }
    if (yvex_remote_catalog_provider_count(catalog) > count)
        yvex_cli_out_writef(fp, "\nshowing %llu ranked results from %llu provider matches · use --all\n",
                            count, yvex_remote_catalog_provider_count(catalog));
    return ferror(fp) ? YVEX_ERR_IO : YVEX_OK;
}
static int remote_catalog_render_audit(
    FILE *fp, const yvex_remote_catalog *catalog,
    const yvex_local_catalog *local_catalog)
{
    unsigned long long model_index;
    yvex_cli_out_writef(fp, "remote_models: %llu\nprovider_matches: %llu\nquery: %s\n",
                        yvex_remote_catalog_count(catalog),
                        yvex_remote_catalog_provider_count(catalog),
                        yvex_remote_catalog_query(catalog));
    for (model_index = 0u; model_index < yvex_remote_catalog_count(catalog); ++model_index) {
        const yvex_remote_model *model = yvex_remote_catalog_at(catalog, model_index);
        remote_local_projection local;
        unsigned int representation_index;
        remote_local_project(&local, model, local_catalog);
        yvex_cli_out_writef(fp, "model[%llu]:\n", model_index);
        yvex_cli_out_writef(fp, "  provider: %s\n  repository: %s\n  author: %s\n",
                            model->provider, model->repository,
                            model->author[0] ? model->author : "unknown");
        yvex_cli_out_writef(fp, "  revision_reference: %s\n  resolved_revision: %s\n",
                            model->revision_reference[0] ? model->revision_reference : "unknown",
                            model->resolved_revision[0] ? model->resolved_revision : "unknown");
        yvex_cli_out_writef(fp, "  family: %s\n  family_evidence: %s\n  architecture: %s\n",
                            model->family[0] ? model->family : "unknown",
                            model->family_evidence[0] ? model->family_evidence : "unknown",
                            model->architecture[0] ? model->architecture : "unknown");
        yvex_cli_out_writef(fp, "  kind: %s\n  kind_evidence: %s\n  kind_provisional: %s\n",
                            yvex_remote_model_kind_name(model->kind), model->kind_evidence,
                            model->kind_provisional ? "true" : "false");
        yvex_cli_out_writef(fp, "  model_identity: %s\n  canonical: %s\n",
                            model->model_identity[0] ? model->model_identity : "unavailable",
                            model->canonical ? "true" : "false");
        yvex_cli_out_writef(fp, "  base_model: %s\n  lineage_relation: %s\n",
                            model->base_model[0] ? model->base_model : "unknown",
                            model->lineage_relation[0] ? model->lineage_relation : "unknown");
        yvex_cli_out_writef(fp, "  product_status: %s\n  support_stage: %s\n  support_reason: %s\n",
                            remote_product_status(model),
                            yvex_model_support_stage_name(model->support_stage),
                            model->support_reason);
        yvex_cli_out_writef(fp, "  local_source: %s\n  local_package: %s\n  engine: %s\n",
                            local.source ? "true" : "false",
                            local.package ? "true" : "false", "not-observed");
        yvex_cli_out_writef(fp, "  local_source_revision: %s\n  local_package_revision: %s\n",
                            local.source_revision[0] ? local.source_revision : "none",
                            local.package_revision[0] ? local.package_revision : "none");
        yvex_cli_out_writef(fp, "  ranking_score: %u\n  provider_rank: %u\n",
                            model->ranking_score, model->provider_rank);
        for (representation_index = 0u; representation_index < model->representation_count;
             ++representation_index) {
            const yvex_model_representation *representation =
                yvex_remote_catalog_representation_at(catalog, model_index,
                                                       representation_index);
            yvex_cli_out_writef(
                fp,
                "  representation[%u]: identity=%s format=%s precision=%s evidence=%s "
                "files=%llu bytes=%llu size_known=%s selector=%s local=%s compatibility=%s "
                "recommendation=%s\n",
                representation_index, representation->identity, representation->format,
                representation->precision[0] ? representation->precision : "unknown",
                representation->precision_evidence[0] ? representation->precision_evidence
                                                       : "unknown",
                representation->file_count, representation->size_bytes,
                representation->size_known ? "true" : "false",
                representation->file_pattern[0] ? representation->file_pattern : "multiple",
                remote_representation_local(&local, representation) ? "true" : "false",
                representation->compatibility,
                representation->recommendation);
        }
        for (representation_index = 0u; representation_index < model->available_file_count;
             ++representation_index) {
            const yvex_remote_file *file =
                yvex_remote_catalog_file_at(catalog, model_index, representation_index);
            yvex_cli_out_writef(fp, "  file[%u]: path=%s kind=%s representation=%s bytes=%llu "
                                    "size_known=%s\n",
                                representation_index, file->path,
                                yvex_remote_file_kind_name(file->kind), file->representation,
                                file->size_bytes, file->size_known ? "true" : "false");
        }
    }
    return ferror(fp) ? YVEX_ERR_IO : YVEX_OK;
}
static void remote_json_text(FILE *fp, const char *key, const char *value, int comma)
{
    yvex_cli_out_writef(fp, "%s\"%s\":", comma ? "," : "", key);
    yvex_file_json_write_string(fp, value);
}
static int remote_catalog_render_json(
    FILE *fp, const yvex_remote_catalog *catalog,
    const yvex_local_catalog *local_catalog)
{
    unsigned long long model_index;
    yvex_cli_out_fputs(
        "{\"schema\":\"yvex.model-catalog-projection.v1\","
        "\"authorities\":[\"remote-provider\",\"local-catalog\"],\"query\":",
        fp);
    yvex_file_json_write_string(fp, yvex_remote_catalog_query(catalog));
    yvex_cli_out_writef(fp, ",\"provider_result_count\":%llu,\"models\":[",
                        yvex_remote_catalog_provider_count(catalog));
    for (model_index = 0u; model_index < yvex_remote_catalog_count(catalog); ++model_index) {
        const yvex_remote_model *model = yvex_remote_catalog_at(catalog, model_index);
        remote_local_projection local;
        unsigned int representation_index;
        remote_local_project(&local, model, local_catalog);
        if (model_index) yvex_cli_out_fputs(",", fp);
        yvex_cli_out_fputs("{", fp);
        remote_json_text(fp, "provider", model->provider, 0);
        remote_json_text(fp, "repository", model->repository, 1);
        remote_json_text(fp, "requested_revision", model->revision_reference, 1);
        remote_json_text(fp, "resolved_revision", model->resolved_revision, 1);
        remote_json_text(fp, "kind", yvex_remote_model_kind_name(model->kind), 1);
        remote_json_text(fp, "kind_evidence", model->kind_evidence, 1);
        remote_json_text(fp, "model_identity", model->model_identity, 1);
        remote_json_text(fp, "family_affinity", model->family, 1);
        remote_json_text(fp, "family_evidence", model->family_evidence, 1);
        remote_json_text(fp, "architecture", model->architecture, 1);
        remote_json_text(fp, "base_model", model->base_model, 1);
        remote_json_text(fp, "support_stage",
                         yvex_model_support_stage_name(model->support_stage), 1);
        remote_json_text(fp, "product_status", remote_product_status(model), 1);
        remote_json_text(fp, "engine_state", "not-observed", 1);
        remote_json_text(fp, "local_source_revision", local.source_revision, 1);
        remote_json_text(fp, "local_package_revision", local.package_revision, 1);
        yvex_cli_out_writef(fp, ",\"local_source\":%s,\"local_package\":%s,"
                                "\"local_related_revision\":%s,"
                                "\"kind_provisional\":%s,\"canonical\":%s,"
                                "\"ranking_score\":%u,\"provider_rank\":%u,"
                                "\"gated_known\":%s,\"gated\":%s,"
                                "\"parameter_count_known\":%s,\"parameter_count\":%llu,"
                                "\"representations\":[",
                            local.source ? "true" : "false",
                            local.package ? "true" : "false",
                            local.related_revision ? "true" : "false",
                            model->kind_provisional ? "true" : "false",
                            model->canonical ? "true" : "false", model->ranking_score,
                            model->provider_rank,
                            model->gated_known ? "true" : "false",
                            model->gated ? "true" : "false",
                            model->parameter_count_known ? "true" : "false",
                            model->parameter_count);
        for (representation_index = 0u; representation_index < model->representation_count;
             ++representation_index) {
            const yvex_model_representation *representation =
                yvex_remote_catalog_representation_at(catalog, model_index,
                                                       representation_index);
            if (representation_index) yvex_cli_out_fputs(",", fp);
            yvex_cli_out_fputs("{", fp);
            remote_json_text(fp, "identity", representation->identity, 0);
            remote_json_text(fp, "format", representation->format, 1);
            remote_json_text(fp, "precision", representation->precision, 1);
            remote_json_text(fp, "precision_evidence", representation->precision_evidence, 1);
            remote_json_text(fp, "file_pattern", representation->file_pattern, 1);
            remote_json_text(fp, "compatibility", representation->compatibility, 1);
            yvex_cli_out_writef(fp, ",\"file_count\":%llu,\"size_bytes\":%llu,"
                                    "\"size_known\":%s,\"provisional\":%s,\"local\":%s}",
                                representation->file_count, representation->size_bytes,
                                representation->size_known ? "true" : "false",
                                representation->provisional ? "true" : "false",
                                remote_representation_local(&local, representation)
                                    ? "true" : "false");
        }
        yvex_cli_out_fputs("],\"files\":[", fp);
        for (representation_index = 0u; representation_index < model->available_file_count;
             ++representation_index) {
            const yvex_remote_file *file =
                yvex_remote_catalog_file_at(catalog, model_index, representation_index);
            if (representation_index) yvex_cli_out_fputs(",", fp);
            yvex_cli_out_fputs("{", fp);
            remote_json_text(fp, "path", file->path, 0);
            remote_json_text(fp, "kind", yvex_remote_file_kind_name(file->kind), 1);
            remote_json_text(fp, "representation", file->representation, 1);
            yvex_cli_out_writef(fp, ",\"size_bytes\":%llu,\"size_known\":%s}",
                                file->size_bytes, file->size_known ? "true" : "false");
        }
        yvex_cli_out_fputs("]}", fp);
    }
    yvex_cli_out_fputs("]}\n", fp);
    return ferror(fp) ? YVEX_ERR_IO : YVEX_OK;
}
int yvex_remote_catalog_render(FILE *fp,
                               const yvex_remote_catalog *catalog,
                               const yvex_local_catalog *local_catalog,
                               yvex_model_catalog_output_mode mode,
                               int representations)
{
    if (!fp || !catalog) return YVEX_ERR_INVALID_ARG;
    if (mode == YVEX_MODEL_CATALOG_OUTPUT_JSON)
        return remote_catalog_render_json(fp, catalog, local_catalog);
    if (mode == YVEX_MODEL_CATALOG_OUTPUT_AUDIT)
        return remote_catalog_render_audit(fp, catalog, local_catalog);
    return remote_catalog_render_table(fp, catalog, local_catalog,
                                       representations);
}
static unsigned long long local_projection_count(const yvex_local_catalog *catalog)
{
    return yvex_local_catalog_source_count(catalog) +
           yvex_local_catalog_package_count(catalog);
}

static const char *local_source_projection_blocker(const yvex_local_source_record *source)
{
    if (source->blocker[0]) return source->blocker;
    if (strcmp(source->acquisition_state, "source-acquired") != 0) return "";
    if (strstr(source->representation, "mixed"))
        return "select one acquired representation before package preparation";
    if (strstr(source->representation, "safetensors"))
        return "compile or prepare an admitted YVEX package";
    if (strstr(source->representation, "gguf"))
        return "inspect GGUF compatibility, then admit or repack";
    return "classify acquired files before package preparation";
}

static void local_source_render_json(FILE *fp,
                                     const yvex_local_source_record *source,
                                     int comma)
{
    const char *blocker = local_source_projection_blocker(source);

    if (comma) yvex_cli_out_fputs(",", fp);
    yvex_cli_out_fputs("{", fp);
    remote_json_text(fp, "name", source->name, 0);
    remote_json_text(fp, "family", source->family, 1);
    remote_json_text(fp, "provider", source->provider, 1);
    remote_json_text(fp, "repository", source->repository, 1);
    remote_json_text(fp, "revision", source->revision, 1);
    remote_json_text(fp, "kind", "acquired-source", 1);
    remote_json_text(fp, "representation", source->representation, 1);
    remote_json_text(fp, "package_state", source->acquisition_state, 1);
    remote_json_text(fp, "verification_state", source->verification_state, 1);
    remote_json_text(fp, "engine_state", "not-applicable", 1);
    remote_json_text(fp, "blocker", blocker, 1);
    yvex_cli_out_writef(fp,
                        ",\"size_bytes\":%llu,\"size_known\":%s,"
                        "\"package_ready\":false}",
                        source->size_bytes, source->size_known ? "true" : "false");
}

static void local_package_render_json(FILE *fp,
                                      const yvex_local_package_record *package,
                                      const char *engine_state,
                                      int comma)
{
    if (comma) yvex_cli_out_fputs(",", fp);
    yvex_cli_out_fputs("{", fp);
    remote_json_text(fp, "name", package->name, 0);
    remote_json_text(fp, "family", package->family, 1);
    remote_json_text(fp, "provider", "", 1);
    remote_json_text(fp, "repository", package->repository, 1);
    remote_json_text(fp, "revision", package->revision, 1);
    remote_json_text(fp, "kind", "package", 1);
    remote_json_text(fp, "representation", package->representation, 1);
    remote_json_text(fp, "package_state", package->package_state, 1);
    remote_json_text(fp, "verification_state", package->verification_state, 1);
    remote_json_text(fp, "engine_state", engine_state, 1);
    remote_json_text(fp, "blocker", package->blocker, 1);
    yvex_cli_out_writef(fp,
                        ",\"size_bytes\":%llu,\"size_known\":%s,"
                        "\"package_ready\":%s}",
                        package->size_bytes, package->size_known ? "true" : "false",
                        package->ready ? "true" : "false");
}

static void local_source_render_audit(FILE *fp,
                                      const yvex_local_source_record *source,
                                      unsigned long long index)
{
    const char *blocker = local_source_projection_blocker(source);

    yvex_cli_out_writef(
        fp,
        "model[%llu]: name=%s family=%s kind=acquired-source representation=%s "
        "package_state=%s verification=%s engine=not-applicable backend=unknown "
        "provider=%s repository=%s revision=%s bytes=%llu blocker=%s path=%s\n",
        index, source->name, source->family, source->representation,
        source->acquisition_state, source->verification_state,
        source->provider[0] ? source->provider : "unknown",
        source->repository[0] ? source->repository : "unknown",
        source->revision[0] ? source->revision : "unknown", source->size_bytes,
        blocker[0] ? blocker : "none", source->path);
}

static void local_package_render_audit(FILE *fp,
                                       const yvex_local_package_record *package,
                                       const char *engine_state,
                                       unsigned long long index)
{
    yvex_cli_out_writef(
        fp,
        "model[%llu]: name=%s family=%s kind=package representation=%s "
        "package_state=%s verification=%s engine=%s backend=%s provider=unknown "
        "repository=%s revision=%s bytes=%llu blocker=%s path=%s\n",
        index, package->name, package->family, package->representation,
        package->package_state, package->verification_state, engine_state,
        package->backend[0] ? package->backend : "unknown",
        package->repository[0] ? package->repository : "unknown",
        package->revision[0] ? package->revision : "unknown", package->size_bytes,
        package->blocker[0] ? package->blocker : "none", package->path);
}

static void local_table_row(FILE *fp,
                            const char *name,
                            const char *family,
                            const char *kind,
                            const char *representation,
                            const char *state,
                            const char *verification,
                            unsigned long long size_bytes,
                            int size_known,
                            const char *engine,
                            const char *backend,
                            const char *blocker)
{
    char size[32];

    if (size_known)
        model_download_format_bytes(size, sizeof(size), size_bytes);
    else
        snprintf(size, sizeof(size), "unknown");
    yvex_cli_out_writef(fp,
                        "%-40.40s %-12.12s %-8.8s %-16.16s %-18.18s %-18.18s "
                        "%10.10s %-12.12s %s%s%s\n",
                        name, family, kind, representation, state, verification, size,
                        engine, backend, blocker[0] ? " · " : "", blocker);
}

int yvex_local_catalog_render(FILE *fp,
                              const yvex_local_catalog *catalog,
                              yvex_cli_engine_state_resolver engine_state,
                              const void *engine_context,
                              int engine_host_observed,
                              yvex_model_catalog_output_mode mode)
{
    unsigned long long source_index;
    unsigned long long package_index;
    unsigned long long row = 0u;

    if (!fp || !catalog) return YVEX_ERR_INVALID_ARG;
    if (mode == YVEX_MODEL_CATALOG_OUTPUT_JSON) {
        yvex_cli_out_writef(fp,
                            "{\"schema\":\"yvex.local-model-catalog.v2\","
                            "\"engine_host_observed\":%s,\"models\":[",
                            engine_host_observed ? "true" : "false");
        for (source_index = 0u;
             source_index < yvex_local_catalog_source_count(catalog);
             ++source_index, ++row) {
            local_source_render_json(
                fp, yvex_local_catalog_source_at(catalog, source_index), row != 0u);
        }
        for (package_index = 0u;
             package_index < yvex_local_catalog_package_count(catalog);
             ++package_index, ++row) {
            const yvex_local_package_record *package =
                yvex_local_catalog_package_at(catalog, package_index);
            const char *state = engine_state
                                    ? engine_state(package, engine_context)
                                    : "not-observed";
            local_package_render_json(fp, package, state, row != 0u);
        }
        yvex_cli_out_fputs("]}\n", fp);
        return ferror(fp) ? YVEX_ERR_IO : YVEX_OK;
    }
    if (mode == YVEX_MODEL_CATALOG_OUTPUT_AUDIT) {
        yvex_cli_out_writef(fp, "local_models: %llu\n", local_projection_count(catalog));
        yvex_cli_out_writef(fp, "engine_host_observed: %s\n",
                            engine_host_observed ? "true" : "false");
        for (source_index = 0u;
             source_index < yvex_local_catalog_source_count(catalog);
             ++source_index, ++row) {
            local_source_render_audit(
                fp, yvex_local_catalog_source_at(catalog, source_index), row);
        }
        for (package_index = 0u;
             package_index < yvex_local_catalog_package_count(catalog);
             ++package_index, ++row) {
            const yvex_local_package_record *package =
                yvex_local_catalog_package_at(catalog, package_index);
            const char *state = engine_state
                                    ? engine_state(package, engine_context)
                                    : "not-observed";
            local_package_render_audit(fp, package, state, row);
        }
        return ferror(fp) ? YVEX_ERR_IO : YVEX_OK;
    }
    yvex_cli_out_writef(fp, "LOCAL MODELS  count=%llu\n\n", local_projection_count(catalog));
    yvex_cli_out_writef(fp,
                        "%-40s %-12s %-8s %-16s %-18s %-18s %10s %-12s %s\n",
                        "MODEL", "FAMILY", "KIND", "REPRESENTATION", "PACKAGE STATE",
                        "VERIFICATION", "SIZE", "ENGINE", "BACKEND / BLOCKER");
    for (source_index = 0u;
         source_index < yvex_local_catalog_source_count(catalog);
         ++source_index) {
        const yvex_local_source_record *source =
            yvex_local_catalog_source_at(catalog, source_index);
        const char *blocker = local_source_projection_blocker(source);
        local_table_row(fp, source->name, source->family, "source",
                        source->representation, source->acquisition_state,
                        source->verification_state, source->size_bytes, source->size_known,
                        "not-applicable", "-", blocker);
    }
    for (package_index = 0u;
         package_index < yvex_local_catalog_package_count(catalog);
         ++package_index) {
        const yvex_local_package_record *package =
            yvex_local_catalog_package_at(catalog, package_index);
        const char *state = engine_state
                                ? engine_state(package, engine_context)
                                : "not-observed";
        local_table_row(fp, package->name, package->family, "package",
                        package->representation, package->package_state,
                        package->verification_state, package->size_bytes,
                        package->size_known, state,
                        package->backend[0] ? package->backend : "-", package->blocker);
    }
    return ferror(fp) ? YVEX_ERR_IO : YVEX_OK;
}
