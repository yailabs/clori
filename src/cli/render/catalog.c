/*
 * Join remote provider truth with local package and live-engine observations for operator
 * presentation. The joined record is a projection and never becomes a model-domain authority.
 */
#include "src/cli/model_artifacts/private.h"

#include <stdio.h>
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
                                 const yvex_local_model_catalog *local)
{
    unsigned long long index;

    memset(projection, 0, sizeof(*projection));
    if (!remote || !local) return;
    for (index = 0ull; index < yvex_local_model_catalog_count(local); ++index) {
        const yvex_local_model *entry = yvex_local_model_catalog_at(local, index);
        char *revision;
        char *representation;

        if (!entry || !entry->repository[0] ||
            strcmp(entry->repository, remote->repository) != 0)
            continue;
        revision = entry->kind == YVEX_LOCAL_MODEL_PACKAGE
                       ? projection->package_revision
                       : projection->source_revision;
        representation = entry->kind == YVEX_LOCAL_MODEL_PACKAGE
                             ? projection->package_representation
                             : projection->source_representation;
        snprintf(revision, YVEX_REMOTE_REVISION_CAP, "%s", entry->revision);
        snprintf(representation, YVEX_REMOTE_PRECISION_CAP, "%s",
                 entry->representation);
        if (!entry->revision[0] || !remote->resolved_revision[0] ||
            strcmp(entry->revision, remote->resolved_revision) != 0) {
            projection->related_revision = 1;
        } else if (entry->kind == YVEX_LOCAL_MODEL_PACKAGE) {
            projection->package = 1;
        } else {
            projection->source = 1;
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
static int remote_catalog_render_table(FILE *fp, const yvex_remote_catalog *catalog,
                                       const yvex_local_model_catalog *local_catalog,
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
    yvex_cli_out_writef(fp, "REMOTE MODELS%s%s%s\n\n",
                        yvex_remote_catalog_query(catalog)[0] ? " · \"" : "",
                        yvex_remote_catalog_query(catalog),
                        yvex_remote_catalog_query(catalog)[0] ? "\"" : "");
    yvex_cli_out_writef(fp, "%-42s %-17s %-13s %8s %-15s %-8s %s\n", "MODEL / REPOSITORY",
                        "KIND", "FAMILY", "PARAMS", "FORMAT", "LOCAL", "YVEX");
    for (row = 0u; row < count; ++row) {
        const yvex_remote_model *model = yvex_remote_catalog_at(catalog, row);
        remote_local_projection local;
        char parameters[32];
        char classes[64];
        char kind[32];
        unsigned int index;
        remote_local_project(&local, model, local_catalog);
        remote_parameter_text(parameters, sizeof(parameters), model);
        classes[0] = '\0';
        for (index = 0u; index < model->representation_count; ++index) {
            const yvex_model_representation *representation =
                yvex_remote_catalog_representation_at(catalog, row, index);
            size_t used = strlen(classes);
            if (!representation || strstr(classes, representation->format)) continue;
            snprintf(classes + used, sizeof(classes) - used, "%s%s", used ? "," : "",
                     representation->format);
        }
        if (!classes[0]) snprintf(classes, sizeof(classes), "unknown");
        snprintf(kind, sizeof(kind), "%s%s", yvex_remote_model_kind_name(model->kind),
                 model->kind_provisional ? " ?" : "");
        yvex_cli_out_writef(fp, "%-42.42s %-17.17s %-13.13s %8s %-15.15s %-8s %s\n",
                            model->repository, kind,
                            model->family[0] ? model->family : "unknown", parameters, classes,
                            local.package
                                ? "package"
                                : (local.source
                                       ? "source"
                                       : (local.related_revision ? "other-rev" : "no")),
                            remote_product_status(model));
    }
    if (yvex_remote_catalog_provider_count(catalog) > count)
        yvex_cli_out_writef(fp, "\nshowing %llu ranked results from %llu provider matches · use --all\n",
                            count, yvex_remote_catalog_provider_count(catalog));
    return ferror(fp) ? YVEX_ERR_IO : YVEX_OK;
}
static int remote_catalog_render_audit(
    FILE *fp, const yvex_remote_catalog *catalog,
    const yvex_local_model_catalog *local_catalog)
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
    const yvex_local_model_catalog *local_catalog)
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
                               const yvex_local_model_catalog *local_catalog,
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
int yvex_local_catalog_render(FILE *fp,
                              const yvex_local_model_catalog *catalog,
                              yvex_cli_engine_state_resolver engine_state,
                              const void *engine_context,
                              int engine_host_observed,
                              yvex_model_catalog_output_mode mode)
{
    unsigned long long index;
    if (!fp || !catalog) return YVEX_ERR_INVALID_ARG;
    if (mode == YVEX_MODEL_CATALOG_OUTPUT_JSON) {
        yvex_cli_out_writef(fp,
                            "{\"schema\":\"yvex.local-model-catalog.v2\","
                            "\"engine_host_observed\":%s,\"models\":[",
                            engine_host_observed ? "true" : "false");
        for (index = 0u; index < yvex_local_model_catalog_count(catalog); ++index) {
            const yvex_local_model *model = yvex_local_model_catalog_at(catalog, index);
            const char *state = engine_state ? engine_state(model, engine_context)
                                             : "not-observed";
            if (index) yvex_cli_out_fputs(",", fp);
            yvex_cli_out_fputs("{", fp);
            remote_json_text(fp, "name", model->name, 0);
            remote_json_text(fp, "family", model->family, 1);
            remote_json_text(fp, "provider", model->provider, 1);
            remote_json_text(fp, "repository", model->repository, 1);
            remote_json_text(fp, "revision", model->revision, 1);
            remote_json_text(fp, "kind",
                             model->kind == YVEX_LOCAL_MODEL_PACKAGE ? "package" : "acquired-source", 1);
            remote_json_text(fp, "representation", model->representation, 1);
            remote_json_text(fp, "package_state", model->package_state, 1);
            remote_json_text(fp, "verification_state", model->verification_state, 1);
            remote_json_text(fp, "engine_state", state, 1);
            remote_json_text(fp, "blocker", model->blocker, 1);
            yvex_cli_out_writef(fp, ",\"size_bytes\":%llu,\"size_known\":%s,"
                                    "\"package_ready\":%s}",
                                model->size_bytes, model->size_known ? "true" : "false",
                                model->package_ready ? "true" : "false");
        }
        yvex_cli_out_fputs("]}\n", fp);
        return ferror(fp) ? YVEX_ERR_IO : YVEX_OK;
    }
    if (mode == YVEX_MODEL_CATALOG_OUTPUT_AUDIT) {
        yvex_cli_out_writef(fp, "local_models: %llu\n",
                            yvex_local_model_catalog_count(catalog));
        yvex_cli_out_writef(fp, "engine_host_observed: %s\n",
                            engine_host_observed ? "true" : "false");
        for (index = 0u; index < yvex_local_model_catalog_count(catalog); ++index) {
            const yvex_local_model *model = yvex_local_model_catalog_at(catalog, index);
            const char *state = engine_state ? engine_state(model, engine_context)
                                             : "not-observed";
            yvex_cli_out_writef(fp, "model[%llu]: name=%s family=%s kind=%s representation=%s "
                                    "package_state=%s verification=%s engine=%s backend=%s "
                                    "provider=%s repository=%s revision=%s bytes=%llu blocker=%s path=%s\n",
                                index, model->name, model->family,
                                model->kind == YVEX_LOCAL_MODEL_PACKAGE ? "package"
                                                                        : "acquired-source",
                                model->representation, model->package_state,
                                model->verification_state, state,
                                model->backend[0] ? model->backend : "unknown",
                                model->provider[0] ? model->provider : "unknown",
                                model->repository[0] ? model->repository : "unknown",
                                model->revision[0] ? model->revision : "unknown", model->size_bytes,
                                model->blocker[0] ? model->blocker : "none", model->path);
        }
        return ferror(fp) ? YVEX_ERR_IO : YVEX_OK;
    }
    yvex_cli_out_writef(fp, "LOCAL MODELS  count=%llu\n\n",
                        yvex_local_model_catalog_count(catalog));
    yvex_cli_out_writef(fp,
                        "%-40s %-12s %-8s %-16s %-18s %-18s %10s %-12s %s\n",
                        "MODEL", "FAMILY", "KIND", "REPRESENTATION", "PACKAGE STATE",
                        "VERIFICATION", "SIZE", "ENGINE", "BACKEND / BLOCKER");
    for (index = 0u; index < yvex_local_model_catalog_count(catalog); ++index) {
        const yvex_local_model *model = yvex_local_model_catalog_at(catalog, index);
        const char *state = engine_state ? engine_state(model, engine_context)
                                         : "not-observed";
        char size[32];
        if (model->size_known)
            model_download_format_bytes(size, sizeof(size), model->size_bytes);
        else
            snprintf(size, sizeof(size), "unknown");
        yvex_cli_out_writef(fp, "%-40.40s %-12.12s %-8.8s %-16.16s %-18.18s %-18.18s "
                                "%10.10s %-12.12s %s%s%s\n",
                            model->name, model->family,
                            model->kind == YVEX_LOCAL_MODEL_PACKAGE ? "package" : "source",
                            model->representation,
                            model->package_state, model->verification_state, size,
                            state,
                            model->backend[0] ? model->backend : "-",
                            model->blocker[0] ? " · " : "", model->blocker);
    }
    return ferror(fp) ? YVEX_ERR_IO : YVEX_OK;
}
