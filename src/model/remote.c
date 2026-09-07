/*
 * Normalize provider discovery into model and representation facts.
 *
 * Provider metadata proves remote existence and exact revision only. Filename-derived GGUF
 * precision remains provisional until artifact admission inspects the acquired container.
 */

#include <yvex/catalog.h>
#include <yvex/internal/model_lifecycle.h>
#include <yvex/internal/provider.h>
#include <yvex/internal/source_distribution.h>

#include <ctype.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <strings.h>

#include <yvex/internal/core.h>
#include <yvex/internal/model.h>
#include <yvex/internal/source_catalog.h>

#define REMOTE_OUTPUT_CAP (8u * 1024u * 1024u)
#define REMOTE_ERROR_CAP (64u * 1024u)
#define REMOTE_MODEL_CAP 1000u
#define REMOTE_FILE_CAP 8192u

typedef struct {
    yvex_remote_model model;
    char safetensors_precision[YVEX_REMOTE_PRECISION_CAP];
    int has_safetensors;
    int has_gguf;
    int tag_adapter;
    int tag_component;
    int tag_delta;
    int tag_conversion;
    int tag_derivative;
} parsed_remote_model;

struct yvex_remote_catalog {
    yvex_remote_model *models;
    unsigned long long model_count;
    unsigned long long model_capacity;
    yvex_model_representation *representations;
    unsigned long long representation_count;
    unsigned long long representation_capacity;
    unsigned long long *representation_offsets;
    yvex_remote_file *files;
    char (*file_sha256)[YVEX_SHA256_HEX_CAP];
    unsigned long long file_count;
    unsigned long long file_capacity;
    unsigned long long *file_offsets;
    unsigned long long provider_result_count;
    char query[257];
};

static int remote_refuse(yvex_error *err, yvex_status status, const char *message)
{
    yvex_error_set(err, status, "remote_model", message);
    return status;
}

static void remote_copy(char *out, size_t capacity, const char *value)
{
    if (!out || !capacity) return;
    snprintf(out, capacity, "%s", value ? value : "");
}

const char *yvex_model_representation_kind_name(yvex_model_representation_kind kind)
{
    switch (kind) {
    case YVEX_MODEL_REPRESENTATION_SAFETENSORS: return "safetensors";
    case YVEX_MODEL_REPRESENTATION_GGUF: return "gguf";
    case YVEX_MODEL_REPRESENTATION_UNKNOWN: return "unknown";
    }
    return "unknown";
}

const char *yvex_remote_model_kind_name(yvex_remote_model_kind kind)
{
    switch (kind) {
    case YVEX_REMOTE_MODEL_FULL: return "full model";
    case YVEX_REMOTE_MODEL_CONVERSION: return "conversion";
    case YVEX_REMOTE_MODEL_ADAPTER: return "adapter / LoRA";
    case YVEX_REMOTE_MODEL_COMPONENT: return "component";
    case YVEX_REMOTE_MODEL_DELTA: return "delta / patch";
    case YVEX_REMOTE_MODEL_DERIVATIVE: return "derivative / fork";
    case YVEX_REMOTE_MODEL_UNKNOWN: return "unknown";
    }
    return "unknown";
}

const char *yvex_remote_file_kind_name(yvex_remote_file_kind kind)
{
    switch (kind) {
    case YVEX_REMOTE_FILE_SAFETENSORS: return "safetensors";
    case YVEX_REMOTE_FILE_GGUF: return "gguf";
    case YVEX_REMOTE_FILE_CONFIGURATION: return "configuration";
    case YVEX_REMOTE_FILE_TOKENIZER: return "tokenizer";
    case YVEX_REMOTE_FILE_SIDECAR: return "sidecar";
    case YVEX_REMOTE_FILE_UNKNOWN: return "unknown";
    }
    return "unknown";
}

const char *yvex_model_support_stage_name(yvex_model_support_stage stage)
{
    switch (stage) {
    case YVEX_MODEL_SUPPORT_REMOTE_ONLY: return "remote-only";
    case YVEX_MODEL_SUPPORT_ARCHITECTURE_RECOGNIZED: return "architecture-recognized";
    case YVEX_MODEL_SUPPORT_SOURCE_INGEST: return "source-ingest";
    case YVEX_MODEL_SUPPORT_SEMANTIC_FAMILY: return "semantic-family";
    case YVEX_MODEL_SUPPORT_PHYSICAL_INSPECTION: return "physical-inspection";
    case YVEX_MODEL_SUPPORT_PACKAGE_PREPARATION: return "package-preparation";
    }
    return "remote-only";
}

static int remote_repository_valid(const char *repository)
{
    const unsigned char *cursor = (const unsigned char *)repository;
    unsigned int slashes = 0u;

    if (!repository || !repository[0] || repository[0] == '/' || strstr(repository, ".."))
        return 0;
    while (*cursor) {
        if (*cursor == '/') {
            if (++slashes > 1u || !cursor[1]) return 0;
        } else if (!(isalnum(*cursor) || *cursor == '-' || *cursor == '_' || *cursor == '.')) {
            return 0;
        }
        cursor++;
    }
    return slashes == 1u;
}

static int remote_text_valid(const char *value, size_t maximum)
{
    size_t index;

    if (!value || !value[0] || strlen(value) > maximum) return 0;
    for (index = 0u; value[index]; ++index) {
        unsigned char byte = (unsigned char)value[index];
        if (byte < 0x20u || byte == 0x7fu) return 0;
    }
    return 1;
}

static int remote_revision_immutable(const char *revision)
{
    size_t index;
    size_t length = revision ? strlen(revision) : 0u;

    if (length != 40u && length != 64u) return 0;
    for (index = 0u; index < length; ++index)
        if (!isxdigit((unsigned char)revision[index])) return 0;
    return 1;
}

static int remote_catalog_reserve_models(yvex_remote_catalog *catalog,
                                         unsigned long long required)
{
    yvex_remote_model *models;
    unsigned long long *offsets;
    unsigned long long *file_offsets;
    unsigned long long capacity;

    if (required <= catalog->model_capacity) return 1;
    capacity = catalog->model_capacity ? catalog->model_capacity * 2u : 16u;
    while (capacity < required) capacity *= 2u;
    if (capacity > REMOTE_MODEL_CAP) capacity = REMOTE_MODEL_CAP;
    if (capacity < required) return 0;
    models = realloc(catalog->models, (size_t)capacity * sizeof(*models));
    if (!models) return 0;
    catalog->models = models;
    offsets = realloc(catalog->representation_offsets, (size_t)capacity * sizeof(*offsets));
    if (!offsets) return 0;
    catalog->representation_offsets = offsets;
    file_offsets = realloc(catalog->file_offsets, (size_t)capacity * sizeof(*file_offsets));
    if (!file_offsets) return 0;
    catalog->file_offsets = file_offsets;
    catalog->model_capacity = capacity;
    return 1;
}

static int remote_catalog_reserve_files(yvex_remote_catalog *catalog,
                                        unsigned long long required)
{
    yvex_remote_file *files;
    char (*identities)[YVEX_SHA256_HEX_CAP];
    unsigned long long capacity;

    if (required <= catalog->file_capacity) return 1;
    capacity = catalog->file_capacity ? catalog->file_capacity * 2u : 64u;
    while (capacity < required) capacity *= 2u;
    if (capacity > REMOTE_FILE_CAP) capacity = REMOTE_FILE_CAP;
    if (capacity < required) return 0;
    files = realloc(catalog->files, (size_t)capacity * sizeof(*files));
    if (!files) return 0;
    catalog->files = files;
    identities = realloc(catalog->file_sha256, (size_t)capacity * sizeof(*identities));
    if (!identities) return 0;
    catalog->file_sha256 = identities;
    catalog->file_capacity = capacity;
    return 1;
}

static int remote_catalog_reserve_representations(yvex_remote_catalog *catalog,
                                                  unsigned long long required)
{
    yvex_model_representation *representations;
    unsigned long long capacity;

    if (required <= catalog->representation_capacity) return 1;
    capacity = catalog->representation_capacity ? catalog->representation_capacity * 2u : 32u;
    while (capacity < required) capacity *= 2u;
    representations = realloc(catalog->representations,
                              (size_t)capacity * sizeof(*representations));
    if (!representations) return 0;
    catalog->representations = representations;
    catalog->representation_capacity = capacity;
    return 1;
}

static int remote_catalog_add_model(yvex_remote_catalog *catalog,
                                    const yvex_remote_model *model,
                                    unsigned long long *index)
{
    if (!remote_catalog_reserve_models(catalog, catalog->model_count + 1u)) return 0;
    *index = catalog->model_count++;
    catalog->models[*index] = *model;
    catalog->models[*index].representation_count = 0u;
    catalog->models[*index].available_file_count = 0u;
    catalog->representation_offsets[*index] = catalog->representation_count;
    catalog->file_offsets[*index] = catalog->file_count;
    return 1;
}

static yvex_remote_file *remote_catalog_add_file(yvex_remote_catalog *catalog,
                                                 unsigned long long model_index)
{
    yvex_remote_file *file;

    if (catalog->models[model_index].available_file_count == UINT_MAX ||
        !remote_catalog_reserve_files(catalog, catalog->file_count + 1u))
        return NULL;
    file = &catalog->files[catalog->file_count++];
    catalog->file_sha256[catalog->file_count - 1u][0] = '\0';
    memset(file, 0, sizeof(*file));
    catalog->models[model_index].available_file_count++;
    return file;
}

static yvex_model_representation *remote_representation_find(
    yvex_remote_catalog *catalog,
    unsigned long long model_index,
    yvex_model_representation_kind kind,
    const char *identity)
{
    yvex_remote_model *model = &catalog->models[model_index];
    unsigned long long offset = catalog->representation_offsets[model_index];
    unsigned int index;

    for (index = 0u; index < model->representation_count; ++index) {
        yvex_model_representation *representation = &catalog->representations[offset + index];
        if (representation->kind == kind && strcmp(representation->identity, identity) == 0)
            return representation;
    }
    return NULL;
}

static yvex_model_representation *remote_catalog_add_representation(
    yvex_remote_catalog *catalog,
    unsigned long long model_index,
    yvex_model_representation_kind kind,
    const char *identity)
{
    yvex_remote_model *model = &catalog->models[model_index];
    yvex_model_representation *representation;

    representation = remote_representation_find(catalog, model_index, kind, identity);
    if (representation) return representation;
    if (model->representation_count >= YVEX_REMOTE_MAX_REPRESENTATIONS ||
        !remote_catalog_reserve_representations(catalog, catalog->representation_count + 1u))
        return NULL;
    representation = &catalog->representations[catalog->representation_count++];
    memset(representation, 0, sizeof(*representation));
    representation->kind = kind;
    remote_copy(representation->identity, sizeof(representation->identity), identity);
    remote_copy(representation->format, sizeof(representation->format),
                yvex_model_representation_kind_name(kind));
    model->representation_count++;
    return representation;
}

static int remote_tag_family(const char *tag, char *family, size_t family_capacity)
{
    struct family_tag {
        const char *prefix;
        const char *family;
    };
    static const struct family_tag tags[] = {
        {"minimax-h3", "minimax-h3"}, {"deepseek", "deepseek"}, {"qwen", "qwen"},
        {"gemma", "gemma"}, {"glm", "glm"},
    };
    size_t index;

    for (index = 0u; index < sizeof(tags) / sizeof(tags[0]); ++index) {
        size_t length = strlen(tags[index].prefix);
        if (strncasecmp(tag, tags[index].prefix, length) == 0) {
            remote_copy(family, family_capacity, tags[index].family);
            return 1;
        }
    }
    return 0;
}

static void remote_architecture_family(yvex_remote_model *model)
{
    char lowered[YVEX_REMOTE_NAME_CAP];
    size_t index;

    if (model->family[0] || !model->architecture[0]) return;
    for (index = 0u; index + 1u < sizeof(lowered) && model->architecture[index]; ++index)
        lowered[index] = (char)tolower((unsigned char)model->architecture[index]);
    lowered[index] = '\0';
    if (strstr(lowered, "qwen")) remote_copy(model->family, sizeof(model->family), "qwen");
    else if (strstr(lowered, "deepseek"))
        remote_copy(model->family, sizeof(model->family), "deepseek");
    else if (strstr(lowered, "gemma"))
        remote_copy(model->family, sizeof(model->family), "gemma");
    else if (strstr(lowered, "glm"))
        remote_copy(model->family, sizeof(model->family), "glm");
    if (model->family[0])
        remote_copy(model->family_evidence, sizeof(model->family_evidence),
                    "provider-architecture");
}

static void remote_tag_kind(parsed_remote_model *parsed, const char *tag)
{
    char lowered[YVEX_REMOTE_REPOSITORY_CAP];
    size_t index;

    for (index = 0u; index + 1u < sizeof(lowered) && tag[index]; ++index)
        lowered[index] = (char)tolower((unsigned char)tag[index]);
    lowered[index] = '\0';
    if (strstr(lowered, "lora") || strstr(lowered, "adapter") ||
        strcmp(lowered, "peft") == 0)
        parsed->tag_adapter = 1;
    if (strstr(lowered, "controlnet") || strstr(lowered, "upscaler") ||
        strstr(lowered, "text-encoder") || strcmp(lowered, "vae") == 0)
        parsed->tag_component = 1;
    if (strstr(lowered, "delta") || strstr(lowered, "patch")) parsed->tag_delta = 1;
    if (strstr(lowered, "conversion") || strstr(lowered, "quantized"))
        parsed->tag_conversion = 1;
    if (strstr(lowered, "finetune") || strstr(lowered, "fine-tune") ||
        strstr(lowered, "derived"))
        parsed->tag_derivative = 1;
}

static int remote_parse_string_array(yvex_json *json,
                                     parsed_remote_model *parsed,
                                     int tags)
{
    yvex_json_iter iter;
    yvex_json_item item;

    if (!yvex_json_iter_begin(json, &iter, YVEX_JSON_COLLECTION_ARRAY)) return 0;
    while ((item = yvex_json_array_value(&iter)) == YVEX_JSON_ITEM_READY) {
        char value[YVEX_REMOTE_REPOSITORY_CAP];
        if (!yvex_json_string(json, value, sizeof(value))) return 0;
        if (!tags) continue;
        remote_tag_kind(parsed, value);
        if (!parsed->model.family[0] && remote_tag_family(value, parsed->model.family,
                                                          sizeof(parsed->model.family)))
            remote_copy(parsed->model.family_evidence, sizeof(parsed->model.family_evidence),
                        "provider-tag");
        if (strcmp(value, "safetensors") == 0) parsed->has_safetensors = 1;
        if (strcmp(value, "gguf") == 0) parsed->has_gguf = 1;
    }
    return item == YVEX_JSON_ITEM_END && !iter.trailing_separator;
}

static int remote_precision_append(char *out, size_t capacity, const char *precision)
{
    const char *cursor = out;
    size_t used = strlen(out);
    int written;

    while (*cursor) {
        const char *end = strchr(cursor, '+');
        size_t token_length = end ? (size_t)(end - cursor) : strlen(cursor);
        if (strlen(precision) == token_length && strncmp(cursor, precision, token_length) == 0)
            return 1;
        cursor = end ? end + 1 : cursor + token_length;
    }
    written = snprintf(out + used, capacity - used, "%s%s", used ? "+" : "", precision);
    return written >= 0 && (size_t)written < capacity - used;
}

static int remote_parse_parameters(yvex_json *json, parsed_remote_model *parsed)
{
    yvex_json_iter iter;
    yvex_json_item item;
    char key[YVEX_JSON_KEY_CAP];

    if (!yvex_json_iter_begin(json, &iter, YVEX_JSON_COLLECTION_OBJECT)) return 0;
    while ((item = yvex_json_object_member(&iter, key, sizeof(key))) == YVEX_JSON_ITEM_READY) {
        unsigned long long ignored;
        if (!yvex_json_u64(json, &ignored) ||
            !remote_precision_append(parsed->safetensors_precision,
                                     sizeof(parsed->safetensors_precision), key))
            return 0;
    }
    return item == YVEX_JSON_ITEM_END && !iter.trailing_separator;
}

static int remote_parse_safetensors(yvex_json *json, parsed_remote_model *parsed)
{
    yvex_json_iter iter;
    yvex_json_item item;
    char key[YVEX_JSON_KEY_CAP];

    parsed->has_safetensors = 1;
    if (!yvex_json_iter_begin(json, &iter, YVEX_JSON_COLLECTION_OBJECT)) return 0;
    while ((item = yvex_json_object_member(&iter, key, sizeof(key))) == YVEX_JSON_ITEM_READY) {
        if (strcmp(key, "parameters") == 0) {
            if (!remote_parse_parameters(json, parsed)) return 0;
        } else if (strcmp(key, "total") == 0) {
            if (!yvex_json_u64(json, &parsed->model.parameter_count)) return 0;
            parsed->model.parameter_count_known = 1;
        } else if (!yvex_json_skip_value(json)) {
            return 0;
        }
    }
    return item == YVEX_JSON_ITEM_END && !iter.trailing_separator;
}

static int remote_parse_transformers(yvex_json *json, parsed_remote_model *parsed)
{
    yvex_json_iter iter;
    yvex_json_item item;
    char key[YVEX_JSON_KEY_CAP];

    if (!yvex_json_iter_begin(json, &iter, YVEX_JSON_COLLECTION_OBJECT)) return 0;
    while ((item = yvex_json_object_member(&iter, key, sizeof(key))) == YVEX_JSON_ITEM_READY) {
        if (strcmp(key, "auto_model") == 0) {
            if (!yvex_json_string(json, parsed->model.architecture,
                                  sizeof(parsed->model.architecture)))
                return 0;
        } else if (!yvex_json_skip_value(json)) {
            return 0;
        }
    }
    return item == YVEX_JSON_ITEM_END && !iter.trailing_separator;
}

static int remote_parse_base_models(yvex_json *json, parsed_remote_model *parsed)
{
    yvex_json_iter object;
    yvex_json_item item;
    char key[YVEX_JSON_KEY_CAP];

    if (!yvex_json_iter_begin(json, &object, YVEX_JSON_COLLECTION_OBJECT)) return 0;
    while ((item = yvex_json_object_member(&object, key, sizeof(key))) == YVEX_JSON_ITEM_READY) {
        if (strcmp(key, "relation") == 0) {
            if (!yvex_json_string(json, parsed->model.lineage_relation,
                                  sizeof(parsed->model.lineage_relation)))
                return 0;
        } else if (strcmp(key, "models") == 0) {
            yvex_json_iter array;
            yvex_json_item model_item;
            if (!yvex_json_iter_begin(json, &array, YVEX_JSON_COLLECTION_ARRAY)) return 0;
            while ((model_item = yvex_json_array_value(&array)) == YVEX_JSON_ITEM_READY) {
                yvex_json_iter model;
                yvex_json_item field;
                if (!yvex_json_iter_begin(json, &model, YVEX_JSON_COLLECTION_OBJECT)) return 0;
                while ((field = yvex_json_object_member(&model, key, sizeof(key))) ==
                       YVEX_JSON_ITEM_READY) {
                    if (strcmp(key, "id") == 0 && !parsed->model.base_model[0]) {
                        if (!yvex_json_string(json, parsed->model.base_model,
                                              sizeof(parsed->model.base_model)))
                            return 0;
                    } else if (!yvex_json_skip_value(json)) {
                        return 0;
                    }
                }
                if (field != YVEX_JSON_ITEM_END || model.trailing_separator) return 0;
            }
            if (model_item != YVEX_JSON_ITEM_END || array.trailing_separator) return 0;
        } else if (!yvex_json_skip_value(json)) {
            return 0;
        }
    }
    return item == YVEX_JSON_ITEM_END && !object.trailing_separator;
}

static int remote_parse_gated(yvex_json *json, yvex_remote_model *model)
{
    yvex_json probe = *json;
    int gated;

    yvex_json_space(&probe);
    if (probe.cursor < probe.end && (*probe.cursor == 't' || *probe.cursor == 'f')) {
        if (!yvex_json_bool(json, &gated)) return 0;
        model->gated = gated;
        model->gated_known = 1;
        return 1;
    }
    if (probe.cursor < probe.end && *probe.cursor == '"') {
        char policy[32];
        if (!yvex_json_string(json, policy, sizeof(policy))) return 0;
        model->gated = 1;
        model->gated_known = 1;
        return 1;
    }
    return yvex_json_skip_value(json);
}

static int remote_parse_model(yvex_json *json, parsed_remote_model *parsed)
{
    yvex_json_iter iter;
    yvex_json_item item;
    char key[YVEX_JSON_KEY_CAP];

    memset(parsed, 0, sizeof(*parsed));
    remote_copy(parsed->model.provider, sizeof(parsed->model.provider), "huggingface");
    remote_copy(parsed->model.revision_reference,
                sizeof(parsed->model.revision_reference), "default");
    if (!yvex_json_iter_begin(json, &iter, YVEX_JSON_COLLECTION_OBJECT)) return 0;
    while ((item = yvex_json_object_member(&iter, key, sizeof(key))) == YVEX_JSON_ITEM_READY) {
        if (strcmp(key, "id") == 0) {
            if (!yvex_json_string(json, parsed->model.repository,
                                  sizeof(parsed->model.repository)))
                return 0;
        } else if (strcmp(key, "author") == 0) {
            if (!yvex_json_string(json, parsed->model.author, sizeof(parsed->model.author)))
                return 0;
        } else if (strcmp(key, "sha") == 0) {
            if (!yvex_json_string(json, parsed->model.resolved_revision,
                                  sizeof(parsed->model.resolved_revision)))
                return 0;
        } else if (strcmp(key, "pipeline_tag") == 0) {
            if (!yvex_json_string(json, parsed->model.pipeline,
                                  sizeof(parsed->model.pipeline)))
                return 0;
        } else if (strcmp(key, "gated") == 0) {
            if (!remote_parse_gated(json, &parsed->model)) return 0;
        } else if (strcmp(key, "tags") == 0) {
            if (!remote_parse_string_array(json, parsed, 1)) return 0;
        } else if (strcmp(key, "safetensors") == 0) {
            if (!remote_parse_safetensors(json, parsed)) return 0;
        } else if (strcmp(key, "transformers_info") == 0) {
            if (!remote_parse_transformers(json, parsed)) return 0;
        } else if (strcmp(key, "base_models") == 0) {
            if (!remote_parse_base_models(json, parsed)) return 0;
        } else if (!yvex_json_skip_value(json)) {
            return 0;
        }
    }
    remote_architecture_family(&parsed->model);
    return item == YVEX_JSON_ITEM_END && !iter.trailing_separator &&
           remote_repository_valid(parsed->model.repository);
}

static const yvex_source_target_identity *remote_qualified_source(
    const yvex_remote_model *model)
{
    return yvex_source_target_identity_find_repository(model->repository);
}

static int remote_qualified_revision(const yvex_remote_model *model,
                                     const yvex_source_target_identity *qualified)
{
    return qualified &&
           strcmp(model->resolved_revision, qualified->upstream_revision) == 0;
}

static int remote_name_hint(const char *repository, const char *needle)
{
    char lowered[YVEX_REMOTE_REPOSITORY_CAP];
    size_t index;

    for (index = 0u; index + 1u < sizeof(lowered) && repository[index]; ++index)
        lowered[index] = (char)tolower((unsigned char)repository[index]);
    lowered[index] = '\0';
    return strstr(lowered, needle) != NULL;
}

static void remote_kind_classify(parsed_remote_model *parsed,
                                 const yvex_source_target_identity *qualified)
{
    yvex_remote_model *model = &parsed->model;

    if (qualified) {
        model->kind = YVEX_REMOTE_MODEL_FULL;
        model->canonical = 1;
        remote_copy(model->kind_evidence, sizeof(model->kind_evidence),
                    "yvex-source-catalog");
    } else if (parsed->tag_adapter || strcasecmp(model->lineage_relation, "adapter") == 0) {
        model->kind = YVEX_REMOTE_MODEL_ADAPTER;
        remote_copy(model->kind_evidence, sizeof(model->kind_evidence), "provider-metadata");
    } else if (parsed->tag_component) {
        model->kind = YVEX_REMOTE_MODEL_COMPONENT;
        remote_copy(model->kind_evidence, sizeof(model->kind_evidence), "provider-metadata");
    } else if (parsed->tag_delta) {
        model->kind = YVEX_REMOTE_MODEL_DELTA;
        remote_copy(model->kind_evidence, sizeof(model->kind_evidence), "provider-metadata");
    } else if (parsed->tag_conversion ||
               strcasecmp(model->lineage_relation, "quantized") == 0) {
        model->kind = YVEX_REMOTE_MODEL_CONVERSION;
        remote_copy(model->kind_evidence, sizeof(model->kind_evidence), "provider-metadata");
    } else if (model->base_model[0] || parsed->tag_derivative) {
        model->kind = YVEX_REMOTE_MODEL_DERIVATIVE;
        remote_copy(model->kind_evidence, sizeof(model->kind_evidence), "provider-lineage");
    } else if (parsed->has_safetensors && model->architecture[0]) {
        model->kind = YVEX_REMOTE_MODEL_FULL;
        remote_copy(model->kind_evidence, sizeof(model->kind_evidence), "provider-model-metadata");
    } else {
        model->kind_provisional = 1;
        remote_copy(model->kind_evidence, sizeof(model->kind_evidence), "repository-name-hint");
        if (remote_name_hint(model->repository, "lora") ||
            remote_name_hint(model->repository, "adapter"))
            model->kind = YVEX_REMOTE_MODEL_ADAPTER;
        else if (remote_name_hint(model->repository, "controlnet") ||
                 remote_name_hint(model->repository, "upscaler") ||
                 remote_name_hint(model->repository, "vae"))
            model->kind = YVEX_REMOTE_MODEL_COMPONENT;
        else if (remote_name_hint(model->repository, "delta") ||
                 remote_name_hint(model->repository, "patch"))
            model->kind = YVEX_REMOTE_MODEL_DELTA;
        else if (remote_name_hint(model->repository, "gguf") ||
                 remote_name_hint(model->repository, "conversion"))
            model->kind = YVEX_REMOTE_MODEL_CONVERSION;
        else
            model->kind = YVEX_REMOTE_MODEL_UNKNOWN;
    }
    if (model->kind == YVEX_REMOTE_MODEL_FULL)
        remote_copy(model->model_identity, sizeof(model->model_identity), model->repository);
}

static void remote_support_classify(parsed_remote_model *parsed)
{
    yvex_remote_model *model = &parsed->model;
    const yvex_source_target_identity *qualified = remote_qualified_source(model);
    int qualified_revision = remote_qualified_revision(model, qualified);

    if (qualified && !model->family[0]) {
        remote_copy(model->family, sizeof(model->family), qualified->family_key);
        remote_copy(model->family_evidence, sizeof(model->family_evidence),
                    "yvex-source-catalog");
    }
    remote_kind_classify(parsed, qualified);
    model->support_stage = YVEX_MODEL_SUPPORT_REMOTE_ONLY;
    remote_copy(model->support_reason, sizeof(model->support_reason),
                "remote availability only; YVEX family support is unknown");
    if (model->family[0]) {
        model->support_stage = YVEX_MODEL_SUPPORT_ARCHITECTURE_RECOGNIZED;
        remote_copy(model->support_reason, sizeof(model->support_reason),
                    "provider metadata identifies a known architecture family");
    }
    if (parsed->has_safetensors) {
        model->support_stage = YVEX_MODEL_SUPPORT_SOURCE_INGEST;
        remote_copy(model->support_reason, sizeof(model->support_reason),
                    "safetensors source can be acquired and inspected; package preparation is separate");
    }
    if (qualified_revision) {
        model->support_stage = YVEX_MODEL_SUPPORT_PACKAGE_PREPARATION;
        remote_copy(model->support_reason, sizeof(model->support_reason),
                    "exact revision matches a YVEX-qualified source/package path");
    } else if (parsed->has_gguf && model->support_stage < YVEX_MODEL_SUPPORT_PHYSICAL_INSPECTION) {
        model->support_stage = YVEX_MODEL_SUPPORT_PHYSICAL_INSPECTION;
        remote_copy(model->support_reason, sizeof(model->support_reason),
                    "GGUF format is recognizable; compatibility requires acquired-container inspection");
    }
}

static int remote_add_parsed(yvex_remote_catalog *catalog,
                             parsed_remote_model *parsed,
                             int search_projection,
                             unsigned long long *model_index)
{
    yvex_model_representation *representation;

    remote_support_classify(parsed);
    if (!remote_catalog_add_model(catalog, &parsed->model, model_index)) return 0;
    if (parsed->has_safetensors) {
        representation = remote_catalog_add_representation(
            catalog, *model_index, YVEX_MODEL_REPRESENTATION_SAFETENSORS,
            "safetensors-source");
        if (!representation) return 0;
        remote_copy(representation->precision, sizeof(representation->precision),
                    parsed->safetensors_precision[0] ? parsed->safetensors_precision : "unknown");
        remote_copy(representation->precision_evidence,
                    sizeof(representation->precision_evidence),
                    parsed->safetensors_precision[0] ? "provider-metadata" : "unknown");
        representation->source_ingest_supported = 1;
        representation->package_preparation_supported =
            remote_qualified_revision(&parsed->model,
                                      remote_qualified_source(&parsed->model));
        remote_copy(representation->compatibility, sizeof(representation->compatibility),
                    representation->package_preparation_supported
                        ? "exact-revision-package-path"
                        : "source-ingest-only");
        remote_copy(representation->recommendation, sizeof(representation->recommendation),
                    representation->package_preparation_supported
                        ? "qualified exact-revision YVEX package path"
                        : "acquire only when source inspection or a later compiler path is intended");
    }
    if (search_projection && parsed->has_gguf) {
        representation = remote_catalog_add_representation(
            catalog, *model_index, YVEX_MODEL_REPRESENTATION_GGUF, "gguf-files");
        if (!representation) return 0;
        remote_copy(representation->precision, sizeof(representation->precision), "unknown");
        remote_copy(representation->precision_evidence,
                    sizeof(representation->precision_evidence), "remote-class-hint");
        remote_copy(representation->compatibility, sizeof(representation->compatibility),
                    "acquire-and-inspect-required");
        remote_copy(representation->recommendation, sizeof(representation->recommendation),
                    "inspect exact files before selecting a physical representation");
        representation->provisional = 1;
        representation->direct_admission_requires_inspection = 1;
    }
    return 1;
}

static int remote_parse_search_output(yvex_remote_catalog *catalog,
                                      const char *output,
                                      size_t output_length,
                                      yvex_error *err)
{
    yvex_json json;
    yvex_json_iter array;
    yvex_json_item item;
    unsigned int source_index = 0u;

    yvex_json_init(&json, output, output_length);
    if (!yvex_json_iter_begin(&json, &array, YVEX_JSON_COLLECTION_ARRAY))
        return remote_refuse(err, YVEX_ERR_FORMAT, "provider search did not return a JSON array");
    while ((item = yvex_json_array_value(&array)) == YVEX_JSON_ITEM_READY) {
        parsed_remote_model parsed;
        if (!remote_parse_model(&json, &parsed))
            return remote_refuse(err, YVEX_ERR_FORMAT, "provider search returned malformed model metadata");
        if (++source_index > REMOTE_MODEL_CAP)
            return remote_refuse(err, YVEX_ERR_BOUNDS, "provider search exceeded the model limit");
        parsed.model.provider_rank = source_index;
        {
            unsigned long long ignored;
            if (!remote_add_parsed(catalog, &parsed, 1, &ignored))
                return remote_refuse(err, YVEX_ERR_NOMEM, "remote catalog allocation failed");
        }
    }
    if (item != YVEX_JSON_ITEM_END || array.trailing_separator || !yvex_json_complete(&json))
        return remote_refuse(err, YVEX_ERR_FORMAT, "provider search JSON is incomplete");
    catalog->provider_result_count = source_index;
    yvex_error_clear(err);
    return YVEX_OK;
}

static const char *remote_extension(const char *path)
{
    const char *slash = strrchr(path, '/');
    const char *dot = strrchr(path, '.');
    return dot && (!slash || dot > slash) ? dot : "";
}

static int remote_file_path_valid(const char *path)
{
    return path && path[0] && path[0] != '/' && !strstr(path, "..") &&
           remote_text_valid(path, YVEX_REMOTE_REPOSITORY_CAP - 1u);
}

static yvex_remote_file_kind remote_file_kind(const char *path)
{
    const char *base = strrchr(path, '/');
    const char *extension = remote_extension(path);

    base = base ? base + 1 : path;
    if (strcasecmp(extension, ".safetensors") == 0) return YVEX_REMOTE_FILE_SAFETENSORS;
    if (strcasecmp(extension, ".gguf") == 0) return YVEX_REMOTE_FILE_GGUF;
    if (strstr(base, "config") && strcasecmp(extension, ".json") == 0)
        return YVEX_REMOTE_FILE_CONFIGURATION;
    if (strstr(base, "tokenizer") || strcmp(base, "vocab.json") == 0 ||
        strcmp(base, "merges.txt") == 0 || strstr(base, "special_tokens") ||
        strstr(base, "chat_template"))
        return YVEX_REMOTE_FILE_TOKENIZER;
    if (strcasecmp(extension, ".json") == 0 || strcasecmp(extension, ".md") == 0 ||
        strcasecmp(extension, ".txt") == 0 || strcasecmp(extension, ".model") == 0 ||
        strcasecmp(extension, ".jinja") == 0)
        return YVEX_REMOTE_FILE_SIDECAR;
    return YVEX_REMOTE_FILE_UNKNOWN;
}

static void remote_upper_basename(char *out, size_t capacity, const char *path)
{
    const char *base = strrchr(path, '/');
    size_t index;

    base = base ? base + 1 : path;
    for (index = 0u; index + 1u < capacity && base[index]; ++index)
        out[index] = (char)toupper((unsigned char)base[index]);
    out[index] = '\0';
}

static void remote_gguf_precision(char *out, size_t capacity, const char *path)
{
    static const char *const qtypes[] = {
        "IQ2_XXS", "IQ3_XXS", "IQ1_M", "IQ1_S", "IQ2_XS", "IQ2_S", "IQ3_S",
        "IQ4_XS", "Q2_K_S", "Q2_K", "Q3_K_M", "Q3_K_S", "Q3_K_L", "Q3_K",
        "Q4_K_M", "Q4_K_S", "Q4_K", "Q5_K_M", "Q5_K_S", "Q5_K", "Q6_K",
        "Q8_0", "Q5_0", "Q5_1", "Q4_0", "Q4_1", "BF16", "F16", "F32",
    };
    char upper[YVEX_REMOTE_REPOSITORY_CAP];
    size_t index;

    remote_copy(out, capacity, "unknown");
    remote_upper_basename(upper, sizeof(upper), path);
    for (index = 0u; index < sizeof(qtypes) / sizeof(qtypes[0]); ++index) {
        if (strstr(upper, qtypes[index])) {
            remote_copy(out, capacity, qtypes[index]);
            return;
        }
    }
}

static int remote_add_file(yvex_remote_catalog *catalog,
                           unsigned long long model_index,
                           const char *path,
                           unsigned long long size,
                           int size_known)
{
    const char *extension = remote_extension(path);
    yvex_model_representation *representation;
    yvex_remote_file *file;
    char identity[YVEX_REMOTE_NAME_CAP];

    if (!remote_file_path_valid(path)) return 0;
    file = remote_catalog_add_file(catalog, model_index);
    if (!file) return 0;
    remote_copy(file->path, sizeof(file->path), path);
    file->kind = remote_file_kind(path);
    file->size_bytes = size;
    file->size_known = size_known;

    if (strcasecmp(extension, ".safetensors") == 0) {
        representation = remote_representation_find(
            catalog, model_index, YVEX_MODEL_REPRESENTATION_SAFETENSORS,
            "safetensors-source");
        if (!representation)
            representation = remote_catalog_add_representation(
                catalog, model_index, YVEX_MODEL_REPRESENTATION_SAFETENSORS,
                "safetensors-source");
        if (!representation) return 0;
        if (!representation->precision[0]) {
            remote_copy(representation->precision, sizeof(representation->precision),
                        "unknown");
            remote_copy(representation->precision_evidence,
                        sizeof(representation->precision_evidence), "not-inspected");
            remote_copy(representation->compatibility,
                        sizeof(representation->compatibility),
                        catalog->models[model_index].family[0]
                            ? "source-ingest-only"
                            : "source-role-inspection-required");
            remote_copy(representation->recommendation,
                        sizeof(representation->recommendation),
                        "inspect repository metadata before selecting source payloads");
            representation->provisional = 1;
        }
        remote_copy(representation->file_pattern, sizeof(representation->file_pattern),
                    "*.safetensors");
        remote_copy(file->representation, sizeof(file->representation),
                    representation->identity);
    } else if (strcasecmp(extension, ".gguf") == 0) {
        char precision[YVEX_REMOTE_PRECISION_CAP];
        remote_gguf_precision(precision, sizeof(precision), path);
        snprintf(identity, sizeof(identity), "gguf-%s", precision);
        representation = remote_catalog_add_representation(
            catalog, model_index, YVEX_MODEL_REPRESENTATION_GGUF, identity);
        if (!representation) return 0;
        remote_copy(representation->precision, sizeof(representation->precision), precision);
        remote_copy(representation->precision_evidence,
                    sizeof(representation->precision_evidence), "filename-hint");
        if (!representation->file_count)
            remote_copy(representation->file_pattern, sizeof(representation->file_pattern), path);
        else if (strcmp(representation->file_pattern, path) != 0)
            representation->file_pattern[0] = '\0';
        remote_copy(representation->compatibility, sizeof(representation->compatibility),
                    "acquire-and-inspect-required");
        remote_copy(representation->recommendation, sizeof(representation->recommendation),
                    "provisional qtype; inspect the acquired GGUF before admission");
        representation->provisional = 1;
        representation->direct_admission_requires_inspection = 1;
        remote_copy(file->representation, sizeof(file->representation),
                    representation->identity);
    } else {
        remote_copy(file->representation, sizeof(file->representation),
                    yvex_remote_file_kind_name(file->kind));
        return 1;
    }
    representation->file_count++;
    if (size_known) {
        if (ULLONG_MAX - representation->size_bytes < size) return 0;
        representation->size_bytes += size;
        representation->size_known = 1;
    }
    return 1;
}

static int remote_parse_content_identity(yvex_json *json, char sha256[YVEX_SHA256_HEX_CAP])
{
    yvex_json_iter object;
    yvex_json_item item;
    char key[YVEX_JSON_KEY_CAP];
    if (!yvex_json_iter_begin(json, &object, YVEX_JSON_COLLECTION_OBJECT)) return 0;
    while ((item = yvex_json_object_member(&object, key, sizeof(key))) == YVEX_JSON_ITEM_READY) {
        if (!strcmp(key, "oid") || !strcmp(key, "sha256")) {
            char value[YVEX_SHA256_HEX_CAP];
            if (!yvex_json_string(json, value, sizeof(value))) return 0;
            if (!yvex_sha256_hex_is_valid(value) || (sha256[0] && strcmp(sha256, value))) return 0;
            remote_copy(sha256, YVEX_SHA256_HEX_CAP, value);
        } else if (!yvex_json_skip_value(json)) return 0;
    }
    return item == YVEX_JSON_ITEM_END && !object.trailing_separator;
}

static int remote_parse_file_object(yvex_json *json,
                                    yvex_remote_catalog *catalog,
                                    unsigned long long model_index)
{
    yvex_json_iter object;
    yvex_json_item item;
    char key[YVEX_JSON_KEY_CAP];
    char path[YVEX_REMOTE_REPOSITORY_CAP] = "";
    char sha256[YVEX_SHA256_HEX_CAP] = "";
    unsigned long long size = 0u;
    int size_known = 0;

    if (!yvex_json_iter_begin(json, &object, YVEX_JSON_COLLECTION_OBJECT)) return 0;
    while ((item = yvex_json_object_member(&object, key, sizeof(key))) == YVEX_JSON_ITEM_READY) {
        if (strcmp(key, "path") == 0) {
            if (!yvex_json_string(json, path, sizeof(path))) return 0;
        } else if (strcmp(key, "size") == 0) {
            if (!yvex_json_u64(json, &size)) return 0;
            size_known = 1;
        } else if (!strcmp(key, "lfs")) {
            if (!remote_parse_content_identity(json, sha256)) return 0;
        } else if (!yvex_json_skip_value(json)) {
            return 0;
        }
    }
    if (item != YVEX_JSON_ITEM_END || object.trailing_separator) return 0;
    if (!path[0]) return 1;
    if (!remote_add_file(catalog, model_index, path, size, size_known)) return 0;
    remote_copy(catalog->file_sha256[catalog->file_count - 1u], YVEX_SHA256_HEX_CAP, sha256);
    return 1;
}

/* A numbered, complete shard population and a standalone file in the same directory
 * are separate acquisition candidates. This is representation evidence only: source
 * admission must still authenticate the index, headers and full tensor population. */
static int remote_shard_name(const char *path, char *prefix, size_t capacity,
                              unsigned int *ordinal, unsigned int *total)
{
    size_t length = strlen(path), index;
    const size_t suffix = sizeof("-00001-of-00003.safetensors") - 1u;
    const char *tail;

    /* -00001-of-00003.safetensors */
    if (length <= suffix) return 0;
    tail = path + length - suffix;
    if (tail[0] != '-' || memcmp(tail + 6u, "-of-", 4u) ||
        strcmp(tail + 15u, ".safetensors")) return 0;
    *ordinal = 0u; *total = 0u;
    for (index = 1u; index < 6u; ++index) {
        if (!isdigit((unsigned char)tail[index]) ||
            !isdigit((unsigned char)tail[index + 9u])) return 0;
        *ordinal = *ordinal * 10u + (unsigned int)(tail[index] - '0');
        *total = *total * 10u + (unsigned int)(tail[index + 9u] - '0');
    }
    if (!*ordinal || *ordinal > *total || *total > REMOTE_FILE_CAP ||
        (size_t)(tail - path) >= capacity) return 0;
    memcpy(prefix, path, (size_t)(tail - path));
    prefix[tail - path] = '\0';
    return 1;
}

static int remote_same_directory(const char *left, const char *right)
{
    const char *a = strrchr(left, '/'), *b = strrchr(right, '/');
    size_t n = a ? (size_t)(a - left) + 1u : 0u;
    size_t m = b ? (size_t)(b - right) + 1u : 0u;
    return n == m && !memcmp(left, right, n);
}

static int remote_separate_safetensors(yvex_remote_catalog *catalog,
                                        unsigned long long model_index,
                                        yvex_error *err)
{
    yvex_remote_model *model = &catalog->models[model_index];
    unsigned long long offset = catalog->file_offsets[model_index];
    unsigned int i, j;
    yvex_model_representation *source = remote_representation_find(
        catalog, model_index, YVEX_MODEL_REPRESENTATION_SAFETENSORS,
        "safetensors-source");
    yvex_model_representation template;

    if (!source) return YVEX_OK;
    template = *source;
    for (i = 0u; i < model->available_file_count; ++i) {
        yvex_remote_file *file = &catalog->files[offset + i];
        char prefix[YVEX_REMOTE_REPOSITORY_CAP];
        unsigned int ordinal, total;
        if (file->kind != YVEX_REMOTE_FILE_SAFETENSORS) continue;
        for (j = 0u; j < i; ++j)
            if (!strcmp(file->path, catalog->files[offset + j].path))
                return remote_refuse(err, YVEX_ERR_FORMAT, "duplicate provider payload path");
        if (remote_shard_name(file->path, prefix, sizeof(prefix), &ordinal, &total)) {
            unsigned int found = 0u, k;
            for (k = 0u; k < model->available_file_count; ++k) {
                char other[YVEX_REMOTE_REPOSITORY_CAP];
                unsigned int number, population;
                if (remote_shard_name(catalog->files[offset + k].path, other,
                                      sizeof(other), &number, &population) &&
                    !strcmp(prefix, other)) {
                    if (population != total)
                        return remote_refuse(err, YVEX_ERR_FORMAT,
                                             "inconsistent provider shard population");
                    found++;
                }
            }
            if (found != total)
                return remote_refuse(err, YVEX_ERR_FORMAT, "incomplete provider shard population");
            continue;
        }
        for (j = 0u; j < model->available_file_count; ++j) {
            const yvex_remote_file *other = &catalog->files[offset + j];
            if (remote_same_directory(file->path, other->path) &&
                remote_shard_name(other->path, prefix, sizeof(prefix), &ordinal, &total)) {
                char identity[YVEX_REMOTE_NAME_CAP], digest[YVEX_SHA256_HEX_BYTES];
                yvex_model_representation *alternative;
                yvex_sha256 hash;
                unsigned char bytes[YVEX_SHA256_DIGEST_BYTES];
                yvex_sha256_init(&hash);
                yvex_sha256_update(&hash, file->path, strlen(file->path));
                yvex_sha256_final(&hash, bytes);
                yvex_sha256_hex(bytes, digest);
                snprintf(identity, sizeof(identity), "safetensors-file-%s", digest);
                alternative = remote_catalog_add_representation(
                    catalog, model_index, YVEX_MODEL_REPRESENTATION_SAFETENSORS, identity);
                if (!alternative) return remote_refuse(err, YVEX_ERR_NOMEM,
                                                       "too many acquisition representations");
                *alternative = template;
                remote_copy(alternative->identity, sizeof(alternative->identity), identity);
                remote_copy(alternative->file_pattern, sizeof(alternative->file_pattern), file->path);
                alternative->file_count = 1u;
                alternative->size_bytes = file->size_bytes;
                alternative->size_known = file->size_known;
                alternative->package_preparation_supported = 0;
                alternative->provisional = 1;
                remote_copy(alternative->recommendation, sizeof(alternative->recommendation),
                            "standalone alternative; tensor topology requires independent admission");
                remote_copy(file->representation, sizeof(file->representation), identity);
                source = remote_representation_find(catalog, model_index,
                    YVEX_MODEL_REPRESENTATION_SAFETENSORS, "safetensors-source");
                source->file_count--;
                source->size_bytes -= file->size_bytes;
                break;
            }
        }
    }
    return YVEX_OK;
}

static int remote_parse_files(yvex_remote_catalog *catalog,
                              unsigned long long model_index,
                              const char *output,
                              size_t output_length,
                              yvex_error *err)
{
    yvex_json json;
    yvex_json_iter array;
    yvex_json_item item;
    unsigned int count = 0u;

    yvex_json_init(&json, output, output_length);
    if (!yvex_json_iter_begin(&json, &array, YVEX_JSON_COLLECTION_ARRAY))
        return remote_refuse(err, YVEX_ERR_FORMAT, "provider file listing is not a JSON array");
    while ((item = yvex_json_array_value(&array)) == YVEX_JSON_ITEM_READY) {
        if (++count > REMOTE_FILE_CAP || !remote_parse_file_object(&json, catalog, model_index))
            return remote_refuse(err, YVEX_ERR_FORMAT, "provider file listing is malformed or oversized");
    }
    if (item != YVEX_JSON_ITEM_END || array.trailing_separator || !yvex_json_complete(&json))
        return remote_refuse(err, YVEX_ERR_FORMAT, "provider file listing JSON is incomplete");
    return remote_separate_safetensors(catalog, model_index, err);
}

static int remote_run_policy(const char *const *arguments,
                             int anonymous, int offline,
                      const char *not_found_reason,
                      char **output,
                      size_t *output_length,
                      yvex_error *err)
{
    yvex_account_observe_options observe;
    yvex_account_observation account;
    yvex_account_capture_options capture;
    char *stdout_bytes;
    char *stderr_bytes;
    size_t index;
    int rc;

    memset(&observe, 0, sizeof(observe));
    observe.provider = YVEX_ACCOUNT_PROVIDER_HUGGINGFACE;
    rc = yvex_account_observe(&observe, &account, err);
    if (rc != YVEX_OK) return rc;
    if (!account.cli_present)
        return remote_refuse(err, YVEX_ERR_STATE, "Hugging Face CLI is unavailable");
    stdout_bytes = calloc(1u, REMOTE_OUTPUT_CAP);
    stderr_bytes = calloc(1u, REMOTE_ERROR_CAP);
    if (!stdout_bytes || !stderr_bytes) {
        free(stdout_bytes);
        free(stderr_bytes);
        return remote_refuse(err, YVEX_ERR_NOMEM, "provider output allocation failed");
    }
    memset(&capture, 0, sizeof(capture));
    capture.args[0] = account.cli_path;
    for (index = 0u; arguments[index] && index + 2u < YVEX_ACCOUNT_ARG_CAP; ++index)
        capture.args[index + 1u] = arguments[index];
    capture.stdout_bytes = stdout_bytes;
    capture.stdout_capacity = REMOTE_OUTPUT_CAP;
    capture.stderr_bytes = stderr_bytes;
    capture.stderr_capacity = REMOTE_ERROR_CAP;
    rc = yvex_provider_capture(&capture, anonymous, offline, err);
    if (rc != YVEX_OK) {
        free(stdout_bytes);
        free(stderr_bytes);
        return rc;
    }
    if (capture.stdout_truncated || capture.stderr_truncated) {
        free(stdout_bytes);
        free(stderr_bytes);
        return remote_refuse(err, YVEX_ERR_BOUNDS, "provider output exceeded the discovery limit");
    }
    if (capture.exit_code != 0) {
        const char *reason = strstr(stderr_bytes, "gated") || strstr(stderr_bytes, "401") ||
                                     strstr(stderr_bytes, "auth")
                                 ? "provider authentication is required"
                                 : strstr(stderr_bytes, "not found") || strstr(stderr_bytes, "404")
                                       ? not_found_reason
                                       : "provider operation failed";
        free(stdout_bytes);
        free(stderr_bytes);
        return remote_refuse(err, YVEX_ERR_STATE, reason);
    }
    free(stderr_bytes);
    *output = stdout_bytes;
    *output_length = capture.stdout_count;
    return YVEX_OK;
}

static int remote_run(const char *const *arguments, const char *not_found_reason,
                        char **output, size_t *output_length, yvex_error *err)
{
    return remote_run_policy(arguments, 0, 0, not_found_reason, output, output_length, err);
}

static void remote_normalize_name(char *out, size_t capacity, const char *value)
{
    size_t written = 0u;

    while (value && *value && written + 1u < capacity) {
        unsigned char byte = (unsigned char)*value++;
        if (isalnum(byte)) out[written++] = (char)tolower(byte);
    }
    out[written] = '\0';
}

static unsigned int remote_rank_score(const yvex_remote_model *model, const char *query)
{
    const char *name = strrchr(model->repository, '/');
    char normalized_query[257];
    char normalized_name[YVEX_REMOTE_REPOSITORY_CAP];
    unsigned int score = 0u;

    remote_normalize_name(normalized_query, sizeof(normalized_query), query);
    remote_normalize_name(normalized_name, sizeof(normalized_name), name ? name + 1 : model->repository);
    if (normalized_query[0] && strcmp(normalized_query, normalized_name) == 0) score += 4000u;
    else if (normalized_query[0] && strstr(normalized_name, normalized_query))
        score += 2500u;
    if (model->canonical) score += 3500u;
    switch (model->kind) {
    case YVEX_REMOTE_MODEL_FULL: score += 2000u; break;
    case YVEX_REMOTE_MODEL_CONVERSION: score += 1200u; break;
    case YVEX_REMOTE_MODEL_DERIVATIVE: score += 500u; break;
    case YVEX_REMOTE_MODEL_ADAPTER: score += 300u; break;
    case YVEX_REMOTE_MODEL_COMPONENT: score += 200u; break;
    case YVEX_REMOTE_MODEL_DELTA: score += 100u; break;
    case YVEX_REMOTE_MODEL_UNKNOWN: break;
    }
    if (model->family[0]) score += 400u;
    score += (unsigned int)model->support_stage * 100u;
    if (model->parameter_count_known) score += 50u;
    if (model->provider_rank && model->provider_rank <= 50u) score += 51u - model->provider_rank;
    return score;
}

typedef struct {
    yvex_remote_model model;
    unsigned long long representation_offset;
    unsigned long long file_offset;
} remote_model_slot;

static int remote_slot_compare(const void *left, const void *right)
{
    const remote_model_slot *a = left;
    const remote_model_slot *b = right;

    if (a->model.ranking_score != b->model.ranking_score)
        return a->model.ranking_score > b->model.ranking_score ? -1 : 1;
    if (a->model.provider_rank != b->model.provider_rank)
        return a->model.provider_rank < b->model.provider_rank ? -1 : 1;
    return strcmp(a->model.repository, b->model.repository);
}

static int remote_catalog_rank(yvex_remote_catalog *catalog, const char *query)
{
    remote_model_slot *slots;
    unsigned long long index;

    if (!catalog->model_count) return 1;
    slots = calloc((size_t)catalog->model_count, sizeof(*slots));
    if (!slots) return 0;
    for (index = 0u; index < catalog->model_count; ++index) {
        catalog->models[index].ranking_score = remote_rank_score(&catalog->models[index], query);
        slots[index].model = catalog->models[index];
        slots[index].representation_offset = catalog->representation_offsets[index];
        slots[index].file_offset = catalog->file_offsets[index];
    }
    qsort(slots, (size_t)catalog->model_count, sizeof(*slots), remote_slot_compare);
    for (index = 0u; index < catalog->model_count; ++index) {
        catalog->models[index] = slots[index].model;
        catalog->representation_offsets[index] = slots[index].representation_offset;
        catalog->file_offsets[index] = slots[index].file_offset;
    }
    free(slots);
    return 1;
}

static void remote_catalog_page(yvex_remote_catalog *catalog,
                                unsigned int skip,
                                unsigned int take)
{
    unsigned long long retained;

    if (skip >= catalog->model_count) {
        catalog->model_count = 0u;
        return;
    }
    retained = catalog->model_count - skip;
    if (retained > take) retained = take;
    memmove(catalog->models, catalog->models + skip, (size_t)retained * sizeof(*catalog->models));
    memmove(catalog->representation_offsets, catalog->representation_offsets + skip,
            (size_t)retained * sizeof(*catalog->representation_offsets));
    memmove(catalog->file_offsets, catalog->file_offsets + skip,
            (size_t)retained * sizeof(*catalog->file_offsets));
    catalog->model_count = retained;
}

int yvex_remote_model_search(yvex_remote_catalog **out,
                             const yvex_remote_search_options *options,
                             yvex_error *err)
{
    yvex_remote_catalog *catalog;
    const char *arguments[24];
    char limit[16];
    char *output = NULL;
    size_t output_length = 0u;
    unsigned int page;
    unsigned int page_size;
    unsigned int cumulative;
    unsigned int provider_limit;
    unsigned int argument_count = 0u;
    int rc;

    if (!out || !options ||
        (options->provider != YVEX_ACCOUNT_PROVIDER_UNKNOWN &&
         options->provider != YVEX_ACCOUNT_PROVIDER_HUGGINGFACE) ||
        (options->query && !remote_text_valid(options->query, 256u)) ||
        (options->author && !remote_text_valid(options->author, 128u)) ||
        (options->filter && !remote_text_valid(options->filter, 128u)))
        return remote_refuse(err, YVEX_ERR_INVALID_ARG, "bounded Hugging Face search options are required");
    *out = NULL;
    page = options->page ? options->page : 1u;
    page_size = options->page_size ? options->page_size : 8u;
    if (page > 20u || page_size > 50u || page > REMOTE_MODEL_CAP / page_size)
        return remote_refuse(err, YVEX_ERR_BOUNDS, "search page exceeds the bounded catalog window");
    cumulative = page * page_size;
    provider_limit = cumulative < 50u ? 50u : cumulative;
    snprintf(limit, sizeof(limit), "%u", provider_limit);
    arguments[argument_count++] = "models";
    arguments[argument_count++] = "list";
    if (options->query && options->query[0]) {
        arguments[argument_count++] = "--search";
        arguments[argument_count++] = options->query;
    }
    if (options->author && options->author[0]) {
        arguments[argument_count++] = "--author";
        arguments[argument_count++] = options->author;
    }
    if (options->filter && options->filter[0]) {
        arguments[argument_count++] = "--filter";
        arguments[argument_count++] = options->filter;
    }
    arguments[argument_count++] = "--limit";
    arguments[argument_count++] = limit;
    arguments[argument_count++] = "--expand";
    arguments[argument_count++] =
        "author,baseModels,gated,pipeline_tag,safetensors,sha,tags,transformersInfo";
    arguments[argument_count++] = "--format";
    arguments[argument_count++] = "json";
    arguments[argument_count] = NULL;
    rc = remote_run(arguments, "remote model was not found",
                    &output, &output_length, err);
    if (rc != YVEX_OK) return rc;
    catalog = calloc(1u, sizeof(*catalog));
    if (!catalog) {
        free(output);
        return remote_refuse(err, YVEX_ERR_NOMEM, "remote catalog allocation failed");
    }
    remote_copy(catalog->query, sizeof(catalog->query), options->query);
    rc = remote_parse_search_output(catalog, output, output_length, err);
    free(output);
    if (rc != YVEX_OK) {
        yvex_remote_catalog_close(catalog);
        return rc;
    }
    if (!remote_catalog_rank(catalog, options->query)) {
        yvex_remote_catalog_close(catalog);
        return remote_refuse(err, YVEX_ERR_NOMEM, "remote ranking allocation failed");
    }
    remote_catalog_page(catalog, (page - 1u) * page_size, page_size);
    *out = catalog;
    return YVEX_OK;
}

int yvex_model_remote_inspect_policy(yvex_remote_catalog **out,
                              const yvex_remote_inspect_options *options,
                              int anonymous, yvex_error *err)
{
    const char *info_arguments[16];
    const char *file_arguments[16];
    yvex_remote_catalog *catalog;
    parsed_remote_model parsed;
    yvex_json json;
    char *output = NULL;
    size_t output_length = 0u;
    unsigned long long model_index;
    unsigned int count = 0u;
    int rc;

    if (!out || !options || !remote_repository_valid(options->repository) ||
        (options->revision && !remote_text_valid(options->revision, 127u)) ||
        (options->provider != YVEX_ACCOUNT_PROVIDER_UNKNOWN &&
         options->provider != YVEX_ACCOUNT_PROVIDER_HUGGINGFACE))
        return remote_refuse(err, YVEX_ERR_INVALID_ARG, "repository and bounded revision are required");
    *out = NULL;
    info_arguments[count++] = "models";
    info_arguments[count++] = "info";
    info_arguments[count++] = options->repository;
    if (options->revision && options->revision[0]) {
        info_arguments[count++] = "--revision";
        info_arguments[count++] = options->revision;
    }
    info_arguments[count++] = "--expand";
    info_arguments[count++] =
        "author,baseModels,gated,pipeline_tag,safetensors,sha,tags,transformersInfo";
    info_arguments[count++] = "--format";
    info_arguments[count++] = "json";
    info_arguments[count] = NULL;
    rc = remote_run_policy(info_arguments, anonymous, 0,
                    options->revision && options->revision[0]
                        ? "remote revision or reference was not found"
                        : "remote model was not found",
                    &output, &output_length, err);
    if (rc != YVEX_OK) return rc;
    yvex_json_init(&json, output, output_length);
    if (!remote_parse_model(&json, &parsed) || !yvex_json_complete(&json)) {
        free(output);
        return remote_refuse(err, YVEX_ERR_FORMAT, "provider inspect returned malformed model metadata");
    }
    free(output);
    if (!remote_revision_immutable(parsed.model.resolved_revision))
        return remote_refuse(err, YVEX_ERR_STATE,
                             "provider inspection did not resolve an immutable revision");
    if (options->revision && remote_revision_immutable(options->revision) &&
        strcmp(options->revision, parsed.model.resolved_revision) != 0)
        return remote_refuse(err, YVEX_ERR_STATE,
                             "provider resolved revision does not match the requested identity");
    if (options->revision && options->revision[0])
        remote_copy(parsed.model.revision_reference,
                    sizeof(parsed.model.revision_reference), options->revision);
    catalog = calloc(1u, sizeof(*catalog));
    if (!catalog || !remote_add_parsed(catalog, &parsed, 0, &model_index)) {
        yvex_remote_catalog_close(catalog);
        return remote_refuse(err, YVEX_ERR_NOMEM, "remote catalog allocation failed");
    }
    count = 0u;
    file_arguments[count++] = "models";
    file_arguments[count++] = "list";
    file_arguments[count++] = options->repository;
    file_arguments[count++] = "-R";
    file_arguments[count++] = "--revision";
    file_arguments[count++] = parsed.model.resolved_revision;
    file_arguments[count++] = "--format";
    file_arguments[count++] = "json";
    file_arguments[count] = NULL;
    rc = remote_run_policy(file_arguments, anonymous, 0,
                    "remote representation listing was not found",
                    &output, &output_length, err);
    if (rc == YVEX_OK)
        rc = remote_parse_files(catalog, model_index, output, output_length, err);
    free(output);
    if (rc != YVEX_OK) {
        yvex_remote_catalog_close(catalog);
        return rc;
    }
    *out = catalog;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_remote_model_inspect(yvex_remote_catalog **out,
                              const yvex_remote_inspect_options *options, yvex_error *err)
{
    return yvex_model_remote_inspect_policy(out, options, 0, err);
}

void yvex_remote_catalog_close(yvex_remote_catalog *catalog)
{
    if (!catalog) return;
    free(catalog->models);
    free(catalog->representations);
    free(catalog->representation_offsets);
    free(catalog->files);
    free(catalog->file_sha256);
    free(catalog->file_offsets);
    free(catalog);
}

const yvex_remote_file *yvex_remote_catalog_file_at(const yvex_remote_catalog *catalog,
                                                    unsigned long long model_index,
                                                    unsigned int file_index)
{
    if (!catalog || model_index >= catalog->model_count ||
        file_index >= catalog->models[model_index].available_file_count)
        return NULL;
    return &catalog->files[catalog->file_offsets[model_index] + file_index];
}

const char *yvex_model_remote_file_sha256(const yvex_remote_catalog *catalog,
                                          unsigned long long model_index, unsigned int file_index)
{
    if (!yvex_remote_catalog_file_at(catalog, model_index, file_index)) return NULL;
    return catalog->file_sha256[catalog->file_offsets[model_index] + file_index];
}

unsigned long long yvex_remote_catalog_count(const yvex_remote_catalog *catalog)
{
    return catalog ? catalog->model_count : 0u;
}

unsigned long long yvex_remote_catalog_provider_count(const yvex_remote_catalog *catalog)
{
    return catalog ? catalog->provider_result_count : 0u;
}

const char *yvex_remote_catalog_query(const yvex_remote_catalog *catalog)
{
    return catalog ? catalog->query : "";
}

const yvex_remote_model *yvex_remote_catalog_at(const yvex_remote_catalog *catalog,
                                                unsigned long long index)
{
    return catalog && index < catalog->model_count ? &catalog->models[index] : NULL;
}

const yvex_model_representation *yvex_remote_catalog_representation_at(
    const yvex_remote_catalog *catalog,
    unsigned long long model_index,
    unsigned int representation_index)
{
    if (!catalog || model_index >= catalog->model_count ||
        representation_index >= catalog->models[model_index].representation_count)
        return NULL;
    return &catalog->representations[catalog->representation_offsets[model_index] +
                                     representation_index];
}

/* Ask the provider's supported offline client for its own cache location. */
static int remote_cached_file(const yvex_model_publication *remote, const char *directory,
                               int anonymous, int *found, yvex_error *err)
{
    const char *arguments[] = {"download", remote->repository, remote->filename,
                               "--revision", remote->revision, "--quiet", NULL};
    char *output = NULL;
    char destination[YVEX_PATH_CAP];
    size_t length = 0u;
    int rc;
    *found = 0;
    rc = remote_run_policy(arguments, anonymous, 1, "cache entry absent", &output, &length, err);
    if (rc != YVEX_OK) { yvex_error_clear(err); return YVEX_OK; }
    while (length && (output[length - 1u] == '\n' || output[length - 1u] == '\r')) output[--length] = '\0';
    if (!length || output[0] != '/' || strchr(output, '\n') || strchr(output, '\r')) {
        free(output);
        return YVEX_OK;
    }
    if (snprintf(destination, sizeof(destination), "%s/%s", directory, remote->filename) >= (int)sizeof(destination)) {
        free(output);
        return remote_refuse(err, YVEX_ERR_BOUNDS, "cache staging path exceeds bound");
    }
    /* Leave an interrupted local-dir transfer to the provider's own recovery path. */
    if (access(destination, F_OK) == 0) { free(output); return YVEX_OK; }
    rc = yvex_source_stage_file(output, destination, err);
    free(output);
    if (rc == YVEX_OK) *found = 1;
    return rc;
}

/* Transfer one pinned representation through the official resumable Hub client. */
int yvex_model_remote_file_download(const yvex_model_publication *remote,
                                    const char *directory, int anonymous, yvex_error *err)
{
    const char *arguments[10];
    char *output = NULL;
    size_t length = 0u;
    int cached = 0, rc;
    if (!remote || !directory || strcmp(remote->provider, "huggingface") ||
        !remote_repository_valid(remote->repository) || !remote_revision_immutable(remote->revision) ||
        !remote->filename[0] || remote->filename[0] == '/' || strstr(remote->filename, ".."))
        return remote_refuse(err, YVEX_ERR_INVALID_ARG, "exact remote file and transient directory required");
    rc = remote_cached_file(remote, directory, anonymous, &cached, err);
    if (rc != YVEX_OK || cached) return rc;
    arguments[0] = "download";
    arguments[1] = remote->repository;
    arguments[2] = remote->filename;
    arguments[3] = "--revision";
    arguments[4] = remote->revision;
    arguments[5] = "--local-dir";
    arguments[6] = directory;
    arguments[7] = "--quiet";
    arguments[8] = NULL;
    rc = remote_run_policy(arguments, anonymous, 0, "remote artifact was not found", &output, &length, err);
    free(output);
    return rc;
}
