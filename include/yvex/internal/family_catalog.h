/* Family registration providers are visible only to the immutable composition catalog. */
#ifndef INCLUDE_YVEX_INTERNAL_FAMILY_CATALOG_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_FAMILY_CATALOG_H_INCLUDED

#include <yvex/internal/graph.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_FAMILY_SOURCE_ADAPTER_SCHEMA_V1 1u
typedef struct {
    void *owner;
    void (*release)(void *owner);
    const struct yvex_source_verification *verification;
    const struct yvex_compilation_source_summary *source_summary;
    const yvex_semantic_model_ir *semantic_model;
    const struct yvex_transform_ir *transform_ir;
    const struct yvex_artifact_lowering_map *lowering;
    char derivation_identity[YVEX_SHA256_HEX_BYTES];
} yvex_family_source_products;
typedef struct {
    unsigned int schema_version;
    const char *target_id, *family;
    const char *tokenizer_architecture, *tokenizer_pre;
    int (*tokenizer_policy)(struct yvex_tokenizer_family_policy *, yvex_error *);
    int (*compile)(yvex_family_source_products *out,
                   const struct yvex_compilation_runtime_binding_request *request,
                   yvex_error *err);
} yvex_family_source_adapter;

#define YVEX_FAMILY_DESCRIPTOR_SCHEMA_V1 1u
typedef const yvex_graph_execution_binding *(*yvex_family_execution_provider)(void);
typedef const yvex_component_variant_adapter *(*yvex_family_component_provider)(void);
typedef const yvex_quant_preset_catalog *(*yvex_family_quant_provider)(void);
typedef const yvex_family_source_adapter *(*yvex_family_source_provider)(void);
typedef struct yvex_family_descriptor {
    unsigned int schema_version;
    const char *target_id, *family, *tokenizer_architecture, *tokenizer_pre;
    yvex_family_execution_provider execution;
    yvex_family_component_provider component;
    yvex_family_quant_provider quant_presets;
    yvex_family_source_provider source;
} yvex_family_descriptor;

static inline const yvex_family_descriptor *yvex_family_descriptor_find_registered(
    const yvex_family_descriptor *const *descriptors,
    unsigned long long count, const char *target_id)
{
    const yvex_family_descriptor *selected = NULL;
    unsigned long long index;

    if (!descriptors || !target_id || !target_id[0]) return NULL;
    for (index = 0ull; index < count; ++index) {
        const yvex_family_descriptor *descriptor = descriptors[index];

        if (!descriptor || descriptor->schema_version != YVEX_FAMILY_DESCRIPTOR_SCHEMA_V1 ||
            !descriptor->target_id || !descriptor->family ||
            (!descriptor->execution && !descriptor->component && !descriptor->source) ||
            strcmp(descriptor->target_id, target_id) != 0)
            continue;
        if (selected) return NULL;
        selected = descriptor;
    }
    return selected;
}

int yvex_family_source_compile(
    const char *target_id, const struct yvex_compilation_runtime_binding_request *request,
    yvex_family_source_products *products, yvex_error *err);
void yvex_family_source_products_release(yvex_family_source_products *products);
int yvex_family_tokenizer_open(
    struct yvex_tokenizer **out, const struct yvex_gguf *gguf, yvex_error *err);

#ifdef __cplusplus
}
#endif
#endif
