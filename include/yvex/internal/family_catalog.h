/* Family registration providers are visible only to the immutable composition catalog. */
#ifndef INCLUDE_YVEX_INTERNAL_FAMILY_CATALOG_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_FAMILY_CATALOG_H_INCLUDED

#include <yvex/internal/graph.h>

#ifdef __cplusplus
extern "C" {
#endif

const yvex_graph_execution_binding *yvex_graph_deepseek_v4_execution_binding(void);
const yvex_component_variant_adapter *yvex_graph_minimax_h3_component_adapter(void);
const yvex_quant_preset_catalog *yvex_graph_deepseek_v4_quant_presets(void);

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
    const char *tokenizer_architecture;
    int (*tokenizer_policy)(struct yvex_tokenizer_family_policy *, yvex_error *);
    int (*compile)(yvex_family_source_products *out,
                   const struct yvex_compilation_runtime_binding_request *request,
                   yvex_error *err);
} yvex_family_source_adapter;
const yvex_family_source_adapter *yvex_graph_minimax_h3_source_adapter(void);
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
