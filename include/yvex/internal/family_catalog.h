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

#ifdef __cplusplus
}
#endif
#endif
