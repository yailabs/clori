/*
 * Compose family registration entrypoints without importing either family's representation.
 * Compiler and product callers resolve immutable typed bindings through this one catalog; adding
 * a family extends composition here rather than teaching a different family how to dispatch it.
 */
#include <yvex/internal/family_catalog.h>

#include <string.h>

typedef const yvex_graph_execution_binding *(*execution_provider)(void);
typedef const yvex_component_variant_adapter *(*component_provider)(void);

static const execution_provider execution_providers[] = {
    yvex_graph_deepseek_v4_execution_binding,
};

static const component_provider component_providers[] = {
    yvex_graph_minimax_h3_component_adapter,
};

const yvex_graph_execution_binding *yvex_graph_execution_find(
    unsigned long long adapter_id, unsigned long long adapter_version,
    const char *target_id)
{
    size_t index;

    for (index = 0u;
         index < sizeof(execution_providers) / sizeof(execution_providers[0]);
         ++index) {
        const yvex_graph_execution_binding *binding = execution_providers[index]();

        if (!binding || binding->schema_version != YVEX_GRAPH_EXECUTION_BINDING_SCHEMA_V1)
            continue;
        if ((target_id && strcmp(target_id, binding->target_id) == 0) ||
            (!target_id && adapter_id == binding->adapter_id &&
             adapter_version == binding->adapter_version))
            return binding;
    }
    return NULL;
}

const yvex_component_variant_adapter *yvex_graph_component_variant_find(
    const char *target_id)
{
    size_t index;

    if (!target_id) return NULL;
    for (index = 0u;
         index < sizeof(component_providers) / sizeof(component_providers[0]);
         ++index) {
        const yvex_component_variant_adapter *adapter = component_providers[index]();

        if (adapter && adapter->schema_version == YVEX_PHYSICAL_VARIANT_SESSION_SCHEMA_V1 &&
            strcmp(target_id, adapter->target_id) == 0)
            return adapter;
    }
    return NULL;
}
