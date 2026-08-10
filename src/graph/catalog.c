/*
 * Compose family registration entrypoints without importing either family's representation.
 * Compiler and product callers resolve immutable typed bindings through this one catalog; adding
 * a family extends composition here rather than teaching a different family how to dispatch it.
 */
#include <yvex/internal/family_catalog.h>
#include <yvex/quant.h>

#include <string.h>

typedef const yvex_graph_execution_binding *(*execution_provider)(void);
typedef const yvex_component_variant_adapter *(*component_provider)(void);
typedef const yvex_quant_preset_catalog *(*quant_preset_provider)(void);

static const execution_provider execution_providers[] = {
    yvex_graph_deepseek_v4_execution_binding,
};

static const component_provider component_providers[] = {
    yvex_graph_minimax_h3_component_adapter,
};

static const quant_preset_provider quant_preset_providers[] = {
    yvex_graph_deepseek_v4_quant_presets,
};

static const yvex_quant_preset_catalog *quant_preset_catalog_at(size_t index)
{
    const yvex_quant_preset_catalog *catalog = index < sizeof(quant_preset_providers) /
        sizeof(quant_preset_providers[0]) ? quant_preset_providers[index]() : NULL;

    return catalog && catalog->schema_version == YVEX_QUANT_PRESET_CATALOG_SCHEMA_V1 &&
                   catalog->target_id && catalog->count && catalog->name && catalog->open
               ? catalog : NULL;
}

unsigned long long yvex_quant_policy_preset_count(void)
{
    unsigned long long count = 0u;
    size_t index;

    for (index = 0u; index < sizeof(quant_preset_providers) /
                                   sizeof(quant_preset_providers[0]); ++index) {
        const yvex_quant_preset_catalog *catalog = quant_preset_catalog_at(index);

        if (catalog) count += catalog->count();
    }
    return count;
}

const char *yvex_quant_policy_preset_name(unsigned long long ordinal)
{
    size_t index;

    for (index = 0u; index < sizeof(quant_preset_providers) /
                                   sizeof(quant_preset_providers[0]); ++index) {
        const yvex_quant_preset_catalog *catalog = quant_preset_catalog_at(index);
        unsigned long long count = catalog ? catalog->count() : 0u;

        if (ordinal < count) return catalog->name(ordinal);
        ordinal -= count;
    }
    return NULL;
}

int yvex_quant_policy_preset_open(
    yvex_quant_policy **out, const char *name, yvex_error *err)
{
    const yvex_quant_preset_catalog *selected = NULL;
    size_t provider;

    if (out) *out = NULL;
    if (!out || !name || !name[0]) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "quant_policy_preset",
                       "out and preset name are required");
        return YVEX_ERR_INVALID_ARG;
    }
    for (provider = 0u; provider < sizeof(quant_preset_providers) /
                                         sizeof(quant_preset_providers[0]); ++provider) {
        const yvex_quant_preset_catalog *catalog = quant_preset_catalog_at(provider);
        unsigned long long preset;

        for (preset = 0u; catalog && preset < catalog->count(); ++preset) {
            const char *candidate = catalog->name(preset);

            if (!candidate || strcmp(candidate, name) != 0) continue;
            if (selected) {
                yvex_error_setf(err, YVEX_ERR_STATE, "quant_policy_preset",
                                "ambiguous quantization preset: %s", name);
                return YVEX_ERR_STATE;
            }
            selected = catalog;
        }
    }
    if (!selected) {
        yvex_error_setf(err, YVEX_ERR_UNSUPPORTED, "quant_policy_preset",
                        "unknown quantization preset: %s", name);
        return YVEX_ERR_UNSUPPORTED;
    }
    return selected->open(out, name, err);
}

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
