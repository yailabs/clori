/*
 * Compose family-owned descriptors without importing any family's representation.
 * The source-ownership projection supplies provider membership, so extending one family does not
 * require another handwritten catalog or switch in this generic owner.
 */
#include <yvex/internal/family_catalog.h>
#include <yvex/internal/compilation.h>
#include <yvex/internal/core.h>
#include <yvex/internal/tokenizer.h>
#include <yvex/gguf.h>
#include <yvex/quant.h>
#include <source/families.h>

#include <stdlib.h>
#include <string.h>

typedef int (*tokenizer_policy_provider)(yvex_tokenizer_family_policy *, yvex_error *);

#define DECLARE_FAMILY_DESCRIPTOR(name) \
    extern const yvex_family_descriptor yvex_graph_family_descriptor_##name;
YVEX_GRAPH_FAMILY_DESCRIPTORS(DECLARE_FAMILY_DESCRIPTOR)
#undef DECLARE_FAMILY_DESCRIPTOR

static const yvex_family_descriptor *const family_descriptors[] = {
#define FAMILY_DESCRIPTOR(name) &yvex_graph_family_descriptor_##name,
    YVEX_GRAPH_FAMILY_DESCRIPTORS(FAMILY_DESCRIPTOR)
#undef FAMILY_DESCRIPTOR
};

_Static_assert(sizeof(family_descriptors) / sizeof(family_descriptors[0]) ==
                   YVEX_GRAPH_FAMILY_DESCRIPTOR_COUNT,
               "generated graph family descriptor count is inconsistent");

static const yvex_family_descriptor *family_descriptor_at(size_t index)
{
    const yvex_family_descriptor *descriptor;
    const yvex_graph_execution_binding *execution;
    const yvex_component_variant_adapter *component;
    const yvex_family_source_adapter *source;

    if (index >= sizeof(family_descriptors) / sizeof(family_descriptors[0])) return NULL;
    descriptor = family_descriptors[index];
    if (!descriptor || descriptor->schema_version != YVEX_FAMILY_DESCRIPTOR_SCHEMA_V1 ||
        !descriptor->target_id || !descriptor->target_id[0] ||
        !descriptor->family || !descriptor->family[0] ||
        (!descriptor->execution && !descriptor->component && !descriptor->source))
        return NULL;
    execution = descriptor->execution ? descriptor->execution() : NULL;
    component = descriptor->component ? descriptor->component() : NULL;
    source = descriptor->source ? descriptor->source() : NULL;
    if ((descriptor->execution &&
         (!execution || execution->schema_version != YVEX_GRAPH_EXECUTION_BINDING_SCHEMA_V1 ||
          !execution->compiler || strcmp(execution->target_id, descriptor->target_id) != 0 ||
          strcmp(execution->compiler->family, descriptor->family) != 0)) ||
        (descriptor->component &&
         (!component || component->schema_version != YVEX_PHYSICAL_VARIANT_SESSION_SCHEMA_V1 ||
          !component->family || strcmp(component->target_id, descriptor->target_id) != 0 ||
          strcmp(component->family, descriptor->family) != 0)) ||
        (descriptor->source &&
         (!source || source->schema_version != YVEX_FAMILY_SOURCE_ADAPTER_SCHEMA_V1 ||
          !source->family || strcmp(source->target_id, descriptor->target_id) != 0 ||
          strcmp(source->family, descriptor->family) != 0)))
        return NULL;
    return descriptor;
}

static const yvex_family_descriptor *family_descriptor_find_target(const char *target_id)
{
    const yvex_family_descriptor *descriptors[YVEX_GRAPH_FAMILY_DESCRIPTOR_COUNT];
    size_t index;

    for (index = 0u; index < YVEX_GRAPH_FAMILY_DESCRIPTOR_COUNT; ++index)
        descriptors[index] = family_descriptor_at(index);
    return yvex_family_descriptor_find_registered(
        descriptors, YVEX_GRAPH_FAMILY_DESCRIPTOR_COUNT, target_id);
}

static int architecture_matches(
    const char *candidate, const char *architecture, unsigned long long count)
{
    return candidate && count == strlen(candidate) &&
           memcmp(candidate, architecture, (size_t)count) == 0;
}

static int catalog_tokenizer_policy(
    yvex_tokenizer_family_policy *policy, const yvex_gguf *gguf, yvex_error *err)
{
    const yvex_gguf_value *value = yvex_gguf_metadata_find(gguf, "general.architecture");
    const char *architecture = NULL;
    unsigned long long count = 0ull;
    tokenizer_policy_provider selected = NULL;
    size_t index;

    if (!policy || !gguf ||
        yvex_gguf_value_as_string(value, &architecture, &count) != YVEX_OK ||
        !architecture || !count) {
        yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "family.tokenizer-catalog",
                       "artifact architecture has no compiled tokenizer policy");
        return YVEX_ERR_UNSUPPORTED;
    }
    for (index = 0u; index < sizeof(family_descriptors) /
                                      sizeof(family_descriptors[0]); ++index) {
        const yvex_family_descriptor *descriptor = family_descriptor_at(index);
        const yvex_graph_execution_binding *binding =
            descriptor && descriptor->execution ? descriptor->execution() : NULL;
        const yvex_family_compiler_adapter *compiler = binding ? binding->compiler : NULL;
        const yvex_family_binding_pipeline *pipeline =
            compiler ? compiler->binding_pipeline : NULL;
        const yvex_family_source_adapter *adapter =
            descriptor && descriptor->source ? descriptor->source() : NULL;

        if (!descriptor || !descriptor->tokenizer_architecture ||
            !architecture_matches(descriptor->tokenizer_architecture, architecture, count))
            continue;
        if (compiler && compiler->tokenizer_policy && pipeline &&
            architecture_matches(pipeline->tokenizer_architecture, architecture, count)) {
            if (selected) goto ambiguous;
            selected = compiler->tokenizer_policy;
            continue;
        }
        if (!adapter || !adapter->tokenizer_policy ||
            !architecture_matches(adapter->tokenizer_architecture, architecture, count))
            continue;
        if (selected) goto ambiguous;
        selected = adapter->tokenizer_policy;
    }
    if (!selected) {
        yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "family.tokenizer-catalog",
                       "artifact architecture has no compiled tokenizer policy");
        return YVEX_ERR_UNSUPPORTED;
    }
    if (!selected(policy, err)) {
        if (yvex_error_code(err) == YVEX_OK)
            yvex_error_set(err, YVEX_ERR_STATE, "family.tokenizer-catalog",
                           "family tokenizer policy compilation failed");
        return yvex_error_code(err);
    }
    return YVEX_OK;
ambiguous:
    yvex_error_set(err, YVEX_ERR_STATE, "family.tokenizer-catalog",
                   "artifact architecture has multiple tokenizer policy owners");
    return YVEX_ERR_STATE;
}

int yvex_family_tokenizer_open(
    yvex_tokenizer **out, const yvex_gguf *gguf, yvex_error *err)
{
    yvex_tokenizer_family_policy policy;
    int rc;

    if (out) *out = NULL;
    rc = catalog_tokenizer_policy(&policy, gguf, err);
    if (rc != YVEX_OK) return rc;
    return yvex_tokenizer_from_compiled_gguf(out, gguf, &policy, err);
}

typedef struct {
    const yvex_family_binding_pipeline *pipeline;
    yvex_family_compilation_source source;
    yvex_semantic_model_ir *semantic_model;
} catalog_source_owner;

static void catalog_source_release(void *pointer)
{
    catalog_source_owner *owner = pointer;

    if (!owner) return;
    yvex_semantic_model_ir_close(&owner->semantic_model);
    if (owner->pipeline && owner->pipeline->source_close)
        owner->pipeline->source_close(owner->source.owner);
    free(owner);
}

static int catalog_execution_source_compile(
    const yvex_graph_execution_binding *execution,
    const yvex_compilation_runtime_binding_request *request,
    yvex_family_source_products *products, yvex_error *err)
{
    const yvex_family_binding_pipeline *pipeline =
        execution && execution->compiler ? execution->compiler->binding_pipeline : NULL;
    catalog_source_owner *owner;
    const yvex_semantic_model_ir_summary *semantic;
    int rc;

    if (!pipeline || pipeline->schema_version != YVEX_FAMILY_BINDING_PIPELINE_SCHEMA_V1 ||
        !pipeline->source_open || !pipeline->source_close || !pipeline->semantic_model_build) {
        yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "family.source-catalog",
                       "family has no complete source compiler adapter");
        return YVEX_ERR_UNSUPPORTED;
    }
    owner = calloc(1u, sizeof(*owner));
    if (!owner) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "family.source-catalog",
                       "source compiler ownership allocation failed");
        return YVEX_ERR_NOMEM;
    }
    owner->pipeline = pipeline;
    rc = pipeline->source_open(&owner->source, request, err);
    if (rc == YVEX_OK)
        rc = pipeline->semantic_model_build(
            &owner->semantic_model, owner->source.verification, err);
    semantic = rc == YVEX_OK
                   ? yvex_semantic_model_ir_summary_get(owner->semantic_model) : NULL;
    if (rc != YVEX_OK || !semantic) {
        catalog_source_release(owner);
        return rc != YVEX_OK ? rc : YVEX_ERR_STATE;
    }
    products->owner = owner;
    products->release = catalog_source_release;
    products->verification = owner->source.verification;
    products->source_summary = owner->source.source_summary;
    products->semantic_model = owner->semantic_model;
    products->transform_ir = owner->source.transform_ir;
    products->lowering = owner->source.artifact_lowering;
    yvex_core_text_copy(products->derivation_identity,
                        sizeof(products->derivation_identity), semantic->identity);
    return YVEX_OK;
}

static const yvex_quant_preset_catalog *quant_preset_catalog_at(size_t index)
{
    const yvex_family_descriptor *descriptor = family_descriptor_at(index);
    const yvex_quant_preset_catalog *catalog =
        descriptor && descriptor->quant_presets ? descriptor->quant_presets() : NULL;

    return catalog && catalog->schema_version == YVEX_QUANT_PRESET_CATALOG_SCHEMA_V1 &&
                   catalog->target_id && catalog->count && catalog->name && catalog->open
               ? catalog : NULL;
}

unsigned long long yvex_quant_policy_preset_count(void)
{
    unsigned long long count = 0u;
    size_t index;

    for (index = 0u; index < sizeof(family_descriptors) /
                                   sizeof(family_descriptors[0]); ++index) {
        const yvex_quant_preset_catalog *catalog = quant_preset_catalog_at(index);

        if (catalog) count += catalog->count();
    }
    return count;
}

const char *yvex_quant_policy_preset_name(unsigned long long ordinal)
{
    size_t index;

    for (index = 0u; index < sizeof(family_descriptors) /
                                   sizeof(family_descriptors[0]); ++index) {
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
    for (provider = 0u; provider < sizeof(family_descriptors) /
                                         sizeof(family_descriptors[0]); ++provider) {
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
    const yvex_family_descriptor *descriptor;
    size_t index;

    if (target_id) {
        descriptor = family_descriptor_find_target(target_id);
        return descriptor && descriptor->execution ? descriptor->execution() : NULL;
    }
    for (index = 0u; index < sizeof(family_descriptors) / sizeof(family_descriptors[0]); ++index) {
        const yvex_family_descriptor *candidate = family_descriptor_at(index);
        const yvex_graph_execution_binding *binding =
            candidate && candidate->execution ? candidate->execution() : NULL;

        if (!binding || binding->schema_version != YVEX_GRAPH_EXECUTION_BINDING_SCHEMA_V1)
            continue;
        if (adapter_id == binding->adapter_id && adapter_version == binding->adapter_version)
            return binding;
    }
    return NULL;
}

const yvex_component_variant_adapter *yvex_graph_component_variant_find(
    const char *target_id)
{
    const yvex_family_descriptor *descriptor;

    if (!target_id) return NULL;
    descriptor = family_descriptor_find_target(target_id);
    return descriptor && descriptor->component ? descriptor->component() : NULL;
}

const yvex_component_variant_adapter *yvex_graph_component_variant_find_family(
    const char *family)
{
    size_t index;

    if (!family) return NULL;
    for (index = 0u; index < sizeof(family_descriptors) / sizeof(family_descriptors[0]); ++index) {
        const yvex_family_descriptor *descriptor = family_descriptor_at(index);
        const yvex_component_variant_adapter *adapter =
            descriptor && descriptor->component ? descriptor->component() : NULL;

        if (adapter && adapter->schema_version == YVEX_PHYSICAL_VARIANT_SESSION_SCHEMA_V1 &&
            adapter->family && strcmp(family, adapter->family) == 0)
            return adapter;
    }
    return NULL;
}

int yvex_family_source_compile(
    const char *target_id, const yvex_compilation_runtime_binding_request *request,
    yvex_family_source_products *products, yvex_error *err)
{
    const yvex_family_descriptor *descriptor;
    const yvex_graph_execution_binding *execution;
    const yvex_family_source_adapter *adapter;

    if (products) memset(products, 0, sizeof(*products));
    if (!target_id || !target_id[0] || !request || !products) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "family.source-catalog",
                       "target, source request, and products are required");
        return YVEX_ERR_INVALID_ARG;
    }
    descriptor = family_descriptor_find_target(target_id);
    execution = descriptor && descriptor->execution ? descriptor->execution() : NULL;
    if (execution)
        return catalog_execution_source_compile(execution, request, products, err);
    adapter = descriptor && descriptor->source ? descriptor->source() : NULL;
    if (adapter && adapter->compile)
        return adapter->compile(products, request, err);
    yvex_error_setf(err, YVEX_ERR_UNSUPPORTED, "family.source-catalog",
                    "no source compiler adapter for target: %s", target_id);
    return YVEX_ERR_UNSUPPORTED;
}

void yvex_family_source_products_release(yvex_family_source_products *products)
{
    if (!products) return;
    if (products->release) products->release(products->owner);
    memset(products, 0, sizeof(*products));
}
