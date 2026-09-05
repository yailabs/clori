/* Derive current deployment launchability from durable profile and execution facts. */
#ifndef INCLUDE_YVEX_INTERNAL_DEPLOYMENT_COMPATIBILITY_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_DEPLOYMENT_COMPATIBILITY_H_INCLUDED

#include <yvex/core.h>
#include <yvex/internal/core.h>
#include <yvex/registry.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_DEPLOYMENT_COMPATIBILITY_SCHEMA_V1 1u
#define YVEX_DEPLOYMENT_COMPATIBILITY_REASON_CAP 256u

typedef enum {
    YVEX_DEPLOYMENT_COMPATIBILITY_CURRENT = 0,
    YVEX_DEPLOYMENT_COMPATIBILITY_INCOMPLETE,
    YVEX_DEPLOYMENT_COMPATIBILITY_MISSING_DEPENDENCY,
    YVEX_DEPLOYMENT_COMPATIBILITY_UNSUPPORTED_TARGET,
    YVEX_DEPLOYMENT_COMPATIBILITY_STALE_BINDING,
    YVEX_DEPLOYMENT_COMPATIBILITY_MALFORMED_BINDING,
    YVEX_DEPLOYMENT_COMPATIBILITY_ARTIFACT_MISMATCH
} yvex_deployment_compatibility_status;

typedef struct {
    unsigned int schema_version;
    yvex_deployment_compatibility_status status;
    int current;
    unsigned long long family_adapter_id, family_adapter_version;
    char runtime_binding_identity[YVEX_SHA256_HEX_BYTES];
    char artifact_identity[YVEX_SHA256_HEX_BYTES];
    char semantic_graph_identity[YVEX_SHA256_HEX_BYTES];
    char executable_graph_identity[YVEX_SHA256_HEX_BYTES];
    char reason[YVEX_DEPLOYMENT_COMPATIBILITY_REASON_CAP];
} yvex_deployment_compatibility;

/* This is an inert compatibility projection: it opens no model payload and
 * creates no engine/session state. Runtime admission repeats the check. */
int yvex_deployment_compatibility_evaluate(
    const yvex_model_registry_entry *entry,
    yvex_deployment_compatibility *result, yvex_error *err);
const char *yvex_deployment_compatibility_status_name(
    yvex_deployment_compatibility_status status);

#ifdef __cplusplus
}
#endif
#endif /* INCLUDE_YVEX_INTERNAL_DEPLOYMENT_COMPATIBILITY_H_INCLUDED */
