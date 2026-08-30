/* Track independently releasable resources owned by one live engine generation. */
#ifndef INCLUDE_YVEX_INTERNAL_ENGINE_RESOURCE_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_ENGINE_RESOURCE_H_INCLUDED

#include <yvex/artifact.h>
#include <yvex/core.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_ENGINE_RESOURCE_NAME_CAP 48u

typedef enum {
    YVEX_ENGINE_RESOURCE_PACKAGE_MAPPING = 0,
    YVEX_ENGINE_RESOURCE_PREPARED_TENSOR,
    YVEX_ENGINE_RESOURCE_PREPARED_GROUP,
    YVEX_ENGINE_RESOURCE_PREPARED_LAYOUT,
    YVEX_ENGINE_RESOURCE_COMPONENT,
    YVEX_ENGINE_RESOURCE_BACKEND_HANDLE,
    YVEX_ENGINE_RESOURCE_EXECUTABLE_CACHE,
    YVEX_ENGINE_RESOURCE_SEQUENCE_STATE,
    YVEX_ENGINE_RESOURCE_WORKSPACE,
    YVEX_ENGINE_RESOURCE_TEMPORARY,
    YVEX_ENGINE_RESOURCE_KIND_COUNT
} yvex_engine_resource_kind;

typedef enum {
    YVEX_ENGINE_RESOURCE_OWNER_PACKAGE = 0,
    YVEX_ENGINE_RESOURCE_OWNER_SPECIALIZATION,
    YVEX_ENGINE_RESOURCE_OWNER_ENGINE,
    YVEX_ENGINE_RESOURCE_OWNER_SESSION,
    YVEX_ENGINE_RESOURCE_OWNER_BACKEND,
    YVEX_ENGINE_RESOURCE_OWNER_EXECUTION
} yvex_engine_resource_owner;

typedef enum {
    YVEX_ENGINE_RESOURCE_LIFETIME_ENGINE = 0,
    YVEX_ENGINE_RESOURCE_LIFETIME_SESSION,
    YVEX_ENGINE_RESOURCE_LIFETIME_EXECUTION,
    YVEX_ENGINE_RESOURCE_LIFETIME_REQUEST,
    YVEX_ENGINE_RESOURCE_LIFETIME_CONDITION,
    YVEX_ENGINE_RESOURCE_LIFETIME_TRAJECTORY,
    YVEX_ENGINE_RESOURCE_LIFETIME_QUANTUM
} yvex_engine_resource_lifetime;

typedef enum {
    YVEX_ENGINE_RESOURCE_NUMERIC_NONE = 0,
    YVEX_ENGINE_RESOURCE_NUMERIC_CANONICAL_PACKAGE,
    YVEX_ENGINE_RESOURCE_NUMERIC_EXACT_SPECIALIZATION,
    YVEX_ENGINE_RESOURCE_NUMERIC_EQUIVALENT_PREPARED,
    YVEX_ENGINE_RESOURCE_NUMERIC_STATE
} yvex_engine_resource_numeric_class;

typedef enum {
    YVEX_ENGINE_RESOURCE_DECLARED = 0,
    YVEX_ENGINE_RESOURCE_READY,
    YVEX_ENGINE_RESOURCE_RELEASING,
    YVEX_ENGINE_RESOURCE_FAILED
} yvex_engine_resource_state;

typedef struct {
    unsigned long long mapped_package_bytes, host_resident_bytes;
    unsigned long long device_resident_bytes, prepared_bytes;
    unsigned long long sequence_state_bytes, workspace_bytes, temporary_bytes;
} yvex_engine_resource_bytes;

typedef struct {
    unsigned long long engine_generation, slot, generation;
} yvex_engine_resource_handle;

typedef int (*yvex_engine_resource_release_fn)(void *context, yvex_error *err);

typedef struct {
    yvex_engine_resource_kind kind;
    yvex_engine_resource_owner owner;
    yvex_engine_resource_lifetime lifetime;
    yvex_engine_resource_numeric_class numeric_class;
    const char *name, *package_identity, *specialization_identity;
    const char *admission_identity;
    yvex_engine_resource_handle dependency;
    yvex_engine_resource_bytes bytes;
    unsigned long long preparation_nanoseconds;
    void *value;
    yvex_engine_resource_release_fn release;
    void *release_context;
    int ready, evictable;
} yvex_engine_resource_request;

typedef struct {
    yvex_engine_resource_handle handle, dependency;
    yvex_engine_resource_kind kind;
    yvex_engine_resource_owner owner;
    yvex_engine_resource_lifetime lifetime;
    yvex_engine_resource_numeric_class numeric_class;
    yvex_engine_resource_state state;
    yvex_engine_resource_bytes bytes;
    unsigned long long consumer_count, dependent_count, acquisition_count;
    unsigned long long preparation_nanoseconds;
    char name[YVEX_ENGINE_RESOURCE_NAME_CAP];
    char package_identity[YVEX_SHA256_HEX_CAP];
    char specialization_identity[YVEX_SHA256_HEX_CAP];
    char admission_identity[YVEX_SHA256_HEX_CAP];
    int evictable;
} yvex_engine_resource_entry;

typedef struct {
    unsigned long long engine_generation, capacity, resource_count;
    unsigned long long generation, registration_count, release_count;
    unsigned long long eviction_count, failed_release_count;
    unsigned long long acquisition_count, preparation_nanoseconds;
    unsigned long long ready_count, failed_count;
    unsigned long long count_by_kind[YVEX_ENGINE_RESOURCE_KIND_COUNT];
    yvex_engine_resource_bytes bytes;
    char engine_identity[YVEX_SHA256_HEX_CAP];
    int closing;
} yvex_engine_resource_summary;

typedef struct yvex_engine_resource_catalog yvex_engine_resource_catalog;

int yvex_runtime_resource_catalog_open(
    yvex_engine_resource_catalog **out, unsigned long long engine_generation,
    const char *engine_identity, unsigned long long capacity, yvex_error *err);
int yvex_runtime_resource_register(
    yvex_engine_resource_catalog *catalog,
    const yvex_engine_resource_request *request,
    yvex_engine_resource_handle *handle, yvex_error *err);
int yvex_runtime_resource_publish_ready(
    yvex_engine_resource_catalog *catalog,
    yvex_engine_resource_handle handle, yvex_error *err);
int yvex_runtime_resource_acquire(
    yvex_engine_resource_catalog *catalog,
    yvex_engine_resource_handle handle, void **value, yvex_error *err);
int yvex_runtime_resource_drop(
    yvex_engine_resource_catalog *catalog,
    yvex_engine_resource_handle handle, yvex_error *err);
int yvex_runtime_resource_evict(
    yvex_engine_resource_catalog *catalog,
    yvex_engine_resource_handle *handle, yvex_error *err);
int yvex_runtime_resource_snapshot(
    const yvex_engine_resource_catalog *catalog,
    yvex_engine_resource_summary *summary,
    yvex_engine_resource_entry *entries, unsigned long long entry_capacity,
    unsigned long long *entry_count, yvex_error *err);
int yvex_runtime_resource_catalog_close(
    yvex_engine_resource_catalog **catalog, yvex_error *err);

#ifdef __cplusplus
}
#endif
#endif /* INCLUDE_YVEX_INTERNAL_ENGINE_RESOURCE_H_INCLUDED */
