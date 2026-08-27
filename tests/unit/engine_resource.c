/* Prove independent engine-resource lifetime without introducing a model-specific cache. */
#include <yvex/internal/engine_resource.h>

#include <string.h>

#include "tests/test.h"

static const char package_identity[] =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
static const char engine_identity[] =
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
static const char specialization_identity[] =
    "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
static const char admission_identity[] =
    "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";

typedef struct {
    unsigned int releases;
    int fail;
} release_probe;

static int release_count(void *context, yvex_error *err)
{
    release_probe *probe = context;
    probe->releases++;
    if (probe->fail) {
        yvex_error_set(err, YVEX_ERR_STATE, "test.engine-resource",
                       "injected resource release failure");
        return YVEX_ERR_STATE;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_test_engine_resource(void)
{
    yvex_engine_resource_catalog *catalog = NULL;
    yvex_engine_resource_handle package = {0}, prepared = {0}, stale;
    yvex_engine_resource_request request = {0};
    yvex_engine_resource_summary summary = {0};
    yvex_engine_resource_entry entries[2] = {0};
    release_probe package_probe = {0}, prepared_probe = {0};
    unsigned long long count = 0ull;
    void *borrowed = NULL;
    yvex_error err;

    YVEX_TEST_ASSERT(
        yvex_runtime_resource_catalog_open(
            &catalog, 17ull, engine_identity, 4ull, &err) == YVEX_OK,
        "engine resource catalog opens for one authenticated generation");
    request.kind = YVEX_ENGINE_RESOURCE_PACKAGE_MAPPING;
    request.owner = YVEX_ENGINE_RESOURCE_OWNER_PACKAGE;
    request.lifetime = YVEX_ENGINE_RESOURCE_LIFETIME_ENGINE;
    request.numeric_class = YVEX_ENGINE_RESOURCE_NUMERIC_CANONICAL_PACKAGE;
    request.name = "canonical-package";
    request.package_identity = package_identity;
    request.bytes.mapped_package_bytes = 4096ull;
    request.value = &package_probe;
    request.release = release_count;
    request.release_context = &package_probe;
    request.ready = 1;
    YVEX_TEST_ASSERT(
        yvex_runtime_resource_register(catalog, &request, &package, &err) ==
            YVEX_OK,
        "canonical package mapping registers independently");

    memset(&request, 0, sizeof(request));
    request.kind = YVEX_ENGINE_RESOURCE_PREPARED_LAYOUT;
    request.owner = YVEX_ENGINE_RESOURCE_OWNER_SPECIALIZATION;
    request.lifetime = YVEX_ENGINE_RESOURCE_LIFETIME_ENGINE;
    request.numeric_class = YVEX_ENGINE_RESOURCE_NUMERIC_EQUIVALENT_PREPARED;
    request.name = "synthetic-layout";
    request.package_identity = package_identity;
    request.specialization_identity = specialization_identity;
    request.admission_identity = admission_identity;
    request.dependency = package;
    request.bytes.host_resident_bytes = 256ull;
    request.bytes.prepared_bytes = 256ull;
    request.value = &prepared_probe;
    request.release = release_count;
    request.release_context = &prepared_probe;
    request.evictable = 1;
    YVEX_TEST_ASSERT(
        yvex_runtime_resource_register(catalog, &request, &prepared, &err) ==
            YVEX_OK,
        "one independently evictable prepared view registers");
    YVEX_TEST_ASSERT(
        yvex_runtime_resource_snapshot(
            catalog, &summary, entries, 2ull, &count, &err) == YVEX_OK &&
            count == 2ull && summary.resource_count == 2ull &&
            summary.ready_count == 1ull &&
            summary.bytes.mapped_package_bytes == 4096ull &&
            summary.bytes.prepared_bytes == 256ull &&
            entries[0].dependent_count == 1ull &&
            entries[1].state == YVEX_ENGINE_RESOURCE_DECLARED,
        "snapshot separates declared prepared work from ready package truth");
    YVEX_TEST_ASSERT(
        yvex_runtime_resource_acquire(catalog, prepared, &borrowed, &err) ==
            YVEX_ERR_STATE,
        "declared resource cannot be consumed before readiness publication");
    YVEX_TEST_ASSERT(
        yvex_runtime_resource_publish_ready(catalog, prepared, &err) ==
            YVEX_OK,
        "prepared resource publishes readiness without changing package identity");

    YVEX_TEST_ASSERT(
        yvex_runtime_resource_acquire(catalog, prepared, &borrowed, &err) ==
                YVEX_OK &&
            borrowed == &prepared_probe,
        "consumer borrows an exact resource generation");
    YVEX_TEST_ASSERT(
        yvex_runtime_resource_evict(catalog, &prepared, &err) ==
                YVEX_ERR_STATE &&
            prepared_probe.releases == 0u,
        "borrowed prepared resource cannot be evicted");
    YVEX_TEST_ASSERT(
        yvex_runtime_resource_drop(catalog, prepared, &err) == YVEX_OK,
        "consumer discharges the exact borrow");
    stale = prepared;
    YVEX_TEST_ASSERT(
        yvex_runtime_resource_evict(catalog, &prepared, &err) == YVEX_OK &&
            !prepared.engine_generation && prepared_probe.releases == 1u,
        "prepared resource evicts independently from package truth");
    YVEX_TEST_ASSERT(
        yvex_runtime_resource_acquire(catalog, stale, &borrowed, &err) ==
            YVEX_ERR_STATE,
        "evicted generation handle is stale");
    YVEX_TEST_ASSERT(
        yvex_runtime_resource_snapshot(
            catalog, &summary, entries, 2ull, &count, &err) == YVEX_OK &&
            count == 1ull && summary.resource_count == 1ull &&
            summary.eviction_count == 1ull &&
            summary.bytes.mapped_package_bytes == 4096ull &&
            summary.bytes.prepared_bytes == 0ull,
        "eviction releases prepared accounting without changing package mapping");

    package_probe.fail = 1;
    YVEX_TEST_ASSERT(
        yvex_runtime_resource_catalog_close(&catalog, &err) ==
                YVEX_ERR_STATE &&
            catalog && package_probe.releases == 1u &&
            yvex_runtime_resource_snapshot(
                catalog, &summary, entries, 2ull, &count, &err) == YVEX_OK &&
            summary.failed_count == 1ull && !summary.ready_count &&
            summary.failed_release_count == 1ull,
        "failed release preserves exact ownership for a retry");
    package_probe.fail = 0;
    YVEX_TEST_ASSERT(
        yvex_runtime_resource_catalog_close(&catalog, &err) == YVEX_OK &&
            !catalog && package_probe.releases == 2u,
        "catalog retries and closes the remaining resource exactly");
    return 0;
}
