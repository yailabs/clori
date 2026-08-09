/*
 * Publish family-compiled products through the runtime-binding codec.
 *
 * The family owns semantic compilation and every temporary product it returns. This bridge checks
 * the adapter boundary, publishes the immutable binding, and releases the borrowed products on
 * every path; it does not inspect or reconstruct family topology.
 */
#include <yvex/internal/runtime.h>

#include <string.h>

int yvex_runtime_binding_compile_publish(
    const yvex_family_compiler_adapter *adapter,
    const struct yvex_compilation_runtime_binding_request *request,
    char path[YVEX_PATH_CAP], int *published, yvex_error *err)
{
    yvex_runtime_binding_prepare_request prepare = {0};
    yvex_runtime_binding_prepare_result result = {0};
    yvex_runtime_binding_failure failure = {0};
    void *owner = NULL;
    int rc;

    if (path) memset(path, 0, YVEX_PATH_CAP);
    if (published) *published = 0;
    if (!adapter || !request || !path || !published ||
        adapter->schema_version != YVEX_FAMILY_COMPILER_SCHEMA_V1 ||
        !adapter->adapter_id || !adapter->adapter_version ||
        !adapter->runtime_binding_compile || !adapter->runtime_binding_release) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG,
                       "runtime.binding.compile-publish",
                       "one exact family compiler adapter and output are required");
        return YVEX_ERR_INVALID_ARG;
    }
    rc = adapter->runtime_binding_compile(request, &prepare, &owner, err);
    if (rc == YVEX_OK &&
        (prepare.family_adapter_id != adapter->adapter_id ||
         prepare.family_adapter_version != adapter->adapter_version)) {
        yvex_error_set(err, YVEX_ERR_STATE,
                       "runtime.binding.compile-publish",
                       "compiled binding inputs disagree with the family adapter identity");
        rc = YVEX_ERR_STATE;
    }
    if (rc == YVEX_OK)
        rc = yvex_runtime_binding_prepare(&prepare, &result, &failure, err);
    if (rc == YVEX_OK) {
        memcpy(path, result.path, YVEX_PATH_CAP);
        *published = result.published;
    }
    adapter->runtime_binding_release(owner);
    return rc;
}
