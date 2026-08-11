/* Backend implementations share concrete lifecycle state through this source-local ABI. */
#ifndef SRC_BACKEND_PRIVATE_H_INCLUDED
#define SRC_BACKEND_PRIVATE_H_INCLUDED

#include <limits.h>
#include <stdatomic.h>

#include <yvex/internal/backend.h>

typedef struct yvex_backend_vtable {
    int (*close)(yvex_backend *, yvex_error *);
    int (*memory_stats)(const yvex_backend *, yvex_backend_memory_stats *, yvex_error *);
    int (*device_info)(const yvex_backend *, yvex_backend_device_info *, yvex_error *);
    int (*bandwidth_probe)(yvex_backend *, yvex_backend_bandwidth_evidence *, yvex_error *);
    int (*tensor_alloc)(yvex_backend *, const yvex_backend_tensor_desc *,
                        yvex_device_tensor **, yvex_error *);
    int (*resident_alloc)(yvex_backend *, const yvex_backend_tensor_desc *,
                          yvex_device_tensor **, unsigned char **, yvex_error *);
    int (*resident_prefetch)(yvex_backend *, yvex_device_tensor *,
                             unsigned long long *, yvex_error *);
    int (*tensor_reserve)(yvex_backend *, const yvex_backend_tensor_desc *,
                          yvex_device_tensor **, unsigned long long *, yvex_error *);
    int (*tensor_commit)(yvex_backend *, yvex_device_tensor *, unsigned long long,
                         unsigned long long, unsigned long long *, yvex_error *);
    int (*tensor_decommit)(yvex_backend *, yvex_device_tensor *, unsigned long long *,
                           yvex_error *);
    int (*tensor_free)(yvex_backend *, yvex_device_tensor *, yvex_error *);
    int (*tensor_write)(yvex_backend *, yvex_device_tensor *, const void *,
                        unsigned long long, yvex_error *);
    int (*tensor_read)(yvex_backend *, const yvex_device_tensor *, void *,
                       unsigned long long, yvex_error *);
    int (*tensor_copy)(yvex_backend *, yvex_device_tensor *, const yvex_device_tensor *,
                       yvex_error *);
    int (*tensor_copy_async)(yvex_backend *, yvex_device_tensor *,
                             const yvex_device_tensor *, yvex_error *);
    int (*sync)(yvex_backend *, yvex_error *);
    int (*query_capability)(const yvex_backend *, yvex_backend_operation_variant,
                            yvex_backend_capability_result *, yvex_error *);
    int (*op_embed)(yvex_backend *, const yvex_device_tensor *, const unsigned int *,
                    unsigned long long, yvex_device_tensor *, yvex_error *);
    int (*op_rms_norm)(yvex_backend *, const yvex_device_tensor *,
                       const yvex_device_tensor *, float, yvex_device_tensor *, yvex_error *);
    int (*op_rope)(yvex_backend *, const yvex_device_tensor *, unsigned long long, float,
                   yvex_device_tensor *, yvex_error *);
    int (*op_matmul)(yvex_backend *, const yvex_device_tensor *,
                     const yvex_device_tensor *, yvex_device_tensor *, yvex_error *);
    int (*op_mlp)(yvex_backend *, const yvex_device_tensor *, const yvex_device_tensor *,
                  const yvex_device_tensor *, const yvex_device_tensor *,
                  const yvex_mlp_options *, yvex_device_tensor *, yvex_device_tensor *,
                  yvex_error *);
    int (*op_attention)(yvex_backend *, const yvex_device_tensor *,
                        const yvex_device_tensor *, const yvex_device_tensor *,
                        unsigned long long, unsigned long long, float, int,
                        yvex_device_tensor *, yvex_device_tensor *, yvex_device_tensor *,
                        yvex_error *);
    int (*host_workspace_alloc)(yvex_backend *, size_t, unsigned char **, yvex_error *);
    int (*host_workspace_free)(yvex_backend *, unsigned char **, yvex_error *);
} yvex_backend_vtable;

struct yvex_backend {
    yvex_backend_kind kind;
    _Atomic(yvex_backend_status) status;
    const yvex_backend_vtable *vtable;
    yvex_backend_memory_stats stats;
    yvex_backend_device_info device_info;
    char device_name_storage[128];
    void *impl;
    struct yvex_backend *resource_owner;
    _Atomic unsigned long long lifecycle;
    unsigned long long tensor_id_next, resident_host_bytes;
    const unsigned char *resident_host_base;
    const yvex_device_tensor *resident_device_tensor;
    unsigned long long resident_device_address, resident_generation;
    const void *state_residency_context;
    yvex_backend_state_resolve_fn state_residency_resolve;
    unsigned long long state_residency_generation;
    const yvex_device_tensor *workspace_device_tensor;
    unsigned long long workspace_device_address, workspace_bytes, workspace_cursor;
    unsigned long long workspace_peak, workspace_generation;
    unsigned char *host_workspace_base;
    unsigned long long host_workspace_bytes, host_workspace_reserved, host_workspace_cursor;
    unsigned long long host_workspace_peak;
    unsigned long long host_workspace_generation, host_workspace_allocation_count;
    int host_workspace_owned, host_workspace_pinned, shared_owner_registered;
    int virtual_tensor_ready;
};

#define YVEX_BACKEND_LIFECYCLE_CLOSING (1ull << 63)
#define YVEX_BACKEND_LIFECYCLE_CHILD_MASK (YVEX_BACKEND_LIFECYCLE_CLOSING - 1ull)

static inline int backend_cleanup_only(const yvex_backend *backend)
{
    return backend &&
           (atomic_load_explicit(&backend->lifecycle, memory_order_acquire) &
            YVEX_BACKEND_LIFECYCLE_CLOSING) != 0ull;
}

static inline int backend_dispatch_admit(const yvex_backend *backend,
                                         const char *where, yvex_error *err)
{
    if (!backend || backend_cleanup_only(backend) ||
        backend->status == YVEX_BACKEND_STATUS_FAILED) {
        yvex_error_set(err, YVEX_ERR_STATE, where,
                       "backend is retained for cleanup only");
        return YVEX_ERR_STATE;
    }
    return YVEX_OK;
}

static inline int backend_variant_supported(const yvex_backend *backend,
                                            yvex_backend_operation_variant variant)
{
    yvex_backend_capability_result result;
    yvex_error err;
    yvex_error_clear(&err);
    return yvex_backend_query_capability(backend, variant, &result, &err) == YVEX_OK &&
           result.state == YVEX_BACKEND_CAPABILITY_SUPPORTED;
}

static inline int backend_tensor_f32_elements(const yvex_device_tensor *tensor,
                                              unsigned long long elements)
{
    return tensor && elements <= ULLONG_MAX / sizeof(float) &&
           tensor->bytes == elements * (unsigned long long)sizeof(float);
}

static inline void backend_memory_acquire(yvex_backend *backend, unsigned long long bytes)
{
    backend->stats.allocated_bytes += bytes;
    backend->stats.allocation_count += 1ull;
    backend->stats.allocation_events += 1ull;
    if (backend->stats.allocated_bytes > backend->stats.peak_allocated_bytes)
        backend->stats.peak_allocated_bytes = backend->stats.allocated_bytes;
}

static inline void backend_memory_release(yvex_backend *backend, unsigned long long bytes)
{
    backend->stats.allocated_bytes =
        backend->stats.allocated_bytes >= bytes ? backend->stats.allocated_bytes - bytes : 0ull;
    if (backend->stats.allocation_count > 0ull) {
        backend->stats.allocation_count -= 1ull;
        backend->stats.release_events += 1ull;
    }
}

static inline int backend_tensor_owner_is(const yvex_backend *backend,
                                          const yvex_device_tensor *tensor)
{
    return backend && tensor && tensor->owner == backend && tensor->owner_id != 0ull;
}

static inline void backend_workspace_reset(yvex_backend *backend)
{
    if (backend) backend->workspace_cursor = 0ull;
}

static inline void backend_host_workspace_reset(yvex_backend *backend)
{
    if (backend) backend->host_workspace_cursor = backend->host_workspace_reserved;
}

#endif /* SRC_BACKEND_PRIVATE_H_INCLUDED */
