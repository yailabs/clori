/*
 * Exercises CUDA tensor allocation, zero-read, write/read, copy, and memory accounting when CUDA
 * is available. Returns 77 when CUDA is unavailable.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <yvex/api.h>

#include "tests/test.h"

static void make_desc(yvex_backend_tensor_desc *desc,
                      const char *name,
                      unsigned long long d0,
                      unsigned long long d1)
{
    memset(desc, 0, sizeof(*desc));
    desc->name = name;
    desc->dtype = YVEX_DTYPE_F32;
    desc->rank = 2;
    desc->dims[0] = d0;
    desc->dims[1] = d1;
    desc->bytes = d0 * d1 * (unsigned long long)sizeof(float);
}

int yvex_cuda_test_tensor(void)
{
    yvex_backend *backend = NULL;
    yvex_device_tensor *a = NULL;
    yvex_device_tensor *b = NULL;
    yvex_backend_options options;
    yvex_backend_tensor_desc desc;
    yvex_backend_memory_stats stats;
    yvex_backend_memory_stats before_failed_release;
    yvex_backend_capability_result capability;
    yvex_error err;
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float out[4] = {9.0f, 9.0f, 9.0f, 9.0f};
    int rc;

    memset(&options, 0, sizeof(options));
    options.kind = YVEX_BACKEND_KIND_CUDA;
    rc = yvex_backend_open(&backend, &options, &err);
    if (rc == YVEX_ERR_UNSUPPORTED) {
        fprintf(stderr, "SKIP: CUDA unavailable: %s\n", yvex_error_message(&err));
        return 77;
    }
    YVEX_TEST_ASSERT(rc == YVEX_OK, "open cuda backend");

    rc = yvex_backend_get_memory_stats(backend, &stats, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "initial stats");
    YVEX_TEST_ASSERT(stats.allocated_bytes == 0, "initial allocated bytes");

    make_desc(&desc, "cuda_a", 2, 2);
    rc = yvex_backend_tensor_alloc(backend, &desc, &a, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate cuda tensor a");
    rc = yvex_backend_tensor_read(backend, a, out, sizeof(out), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "read zero initialized");
    YVEX_TEST_ASSERT(out[0] == 0.0f && out[3] == 0.0f, "zero initialized values");

    rc = yvex_backend_tensor_write(backend, a, data, sizeof(data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "write cuda tensor");
    memset(out, 0, sizeof(out));
    rc = yvex_backend_tensor_read(backend, a, out, sizeof(out), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "read cuda tensor");
    YVEX_TEST_ASSERT(memcmp(data, out, sizeof(data)) == 0, "read equals written");

    make_desc(&desc, "cuda_b", 2, 2);
    rc = yvex_backend_tensor_alloc(backend, &desc, &b, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate cuda tensor b");
    rc = yvex_backend_tensor_copy(backend, b, a, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "copy cuda tensor");
    memset(out, 0, sizeof(out));
    rc = yvex_backend_tensor_read(backend, b, out, sizeof(out), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "read copied cuda tensor");
    YVEX_TEST_ASSERT(memcmp(data, out, sizeof(data)) == 0, "copy equals source");

    rc = yvex_backend_tensor_release(backend, &b, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && b == NULL, "checked release cuda tensor b");
    rc = yvex_backend_tensor_release(backend, &a, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && a == NULL, "checked release cuda tensor a");
    rc = yvex_backend_get_memory_stats(backend, &stats, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "final stats");
    YVEX_TEST_ASSERT(stats.allocated_bytes == 0, "allocated bytes return to zero");
    YVEX_TEST_ASSERT(stats.allocation_count == 0, "allocation count returns to zero");
    yvex_backend_close(backend);

    backend = NULL;
    rc = yvex_backend_open(&backend, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "reopen cuda backend for sync failure");
    make_desc(&desc, "cuda_sync_failure", 2, 2);
    rc = yvex_backend_tensor_alloc(backend, &desc, &a, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate sync failure tensor");
    YVEX_TEST_ASSERT(setenv("YVEX_TEST_CUDA_SYNC_FAILURE", "tensor-write", 1) == 0,
                     "set tensor write sync failure");
    rc = yvex_backend_tensor_write(backend, a, data, sizeof(data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_BACKEND, "tensor write sync failure is returned");
    YVEX_TEST_ASSERT(!yvex_device_tensor_is_written(a),
                     "failed synchronized write remains unwritten");
    rc = yvex_backend_query_capability(backend, YVEX_BACKEND_VARIANT_TENSOR_WRITE,
                                       &capability, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK &&
                     capability.state == YVEX_BACKEND_CAPABILITY_FAILED &&
                     capability.reason ==
                         YVEX_BACKEND_CAPABILITY_REASON_SYNCHRONIZATION_FAILED,
                     "write sync failure demotes exact capability");
    rc = yvex_backend_query_capability(backend, YVEX_BACKEND_VARIANT_TENSOR_READ,
                                       &capability, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK &&
                     capability.state == YVEX_BACKEND_CAPABILITY_FAILED &&
                     capability.reason ==
                         YVEX_BACKEND_CAPABILITY_REASON_SYNCHRONIZATION_FAILED,
                     "context synchronization failure blocks other dispatch");
    YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_CUDA_SYNC_FAILURE") == 0,
                     "clear tensor write sync failure");
    rc = yvex_backend_tensor_release(backend, &a, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && a == NULL, "release after sync failure");
    yvex_backend_close(backend);

    backend = NULL;
    rc = yvex_backend_open(&backend, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "reopen cuda backend for cleanup failure");
    make_desc(&desc, "cuda_cleanup_failure", 2, 2);
    rc = yvex_backend_tensor_alloc(backend, &desc, &a, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate cleanup failure tensor");
    rc = yvex_backend_get_memory_stats(backend, &before_failed_release, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "stats before failed release");
    YVEX_TEST_ASSERT(setenv("YVEX_TEST_CUDA_CLEANUP_FAILURE", "tensor-alloc", 1) == 0,
                     "set tensor cleanup failure");
    rc = yvex_backend_tensor_release(backend, &a, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_BACKEND && a != NULL,
                     "failed release preserves owned tensor");
    rc = yvex_backend_get_memory_stats(backend, &stats, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "stats after failed release");
    YVEX_TEST_ASSERT(stats.allocated_bytes == before_failed_release.allocated_bytes &&
                     stats.allocation_count == before_failed_release.allocation_count,
                     "failed release preserves truthful allocation accounting");
    YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_CUDA_CLEANUP_FAILURE") == 0,
                     "clear tensor cleanup failure");
    rc = yvex_backend_tensor_release(backend, &a, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && a == NULL, "retry release succeeds");
    rc = yvex_backend_get_memory_stats(backend, &stats, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && stats.allocated_bytes == 0 &&
                     stats.allocation_count == 0,
                     "retry release restores zero accounting");
    yvex_backend_close(backend);
    return 0;
}
