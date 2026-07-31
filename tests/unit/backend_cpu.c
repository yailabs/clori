/*
 * Exercises backend layer opens the CPU reference backend and supports tensor allocation, memory
 * stats, read/write, copy, sync, and memory-limit errors.
 */
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
    desc->bytes = d0 * d1 * 4u;
}

static int test_open_and_unsupported(void)
{
    yvex_backend *backend = NULL;
    yvex_backend_options options;
    yvex_error err;
    int rc;

    rc = yvex_backend_open_cpu(&backend, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "open cpu backend");
    YVEX_TEST_ASSERT(yvex_backend_kind_of(backend) == YVEX_BACKEND_KIND_CPU, "cpu kind");
    YVEX_TEST_ASSERT(yvex_backend_status_of(backend) == YVEX_BACKEND_STATUS_READY, "cpu ready");
    YVEX_TEST_ASSERT_STREQ(yvex_backend_kind_name(YVEX_BACKEND_KIND_CPU), "cpu", "cpu kind name");
    YVEX_TEST_ASSERT_STREQ(yvex_backend_status_name(YVEX_BACKEND_STATUS_READY), "ready", "ready name");
    YVEX_TEST_ASSERT(yvex_backend_sync(backend, &err) == YVEX_OK, "cpu sync no-op");
    yvex_backend_close(backend);

    memset(&options, 0, sizeof(options));
    options.kind = YVEX_BACKEND_KIND_CUDA;
    rc = yvex_backend_open(&backend, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK || rc == YVEX_ERR_UNSUPPORTED,
                     "cuda opens or reports unsupported cleanly");
    if (rc == YVEX_OK) {
        yvex_backend_status status = yvex_backend_status_of(backend);

        YVEX_TEST_ASSERT(yvex_backend_kind_of(backend) == YVEX_BACKEND_KIND_CUDA, "cuda kind");
        YVEX_TEST_ASSERT(status == YVEX_BACKEND_STATUS_READY ||
                             status == YVEX_BACKEND_STATUS_CONTEXT_READY,
                         "cuda context or kernels ready");
        yvex_backend_close(backend);
    } else {
        YVEX_TEST_ASSERT(backend == NULL, "unsupported backend remains null");
    }
    return 0;
}

static int test_tensor_memory_and_copy(void)
{
    yvex_backend *backend = NULL;
    yvex_device_tensor *a = NULL;
    yvex_device_tensor *b = NULL;
    yvex_backend_tensor_desc desc;
    yvex_backend_memory_stats stats;
    yvex_error err;
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float out[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int rc;

    YVEX_TEST_ASSERT(yvex_backend_open_cpu(&backend, &err) == YVEX_OK, "open cpu");
    rc = yvex_backend_get_memory_stats(backend, &stats, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "initial memory stats");
    YVEX_TEST_ASSERT(stats.allocated_bytes == 0, "initial allocated zero");
    YVEX_TEST_ASSERT(stats.allocation_count == 0, "initial allocation count zero");

    make_desc(&desc, "a", 2, 2);
    rc = yvex_backend_tensor_alloc(backend, &desc, &a, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate tensor a");
    YVEX_TEST_ASSERT_STREQ(yvex_device_tensor_name(a), "a", "tensor name");
    YVEX_TEST_ASSERT(yvex_device_tensor_bytes(a) == 16, "tensor bytes");

    rc = yvex_backend_get_memory_stats(backend, &stats, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "post alloc memory stats");
    YVEX_TEST_ASSERT(stats.allocated_bytes == 16, "allocated bytes increments");
    YVEX_TEST_ASSERT(stats.allocation_count == 1, "allocation count increments");
    YVEX_TEST_ASSERT(stats.peak_allocated_bytes == 16, "peak bytes updates");

    rc = yvex_backend_tensor_read(backend, a, out, sizeof(out), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "zero read before write succeeds");
    YVEX_TEST_ASSERT(out[0] == 0.0f && out[3] == 0.0f, "read before write is zero");

    rc = yvex_backend_tensor_write(backend, a, data, sizeof(data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "write full tensor");
    memset(out, 0, sizeof(out));
    rc = yvex_backend_tensor_read(backend, a, out, sizeof(out), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "read full tensor");
    YVEX_TEST_ASSERT(memcmp(data, out, sizeof(data)) == 0, "read equals written");

    make_desc(&desc, "b", 2, 2);
    rc = yvex_backend_tensor_alloc(backend, &desc, &b, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate tensor b");
    rc = yvex_backend_tensor_copy(backend, b, a, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "copy tensor");
    memset(out, 0, sizeof(out));
    rc = yvex_backend_tensor_read(backend, b, out, sizeof(out), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "read copied tensor");
    YVEX_TEST_ASSERT(memcmp(data, out, sizeof(data)) == 0, "copy equals source");

    rc = yvex_backend_tensor_write(backend, a, data, sizeof(data) - 4u, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_BOUNDS, "write length mismatch fails");

    yvex_backend_tensor_free(backend, b);
    yvex_backend_tensor_free(backend, a);
    rc = yvex_backend_get_memory_stats(backend, &stats, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "post free memory stats");
    YVEX_TEST_ASSERT(stats.allocated_bytes == 0, "allocated bytes decrements");
    YVEX_TEST_ASSERT(stats.allocation_count == 0, "allocation count decrements");
    YVEX_TEST_ASSERT(stats.peak_allocated_bytes == 32, "peak preserves high water");
    yvex_backend_close(backend);
    return 0;
}

static int test_memory_limit_and_invalid_args(void)
{
    yvex_backend *backend = NULL;
    yvex_backend_options options;
    yvex_backend_tensor_desc desc;
    yvex_device_tensor *tensor = NULL;
    yvex_error err;
    int rc;

    memset(&options, 0, sizeof(options));
    options.kind = YVEX_BACKEND_KIND_CPU;
    options.memory_limit_bytes = 8;
    rc = yvex_backend_open(&backend, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "open limited cpu");

    make_desc(&desc, "limited", 2, 2);
    rc = yvex_backend_tensor_alloc(backend, &desc, &tensor, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_NOMEM, "memory limit rejects allocation");
    YVEX_TEST_ASSERT(tensor == NULL, "limited tensor remains null");

    rc = yvex_backend_tensor_alloc(backend, NULL, &tensor, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_INVALID_ARG, "null desc invalid");

    yvex_backend_close(backend);
    return 0;
}

int yvex_test_backend_cpu(void)
{
    if (test_open_and_unsupported() != 0) return 1;
    if (test_tensor_memory_and_copy() != 0) return 1;
    if (test_memory_limit_and_invalid_args() != 0) return 1;
    return 0;
}
