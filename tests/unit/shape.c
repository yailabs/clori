/*
 * Exercises graph planner shape helpers reject invalid ranks/dimensions and compute
 * deterministic products used by graph and memory-plan estimates.
 */
#include <limits.h>

#include <yvex/api.h>

#include "tests/test.h"

static int test_shape_product(void)
{
    unsigned long long dims[2] = {4, 8};
    unsigned long long product = 0;
    yvex_error err;
    int rc;

    rc = yvex_shape_product(dims, 2, &product, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "shape product succeeds");
    YVEX_TEST_ASSERT(product == 32, "shape product value");
    return 0;
}

static int test_shape_invalid(void)
{
    unsigned long long zero_dim[2] = {4, 0};
    unsigned long long overflow_dims[2] = {ULLONG_MAX, 2};
    unsigned long long rank_dims[5] = {1, 1, 1, 1, 1};
    unsigned long long product = 0;
    yvex_error err;
    int rc;

    rc = yvex_shape_product(zero_dim, 2, &product, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_FORMAT, "zero dimension fails");

    rc = yvex_shape_product(overflow_dims, 2, &product, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_BOUNDS, "overflow dimension product fails");

    rc = yvex_shape_product(rank_dims, 5, &product, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_FORMAT, "rank overflow fails");
    return 0;
}

static int test_shape_copy_equal(void)
{
    unsigned long long src[2] = {1, 4};
    unsigned long long dst[YVEX_GRAPH_MAX_DIMS];
    yvex_error err;
    int rc;

    rc = yvex_shape_copy(dst, YVEX_GRAPH_MAX_DIMS, src, 2, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "shape copy succeeds");
    YVEX_TEST_ASSERT(yvex_shape_equal(src, 2, dst, 2), "shape equal succeeds");
    YVEX_TEST_ASSERT(!yvex_shape_equal(src, 2, dst, 3), "rank mismatch not equal");
    return 0;
}

static int test_fixture_embedding_shape(void)
{
    unsigned long long tensor_dims[2] = {4, 8};
    unsigned long long hidden[2] = {1, tensor_dims[0]};
    unsigned long long expected[2] = {1, 4};

    YVEX_TEST_ASSERT(yvex_shape_equal(hidden, 2, expected, 2), "embedding hidden shape is [seq, hidden]");
    return 0;
}

int yvex_test_shape(void)
{
    if (test_shape_product() != 0) return 1;
    if (test_shape_invalid() != 0) return 1;
    if (test_shape_copy_equal() != 0) return 1;
    if (test_fixture_embedding_shape() != 0) return 1;
    return 0;
}
