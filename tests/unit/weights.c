
#include <yvex/api.h>

#include "tests/test.h"

static int test_null_accessors(void)
{
    yvex_error err;
    yvex_materialize_summary summary;

    YVEX_TEST_ASSERT(yvex_weight_table_count(NULL) == 0, "null count");
    YVEX_TEST_ASSERT(yvex_weight_table_at(NULL, 0) == NULL, "null at");
    YVEX_TEST_ASSERT(yvex_weight_table_find(NULL, "x") == NULL, "null find");
    YVEX_TEST_ASSERT_STREQ(yvex_weight_status_name(YVEX_WEIGHT_STATUS_MATERIALIZED),
                           "materialized", "status name");
    YVEX_TEST_ASSERT_STREQ(yvex_weight_residency_name(YVEX_WEIGHT_RESIDENCY_CPU_BACKEND),
                           "cpu_backend", "residency name");
    YVEX_TEST_ASSERT_STREQ(yvex_weight_name(NULL), "", "null weight name");
    YVEX_TEST_ASSERT(yvex_weight_dtype(NULL) == YVEX_DTYPE_UNKNOWN, "null weight dtype");
    YVEX_TEST_ASSERT(yvex_weight_role(NULL) == YVEX_TENSOR_ROLE_UNKNOWN, "null weight role");
    YVEX_TEST_ASSERT(yvex_weight_bytes(NULL) == 0, "null weight bytes");
    YVEX_TEST_ASSERT(yvex_weight_device_tensor(NULL) == NULL, "null device tensor");
    YVEX_TEST_ASSERT(yvex_weight_table_get_summary(NULL, &summary, &err) == YVEX_ERR_INVALID_ARG,
                     "null summary invalid");
    return 0;
}

int yvex_test_weights(void)
{
    if (test_null_accessors() != 0) return 1;
    return 0;
}
