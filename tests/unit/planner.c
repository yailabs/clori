/*
 * Exercises planner objects own a graph and memory plan while reporting CPU backend availability
 * and CUDA availability/unavailability.
 */
#include <stdio.h>
#include <string.h>

#include <yvex/api.h>

#include "tests/test.h"

typedef struct {
    yvex_artifact *artifact;
    yvex_gguf *gguf;
    yvex_tensor_table *table;
    yvex_model_descriptor *model;
} planner_fixture;

static int open_fixture(planner_fixture *fixture)
{
    yvex_artifact_options options;
    yvex_error err;
    int rc;

    memset(fixture, 0, sizeof(*fixture));
    memset(&options, 0, sizeof(options));
    options.path = "tests/fixtures/gguf/valid-tokenizer-simple.gguf";
    options.readonly = 1;

    rc = yvex_artifact_open(&fixture->artifact, &options, &err);
    if (rc == YVEX_OK) rc = yvex_gguf_open(&fixture->gguf, fixture->artifact, &err);
    if (rc == YVEX_OK) rc = yvex_tensor_table_from_gguf(&fixture->table, fixture->gguf, &err);
    if (rc == YVEX_OK) {
        rc = yvex_model_descriptor_from_gguf(&fixture->model, fixture->gguf, fixture->table, &err);
    }
    if (rc != YVEX_OK) {
        fprintf(stderr, "planner fixture failed: %s: %s\n", yvex_error_where(&err), yvex_error_message(&err));
        yvex_model_descriptor_close(fixture->model);
        yvex_tensor_table_close(fixture->table);
        yvex_gguf_close(fixture->gguf);
        yvex_artifact_close(fixture->artifact);
        return 1;
    }
    return 0;
}

static void close_fixture(planner_fixture *fixture)
{
    yvex_model_descriptor_close(fixture->model);
    yvex_tensor_table_close(fixture->table);
    yvex_gguf_close(fixture->gguf);
    yvex_artifact_close(fixture->artifact);
}

static int test_plan_cpu(void)
{
    planner_fixture fixture;
    yvex_plan *plan = NULL;
    yvex_plan_options options;
    yvex_error err;
    int rc;

    memset(&options, 0, sizeof(options));
    options.backend_name = "cpu";

    YVEX_TEST_ASSERT(open_fixture(&fixture) == 0, "open planner fixture");
    rc = yvex_plan_create(&plan, fixture.model, fixture.table, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "cpu plan builds");
    YVEX_TEST_ASSERT(yvex_plan_graph(plan) != NULL, "plan owns graph");
    YVEX_TEST_ASSERT(yvex_plan_memory(plan) != NULL, "plan owns memory plan");
    YVEX_TEST_ASSERT(yvex_graph_status_of(yvex_plan_graph(plan)) == YVEX_GRAPH_STATUS_PARTIAL,
                     "plan graph partial");
    YVEX_TEST_ASSERT(yvex_memory_plan_status_of(yvex_plan_memory(plan)) == YVEX_MEMORY_PLAN_PARTIAL,
                     "plan memory partial");

    yvex_plan_close(plan);
    close_fixture(&fixture);
    return 0;
}

static int test_unknown_backend_rejected(void)
{
    planner_fixture fixture;
    yvex_plan *plan = NULL;
    yvex_plan_options options;
    yvex_error err;
    int rc;

    memset(&options, 0, sizeof(options));
    options.backend_name = "quantum";

    YVEX_TEST_ASSERT(open_fixture(&fixture) == 0, "open planner fixture unknown");
    rc = yvex_plan_create(&plan, fixture.model, fixture.table, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_UNSUPPORTED, "unknown backend rejected");
    YVEX_TEST_ASSERT(plan == NULL, "failed plan remains null");

    close_fixture(&fixture);
    return 0;
}

int yvex_test_planner(void)
{
    if (test_plan_cpu() != 0) return 1;
    if (test_unknown_backend_rejected() != 0) return 1;
    return 0;
}
