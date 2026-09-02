/*
 * Exercises graph planner builds a deterministic partial graph from the current descriptor
 * fixture and reports missing required tensor roles.
 */
#include <stdio.h>
#include <string.h>

#include <yvex/api.h>
#include <yvex/internal/family_catalog.h>

#include "src/graph/private.h"
#include "tests/test.h"

typedef struct {
    yvex_artifact *artifact;
    yvex_gguf *gguf;
    yvex_tensor_table *table;
    yvex_model_descriptor *model;
} graph_fixture;

static int open_fixture(graph_fixture *fixture)
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
        fprintf(stderr, "fixture open failed: %s: %s\n", yvex_error_where(&err), yvex_error_message(&err));
        yvex_model_descriptor_close(fixture->model);
        yvex_tensor_table_close(fixture->table);
        yvex_gguf_close(fixture->gguf);
        yvex_artifact_close(fixture->artifact);
        return 1;
    }
    return 0;
}

static void close_fixture(graph_fixture *fixture)
{
    yvex_model_descriptor_close(fixture->model);
    yvex_tensor_table_close(fixture->table);
    yvex_gguf_close(fixture->gguf);
    yvex_artifact_close(fixture->artifact);
    memset(fixture, 0, sizeof(*fixture));
}

static int test_graph_from_fixture(void)
{
    graph_fixture fixture;
    yvex_graph *graph = NULL;
    yvex_graph_build_options options;
    const yvex_graph_value_info *value;
    const yvex_graph_op_info *op;
    const yvex_graph_missing_required *missing;
    yvex_error err;
    int rc;

    memset(&options, 0, sizeof(options));
    options.sequence_length = 1;
    options.include_prefill_path = 1;

    YVEX_TEST_ASSERT(open_fixture(&fixture) == 0, "open graph fixture");
    rc = yvex_graph_build_for_model(&graph, fixture.model, fixture.table, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "graph builds");

    YVEX_TEST_ASSERT(yvex_graph_status_of(graph) == YVEX_GRAPH_STATUS_PARTIAL, "graph is partial");
    YVEX_TEST_ASSERT(yvex_graph_value_count(graph) == 3, "graph value count");
    YVEX_TEST_ASSERT(yvex_graph_op_count(graph) == 1, "graph op count");
    YVEX_TEST_ASSERT(yvex_graph_missing_required_count(graph) == 2, "missing required count");

    value = yvex_graph_value_at(graph, 0);
    YVEX_TEST_ASSERT(value != NULL, "value 0 exists");
    YVEX_TEST_ASSERT(value->kind == YVEX_VALUE_TOKEN_IDS, "value 0 token ids");
    YVEX_TEST_ASSERT(value->rank == 1 && value->dims[0] == 1, "token ids shape");

    value = yvex_graph_value_at(graph, 1);
    YVEX_TEST_ASSERT(value != NULL, "value 1 exists");
    YVEX_TEST_ASSERT(value->kind == YVEX_VALUE_WEIGHT, "value 1 weight");
    YVEX_TEST_ASSERT_STREQ(value->name, "token_embd.weight", "weight value name");
    YVEX_TEST_ASSERT(value->rank == 2 && value->dims[0] == 4 && value->dims[1] == 8, "weight dims");

    value = yvex_graph_value_at(graph, 2);
    YVEX_TEST_ASSERT(value != NULL, "value 2 exists");
    YVEX_TEST_ASSERT(value->kind == YVEX_VALUE_ACTIVATION, "value 2 activation");
    YVEX_TEST_ASSERT(value->rank == 2 && value->dims[0] == 1 && value->dims[1] == 4,
                     "hidden activation shape");

    op = yvex_graph_op_at(graph, 0);
    YVEX_TEST_ASSERT(op != NULL, "op 0 exists");
    YVEX_TEST_ASSERT(op->kind == YVEX_OP_EMBED, "embed op");
    YVEX_TEST_ASSERT(op->status == YVEX_OP_STATUS_PLANNED, "embed planned");

    missing = yvex_graph_missing_required_at(graph, 0);
    YVEX_TEST_ASSERT(missing != NULL, "missing 0 exists");
    YVEX_TEST_ASSERT(missing->role == YVEX_TENSOR_ROLE_OUTPUT_NORM, "missing output norm");
    missing = yvex_graph_missing_required_at(graph, 1);
    YVEX_TEST_ASSERT(missing != NULL, "missing 1 exists");
    YVEX_TEST_ASSERT(missing->role == YVEX_TENSOR_ROLE_OUTPUT_HEAD, "missing output head");

    yvex_graph_close(graph);
    close_fixture(&fixture);
    return 0;
}

static int component_cancelled(void *context)
{
    return context && *(const int *)context;
}

static int test_component_execution_transaction(void)
{
    yvex_component_plan plan = {.workspace_bytes = 32ull};
    int cancelled = 0;
    yvex_component_execution_request request = {
        .plan = &plan,
        .cancelled = component_cancelled,
        .cancellation_context = &cancelled,
    };
    yvex_component_execution_result result;
    yvex_component_failure failure;
    yvex_graph_component_execution execution;
    yvex_graph_component_buffer first = {0}, refused = {0};
    const yvex_graph_component_api *component = yvex_graph_component_api_get();
    yvex_error err;

    component->execution_open(&execution, NULL, &request, &result, &failure, &err,
                              "test.graph.component");
    YVEX_TEST_ASSERT(component->buffer_open(&execution, 4ull, &first) == YVEX_OK &&
                         first.data && first.count == 4ull &&
                         result.peak_workspace_bytes == 16ull &&
                         execution.workspace.live_bytes == 16ull,
                     "generic component transaction accounts owned workspace");
    YVEX_TEST_ASSERT(component->buffer_open(&execution, 5ull, &refused) == YVEX_ERR_BOUNDS &&
                         !refused.data && failure.code == YVEX_COMPONENT_FAILURE_BUDGET &&
                         failure.expected == 32ull && failure.actual == 36ull &&
                         strcmp(yvex_error_where(&err), "test.graph.component") == 0,
                     "generic component transaction publishes bounded typed refusal");
    component->buffer_close(&execution, &first);
    YVEX_TEST_ASSERT(!first.data && execution.workspace.live_bytes == 0ull,
                     "generic component transaction releases its workspace ownership");
    cancelled = 1;
    YVEX_TEST_ASSERT(component->cancel_check(&execution, "component test cancelled") ==
                             YVEX_ERR_CANCELLED &&
                         failure.code == YVEX_COMPONENT_FAILURE_CANCELLED &&
                         strcmp(failure.reason, "component test cancelled") == 0,
                     "generic component transaction owns cancellation projection");
    return 0;
}

static int synthetic_family_compile(
    yvex_family_source_products *out,
    const yvex_compilation_runtime_binding_request *request,
    yvex_error *err)
{
    (void)out;
    (void)request;
    yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "test.synthetic-family",
                   "synthetic descriptor does not execute a model");
    return YVEX_ERR_UNSUPPORTED;
}

static const yvex_family_source_adapter *synthetic_family_source(void)
{
    static const yvex_family_source_adapter source = {
        .schema_version = YVEX_FAMILY_SOURCE_ADAPTER_SCHEMA_V1,
        .target_id = "synthetic-family-test",
        .family = "synthetic",
        .tokenizer_architecture = "synthetic",
        .tokenizer_pre = "synthetic",
        .compile = synthetic_family_compile,
    };

    return &source;
}

static int test_family_descriptor_registration(void)
{
    static const yvex_family_descriptor descriptor = {
        .schema_version = YVEX_FAMILY_DESCRIPTOR_SCHEMA_V1,
        .target_id = "synthetic-family-test",
        .family = "synthetic",
        .tokenizer_architecture = "synthetic",
        .tokenizer_pre = "synthetic",
        .source = synthetic_family_source,
    };
    const yvex_family_descriptor *registered[] = {&descriptor};
    const yvex_family_descriptor *duplicate[] = {&descriptor, &descriptor};

    YVEX_TEST_ASSERT(
        yvex_family_descriptor_find_registered(registered, 1ull,
                                               "synthetic-family-test") == &descriptor,
        "synthetic family descriptor registers without a central catalog edit");
    YVEX_TEST_ASSERT(
        yvex_family_descriptor_find_registered(registered, 1ull, "unknown") == NULL,
        "family descriptor discovery rejects an unknown target");
    YVEX_TEST_ASSERT(
        yvex_family_descriptor_find_registered(duplicate, 2ull,
                                               "synthetic-family-test") == NULL,
        "family descriptor discovery rejects ambiguous registration");
    return 0;
}

int yvex_test_graph(void)
{
    if (test_graph_from_fixture() != 0) return 1;
    if (test_component_execution_transaction() != 0) return 1;
    if (test_family_descriptor_registration() != 0) return 1;
    return 0;
}
