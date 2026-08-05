/*
 * Fixtures remain deterministic and inspect only typed immutable APIs. Test mutation
 * deliberately probes refusal and never becomes an API.
 */
#include "tests/test.h"

#include <yvex/internal/compilation.h>

#include <stdlib.h>
#include <string.h>

#define FIXTURE_LOGICAL_ID \
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
#define FIXTURE_PAYLOAD_ID \
    "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"

typedef struct {
    unsigned int calls;
    unsigned int fail_at;
    unsigned int live;
} transform_test_allocator;

static void *transform_test_allocate(size_t size, void *context)
{
    transform_test_allocator *state =
        (transform_test_allocator *)context;
    void *allocation;
    if (state->calls++ == state->fail_at) return NULL;
    allocation = malloc(size);
    if (allocation) state->live++;
    return allocation;
}

static void transform_test_release(void *allocation, void *context)
{
    transform_test_allocator *state =
        (transform_test_allocator *)context;
    if (!allocation) return;
    free(allocation);
    state->live--;
}

static void transform_test_header(yvex_transform_header *header,
                                  unsigned long long sources,
                                  unsigned long long terminals)
{
    memset(header, 0, sizeof(*header));
    header->schema_version = YVEX_TRANSFORM_IR_SCHEMA_VERSION;
    header->logical_model_identity = FIXTURE_LOGICAL_ID;
    header->source_snapshot_identity = 0x1020304050607080ull;
    header->coverage_identity = 0x8877665544332211ull;
    header->required_payload_identity = FIXTURE_PAYLOAD_ID;
    header->payload_trust_class = "local_payload_snapshot_sealed";
    header->expected_source_count = sources;
    header->expected_terminal_count = terminals;
}

static yvex_transform_shape transform_test_shape(unsigned long long d0,
                                                 unsigned long long d1)
{
    yvex_transform_shape shape;
    memset(&shape, 0, sizeof(shape));
    shape.rank = d1 ? 2u : 1u;
    shape.dims[0] = d0;
    if (d1) shape.dims[1] = d1;
    return shape;
}

static yvex_transform_precision_constraint transform_test_exact(
    unsigned int physical)
{
    yvex_transform_precision_constraint precision;
    memset(&precision, 0, sizeof(precision));
    precision.flags = YVEX_TRANSFORM_PRECISION_EXACT;
    precision.allowed_physical_classes = physical;
    return precision;
}

static yvex_transform_precision_constraint transform_test_paired(void)
{
    yvex_transform_precision_constraint precision;
    memset(&precision, 0, sizeof(precision));
    precision.flags = YVEX_TRANSFORM_PRECISION_SCALE_PAIRED |
                      YVEX_TRANSFORM_PRECISION_QUANTIZABLE_WEIGHT |
                      YVEX_TRANSFORM_PRECISION_REFERENCE_COMPUTE;
    precision.allowed_physical_classes =
        YVEX_TRANSFORM_PHYSICAL_F32 | YVEX_TRANSFORM_PHYSICAL_QUANTIZED;
    precision.approximation_allowed = 1;
    precision.reference_compute_required = 1;
    return precision;
}

static void transform_test_source_spec(
    yvex_transform_source_spec *source,
    const char *name,
    unsigned long long tensor_index,
    yvex_native_dtype source_dtype,
    yvex_transform_dtype value_dtype,
    yvex_transform_shape shape,
    unsigned long long expert)
{
    memset(source, 0, sizeof(*source));
    source->source_name = name;
    source->shard_name = tensor_index & 1u ? "model-00002.safetensors"
                                          : "model-00001.safetensors";
    source->source_tensor_index = tensor_index;
    source->requirement_index = tensor_index;
    source->source_snapshot_identity = 0x1020304050607080ull;
    source->source_dtype = source_dtype;
    source->value_dtype = value_dtype;
    source->shape = shape;
    source->relative_begin = tensor_index * 4096u + 1u;
    source->relative_end = source->relative_begin + 64u;
    source->requirement_identity = tensor_index + 0x1000u;
    source->scope = YVEX_TRANSFORM_SCOPE_MAIN_LAYER;
    source->subsystem = YVEX_TRANSFORM_SUBSYSTEM_ATTENTION;
    source->role_hint = YVEX_TENSOR_ROLE_ATTENTION_Q_A;
    source->layer_index = 0u;
    source->auxiliary_index = YVEX_TRANSFORM_IR_NO_ID;
    source->expert_index = expert;
    source->required_uses = 1u;
}

static int transform_test_add_source(
    yvex_transform_builder *builder,
    const char *name,
    unsigned long long tensor_index,
    yvex_native_dtype source_dtype,
    yvex_transform_dtype value_dtype,
    yvex_transform_shape shape,
    unsigned long long expert,
    unsigned long long *value,
    yvex_transform_failure *failure,
    yvex_error *err)
{
    yvex_transform_source_spec source;
    transform_test_source_spec(&source, name, tensor_index, source_dtype,
                               value_dtype, shape, expert);
    return yvex_transform_builder_add_source(
        builder, &source, value, failure, err);
}

static void transform_test_terminal_spec(
    yvex_transform_value_spec *value,
    unsigned long long ordinal,
    yvex_tensor_role role,
    yvex_transform_shape shape,
    yvex_transform_dtype dtype,
    yvex_transform_precision_constraint precision)
{
    memset(value, 0, sizeof(*value));
    value->kind = YVEX_TRANSFORM_VALUE_TERMINAL;
    value->semantic_id = 0x2000u + ordinal;
    value->canonical_ordinal = ordinal;
    value->shape = shape;
    value->dtype = dtype;
    value->precision = precision;
    value->logical_key.scope = YVEX_TRANSFORM_SCOPE_MAIN_LAYER;
    value->logical_key.subsystem = YVEX_TRANSFORM_SUBSYSTEM_ATTENTION;
    value->logical_key.role = role;
    value->logical_key.layer_index = 0u;
    value->logical_key.auxiliary_index = YVEX_TRANSFORM_IR_NO_ID;
    value->logical_key.group_index = ordinal;
}

static int transform_test_add_terminal(
    yvex_transform_builder *builder,
    unsigned long long ordinal,
    yvex_tensor_role role,
    yvex_transform_shape shape,
    yvex_transform_dtype dtype,
    yvex_transform_precision_constraint precision,
    yvex_transform_node_spec *node,
    unsigned long long *terminal,
    yvex_transform_failure *failure,
    yvex_error *err)
{
    yvex_transform_value_spec value;
    unsigned long long node_id;
    int rc;

    transform_test_terminal_spec(&value, ordinal, role, shape, dtype,
                                 precision);
    rc = yvex_transform_builder_declare_value(
        builder, &value, terminal, failure, err);
    if (rc != YVEX_OK) return rc;
    node->output_value_id = *terminal;
    return yvex_transform_builder_add_node(
        builder, node, &node_id, failure, err);
}

static int transform_test_build_direct(int reverse,
                                       unsigned long long width,
                                       const yvex_transform_builder_options *options,
                                       yvex_transform_ir **out,
                                       yvex_transform_failure *failure)
{
    yvex_transform_header header;
    yvex_transform_builder *builder = NULL;
    yvex_transform_shape shape = transform_test_shape(width, 0u);
    yvex_transform_precision_constraint precision =
        transform_test_exact(YVEX_TRANSFORM_PHYSICAL_F32);
    yvex_error err;
    unsigned long long sources[2];
    unsigned long long terminals[2];
    unsigned int step;
    int rc;

    if (out) *out = NULL;
    transform_test_header(&header, 2u, 2u);
    yvex_error_clear(&err);
    rc = yvex_transform_builder_create(
        &builder, &header, options, failure, &err);
    for (step = 0u; rc == YVEX_OK && step < 2u; ++step) {
        unsigned int semantic = reverse ? 1u - step : step;
        const char *name = semantic ? "source.beta" : "source.alpha";
        yvex_transform_node_spec node;
        rc = transform_test_add_source(
            builder, name, semantic, YVEX_NATIVE_DTYPE_F32,
            YVEX_TRANSFORM_DTYPE_F32, shape, YVEX_TRANSFORM_IR_NO_ID,
            &sources[semantic], failure, &err);
        if (rc != YVEX_OK) break;
        memset(&node, 0, sizeof(node));
        node.kind = YVEX_TRANSFORM_OP_IDENTITY;
        node.input_value_ids = &sources[semantic];
        node.input_count = 1u;
        node.numeric = YVEX_TRANSFORM_NUMERIC_EXACT;
        node.ordering = YVEX_TRANSFORM_ORDER_INPUT;
        node.payload_execution_required = 1;
        rc = transform_test_add_terminal(
            builder, semantic,
            semantic ? YVEX_TENSOR_ROLE_ATTENTION_Q_B
                     : YVEX_TENSOR_ROLE_ATTENTION_Q_A,
            shape, YVEX_TRANSFORM_DTYPE_F32, precision, &node,
            &terminals[semantic], failure, &err);
    }
    if (rc == YVEX_OK)
        rc = yvex_transform_builder_seal(builder, out, failure, &err);
    yvex_transform_builder_release(&builder);
    return rc;
}

static int transform_test_build_direct_variant(
    yvex_native_dtype native_dtype,
    yvex_transform_dtype value_dtype,
    int swap_edges,
    yvex_transform_ir **out,
    yvex_transform_failure *failure)
{
    yvex_transform_header header;
    yvex_transform_builder *builder = NULL;
    yvex_transform_shape shape = transform_test_shape(8u, 0u);
    yvex_transform_precision_constraint precision = transform_test_exact(
        value_dtype == YVEX_TRANSFORM_DTYPE_BF16
            ? YVEX_TRANSFORM_PHYSICAL_BF16 : YVEX_TRANSFORM_PHYSICAL_F32);
    yvex_transform_node_spec node;
    yvex_error err;
    unsigned long long sources[2];
    unsigned long long terminal;
    unsigned int index;
    int rc;

    if (out) *out = NULL;
    transform_test_header(&header, 2u, 2u);
    yvex_error_clear(&err);
    rc = yvex_transform_builder_create(
        &builder, &header, NULL, failure, &err);
    for (index = 0u; rc == YVEX_OK && index < 2u; ++index)
        rc = transform_test_add_source(
            builder, index ? "variant.beta" : "variant.alpha", index,
            native_dtype, value_dtype, shape, YVEX_TRANSFORM_IR_NO_ID,
            &sources[index], failure, &err);
    for (index = 0u; rc == YVEX_OK && index < 2u; ++index) {
        unsigned long long input = sources[swap_edges ? 1u - index : index];
        memset(&node, 0, sizeof(node));
        node.kind = YVEX_TRANSFORM_OP_IDENTITY;
        node.input_value_ids = &input;
        node.input_count = 1u;
        node.numeric = YVEX_TRANSFORM_NUMERIC_EXACT;
        node.ordering = YVEX_TRANSFORM_ORDER_INPUT;
        node.payload_execution_required = 1;
        rc = transform_test_add_terminal(
            builder, index,
            index ? YVEX_TENSOR_ROLE_ATTENTION_Q_B
                  : YVEX_TENSOR_ROLE_ATTENTION_Q_A,
            shape, value_dtype, precision, &node, &terminal, failure, &err);
    }
    if (rc == YVEX_OK)
        rc = yvex_transform_builder_seal(builder, out, failure, &err);
    yvex_transform_builder_release(&builder);
    return rc;
}

/* Builds one valid concatenate plan whose axis is part of canonical identity. */
static int transform_test_build_axis_variant(
    unsigned int axis,
    yvex_transform_ir **out,
    yvex_transform_failure *failure)
{
    yvex_transform_header header;
    yvex_transform_builder *builder = NULL;
    yvex_transform_shape input_shape = transform_test_shape(2u, 2u);
    yvex_transform_shape output_shape = input_shape;
    yvex_transform_precision_constraint precision =
        transform_test_exact(YVEX_TRANSFORM_PHYSICAL_F32);
    yvex_transform_node_spec node;
    yvex_error err;
    unsigned long long inputs[2];
    unsigned long long terminal;
    unsigned int index;
    int rc;

    if (out) *out = NULL;
    output_shape.dims[axis] = 4u;
    transform_test_header(&header, 2u, 1u);
    yvex_error_clear(&err);
    rc = yvex_transform_builder_create(
        &builder, &header, NULL, failure, &err);
    for (index = 0u; rc == YVEX_OK && index < 2u; ++index)
        rc = transform_test_add_source(
            builder, index ? "axis.beta" : "axis.alpha", index,
            YVEX_NATIVE_DTYPE_F32, YVEX_TRANSFORM_DTYPE_F32, input_shape,
            YVEX_TRANSFORM_IR_NO_ID, &inputs[index], failure, &err);
    memset(&node, 0, sizeof(node));
    node.kind = YVEX_TRANSFORM_OP_CONCATENATE;
    node.input_value_ids = inputs;
    node.input_count = 2u;
    node.axis = axis;
    node.numeric = YVEX_TRANSFORM_NUMERIC_EXACT;
    node.ordering = YVEX_TRANSFORM_ORDER_AXIS;
    node.payload_execution_required = 1;
    if (rc == YVEX_OK)
        rc = transform_test_add_terminal(
            builder, 0u, YVEX_TENSOR_ROLE_ATTENTION_Q_A, output_shape,
            YVEX_TRANSFORM_DTYPE_F32, precision, &node, &terminal,
            failure, &err);
    if (rc == YVEX_OK)
        rc = yvex_transform_builder_seal(builder, out, failure, &err);
    yvex_transform_builder_release(&builder);
    return rc;
}

static int test_transform_identity_and_lifecycle(void)
{
    yvex_transform_ir *forward = NULL;
    yvex_transform_ir *reverse = NULL;
    yvex_transform_ir *changed = NULL;
    yvex_transform_ir *edge_changed = NULL;
    yvex_transform_ir *dtype_changed = NULL;
    yvex_transform_ir *axis_zero = NULL;
    yvex_transform_ir *axis_one = NULL;
    yvex_transform_failure failure;
    const yvex_transform_ir_summary *summary;

    YVEX_TEST_ASSERT(transform_test_build_direct(
                         0, 8u, NULL, &forward, &failure) == YVEX_OK && forward,
                     "direct sealed IR builds");
    YVEX_TEST_ASSERT(transform_test_build_direct(
                         1, 8u, NULL, &reverse, &failure) == YVEX_OK && reverse,
                     "reverse-registration sealed IR builds");
    summary = yvex_transform_ir_summary_get(forward);
    YVEX_TEST_ASSERT(summary && summary->complete &&
                         summary->source_value_count == 2u &&
                         summary->node_count == 2u && summary->edge_count == 2u &&
                         summary->terminal_count == 2u &&
                         summary->maximum_depth == 1u &&
                         summary->payload_bytes_read == 0u,
                     "sealed direct graph accounting is exact");
    YVEX_TEST_ASSERT_STREQ(
        summary->transform_identity,
        yvex_transform_ir_summary_get(reverse)->transform_identity,
        "semantic identity is insertion-order independent");
    YVEX_TEST_ASSERT(transform_test_build_direct(
                         0, 9u, NULL, &changed, &failure) == YVEX_OK && changed,
                     "semantic mutation graph builds");
    YVEX_TEST_ASSERT(strcmp(
                         summary->transform_identity,
                         yvex_transform_ir_summary_get(changed)->
                             transform_identity) != 0,
                     "shape mutation changes canonical identity");
    YVEX_TEST_ASSERT(transform_test_build_direct_variant(
                         YVEX_NATIVE_DTYPE_F32, YVEX_TRANSFORM_DTYPE_F32,
                         1, &edge_changed, &failure) == YVEX_OK &&
                         transform_test_build_direct_variant(
                             YVEX_NATIVE_DTYPE_BF16,
                             YVEX_TRANSFORM_DTYPE_BF16, 0,
                             &dtype_changed, &failure) == YVEX_OK &&
                         strcmp(summary->transform_identity,
                                yvex_transform_ir_summary_get(edge_changed)->
                                    transform_identity) != 0 &&
                         strcmp(summary->transform_identity,
                                yvex_transform_ir_summary_get(dtype_changed)->
                                    transform_identity) != 0,
                     "edge and dtype mutations change canonical identity");
    YVEX_TEST_ASSERT(transform_test_build_axis_variant(
                         0u, &axis_zero, &failure) == YVEX_OK &&
                         transform_test_build_axis_variant(
                             1u, &axis_one, &failure) == YVEX_OK &&
                         strcmp(yvex_transform_ir_summary_get(axis_zero)->
                                    transform_identity,
                                yvex_transform_ir_summary_get(axis_one)->
                                    transform_identity) != 0,
                     "axis mutation changes canonical identity");
    YVEX_TEST_ASSERT(yvex_transform_ir_source_find(
                         forward, "source.alpha") != NULL &&
                         yvex_transform_ir_terminal_at(forward, 0u) != NULL &&
                         yvex_transform_ir_node_topological_at(
                             forward, 1u) != NULL,
                     "sealed indexes and deterministic traversal are usable");
    yvex_transform_ir_release(&changed);
    yvex_transform_ir_release(&axis_one);
    yvex_transform_ir_release(&axis_zero);
    yvex_transform_ir_release(&dtype_changed);
    yvex_transform_ir_release(&edge_changed);
    yvex_transform_ir_release(&changed);
    yvex_transform_ir_release(&reverse);
    yvex_transform_ir_release(&forward);
    YVEX_TEST_ASSERT(!forward && !reverse && !changed,
                     "IR release is idempotent through owned pointer API");
    return 0;
}

static int test_transform_operation_suite(void)
{
    const unsigned long long expert_count = 256u;
    const unsigned long long source_count = 2u + 1u + 1u + 1u + 2u +
                                            2u + 2u + expert_count * 2u;
    yvex_transform_header header;
    yvex_transform_builder *builder = NULL;
    yvex_transform_ir *ir = NULL;
    yvex_transform_failure failure;
    yvex_error err;
    unsigned long long source_ordinal = 0u;
    unsigned long long terminal_ordinal = 0u;
    unsigned long long input[512];
    unsigned long long terminal;
    yvex_transform_node_spec node;
    yvex_transform_precision_constraint precision;
    yvex_transform_shape shape;
    unsigned long long index;
    int rc;

    transform_test_header(&header, source_count, 8u);
    yvex_error_clear(&err);
    rc = yvex_transform_builder_create(
        &builder, &header, NULL, &failure, &err);

#define ADD_SOURCE(name, native, logical, input_shape, expert_id, target) do { \
    if (rc == YVEX_OK) rc = transform_test_add_source(                       \
        builder, name, source_ordinal++, native, logical, input_shape,        \
        expert_id, target, &failure, &err);                                   \
} while (0)
#define ADD_TERMINAL(role_id, output_shape, output_dtype, output_precision) do {\
    if (rc == YVEX_OK) rc = transform_test_add_terminal(                      \
        builder, terminal_ordinal++, role_id, output_shape, output_dtype,     \
        output_precision, &node, &terminal, &failure, &err);                  \
} while (0)
    shape = transform_test_shape(256u, 256u);
    ADD_SOURCE("pair.weight", YVEX_NATIVE_DTYPE_F8_E4M3,
               YVEX_TRANSFORM_DTYPE_FP8_E4M3, shape,
               YVEX_TRANSFORM_IR_NO_ID, &input[0]);
    shape = transform_test_shape(2u, 2u);
    ADD_SOURCE("pair.scale", YVEX_NATIVE_DTYPE_F8_E8M0,
               YVEX_TRANSFORM_DTYPE_E8M0_SCALE, shape,
               YVEX_TRANSFORM_IR_NO_ID, &input[1]);
    memset(&node, 0, sizeof(node));
    node.kind = YVEX_TRANSFORM_OP_DECODE_SCALE_PAIR;
    node.input_value_ids = input;
    node.input_count = 2u;
    node.scale_block_rows = 128u;
    node.scale_block_columns = 128u;
    node.numeric = YVEX_TRANSFORM_NUMERIC_LOSSLESS;
    node.ordering = YVEX_TRANSFORM_ORDER_INPUT;
    node.payload_execution_required = 1;
    shape = transform_test_shape(256u, 256u);
    precision = transform_test_paired();
    ADD_TERMINAL(YVEX_TENSOR_ROLE_ATTENTION_Q_A, shape,
                 YVEX_TRANSFORM_DTYPE_REAL, precision);

    shape = transform_test_shape(4u, 4u);
    ADD_SOURCE("cast.table", YVEX_NATIVE_DTYPE_I64,
               YVEX_TRANSFORM_DTYPE_I64, shape,
               YVEX_TRANSFORM_IR_NO_ID, &input[0]);
    memset(&node, 0, sizeof(node));
    node.kind = YVEX_TRANSFORM_OP_CHECKED_CAST;
    node.input_value_ids = input;
    node.input_count = 1u;
    node.numeric = YVEX_TRANSFORM_NUMERIC_RANGE_PROOF;
    node.ordering = YVEX_TRANSFORM_ORDER_INPUT;
    node.payload_execution_required = 1;
    memset(&precision, 0, sizeof(precision));
    precision.flags = YVEX_TRANSFORM_PRECISION_LOSSLESS |
                      YVEX_TRANSFORM_PRECISION_RANGE_PROOF |
                      YVEX_TRANSFORM_PRECISION_INTEGER_ONLY;
    precision.allowed_physical_classes = YVEX_TRANSFORM_PHYSICAL_I32;
    precision.range_proof_required = 1;
    ADD_TERMINAL(YVEX_TENSOR_ROLE_MOE_ROUTER_TABLE, shape,
                 YVEX_TRANSFORM_DTYPE_I32, precision);

    shape = transform_test_shape(2u, 6u);
    ADD_SOURCE("reshape.input", YVEX_NATIVE_DTYPE_F32,
               YVEX_TRANSFORM_DTYPE_F32, shape,
               YVEX_TRANSFORM_IR_NO_ID, &input[0]);
    memset(&node, 0, sizeof(node));
    node.kind = YVEX_TRANSFORM_OP_RESHAPE;
    node.input_value_ids = input;
    node.input_count = 1u;
    node.numeric = YVEX_TRANSFORM_NUMERIC_EXACT;
    node.ordering = YVEX_TRANSFORM_ORDER_INPUT;
    node.payload_execution_required = 1;
    shape = transform_test_shape(3u, 4u);
    precision = transform_test_exact(YVEX_TRANSFORM_PHYSICAL_F32);
    ADD_TERMINAL(YVEX_TENSOR_ROLE_ATTENTION_Q_B, shape,
                 YVEX_TRANSFORM_DTYPE_F32, precision);

    shape = transform_test_shape(2u, 3u);
    ADD_SOURCE("transpose.input", YVEX_NATIVE_DTYPE_F32,
               YVEX_TRANSFORM_DTYPE_F32, shape,
               YVEX_TRANSFORM_IR_NO_ID, &input[0]);
    memset(&node, 0, sizeof(node));
    node.kind = YVEX_TRANSFORM_OP_TRANSPOSE;
    node.input_value_ids = input;
    node.input_count = 1u;
    node.permutation_rank = 2u;
    node.permutation[0] = 1u;
    node.permutation[1] = 0u;
    node.numeric = YVEX_TRANSFORM_NUMERIC_EXACT;
    node.ordering = YVEX_TRANSFORM_ORDER_AXIS;
    node.payload_execution_required = 1;
    shape = transform_test_shape(3u, 2u);
    ADD_TERMINAL(YVEX_TENSOR_ROLE_ATTENTION_KV, shape,
                 YVEX_TRANSFORM_DTYPE_F32, precision);

    shape = transform_test_shape(2u, 3u);
    ADD_SOURCE("concat.a", YVEX_NATIVE_DTYPE_F32,
               YVEX_TRANSFORM_DTYPE_F32, shape,
               YVEX_TRANSFORM_IR_NO_ID, &input[0]);
    shape = transform_test_shape(4u, 3u);
    ADD_SOURCE("concat.b", YVEX_NATIVE_DTYPE_F32,
               YVEX_TRANSFORM_DTYPE_F32, shape,
               YVEX_TRANSFORM_IR_NO_ID, &input[1]);
    memset(&node, 0, sizeof(node));
    node.kind = YVEX_TRANSFORM_OP_CONCATENATE;
    node.input_value_ids = input;
    node.input_count = 2u;
    node.axis = 0u;
    node.numeric = YVEX_TRANSFORM_NUMERIC_EXACT;
    node.ordering = YVEX_TRANSFORM_ORDER_AXIS;
    node.payload_execution_required = 1;
    shape = transform_test_shape(6u, 3u);
    ADD_TERMINAL(YVEX_TENSOR_ROLE_ATTENTION_OUT_A, shape,
                 YVEX_TRANSFORM_DTYPE_F32, precision);

    shape = transform_test_shape(2u, 3u);
    ADD_SOURCE("stack.a", YVEX_NATIVE_DTYPE_F32,
               YVEX_TRANSFORM_DTYPE_F32, shape,
               YVEX_TRANSFORM_IR_NO_ID, &input[0]);
    ADD_SOURCE("stack.b", YVEX_NATIVE_DTYPE_F32,
               YVEX_TRANSFORM_DTYPE_F32, shape,
               YVEX_TRANSFORM_IR_NO_ID, &input[1]);
    memset(&node, 0, sizeof(node));
    node.kind = YVEX_TRANSFORM_OP_STACK;
    node.input_value_ids = input;
    node.input_count = 2u;
    node.axis = 0u;
    node.numeric = YVEX_TRANSFORM_NUMERIC_EXACT;
    node.ordering = YVEX_TRANSFORM_ORDER_AXIS;
    node.payload_execution_required = 1;
    memset(&shape, 0, sizeof(shape));
    shape.rank = 3u;
    shape.dims[0] = 2u;
    shape.dims[1] = 2u;
    shape.dims[2] = 3u;
    ADD_TERMINAL(YVEX_TENSOR_ROLE_ATTENTION_OUT_B, shape,
                 YVEX_TRANSFORM_DTYPE_F32, precision);

    shape = transform_test_shape(2u, 3u);
    ADD_SOURCE("aggregate.a", YVEX_NATIVE_DTYPE_F32,
               YVEX_TRANSFORM_DTYPE_F32, shape,
               YVEX_TRANSFORM_IR_NO_ID, &input[0]);
    ADD_SOURCE("aggregate.b", YVEX_NATIVE_DTYPE_F32,
               YVEX_TRANSFORM_DTYPE_F32, shape,
               YVEX_TRANSFORM_IR_NO_ID, &input[1]);
    memset(&node, 0, sizeof(node));
    node.kind = YVEX_TRANSFORM_OP_AGGREGATE;
    node.input_value_ids = input;
    node.input_count = 2u;
    node.numeric = YVEX_TRANSFORM_NUMERIC_EXACT;
    node.ordering = YVEX_TRANSFORM_ORDER_INPUT;
    node.payload_execution_required = 1;
    ADD_TERMINAL(YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_KV, shape,
                 YVEX_TRANSFORM_DTYPE_F32, precision);

    for (index = 0u; rc == YVEX_OK && index < expert_count; ++index) {
        char weight[64];
        char scale[64];
        (void)snprintf(weight, sizeof(weight), "expert.%llu.weight", index);
        (void)snprintf(scale, sizeof(scale), "expert.%llu.scale", index);
        shape = transform_test_shape(4u, 16u);
        ADD_SOURCE(weight, YVEX_NATIVE_DTYPE_I8,
                   YVEX_TRANSFORM_DTYPE_PACKED_FP4, shape, index,
                   &input[index * 2u]);
        shape = transform_test_shape(4u, 1u);
        ADD_SOURCE(scale, YVEX_NATIVE_DTYPE_F8_E8M0,
                   YVEX_TRANSFORM_DTYPE_E8M0_SCALE, shape, index,
                   &input[index * 2u + 1u]);
    }
    memset(&node, 0, sizeof(node));
    node.kind = YVEX_TRANSFORM_OP_EXPERT_AGGREGATE;
    node.input_value_ids = input;
    node.input_count = expert_count * 2u;
    node.axis = 0u;
    node.expert_count = expert_count;
    node.packing_factor = 2u;
    node.scale_group_width = 32u;
    node.numeric = YVEX_TRANSFORM_NUMERIC_LOSSLESS;
    node.ordering = YVEX_TRANSFORM_ORDER_EXPERT_INDEX;
    node.payload_execution_required = 1;
    memset(&shape, 0, sizeof(shape));
    shape.rank = 3u;
    shape.dims[0] = expert_count;
    shape.dims[1] = 4u;
    shape.dims[2] = 32u;
    precision = transform_test_paired();
    ADD_TERMINAL(YVEX_TENSOR_ROLE_MOE_EXPERT_GATE, shape,
                 YVEX_TRANSFORM_DTYPE_REAL, precision);
#undef ADD_TERMINAL
#undef ADD_SOURCE
    YVEX_TEST_ASSERT(rc == YVEX_OK && source_ordinal == source_count &&
                         terminal_ordinal == 8u,
                     "all closed operation fixtures register");
    rc = yvex_transform_builder_seal(builder, &ir, &failure, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && ir,
                     "closed operation taxonomy seals");
    YVEX_TEST_ASSERT(yvex_transform_ir_summary_get(ir)->maximum_fan_in == 512u &&
                         yvex_transform_ir_summary_get(ir)->edge_count ==
                             source_count &&
                         yvex_transform_ir_summary_get(ir)->
                             operation_counts[YVEX_TRANSFORM_OP_EXPERT_AGGREGATE] == 1u &&
                         yvex_transform_ir_summary_get(ir)->
                             operation_counts[YVEX_TRANSFORM_OP_DECODE_SCALE_PAIR] == 1u,
                     "large fan-in and operation accounting are exact");
    yvex_transform_ir_release(&ir);
    yvex_transform_builder_release(&builder);
    return 0;
}

static int test_transform_negative_graphs(void)
{
    yvex_transform_header header;
    yvex_transform_builder *builder = NULL;
    yvex_transform_ir *ir = NULL;
    yvex_transform_failure failure;
    yvex_error err;
    yvex_transform_shape shape = transform_test_shape(2u, 2u);
    yvex_transform_node_spec node;
    yvex_transform_source_spec source_spec;
    yvex_transform_value_spec value_spec;
    yvex_transform_precision_constraint precision =
        transform_test_exact(YVEX_TRANSFORM_PHYSICAL_F32);
    unsigned long long source;
    unsigned long long terminal;
    unsigned long long missing = 999u;
    int rc;

    transform_test_header(&header, 1u, 1u);
    YVEX_TEST_ASSERT(yvex_transform_builder_create(
                         &builder, &header, NULL, &failure, &err) == YVEX_OK,
                     "negative unresolved builder opens");
    YVEX_TEST_ASSERT(transform_test_add_source(
                         builder, "negative.source", 0u,
                         YVEX_NATIVE_DTYPE_F32, YVEX_TRANSFORM_DTYPE_F32,
                         shape, YVEX_TRANSFORM_IR_NO_ID, &source,
                         &failure, &err) == YVEX_OK,
                     "negative source registers");
    memset(&node, 0, sizeof(node));
    node.kind = YVEX_TRANSFORM_OP_IDENTITY;
    node.input_value_ids = &missing;
    node.input_count = 1u;
    node.numeric = YVEX_TRANSFORM_NUMERIC_EXACT;
    node.ordering = YVEX_TRANSFORM_ORDER_INPUT;
    node.payload_execution_required = 1;
    YVEX_TEST_ASSERT(transform_test_add_terminal(
                         builder, 0u, YVEX_TENSOR_ROLE_ATTENTION_Q_A,
                         shape, YVEX_TRANSFORM_DTYPE_F32, precision, &node,
                         &terminal, &failure, &err) == YVEX_OK,
                     "unresolved node registers before seal");
    rc = yvex_transform_builder_seal(builder, &ir, &failure, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK && !ir &&
                         failure.code == YVEX_TRANSFORM_FAILURE_UNRESOLVED_EDGE,
                     "unresolved edge refuses sealing");
    yvex_transform_builder_release(&builder);

    transform_test_header(&header, 1u, 1u);
    header.schema_version = YVEX_TRANSFORM_IR_COMPONENT_SCHEMA_VERSION + 1u;
    YVEX_TEST_ASSERT(yvex_transform_builder_create(
                         &builder, &header, NULL, &failure, &err) != YVEX_OK &&
                         !builder &&
                         failure.code ==
                             YVEX_TRANSFORM_FAILURE_SCHEMA_UNSUPPORTED,
                     "unsupported schema refuses before allocation");
    transform_test_header(&header, 1u, 1u);
    YVEX_TEST_ASSERT(yvex_transform_builder_create(
                         &builder, &header, NULL, &failure, &err) == YVEX_OK &&
                         transform_test_add_source(
                             builder, "unsupported.f64", 0u,
                             YVEX_NATIVE_DTYPE_F64,
                             YVEX_TRANSFORM_DTYPE_F32, shape,
                             YVEX_TRANSFORM_IR_NO_ID, &source, &failure,
                             &err) != YVEX_OK &&
                         failure.code ==
                             YVEX_TRANSFORM_FAILURE_UNSUPPORTED_SOURCE_DTYPE,
                     "unrepresented source storage dtype refuses registration");
    yvex_transform_builder_release(&builder);
    transform_test_header(&header, 1u, 1u);
    YVEX_TEST_ASSERT(yvex_transform_builder_create(
                         &builder, &header, NULL, &failure, &err) == YVEX_OK,
                     "invalid source-scope builder opens");
    transform_test_source_spec(
        &source_spec, "invalid.scope", 0u, YVEX_NATIVE_DTYPE_F32,
        YVEX_TRANSFORM_DTYPE_F32, shape, YVEX_TRANSFORM_IR_NO_ID);
    source_spec.scope = YVEX_TRANSFORM_SCOPE_GLOBAL;
    YVEX_TEST_ASSERT(yvex_transform_builder_add_source(
                         builder, &source_spec, &source, &failure, &err) !=
                         YVEX_OK &&
                         failure.code ==
                             YVEX_TRANSFORM_FAILURE_INVALID_LOGICAL_KEY,
                     "inconsistent source scope sentinels refuse registration");
    yvex_transform_builder_release(&builder);

    transform_test_header(&header, 1u, 1u);
    YVEX_TEST_ASSERT(yvex_transform_builder_create(
                         &builder, &header, NULL, &failure, &err) == YVEX_OK &&
                         transform_test_add_source(
                             builder, "invalid.terminal.source", 0u,
                             YVEX_NATIVE_DTYPE_F32,
                             YVEX_TRANSFORM_DTYPE_F32, shape,
                             YVEX_TRANSFORM_IR_NO_ID, &source, &failure,
                             &err) == YVEX_OK,
                     "invalid terminal-key fixture source registers");
    memset(&value_spec, 0, sizeof(value_spec));
    value_spec.kind = YVEX_TRANSFORM_VALUE_TERMINAL;
    value_spec.semantic_id = 1u;
    value_spec.canonical_ordinal = 0u;
    value_spec.shape = shape;
    value_spec.dtype = YVEX_TRANSFORM_DTYPE_F32;
    value_spec.precision = precision;
    value_spec.logical_key.scope = YVEX_TRANSFORM_SCOPE_MAIN_LAYER;
    value_spec.logical_key.subsystem = YVEX_TRANSFORM_SUBSYSTEM_ATTENTION;
    value_spec.logical_key.role = YVEX_TENSOR_ROLE_UNKNOWN;
    value_spec.logical_key.layer_index = 0u;
    value_spec.logical_key.auxiliary_index = YVEX_TRANSFORM_IR_NO_ID;
    YVEX_TEST_ASSERT(yvex_transform_builder_declare_value(
                         builder, &value_spec, &terminal, &failure, &err) !=
                         YVEX_OK &&
                         failure.code ==
                             YVEX_TRANSFORM_FAILURE_INVALID_LOGICAL_KEY,
                     "invalid logical tensor role refuses declaration");
    yvex_transform_builder_release(&builder);
    transform_test_header(&header, 1u, 1u);
    {
        yvex_transform_builder_options options;
        memset(&options, 0, sizeof(options));
        yvex_transform_budget_default(&options.budget);
        options.budget.maximum_sources = 1u;
        options.budget.maximum_values = 1u;
        options.budget.maximum_terminals = 1u;
        options.budget.maximum_owned_bytes = sizeof(void *);
        YVEX_TEST_ASSERT(yvex_transform_builder_create(
                             &builder, &header, &options, &failure, &err) !=
                             YVEX_OK && !builder &&
                             failure.code ==
                                 YVEX_TRANSFORM_FAILURE_RESOURCE_BUDGET,
                         "insufficient owned-byte budget refuses construction");
    }
    return 0;
}

static yvex_transform_failure_code transform_test_operation_failure(
    yvex_transform_operation_kind kind,
    unsigned long long input_count,
    yvex_transform_shape input_shape,
    yvex_transform_shape output_shape,
    unsigned int axis,
    unsigned int permutation_rank,
    unsigned int permutation_zero,
    unsigned int permutation_one)
{
    static const char *const names[] = {
        "invalid.input.0", "invalid.input.1", "invalid.input.2"
    };
    yvex_transform_header header;
    yvex_transform_builder *builder = NULL;
    yvex_transform_ir *ir = NULL;
    yvex_transform_failure failure;
    yvex_transform_node_spec node;
    yvex_transform_precision_constraint precision =
        transform_test_exact(YVEX_TRANSFORM_PHYSICAL_F32);
    yvex_error err;
    unsigned long long inputs[3];
    unsigned long long terminal;
    unsigned long long index;
    int rc;

    transform_test_header(&header, input_count, 1u);
    yvex_error_clear(&err);
    rc = yvex_transform_builder_create(
        &builder, &header, NULL, &failure, &err);
    for (index = 0u; rc == YVEX_OK && index < input_count; ++index)
        rc = transform_test_add_source(
            builder, names[index], index, YVEX_NATIVE_DTYPE_F32,
            YVEX_TRANSFORM_DTYPE_F32, input_shape,
            YVEX_TRANSFORM_IR_NO_ID, &inputs[index], &failure, &err);
    memset(&node, 0, sizeof(node));
    node.kind = kind;
    node.input_value_ids = inputs;
    node.input_count = input_count;
    node.axis = axis;
    node.permutation_rank = permutation_rank;
    node.permutation[0] = permutation_zero;
    node.permutation[1] = permutation_one;
    node.numeric = YVEX_TRANSFORM_NUMERIC_EXACT;
    node.ordering = kind == YVEX_TRANSFORM_OP_TRANSPOSE ||
                            kind == YVEX_TRANSFORM_OP_CONCATENATE
                        ? YVEX_TRANSFORM_ORDER_AXIS
                        : YVEX_TRANSFORM_ORDER_INPUT;
    node.payload_execution_required = 1;
    if (rc == YVEX_OK)
        rc = transform_test_add_terminal(
            builder, 0u, YVEX_TENSOR_ROLE_ATTENTION_Q_A, output_shape,
            YVEX_TRANSFORM_DTYPE_F32, precision, &node, &terminal,
            &failure, &err);
    if (rc == YVEX_OK)
        rc = yvex_transform_builder_seal(builder, &ir, &failure, &err);
    yvex_transform_ir_release(&ir);
    yvex_transform_builder_release(&builder);
    return rc == YVEX_OK ? YVEX_TRANSFORM_FAILURE_NONE : failure.code;
}

static int test_transform_operation_refusals(void)
{
    yvex_transform_shape square = transform_test_shape(2u, 2u);
    yvex_transform_shape mismatch = transform_test_shape(3u, 2u);
    yvex_transform_shape overflow;
    yvex_transform_failure failure;
    yvex_error err;
    unsigned long long elements = 0u;

    YVEX_TEST_ASSERT(transform_test_operation_failure(
                         YVEX_TRANSFORM_OP_IDENTITY, 2u, square, square,
                         0u, 0u, 0u, 0u) ==
                         YVEX_TRANSFORM_FAILURE_INVALID_ARITY,
                     "identity refuses invalid arity");
    YVEX_TEST_ASSERT(transform_test_operation_failure(
                         YVEX_TRANSFORM_OP_RESHAPE, 1u, square, mismatch,
                         0u, 0u, 0u, 0u) ==
                         YVEX_TRANSFORM_FAILURE_ELEMENT_COUNT_MISMATCH,
                     "reshape refuses changed element count");
    YVEX_TEST_ASSERT(transform_test_operation_failure(
                         YVEX_TRANSFORM_OP_TRANSPOSE, 1u, square, square,
                         0u, 2u, 0u, 0u) ==
                         YVEX_TRANSFORM_FAILURE_INVALID_PERMUTATION,
                     "transpose refuses duplicate permutation axes");
    YVEX_TEST_ASSERT(transform_test_operation_failure(
                         YVEX_TRANSFORM_OP_TRANSPOSE, 1u, square, square,
                         0u, 1u, 0u, 0u) ==
                         YVEX_TRANSFORM_FAILURE_INVALID_RANK,
                     "transpose refuses a permutation with the wrong rank");
    YVEX_TEST_ASSERT(transform_test_operation_failure(
                         YVEX_TRANSFORM_OP_CONCATENATE, 2u, square, square,
                         2u, 0u, 0u, 0u) ==
                         YVEX_TRANSFORM_FAILURE_INVALID_AXIS,
                     "concatenate refuses an out-of-rank axis");
    YVEX_TEST_ASSERT(transform_test_operation_failure(
                         YVEX_TRANSFORM_OP_AGGREGATE, 2u, square, mismatch,
                         0u, 0u, 0u, 0u) ==
                         YVEX_TRANSFORM_FAILURE_INVALID_AGGREGATION,
                     "aggregate refuses an incompatible output shape");
    memset(&overflow, 0, sizeof(overflow));
    overflow.rank = 2u;
    overflow.dims[0] = ~0ull;
    overflow.dims[1] = 2u;
    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(yvex_transform_shape_element_count(
                         &overflow, &elements, &failure, &err) != YVEX_OK &&
                         elements == 0u &&
                         failure.code ==
                             YVEX_TRANSFORM_FAILURE_DIMENSION_OVERFLOW,
                     "shape product overflow is checked and typed");
    YVEX_TEST_ASSERT(transform_test_operation_failure(
                         YVEX_TRANSFORM_OP_IDENTITY, 1u, overflow, overflow,
                         0u, 0u, 0u, 0u) ==
                         YVEX_TRANSFORM_FAILURE_DIMENSION_OVERFLOW,
                     "sealing validates element geometry for every value");
    return 0;
}

/* Covers missing ownership, exact source use, terminal uniqueness, and contracts. */
static int test_transform_ownership_refusals(void)
{
    yvex_transform_header header;
    yvex_transform_builder *builder = NULL;
    yvex_transform_ir *ir = NULL;
    yvex_transform_failure failure;
    yvex_transform_node_spec node;
    yvex_transform_value_spec value;
    yvex_transform_precision_constraint precision =
        transform_test_exact(YVEX_TRANSFORM_PHYSICAL_F32);
    yvex_transform_shape shape = transform_test_shape(2u, 2u);
    yvex_error err;
    unsigned long long sources[2];
    unsigned long long values[2];
    unsigned long long node_id;

    transform_test_header(&header, 1u, 1u);
    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(yvex_transform_builder_create(
                         &builder, &header, NULL, &failure, &err) == YVEX_OK &&
                         transform_test_add_source(
                             builder, "missing.producer.source", 0u,
                             YVEX_NATIVE_DTYPE_F32, YVEX_TRANSFORM_DTYPE_F32,
                             shape, YVEX_TRANSFORM_IR_NO_ID, &sources[0],
                             &failure, &err) == YVEX_OK,
                     "missing-producer fixture source registers");
    memset(&value, 0, sizeof(value));
    value.kind = YVEX_TRANSFORM_VALUE_INTERMEDIATE;
    value.semantic_id = 0x3000u;
    value.canonical_ordinal = YVEX_TRANSFORM_IR_NO_ID;
    value.shape = shape;
    value.dtype = YVEX_TRANSFORM_DTYPE_F32;
    value.precision = precision;
    YVEX_TEST_ASSERT(yvex_transform_builder_declare_value(
                         builder, &value, &values[0], &failure, &err) ==
                         YVEX_OK,
                     "missing-producer intermediate declares");
    transform_test_terminal_spec(
        &value, 0u, YVEX_TENSOR_ROLE_ATTENTION_Q_A, shape,
        YVEX_TRANSFORM_DTYPE_F32, precision);
    YVEX_TEST_ASSERT(yvex_transform_builder_declare_value(
                         builder, &value, &values[1], &failure, &err) ==
                         YVEX_OK,
                     "unproduced terminal declares");
    memset(&node, 0, sizeof(node));
    node.kind = YVEX_TRANSFORM_OP_IDENTITY;
    node.output_value_id = values[0];
    node.input_value_ids = &sources[0];
    node.input_count = 1u;
    node.numeric = YVEX_TRANSFORM_NUMERIC_EXACT;
    node.ordering = YVEX_TRANSFORM_ORDER_INPUT;
    node.payload_execution_required = 1;
    YVEX_TEST_ASSERT(yvex_transform_builder_add_node(
                         builder, &node, &node_id, &failure, &err) == YVEX_OK &&
                         yvex_transform_builder_seal(
                             builder, &ir, &failure, &err) != YVEX_OK && !ir &&
                         failure.code ==
                             YVEX_TRANSFORM_FAILURE_MISSING_PRODUCER,
                     "a non-source value without one producer refuses sealing");
    yvex_transform_builder_release(&builder);

    transform_test_header(&header, 2u, 1u);
    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(yvex_transform_builder_create(
                         &builder, &header, NULL, &failure, &err) == YVEX_OK &&
                         transform_test_add_source(
                             builder, "consumed.source", 0u,
                             YVEX_NATIVE_DTYPE_F32, YVEX_TRANSFORM_DTYPE_F32,
                             shape, YVEX_TRANSFORM_IR_NO_ID, &sources[0],
                             &failure, &err) == YVEX_OK &&
                         transform_test_add_source(
                             builder, "unconsumed.source", 1u,
                             YVEX_NATIVE_DTYPE_F32, YVEX_TRANSFORM_DTYPE_F32,
                             shape, YVEX_TRANSFORM_IR_NO_ID, &sources[1],
                             &failure, &err) == YVEX_OK,
                     "unconsumed-source fixture registers both requirements");
    memset(&node, 0, sizeof(node));
    node.kind = YVEX_TRANSFORM_OP_IDENTITY;
    node.input_value_ids = &sources[0];
    node.input_count = 1u;
    node.numeric = YVEX_TRANSFORM_NUMERIC_EXACT;
    node.ordering = YVEX_TRANSFORM_ORDER_INPUT;
    node.payload_execution_required = 1;
    YVEX_TEST_ASSERT(transform_test_add_terminal(
                         builder, 0u, YVEX_TENSOR_ROLE_ATTENTION_Q_A, shape,
                         YVEX_TRANSFORM_DTYPE_F32, precision, &node,
                         &values[0], &failure, &err) == YVEX_OK &&
                         yvex_transform_builder_seal(
                             builder, &ir, &failure, &err) != YVEX_OK && !ir &&
                         failure.code ==
                             YVEX_TRANSFORM_FAILURE_UNCONSUMED_SOURCE,
                     "every required source must be consumed exactly once");
    yvex_transform_builder_release(&builder);

    transform_test_header(&header, 2u, 2u);
    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(yvex_transform_builder_create(
                         &builder, &header, NULL, &failure, &err) == YVEX_OK &&
                         transform_test_add_source(
                             builder, "duplicate.terminal.source.0", 0u,
                             YVEX_NATIVE_DTYPE_F32, YVEX_TRANSFORM_DTYPE_F32,
                             shape, YVEX_TRANSFORM_IR_NO_ID, &sources[0],
                             &failure, &err) == YVEX_OK &&
                         transform_test_add_source(
                             builder, "duplicate.terminal.source.1", 1u,
                             YVEX_NATIVE_DTYPE_F32, YVEX_TRANSFORM_DTYPE_F32,
                             shape, YVEX_TRANSFORM_IR_NO_ID, &sources[1],
                             &failure, &err) == YVEX_OK,
                     "duplicate-terminal fixture sources register");
    transform_test_terminal_spec(
        &value, 0u, YVEX_TENSOR_ROLE_ATTENTION_Q_A, shape,
        YVEX_TRANSFORM_DTYPE_F32, precision);
    YVEX_TEST_ASSERT(yvex_transform_builder_declare_value(
                         builder, &value, &values[0], &failure, &err) ==
                         YVEX_OK,
                     "first terminal key declares");
    transform_test_terminal_spec(
        &value, 1u, YVEX_TENSOR_ROLE_ATTENTION_Q_A, shape,
        YVEX_TRANSFORM_DTYPE_F32, precision);
    value.logical_key.group_index = 0u;
    YVEX_TEST_ASSERT(yvex_transform_builder_declare_value(
                         builder, &value, &values[1], &failure, &err) ==
                         YVEX_OK,
                     "duplicate logical terminal key registers before seal");
    memset(&node, 0, sizeof(node));
    node.kind = YVEX_TRANSFORM_OP_IDENTITY;
    node.input_count = 1u;
    node.numeric = YVEX_TRANSFORM_NUMERIC_EXACT;
    node.ordering = YVEX_TRANSFORM_ORDER_INPUT;
    node.payload_execution_required = 1;
    node.output_value_id = values[0];
    node.input_value_ids = &sources[0];
    YVEX_TEST_ASSERT(yvex_transform_builder_add_node(
                         builder, &node, &node_id, &failure, &err) == YVEX_OK,
                     "first duplicate-key producer registers");
    node.output_value_id = values[1];
    node.input_value_ids = &sources[1];
    YVEX_TEST_ASSERT(yvex_transform_builder_add_node(
                         builder, &node, &node_id, &failure, &err) == YVEX_OK &&
                         yvex_transform_builder_seal(
                             builder, &ir, &failure, &err) != YVEX_OK && !ir &&
                         failure.code ==
                             YVEX_TRANSFORM_FAILURE_DUPLICATE_TERMINAL,
                     "duplicate logical terminal ownership refuses sealing");
    yvex_transform_builder_release(&builder);

    transform_test_header(&header, 1u, 1u);
    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(yvex_transform_builder_create(
                         &builder, &header, NULL, &failure, &err) == YVEX_OK &&
                         transform_test_add_source(
                             builder, "unsupported.contract.source", 0u,
                             YVEX_NATIVE_DTYPE_F32, YVEX_TRANSFORM_DTYPE_F32,
                             shape, YVEX_TRANSFORM_IR_NO_ID, &sources[0],
                             &failure, &err) == YVEX_OK,
                     "unsupported-operation contract fixture opens");
    memset(&node, 0, sizeof(node));
    node.kind = YVEX_TRANSFORM_OP_IDENTITY;
    node.input_value_ids = &sources[0];
    node.input_count = 1u;
    node.numeric = YVEX_TRANSFORM_NUMERIC_EXACT;
    node.ordering = YVEX_TRANSFORM_ORDER_INPUT;
    node.payload_execution_required = 0;
    YVEX_TEST_ASSERT(transform_test_add_terminal(
                         builder, 0u, YVEX_TENSOR_ROLE_ATTENTION_Q_A, shape,
                         YVEX_TRANSFORM_DTYPE_F32, precision, &node,
                         &values[0], &failure, &err) == YVEX_OK &&
                         yvex_transform_builder_seal(
                             builder, &ir, &failure, &err) != YVEX_OK && !ir &&
                         failure.code ==
                             YVEX_TRANSFORM_FAILURE_UNSUPPORTED_OPERATION,
                     "an operation with an unsupported execution contract refuses");
    yvex_transform_builder_release(&builder);
    return 0;
}

static yvex_transform_failure_code transform_test_expert_failure(
    unsigned long long second_expert,
    unsigned long long declared_experts)
{
    yvex_transform_header header;
    yvex_transform_builder *builder = NULL;
    yvex_transform_ir *ir = NULL;
    yvex_transform_failure failure;
    yvex_transform_node_spec node;
    yvex_transform_precision_constraint precision = transform_test_paired();
    yvex_transform_shape weight = transform_test_shape(4u, 16u);
    yvex_transform_shape scale = transform_test_shape(4u, 1u);
    yvex_transform_shape output;
    yvex_error err;
    unsigned long long inputs[4];
    unsigned long long terminal;
    int rc;

    transform_test_header(&header, 4u, 1u);
    yvex_error_clear(&err);
    rc = yvex_transform_builder_create(
        &builder, &header, NULL, &failure, &err);
    if (rc == YVEX_OK)
        rc = transform_test_add_source(
            builder, "expert.0.weight", 0u, YVEX_NATIVE_DTYPE_I8,
            YVEX_TRANSFORM_DTYPE_PACKED_FP4, weight, 0u, &inputs[0],
            &failure, &err);
    if (rc == YVEX_OK)
        rc = transform_test_add_source(
            builder, "expert.0.scale", 1u, YVEX_NATIVE_DTYPE_F8_E8M0,
            YVEX_TRANSFORM_DTYPE_E8M0_SCALE, scale, 0u, &inputs[1],
            &failure, &err);
    if (rc == YVEX_OK)
        rc = transform_test_add_source(
            builder, "expert.1.weight", 2u, YVEX_NATIVE_DTYPE_I8,
            YVEX_TRANSFORM_DTYPE_PACKED_FP4, weight, second_expert,
            &inputs[2], &failure, &err);
    if (rc == YVEX_OK)
        rc = transform_test_add_source(
            builder, "expert.1.scale", 3u, YVEX_NATIVE_DTYPE_F8_E8M0,
            YVEX_TRANSFORM_DTYPE_E8M0_SCALE, scale, second_expert,
            &inputs[3], &failure, &err);
    memset(&node, 0, sizeof(node));
    node.kind = YVEX_TRANSFORM_OP_EXPERT_AGGREGATE;
    node.input_value_ids = inputs;
    node.input_count = 4u;
    node.axis = 0u;
    node.expert_count = declared_experts;
    node.packing_factor = 2u;
    node.scale_group_width = 32u;
    node.numeric = YVEX_TRANSFORM_NUMERIC_LOSSLESS;
    node.ordering = YVEX_TRANSFORM_ORDER_EXPERT_INDEX;
    node.payload_execution_required = 1;
    memset(&output, 0, sizeof(output));
    output.rank = 3u;
    output.dims[0] = declared_experts;
    output.dims[1] = 4u;
    output.dims[2] = 32u;
    if (rc == YVEX_OK)
        rc = transform_test_add_terminal(
            builder, 0u, YVEX_TENSOR_ROLE_MOE_EXPERT_GATE, output,
            YVEX_TRANSFORM_DTYPE_REAL, precision, &node, &terminal,
            &failure, &err);
    if (rc == YVEX_OK)
        rc = yvex_transform_builder_seal(builder, &ir, &failure, &err);
    yvex_transform_ir_release(&ir);
    yvex_transform_builder_release(&builder);
    return rc == YVEX_OK ? YVEX_TRANSFORM_FAILURE_NONE : failure.code;
}

/* Covers duplicate ownership, producer uniqueness, cycles, and post-seal refusal. */
static int test_transform_graph_invariants(void)
{
    yvex_transform_header header;
    yvex_transform_builder *builder = NULL;
    yvex_transform_ir *ir = NULL;
    yvex_transform_failure failure;
    yvex_transform_node_spec node;
    yvex_transform_value_spec value;
    yvex_transform_precision_constraint precision =
        transform_test_exact(YVEX_TRANSFORM_PHYSICAL_F32);
    yvex_transform_shape shape = transform_test_shape(2u, 2u);
    yvex_error err;
    unsigned long long sources[2];
    unsigned long long values[2];
    unsigned long long inputs[2];
    unsigned long long node_id;
    transform_test_header(&header, 2u, 1u);
    YVEX_TEST_ASSERT(yvex_transform_builder_create(
                         &builder, &header, NULL, &failure, &err) == YVEX_OK &&
                         transform_test_add_source(
                             builder, "duplicate.source", 0u,
                             YVEX_NATIVE_DTYPE_F32, YVEX_TRANSFORM_DTYPE_F32,
                             shape, YVEX_TRANSFORM_IR_NO_ID, &sources[0],
                             &failure, &err) == YVEX_OK &&
                         transform_test_add_source(
                             builder, "duplicate.source", 1u,
                             YVEX_NATIVE_DTYPE_F32, YVEX_TRANSFORM_DTYPE_F32,
                             shape, YVEX_TRANSFORM_IR_NO_ID, &sources[1],
                             &failure, &err) == YVEX_OK,
                     "duplicate source fixture registers before seal");
    memset(&node, 0, sizeof(node));
    node.kind = YVEX_TRANSFORM_OP_AGGREGATE;
    node.input_value_ids = sources;
    node.input_count = 2u;
    node.numeric = YVEX_TRANSFORM_NUMERIC_EXACT;
    node.ordering = YVEX_TRANSFORM_ORDER_INPUT;
    node.payload_execution_required = 1;
    YVEX_TEST_ASSERT(transform_test_add_terminal(
                         builder, 0u, YVEX_TENSOR_ROLE_ATTENTION_Q_A, shape,
                         YVEX_TRANSFORM_DTYPE_F32, precision, &node, &values[0],
                         &failure, &err) == YVEX_OK &&
                         yvex_transform_builder_seal(
                             builder, &ir, &failure, &err) != YVEX_OK && !ir &&
                         failure.code ==
                             YVEX_TRANSFORM_FAILURE_DUPLICATE_SOURCE,
                     "duplicate source ownership refuses sealing");
    yvex_transform_builder_release(&builder);

    transform_test_header(&header, 2u, 1u);
    YVEX_TEST_ASSERT(yvex_transform_builder_create(
                         &builder, &header, NULL, &failure, &err) == YVEX_OK &&
                         transform_test_add_source(
                             builder, "producer.source.0", 0u,
                             YVEX_NATIVE_DTYPE_F32, YVEX_TRANSFORM_DTYPE_F32,
                             shape, YVEX_TRANSFORM_IR_NO_ID, &sources[0],
                             &failure, &err) == YVEX_OK &&
                         transform_test_add_source(
                             builder, "producer.source.1", 1u,
                             YVEX_NATIVE_DTYPE_F32, YVEX_TRANSFORM_DTYPE_F32,
                             shape, YVEX_TRANSFORM_IR_NO_ID, &sources[1],
                             &failure, &err) == YVEX_OK,
                     "multiple-producer fixture sources register");
    memset(&value, 0, sizeof(value));
    value.kind = YVEX_TRANSFORM_VALUE_TERMINAL;
    value.semantic_id = 1u;
    value.canonical_ordinal = 0u;
    value.shape = shape;
    value.dtype = YVEX_TRANSFORM_DTYPE_F32;
    value.precision = precision;
    value.logical_key.scope = YVEX_TRANSFORM_SCOPE_MAIN_LAYER;
    value.logical_key.subsystem = YVEX_TRANSFORM_SUBSYSTEM_ATTENTION;
    value.logical_key.role = YVEX_TENSOR_ROLE_ATTENTION_Q_A;
    value.logical_key.layer_index = 0u;
    value.logical_key.auxiliary_index = YVEX_TRANSFORM_IR_NO_ID;
    YVEX_TEST_ASSERT(yvex_transform_builder_declare_value(
                         builder, &value, &values[0], &failure, &err) == YVEX_OK,
                     "multiple-producer terminal declares once");
    memset(&node, 0, sizeof(node));
    node.kind = YVEX_TRANSFORM_OP_IDENTITY;
    node.output_value_id = values[0];
    node.input_value_ids = &sources[0];
    node.input_count = 1u;
    node.numeric = YVEX_TRANSFORM_NUMERIC_EXACT;
    node.ordering = YVEX_TRANSFORM_ORDER_INPUT;
    node.payload_execution_required = 1;
    YVEX_TEST_ASSERT(yvex_transform_builder_add_node(
                         builder, &node, &node_id, &failure, &err) == YVEX_OK,
                     "first terminal producer registers");
    node.input_value_ids = &sources[1];
    YVEX_TEST_ASSERT(yvex_transform_builder_add_node(
                         builder, &node, &node_id, &failure, &err) == YVEX_OK &&
                         yvex_transform_builder_seal(
                             builder, &ir, &failure, &err) != YVEX_OK && !ir &&
                         failure.code ==
                             YVEX_TRANSFORM_FAILURE_MULTIPLE_PRODUCERS,
                     "second producer for one value refuses sealing");
    yvex_transform_builder_release(&builder);

    transform_test_header(&header, 2u, 1u);
    YVEX_TEST_ASSERT(yvex_transform_builder_create(
                         &builder, &header, NULL, &failure, &err) == YVEX_OK &&
                         transform_test_add_source(
                             builder, "cycle.source.0", 0u,
                             YVEX_NATIVE_DTYPE_F32, YVEX_TRANSFORM_DTYPE_F32,
                             shape, YVEX_TRANSFORM_IR_NO_ID, &sources[0],
                             &failure, &err) == YVEX_OK &&
                         transform_test_add_source(
                             builder, "cycle.source.1", 1u,
                             YVEX_NATIVE_DTYPE_F32, YVEX_TRANSFORM_DTYPE_F32,
                             shape, YVEX_TRANSFORM_IR_NO_ID, &sources[1],
                             &failure, &err) == YVEX_OK,
                     "cycle fixture sources register");
    memset(&value, 0, sizeof(value));
    value.kind = YVEX_TRANSFORM_VALUE_INTERMEDIATE;
    value.semantic_id = 11u;
    value.canonical_ordinal = YVEX_TRANSFORM_IR_NO_ID;
    value.shape = shape;
    value.dtype = YVEX_TRANSFORM_DTYPE_F32;
    value.precision = precision;
    YVEX_TEST_ASSERT(yvex_transform_builder_declare_value(
                         builder, &value, &values[0], &failure, &err) == YVEX_OK,
                     "cycle intermediate declares");
    value.kind = YVEX_TRANSFORM_VALUE_TERMINAL;
    value.semantic_id = 12u;
    value.canonical_ordinal = 0u;
    value.logical_key.scope = YVEX_TRANSFORM_SCOPE_MAIN_LAYER;
    value.logical_key.subsystem = YVEX_TRANSFORM_SUBSYSTEM_ATTENTION;
    value.logical_key.role = YVEX_TENSOR_ROLE_ATTENTION_Q_A;
    value.logical_key.layer_index = 0u;
    value.logical_key.auxiliary_index = YVEX_TRANSFORM_IR_NO_ID;
    YVEX_TEST_ASSERT(yvex_transform_builder_declare_value(
                         builder, &value, &values[1], &failure, &err) == YVEX_OK,
                     "cycle terminal declares");
    inputs[0] = sources[0];
    inputs[1] = values[1];
    memset(&node, 0, sizeof(node));
    node.kind = YVEX_TRANSFORM_OP_AGGREGATE;
    node.output_value_id = values[0];
    node.input_value_ids = inputs;
    node.input_count = 2u;
    node.numeric = YVEX_TRANSFORM_NUMERIC_EXACT;
    node.ordering = YVEX_TRANSFORM_ORDER_INPUT;
    node.payload_execution_required = 1;
    YVEX_TEST_ASSERT(yvex_transform_builder_add_node(
                         builder, &node, &node_id, &failure, &err) == YVEX_OK,
                     "cycle first edge registers");
    inputs[0] = sources[1];
    inputs[1] = values[0];
    node.output_value_id = values[1];
    YVEX_TEST_ASSERT(yvex_transform_builder_add_node(
                         builder, &node, &node_id, &failure, &err) == YVEX_OK &&
                         yvex_transform_builder_seal(
                             builder, &ir, &failure, &err) != YVEX_OK && !ir &&
                         failure.code == YVEX_TRANSFORM_FAILURE_CYCLE,
                     "iterative topology validation refuses a cycle");
    yvex_transform_builder_release(&builder);

    transform_test_header(&header, 1u, 1u);
    YVEX_TEST_ASSERT(yvex_transform_builder_create(
                         &builder, &header, NULL, &failure, &err) == YVEX_OK &&
                         transform_test_add_source(
                             builder, "sealed.source", 0u,
                             YVEX_NATIVE_DTYPE_F32, YVEX_TRANSFORM_DTYPE_F32,
                             shape, YVEX_TRANSFORM_IR_NO_ID, &sources[0],
                             &failure, &err) == YVEX_OK,
                     "post-seal fixture opens");
    memset(&node, 0, sizeof(node));
    node.kind = YVEX_TRANSFORM_OP_IDENTITY;
    node.input_value_ids = &sources[0];
    node.input_count = 1u;
    node.numeric = YVEX_TRANSFORM_NUMERIC_EXACT;
    node.ordering = YVEX_TRANSFORM_ORDER_INPUT;
    node.payload_execution_required = 1;
    YVEX_TEST_ASSERT(transform_test_add_terminal(
                         builder, 0u, YVEX_TENSOR_ROLE_ATTENTION_Q_A, shape,
                         YVEX_TRANSFORM_DTYPE_F32, precision, &node, &values[0],
                         &failure, &err) == YVEX_OK &&
                         yvex_transform_builder_seal(
                             builder, &ir, &failure, &err) == YVEX_OK && ir,
                     "post-seal fixture seals");
    YVEX_TEST_ASSERT(transform_test_add_source(
                         builder, "late.source", 1u, YVEX_NATIVE_DTYPE_F32,
                         YVEX_TRANSFORM_DTYPE_F32, shape,
                         YVEX_TRANSFORM_IR_NO_ID, &sources[1], &failure,
                         &err) != YVEX_OK &&
                         failure.code == YVEX_TRANSFORM_FAILURE_INVALID_STATE,
                     "sealed builder refuses mutation");
    yvex_transform_ir_release(&ir);
    yvex_transform_builder_release(&builder);

    YVEX_TEST_ASSERT(transform_test_expert_failure(0u, 2u) ==
                         YVEX_TRANSFORM_FAILURE_DUPLICATE_EXPERT,
                     "expert aggregate refuses duplicate expert index");
    YVEX_TEST_ASSERT(transform_test_expert_failure(2u, 2u) ==
                         YVEX_TRANSFORM_FAILURE_MISSING_EXPERT,
                     "expert aggregate refuses a missing expert index");
    YVEX_TEST_ASSERT(transform_test_expert_failure(1u, 3u) ==
                         YVEX_TRANSFORM_FAILURE_INVALID_AGGREGATION,
                     "expert aggregate refuses wrong cardinality");
    return 0;
}

static int test_transform_allocation_sweep(void)
{
    unsigned int fail_at;
    unsigned int failures = 0u;
    unsigned int successes = 0u;

    for (fail_at = 0u; fail_at < 48u; ++fail_at) {
        transform_test_allocator state;
        yvex_transform_builder_options options;
        yvex_transform_ir *ir = NULL;
        yvex_transform_failure failure;
        int rc;

        memset(&state, 0, sizeof(state));
        state.fail_at = fail_at;
        memset(&options, 0, sizeof(options));
        options.allocator.allocate = transform_test_allocate;
        options.allocator.release = transform_test_release;
        options.allocator.context = &state;
        yvex_transform_budget_default(&options.budget);
        rc = transform_test_build_direct(
            0, 8u, &options, &ir, &failure);
        if (rc == YVEX_OK) {
            successes++;
            yvex_transform_ir_release(&ir);
        } else {
            failures++;
            YVEX_TEST_ASSERT(!ir &&
                                 failure.code ==
                                     YVEX_TRANSFORM_FAILURE_ALLOCATION,
                             "allocation seam returns typed refusal");
        }
        YVEX_TEST_ASSERT(state.live == 0u,
                         "every allocation transition unwinds completely");
    }
    YVEX_TEST_ASSERT(failures > 20u && successes > 0u,
                     "allocation sweep covers build, seal, identity, and release");
    return 0;
}

int yvex_test_transform_ir(void)
{
    if (test_transform_identity_and_lifecycle() != 0) return 1;
    if (test_transform_operation_suite() != 0) return 1;
    if (test_transform_negative_graphs() != 0) return 1;
    if (test_transform_operation_refusals() != 0) return 1;
    if (test_transform_ownership_refusals() != 0) return 1;
    if (test_transform_graph_invariants() != 0) return 1;
    if (test_transform_allocation_sweep() != 0) return 1;
    return 0;
}
