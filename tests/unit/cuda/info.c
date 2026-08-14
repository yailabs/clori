/*
 * Exercises the CUDA backend CUDA device probe opens a real CUDA backend when the local
 * driver/device is available. Returns 77 when CUDA is unavailable.
 */
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <yvex/api.h>
#include <yvex/qtype.h>
#include <yvex/internal/backend.h>
#include <yvex/internal/graph.h>
#include <yvex/internal/moe.h>
#include <yvex/internal/quant_numeric.h>

#include "src/backend/cuda/private.h"
#include "tests/test.h"

static int assert_supported_variant(const yvex_backend *backend,
                                    yvex_backend_operation_variant variant)
{
    yvex_backend_capability_result result;
    yvex_error err;
    int rc = yvex_backend_query_capability(backend, variant, &result, &err);

    YVEX_TEST_ASSERT(rc == YVEX_OK, "query exact CUDA capability");
    YVEX_TEST_ASSERT(result.state == YVEX_BACKEND_CAPABILITY_SUPPORTED,
                     "exact CUDA variant supported");
    YVEX_TEST_ASSERT(result.reason == YVEX_BACKEND_CAPABILITY_REASON_NONE,
                     "supported CUDA variant has no refusal reason");
    YVEX_TEST_ASSERT(result.context_available, "CUDA variant has context");
    if (variant >= YVEX_BACKEND_VARIANT_EMBED_F32_TO_F32) {
        YVEX_TEST_ASSERT(result.kernel_bundle_available,
                         "kernel CUDA variant has admitted generated bundle");
        YVEX_TEST_ASSERT(result.function_available,
                         "kernel CUDA variant has resolved function");
    }
    return 0;
}

/* Contract: proves atomic bundle rejection, cleared handles, and clean retry. */
static int assert_bundle_rollback(const char *failure,
                                  yvex_backend_capability_reason expected_reason,
                                  yvex_backend_operation_variant variant)
{
    yvex_backend *backend = NULL;
    yvex_backend_options options;
    yvex_backend_capability_result result;
    yvex_error err;
    int rc;

    memset(&options, 0, sizeof(options));
    options.kind = YVEX_BACKEND_KIND_CUDA;
    YVEX_TEST_ASSERT(setenv("YVEX_TEST_CUDA_BUNDLE_FAILURE", failure, 1) == 0,
                     "set bundle failure injection");
    rc = yvex_backend_open(&backend, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "context survives rejected kernel bundle");
    YVEX_TEST_ASSERT(yvex_backend_status_of(backend) == YVEX_BACKEND_STATUS_CONTEXT_READY,
                     "rejected bundle leaves context-only status");
    rc = yvex_backend_query_capability(backend, YVEX_BACKEND_VARIANT_TENSOR_ALLOC,
                                       &result, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK &&
                     result.state == YVEX_BACKEND_CAPABILITY_SUPPORTED,
                     "Driver memory capability survives bundle rejection");
    rc = yvex_backend_query_capability(backend, variant, &result, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "query rejected kernel capability");
    YVEX_TEST_ASSERT(result.state == YVEX_BACKEND_CAPABILITY_FAILED,
                     "rejected bundle cannot support kernel variant");
    YVEX_TEST_ASSERT(result.reason == expected_reason,
                     "rejected bundle exposes typed reason");
    YVEX_TEST_ASSERT(!result.kernel_bundle_available && !result.function_available,
                     "rejected bundle leaves no admitted handle");
    yvex_backend_close(backend);
    YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_CUDA_BUNDLE_FAILURE") == 0,
                     "clear bundle failure injection");
    return 0;
}

typedef struct {
    float *arena;
    unsigned long long used, device_base;
} moe_fixture_weights;

static void moe_fixture_weight(moe_fixture_weights *fixture,
                               yvex_moe_weight_view *view, yvex_tensor_role role,
                               unsigned long long rows, unsigned long long width,
                               const float *values)
{
    unsigned long long count = rows * width;
    float *destination = fixture->arena + fixture->used;
    memcpy(destination, values, (size_t)count * sizeof(*values));
    memset(view, 0, sizeof(*view));
    view->tensor_id = fixture->used + 1ull;
    view->expert_index = YVEX_MOE_NO_TENSOR;
    view->role = role;
    view->qtype = YVEX_GGUF_QTYPE_F32;
    view->encoded = (const unsigned char *)destination;
    view->encoded_bytes = (size_t)count * sizeof(*values);
    view->row_bytes = width * sizeof(*values);
    view->row_width = width;
    view->row_count = rows;
    view->device_address = fixture->device_base + fixture->used * sizeof(*values);
    fixture->used += count;
}

static yvex_moe_weight_view moe_fixture_expert(const yvex_moe_weight_view *aggregate,
                                               unsigned long long expert,
                                               unsigned long long rows)
{
    yvex_moe_weight_view view = *aggregate;
    unsigned long long bytes = rows * aggregate->row_bytes;
    view.expert_index = expert;
    view.encoded += expert * bytes;
    view.device_address += expert * bytes;
    view.encoded_bytes = (size_t)bytes;
    view.row_count = rows;
    return view;
}

static void moe_fixture_result(yvex_moe_layer_result *result, float combined[2],
                               float routed[2], float shared[2], float post[1],
                               float combination[1])
{
    memset(result, 0, sizeof(*result));
    result->combined_output = combined;
    result->combined_capacity = 2ull;
    result->routed_output = routed;
    result->routed_capacity = 2ull;
    result->shared_output = shared;
    result->shared_capacity = 2ull;
    result->post = post;
    result->post_capacity = 1ull;
    result->combination = combination;
    result->combination_capacity = 1ull;
}

typedef struct {
    unsigned char *arena;
    unsigned long long used, capacity, device_base;
} moe_encoded_fixture;

static int moe_encoded_reserve(moe_encoded_fixture *fixture,
                               unsigned long long bytes,
                               unsigned long long *offset)
{
    unsigned long long aligned;
    if (!fixture || !bytes || !offset ||
        !yvex_core_u64_add(fixture->used, 15ull, &aligned)) return 0;
    aligned &= ~15ull;
    if (aligned > fixture->capacity || bytes > fixture->capacity - aligned) return 0;
    *offset = aligned;
    fixture->used = aligned + bytes;
    return 1;
}

static int moe_encoded_f32_weight(
    moe_encoded_fixture *fixture, yvex_moe_weight_view *view,
    yvex_tensor_role role, unsigned long long rows,
    unsigned long long width, const float *values)
{
    unsigned long long elements, bytes, offset;
    if (!fixture || !view || !rows || !width || !values ||
        !yvex_core_u64_mul(rows, width, &elements) ||
        !yvex_core_u64_mul(elements, sizeof(*values), &bytes) ||
        bytes > SIZE_MAX || !moe_encoded_reserve(fixture, bytes, &offset)) return 0;
    memcpy(fixture->arena + offset, values, (size_t)bytes);
    memset(view, 0, sizeof(*view));
    view->tensor_id = offset + 1ull;
    view->expert_index = YVEX_MOE_NO_TENSOR;
    view->role = role;
    view->qtype = YVEX_GGUF_QTYPE_F32;
    view->activation = YVEX_EXECUTION_ACTIVATION_DEVICE_F32;
    view->encoded = fixture->arena + offset;
    view->encoded_bytes = (size_t)bytes;
    view->row_bytes = width * sizeof(*values);
    view->row_width = width;
    view->row_count = rows;
    view->device_address = fixture->device_base + offset;
    return 1;
}

static int moe_encoded_weight(
    moe_encoded_fixture *fixture, yvex_moe_weight_view *view,
    yvex_tensor_role role, yvex_gguf_qtype_id qtype, unsigned long long rows,
    unsigned long long width, const float *row_values)
{
    const yvex_gguf_qtype_geometry *geometry =
        yvex_gguf_qtype_geometry_find(qtype);
    yvex_quant_failure failure;
    yvex_error err;
    float calibration[YVEX_QUANT_IQ2_XXS_ELEMENTS];
    unsigned long long row_bytes, bytes, offset, row, block;
    if (!fixture || !view || !geometry || !rows || !width || !row_values ||
        width % geometry->block_size ||
        !yvex_core_u64_mul(width / geometry->block_size,
                           geometry->bytes_per_block, &row_bytes) ||
        !yvex_core_u64_mul(rows, row_bytes, &bytes) ||
        bytes > SIZE_MAX || !moe_encoded_reserve(fixture, bytes, &offset)) return 0;
    for (block = 0ull; block < YVEX_QUANT_IQ2_XXS_ELEMENTS; ++block)
        calibration[block] = 1.0f;
    for (row = 0ull; row < rows; ++row)
        for (block = 0ull; block < width / geometry->block_size; ++block) {
            size_t wrote = 0u;
            const float *source = row_values + block * geometry->block_size;
            unsigned char *destination = fixture->arena + offset + row * row_bytes +
                                         block * geometry->bytes_per_block;
            int rc = qtype == YVEX_GGUF_QTYPE_IQ2_XXS
                         ? yvex_quant_encode_block_weighted(
                               qtype, source, calibration, geometry->block_size, destination,
                               geometry->bytes_per_block, &wrote, &failure, &err)
                         : yvex_quant_encode_block(qtype, source, geometry->block_size,
                                                   destination, geometry->bytes_per_block,
                                                   &wrote, &failure, &err);
            if (rc != YVEX_OK ||
                wrote != geometry->bytes_per_block) return 0;
        }
    memset(view, 0, sizeof(*view));
    view->tensor_id = offset + 1ull;
    view->expert_index = YVEX_MOE_NO_TENSOR;
    view->role = role;
    view->qtype = qtype;
    view->activation = YVEX_EXECUTION_ACTIVATION_DEVICE_F32;
    view->encoded = fixture->arena + offset;
    view->encoded_bytes = (size_t)bytes;
    view->row_bytes = row_bytes;
    view->row_width = width;
    view->row_count = rows;
    view->device_address = fixture->device_base + offset;
    return 1;
}

static int moe_derived_weight(
    moe_encoded_fixture *fixture, const yvex_backend_moe_operations *operations,
    yvex_moe_weight_view *view, unsigned long long experts,
    yvex_execution_consumer_class consumer, yvex_error *err)
{
    yvex_physical_execution_decision decision = {0};
    unsigned long long storage = 0ull, offset;
    if (!fixture || !operations || !operations->derived_layout_plan ||
        !operations->derived_layout_build || !view || !experts ||
        view->row_count % experts)
        return 0;
    decision.schema_version = YVEX_PHYSICAL_EXECUTION_SCHEMA_V2;
    decision.terminal_tensor_id = view->tensor_id;
    decision.role = view->role;
    decision.expert_count = experts;
    decision.canonical_qtype = view->qtype;
    decision.canonical_row_width = view->row_width;
    decision.canonical_row_count = view->row_count;
    decision.encoded_bytes = view->encoded_bytes;
    decision.consumer = consumer;
    decision.layout = YVEX_EXECUTION_LAYOUT_DERIVED_BACKEND;
    decision.derived_asset_required = 1;
    if (operations->derived_layout_plan(&decision, &storage, err) != YVEX_OK ||
        !yvex_core_u64_add(fixture->used, 63ull, &offset))
        return 0;
    offset &= ~63ull;
    if (offset > fixture->capacity || storage > fixture->capacity - offset ||
        operations->derived_layout_build(
            &decision, view->encoded, view->encoded_bytes,
            fixture->arena + offset, storage, err) != YVEX_OK)
        return 0;
    fixture->used = offset + storage;
    view->encoded = fixture->arena + offset;
    view->storage_bytes = (size_t)storage;
    view->layout = YVEX_EXECUTION_LAYOUT_DERIVED_BACKEND;
    view->device_address = fixture->device_base + offset;
    return 1;
}

static void moe_encoded_output(
    yvex_moe_row_batch_output *output, float *combined,
    float *routed, float *shared, float *post, float *combination,
    unsigned long long *selected, float *weights,
    unsigned long long rows, unsigned long long hidden,
    unsigned long long streams, unsigned long long pairs)
{
    memset(output, 0, sizeof(*output));
    output->combined_rows = combined;
    output->combined_capacity = rows * hidden;
    output->routed_rows = routed;
    output->routed_capacity = rows * hidden;
    output->shared_rows = shared;
    output->shared_capacity = rows * hidden;
    output->post_rows = post;
    output->post_capacity = rows * streams;
    output->combination_rows = combination;
    output->combination_capacity = rows * streams * streams;
    output->selected_experts = selected;
    output->selected_weights = weights;
    output->selection_capacity = pairs;
}

static int assert_encoded_moe(yvex_backend *backend)
{
    enum { ROWS = 192, WIDTH = 512, EXPERTS = 256, TOPK = 6, PAIRS = ROWS * TOPK };
    yvex_backend_tensor_desc descriptor = {0};
    unsigned char *workspace_poison = NULL;
    yvex_device_tensor *anchor = NULL, *input = NULL, *small_input = NULL;
    yvex_device_tensor *reference_output = NULL, *encoded_output = NULL;
    yvex_device_tensor *small_output = NULL, *one_input = NULL, *one_output = NULL;
    yvex_device_tensor *workspace = NULL;
    yvex_backend_moe_execution *execution = NULL;
    yvex_moe_layer_plan layer = {0};
    yvex_moe_layer_job job = {0};
    yvex_moe_row_batch rows = {0};
    yvex_moe_row_batch_output output = {0};
    yvex_moe_row_batch_result result = {0};
    const yvex_backend_moe_operations *operations;
    moe_encoded_fixture fixture = {0};
    yvex_moe_weight_view canonical_routed[3];
    yvex_error err;
    float mhc[3 * WIDTH] = {0};
    float scale[3] = {1.0f, 1.0f, 1.0f};
    float base[3] = {0};
    float norm[WIDTH], router[EXPERTS * WIDTH], router_bias[EXPERTS] = {0}, expert_row[WIDTH];
    float input_rows[ROWS * WIDTH], reference[ROWS * WIDTH], encoded[ROWS * WIDTH];
    float tensorcore[ROWS * WIDTH];
    float combined[ROWS * WIDTH], routed[ROWS * WIDTH], shared[ROWS * WIDTH];
    float post[ROWS], combination[ROWS], selected_weights[PAIRS];
    float one_device[WIDTH];
    unsigned long long selected[PAIRS];
    unsigned int token_ids[ROWS] = {7u, 11u};
    unsigned long long address = 0ull, workspace_bytes = 0ull, slot, index, row;
    float maximum_error = 0.0f;
    int native, rc;
    operations = yvex_backend_moe_operations_get(backend);
    YVEX_TEST_ASSERT(operations, "obtain CUDA width-N MoE operations");
    for (index = 0ull; index < WIDTH; ++index) {
        norm[index] = 1.0f;
        expert_row[index] = ((float)((index * 11ull + 5ull) % 31ull) - 15.0f) / 32.0f;
        for (row = 0ull; row < ROWS; ++row)
            input_rows[row * WIDTH + index] = (row + index * 7ull) & 1ull ? -1.0f : 1.0f;
    }
    memset(router, 0, sizeof(router));
    router[0] = 1.0f;
    router[WIDTH] = -1.0f;
    descriptor.name = "encoded-moe-residency";
    descriptor.dtype = YVEX_DTYPE_I8;
    descriptor.rank = 1u;
    descriptor.dims[0] = descriptor.bytes = 128ull * 1024ull * 1024ull;
    YVEX_TEST_ASSERT(
        yvex_backend_resident_alloc(backend, &descriptor, &anchor,
                                    &fixture.arena, &err) == YVEX_OK,
        "allocate encoded MoE residency");
    fixture.capacity = descriptor.bytes;
    memset(fixture.arena, 0, (size_t)fixture.capacity);
    YVEX_TEST_ASSERT(
        yvex_backend_resident_attach(backend, fixture.arena, fixture.capacity,
                                     anchor, 19ull, &err) == YVEX_OK &&
            yvex_backend_resident_resolve(backend, fixture.arena, fixture.capacity,
                                          &address) == YVEX_BACKEND_RESIDENT_HIT,
        "attach encoded MoE residency once");
    fixture.device_base = address;
    layer.schema_version = YVEX_MOE_PLAN_SCHEMA_V1;
    layer.router_class = YVEX_MOE_ROUTER_LEARNED_HIDDEN_STATE;
    layer.scoring = YVEX_MOE_SCORING_SQRT_SOFTPLUS;
    layer.topk_policy = YVEX_MOE_TOPK_NOAUX_TC;
    layer.activation = YVEX_MOE_ACTIVATION_SILU;
    layer.hidden_width = layer.expanded_width = WIDTH;
    layer.residual_streams = 1ull;
    layer.mhc_mixing_rows = 3ull;
    layer.mhc_sinkhorn_iterations = 1ull;
    layer.routed_experts = EXPERTS;
    layer.shared_experts = 1ull;
    layer.experts_per_token = TOPK;
    layer.expert_intermediate_width = layer.shared_intermediate_width = WIDTH;
    layer.correction_bias_width = EXPERTS;
    layer.rms_epsilon = layer.mhc_epsilon = 0.00001;
    layer.mhc_post_multiplier = layer.routed_scaling_factor = 1.0;
    layer.activation_limit = 10.0;
    layer.requires_correction_bias = layer.normalize_topk_probabilities = 1;
    for (slot = 0ull; slot < YVEX_MOE_WEIGHT_COUNT; ++slot)
        layer.tensor_ids[slot] = YVEX_MOE_NO_TENSOR;
    job.layer = &layer;
    job.expanded_input = input_rows;
    job.token_id_present = 1;
    job.evidence_level = YVEX_ATTENTION_EVIDENCE_SUMMARY;
    YVEX_TEST_ASSERT(
        moe_encoded_f32_weight(&fixture, &job.weights[YVEX_MOE_WEIGHT_MHC_FUNCTION],
                                YVEX_TENSOR_ROLE_HC_FFN_FUNCTION, 3ull, WIDTH, mhc) &&
            moe_encoded_f32_weight(&fixture, &job.weights[YVEX_MOE_WEIGHT_MHC_SCALE],
                                    YVEX_TENSOR_ROLE_HC_FFN_SCALE, 1ull, 3ull, scale) &&
            moe_encoded_f32_weight(&fixture, &job.weights[YVEX_MOE_WEIGHT_MHC_BASE],
                                    YVEX_TENSOR_ROLE_HC_FFN_BASE, 1ull, 3ull, base) &&
            moe_encoded_f32_weight(&fixture, &job.weights[YVEX_MOE_WEIGHT_FFN_NORM],
                                    YVEX_TENSOR_ROLE_FFN_NORM, 1ull, WIDTH, norm) &&
            moe_encoded_f32_weight(&fixture, &job.weights[YVEX_MOE_WEIGHT_ROUTER],
                                    YVEX_TENSOR_ROLE_MOE_ROUTER, EXPERTS, WIDTH, router) &&
            moe_encoded_f32_weight(&fixture, &job.weights[YVEX_MOE_WEIGHT_ROUTER_BIAS],
                                    YVEX_TENSOR_ROLE_MOE_ROUTER_BIAS, 1ull, EXPERTS,
                                    router_bias),
        "encode auxiliary MoE weights");
    YVEX_TEST_ASSERT(
        moe_encoded_weight(&fixture, &job.weights[YVEX_MOE_WEIGHT_ROUTED_GATE],
                           YVEX_TENSOR_ROLE_MOE_EXPERT_GATE,
                           YVEX_GGUF_QTYPE_IQ2_XXS,
                           EXPERTS * WIDTH, WIDTH, expert_row) &&
            moe_encoded_weight(&fixture, &job.weights[YVEX_MOE_WEIGHT_ROUTED_UP],
                               YVEX_TENSOR_ROLE_MOE_EXPERT_UP,
                               YVEX_GGUF_QTYPE_IQ2_XXS,
                               EXPERTS * WIDTH, WIDTH, expert_row) &&
            moe_encoded_weight(&fixture, &job.weights[YVEX_MOE_WEIGHT_ROUTED_DOWN],
                               YVEX_TENSOR_ROLE_MOE_EXPERT_DOWN,
                               YVEX_GGUF_QTYPE_Q2_K,
                               EXPERTS * WIDTH, WIDTH, expert_row) &&
            moe_encoded_weight(&fixture, &job.weights[YVEX_MOE_WEIGHT_SHARED_GATE],
                               YVEX_TENSOR_ROLE_MOE_SHARED_EXPERT_GATE,
                               YVEX_GGUF_QTYPE_MXFP4,
                               WIDTH, WIDTH, expert_row) &&
            moe_encoded_weight(&fixture, &job.weights[YVEX_MOE_WEIGHT_SHARED_UP],
                               YVEX_TENSOR_ROLE_MOE_SHARED_EXPERT_UP,
                               YVEX_GGUF_QTYPE_MXFP4,
                               WIDTH, WIDTH, expert_row) &&
            moe_encoded_weight(&fixture, &job.weights[YVEX_MOE_WEIGHT_SHARED_DOWN],
                               YVEX_TENSOR_ROLE_MOE_SHARED_EXPERT_DOWN,
                               YVEX_GGUF_QTYPE_Q8_0,
                               WIDTH, WIDTH, expert_row),
        "encode mixed low-bit expert-major MoE weights");
    for (slot = 0ull; slot < YVEX_MOE_WEIGHT_COUNT; ++slot)
        if (job.weights[slot].device_address)
            layer.tensor_ids[slot] = job.weights[slot].tensor_id;
    YVEX_TEST_ASSERT(
        operations->workspace_required(&layer, ROWS, &workspace_bytes, &err) == YVEX_OK &&
            workspace_bytes != 0ull,
        "derive encoded MoE workspace");
#define ALLOCATE_ENCODED_TENSOR(owner_, name_, bytes_)                                     \
    do {                                                                                   \
        memset(&descriptor, 0, sizeof(descriptor));                                        \
        descriptor.name = (name_);                                                         \
        descriptor.dtype = YVEX_DTYPE_F32;                                                 \
        descriptor.rank = 1u;                                                              \
        descriptor.dims[0] = (bytes_) / sizeof(float);                                     \
        descriptor.bytes = (bytes_);                                                       \
        YVEX_TEST_ASSERT(                                                                  \
            yvex_backend_tensor_alloc(backend, &descriptor, &(owner_), &err) == YVEX_OK,  \
            "allocate encoded MoE device tensor");                                      \
    } while (0)
    ALLOCATE_ENCODED_TENSOR(input, "encoded-moe-input", sizeof(input_rows));
    ALLOCATE_ENCODED_TENSOR(small_input, "encoded-moe-small-input",
                            2ull * WIDTH * sizeof(float));
    ALLOCATE_ENCODED_TENSOR(reference_output, "encoded-moe-reference",
                            sizeof(reference));
    ALLOCATE_ENCODED_TENSOR(encoded_output, "encoded-moe-output", sizeof(encoded));
    ALLOCATE_ENCODED_TENSOR(small_output, "encoded-moe-small-output",
                            2ull * WIDTH * sizeof(float));
    ALLOCATE_ENCODED_TENSOR(one_input, "encoded-moe-one-input", WIDTH * sizeof(float));
    ALLOCATE_ENCODED_TENSOR(one_output, "encoded-moe-one-output", WIDTH * sizeof(float));
    ALLOCATE_ENCODED_TENSOR(workspace, "encoded-moe-workspace", workspace_bytes);
#undef ALLOCATE_ENCODED_TENSOR
    workspace_poison = malloc((size_t)workspace_bytes);
    YVEX_TEST_ASSERT(workspace_poison, "allocate encoded MoE workspace poison");
    memset(workspace_poison, 0xa5, (size_t)workspace_bytes);
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_write(backend, input, input_rows, sizeof(input_rows), &err) ==
                YVEX_OK &&
            yvex_backend_tensor_write(backend, small_input, input_rows,
                                      2ull * WIDTH * sizeof(float), &err) == YVEX_OK &&
            yvex_backend_tensor_write(backend, workspace, workspace_poison,
                                      workspace_bytes, &err) == YVEX_OK &&
            yvex_backend_workspace_attach(backend, workspace, 2ull, &err) == YVEX_OK,
        "prepare encoded MoE device state");
    free(workspace_poison);
    workspace_poison = NULL;
    rows.schema_version = YVEX_MOE_ROW_BATCH_SCHEMA_V1;
    rows.row_count = ROWS;
    rows.row_width = rows.row_stride = WIDTH;
    rows.expanded_rows = input_rows;
    rows.device_rows = input;
    rows.device_outputs = reference_output;
    rows.token_ids = token_ids;
    rows.token_ids_present = 1;
    rows.execution_class = YVEX_EXECUTION_CLASS_DEVICE_NATIVE;
    moe_encoded_output(&output, combined, routed, shared, post, combination,
                       selected, selected_weights, ROWS, WIDTH, 1ull, PAIRS);
    rc = operations->execute_rows(backend, &job, &rows, &output, &result, &err);
    YVEX_TEST_ASSERT(
        rc == YVEX_OK && result.schema_version == YVEX_MOE_ROW_BATCH_RESULT_SCHEMA_V3 &&
            result.tensor_core_launches == 0ull &&
            result.graph_launches == 1ull && result.graph_captures == 1ull &&
            result.graph_replays == 1ull &&
            yvex_backend_tensor_read(backend, reference_output, reference,
                                     sizeof(reference), &err) == YVEX_OK,
        "execute graph-compiled portable low-bit width-N MoE oracle");
    for (row = 0ull; row < ROWS; ++row) {
        yvex_moe_router_result cpu_router = {0};
        YVEX_TEST_ASSERT(yvex_moe_route_cpu(&job, input_rows + row * WIDTH,
                                            &cpu_router, &err) == YVEX_OK,
                         "execute independent CPU router oracle");
        for (index = 0ull; index < TOPK; ++index)
            YVEX_TEST_ASSERT(
                selected[row * TOPK + index] == cpu_router.selected_experts[index] &&
                    fabsf(selected_weights[row * TOPK + index] - cpu_router.selected_weights[index]) <=
                        1e-6f,
                "CUDA route selection and weights match the CPU oracle");
    }
    for (slot = YVEX_MOE_WEIGHT_ROUTED_GATE;
         slot <= YVEX_MOE_WEIGHT_SHARED_DOWN; ++slot) {
        job.weights[slot].activation = YVEX_EXECUTION_ACTIVATION_DEVICE_ENCODED;
        job.weights[slot].kernel_family = YVEX_MOE_KERNEL_PORTABLE_EXPERT_ROW;
    }
    native = yvex_cuda_state(backend)->kernel_bundle_native;
    rows.row_count = 2ull;
    rows.device_rows = small_input;
    rows.device_outputs = small_output;
    memset(&result, 0, sizeof(result));
    rc = operations->execute_rows(backend, &job, &rows, &output, &result, &err);
    if (native) {
        YVEX_TEST_ASSERT(
            rc == YVEX_OK && result.schema_version == YVEX_MOE_ROW_BATCH_RESULT_SCHEMA_V3 &&
                result.tensor_core_launches == 0ull &&
                yvex_device_tensor_is_written(small_output) &&
                yvex_backend_tensor_read(backend, small_output, encoded,
                                         2ull * WIDTH * sizeof(float), &err) == YVEX_OK,
            "execute source-faithful encoded DP4A MoE for a sparse row batch");
        for (index = 0ull; index < 2ull * WIDTH; ++index) {
            float difference = fabsf(reference[index] - encoded[index]);
            if (difference > maximum_error) maximum_error = difference;
        }
        YVEX_TEST_ASSERT(maximum_error <= 0.01f,
                         "native small-row encoded MoE matches portable oracle");
        rows.row_count = ROWS;
        rows.device_rows = input;
        rows.device_outputs = encoded_output;
        encoded_output->is_written = 0;
        memset(&result, 0, sizeof(result));
        rc = operations->execute_rows(backend, &job, &rows, &output, &result, &err);
        YVEX_TEST_ASSERT(
            rc == YVEX_OK && result.schema_version == YVEX_MOE_ROW_BATCH_RESULT_SCHEMA_V3 &&
                result.tensor_core_launches == 0ull &&
                yvex_device_tensor_is_written(encoded_output) &&
                yvex_backend_tensor_read(backend, encoded_output, encoded,
                                         sizeof(encoded), &err) == YVEX_OK,
            "preserve source-faithful encoded DP4A MoE across a complete row batch");
        maximum_error = 0.0f;
        for (index = 0ull; index < ROWS * WIDTH; ++index) {
            float difference = fabsf(reference[index] - encoded[index]);
            if (difference > maximum_error) maximum_error = difference;
        }
        YVEX_TEST_ASSERT(maximum_error <= 0.01f,
                         "native grouped encoded MoE matches portable oracle");
        for (slot = YVEX_MOE_WEIGHT_ROUTED_GATE;
             slot <= YVEX_MOE_WEIGHT_SHARED_DOWN; ++slot)
            job.weights[slot].kernel_family =
                YVEX_MOE_KERNEL_SM121_ROW_REGIME_EXPERT;
        rows.row_count = 2ull;
        rows.device_rows = small_input;
        rows.device_outputs = small_output;
        small_output->is_written = 0;
        memset(&result, 0, sizeof(result));
        rc = operations->execute_rows(backend, &job, &rows, &output, &result, &err);
        YVEX_TEST_ASSERT(
            rc == YVEX_OK && result.schema_version == YVEX_MOE_ROW_BATCH_RESULT_SCHEMA_V3 &&
                result.tensor_core_launches == 0ull &&
                yvex_device_tensor_is_written(small_output) &&
                yvex_backend_tensor_read(backend, small_output, tensorcore,
                                         2ull * WIDTH * sizeof(float), &err) == YVEX_OK,
            "row-regime capability selects encoded DP4A for sparse MoE");
        maximum_error = 0.0f;
        for (index = 0ull; index < 2ull * WIDTH; ++index) {
            float difference = fabsf(encoded[index] - tensorcore[index]);
            if (difference > maximum_error) maximum_error = difference;
        }
        YVEX_TEST_ASSERT(maximum_error <= 1e-6f,
                         "sparse row-regime MoE matches its encoded oracle");
        rows.row_count = ROWS;
        rows.device_rows = input;
        rows.device_outputs = reference_output;
        reference_output->is_written = 0;
        memset(&result, 0, sizeof(result));
        rc = operations->execute_rows(backend, &job, &rows, &output, &result, &err);
        YVEX_TEST_ASSERT(
            rc == YVEX_OK && result.schema_version == YVEX_MOE_ROW_BATCH_RESULT_SCHEMA_V3 &&
                result.tensor_core_launches == 0ull &&
                yvex_device_tensor_is_written(reference_output) &&
                yvex_backend_tensor_read(backend, reference_output, tensorcore,
                                         sizeof(tensorcore), &err) == YVEX_OK,
            "row-regime capability remains on its compiler-selected encoded path");
        maximum_error = 0.0f;
        for (index = 0ull; index < ROWS * WIDTH; ++index) {
            float difference = fabsf(encoded[index] - tensorcore[index]);
            if (difference > maximum_error) maximum_error = difference;
        }
        YVEX_TEST_ASSERT(maximum_error <= 1e-6f,
                         "grouped row-regime MoE matches its encoded oracle");
        for (slot = YVEX_MOE_WEIGHT_ROUTED_GATE;
             slot <= YVEX_MOE_WEIGHT_SHARED_DOWN; ++slot)
            job.weights[slot].kernel_family =
                YVEX_MOE_KERNEL_SM121_TENSORCORE_EXPERT;
        rows.row_count = 2ull;
        rows.device_rows = small_input;
        rows.device_outputs = small_output;
        small_output->is_written = 0;
        memset(&result, 0, sizeof(result));
        rc = operations->execute_rows(backend, &job, &rows, &output, &result, &err);
        YVEX_TEST_ASSERT(
            rc == YVEX_OK && result.schema_version == YVEX_MOE_ROW_BATCH_RESULT_SCHEMA_V3 &&
                result.tensor_core_launches == 4ull &&
                yvex_device_tensor_is_written(small_output),
            "execute compiler-selected sparse-row Tensor Core MoE");
        rows.row_count = ROWS;
        rows.device_rows = input;
        rows.device_outputs = reference_output;
        reference_output->is_written = 0;
        memset(&result, 0, sizeof(result));
        rc = operations->execute_rows(backend, &job, &rows, &output, &result, &err);
        YVEX_TEST_ASSERT(
            rc == YVEX_OK && result.schema_version == YVEX_MOE_ROW_BATCH_RESULT_SCHEMA_V3 &&
                result.tensor_core_launches == 4ull &&
                result.graph_launches == 1ull && result.graph_captures == 1ull &&
                result.graph_replays == 1ull &&
                yvex_device_tensor_is_written(reference_output) &&
                yvex_backend_tensor_read(backend, reference_output, tensorcore,
                                         sizeof(tensorcore), &err) == YVEX_OK,
            "execute compiler-selected native grouped Tensor Core MoE");
        maximum_error = 0.0f;
        for (index = 0ull; index < ROWS * WIDTH; ++index) {
            float difference = fabsf(encoded[index] - tensorcore[index]);
            if (difference > maximum_error) maximum_error = difference;
        }
        YVEX_TEST_ASSERT(maximum_error <= 1e-6f,
                         "Tensor Core MoE matches the encoded DP4A oracle");
        reference_output->is_written = 0;
        memset(&result, 0, sizeof(result));
        rc = operations->execute_rows(backend, &job, &rows, &output, &result, &err);
        YVEX_TEST_ASSERT(
            rc == YVEX_OK && result.graph_launches == 1ull &&
                result.graph_captures == 0ull && result.graph_replays == 1ull &&
                result.tensor_core_launches == 4ull &&
                yvex_device_tensor_is_written(reference_output),
            "replay compiled Tensor Core MoE without recapturing its launch graph");
        for (slot = 0ull; slot < 3ull; ++slot)
            canonical_routed[slot] =
                job.weights[YVEX_MOE_WEIGHT_ROUTED_GATE + slot];
        YVEX_TEST_ASSERT(
            moe_derived_weight(
                &fixture, operations,
                &job.weights[YVEX_MOE_WEIGHT_ROUTED_GATE], EXPERTS,
                YVEX_EXECUTION_CONSUMER_ROUTED_GATE_UP, &err) &&
            moe_derived_weight(
                &fixture, operations,
                &job.weights[YVEX_MOE_WEIGHT_ROUTED_UP], EXPERTS,
                YVEX_EXECUTION_CONSUMER_ROUTED_GATE_UP, &err) &&
            moe_derived_weight(
                &fixture, operations,
                &job.weights[YVEX_MOE_WEIGHT_ROUTED_DOWN], EXPERTS,
                YVEX_EXECUTION_CONSUMER_ROUTED_DOWN, &err),
            "build deterministic aligned routed-expert layouts");
        for (slot = YVEX_MOE_WEIGHT_ROUTED_GATE;
             slot <= YVEX_MOE_WEIGHT_ROUTED_DOWN; ++slot)
            YVEX_TEST_ASSERT(
                job.weights[slot].storage_bytes == job.weights[slot].encoded_bytes,
                "size-neutral derived layouts remain distinct physical assets");
        reference_output->is_written = 0;
        memset(&result, 0, sizeof(result));
        rc = operations->execute_rows(backend, &job, &rows, &output, &result, &err);
        YVEX_TEST_ASSERT(
            rc == YVEX_OK && result.tensor_core_launches == 4ull &&
                result.graph_launches == 1ull && result.graph_captures == 1ull &&
                yvex_device_tensor_is_written(reference_output) &&
                yvex_backend_tensor_read(backend, reference_output, encoded,
                                         sizeof(encoded), &err) == YVEX_OK,
            "execute a separately keyed derived routed-expert graph");
        maximum_error = 0.0f;
        for (index = 0ull; index < ROWS * WIDTH; ++index) {
            float difference = fabsf(encoded[index] - tensorcore[index]);
            if (difference > maximum_error) maximum_error = difference;
        }
        YVEX_TEST_ASSERT(maximum_error <= 1e-6f,
                         "derived routed-expert layout matches canonical Tensor Core numerics");
        {
            yvex_moe_layer_result single = {0};
            single.combined_output = combined;
            single.combined_capacity = WIDTH;
            single.routed_output = routed;
            single.routed_capacity = WIDTH;
            single.shared_output = shared;
            single.shared_capacity = WIDTH;
            single.post = post;
            single.post_capacity = 1ull;
            single.combination = combination;
            single.combination_capacity = 1ull;
            job.device_input = one_input;
            job.device_output = one_output;
            one_output->is_written = 0;
            YVEX_TEST_ASSERT(
                yvex_backend_tensor_write(
                    backend, one_input, input_rows, WIDTH * sizeof(float), &err) == YVEX_OK &&
                yvex_backend_moe_begin(
                    &execution, backend, &job, &single, &err) == YVEX_OK &&
                yvex_backend_moe_add_expert(
                    execution, &job.weights[YVEX_MOE_WEIGHT_SHARED_GATE],
                    &job.weights[YVEX_MOE_WEIGHT_SHARED_UP],
                    &job.weights[YVEX_MOE_WEIGHT_SHARED_DOWN], 1.0f, 1, &err) == YVEX_OK &&
                yvex_backend_moe_finish(execution, &single, &err) == YVEX_OK &&
                yvex_backend_moe_close(&execution, &err) == YVEX_OK &&
                yvex_backend_tensor_read(
                    backend, one_output, one_device, sizeof(one_device), &err) == YVEX_OK,
                "execute derived routed experts through single-row production MoE");
            maximum_error = 0.0f;
            for (index = 0ull; index < WIDTH; ++index) {
                float difference = fabsf(one_device[index] - encoded[index]);
                if (difference > maximum_error) maximum_error = difference;
            }
            YVEX_TEST_ASSERT(
                maximum_error <= 1e-6f,
                "single-row and width-N derived MoE preserve identical numerics");
            job.device_input = NULL;
            job.device_output = NULL;
        }
        job.weights[YVEX_MOE_WEIGHT_ROUTED_DOWN].storage_bytes--;
        reference_output->is_written = 0;
        memset(&result, 0, sizeof(result));
        rc = operations->execute_rows(backend, &job, &rows, &output, &result, &err);
        YVEX_TEST_ASSERT(
            rc != YVEX_OK && !result.completed &&
                !yvex_device_tensor_is_written(reference_output),
            "inexact derived storage refuses without output publication");
        for (slot = 0ull; slot < 3ull; ++slot)
            job.weights[YVEX_MOE_WEIGHT_ROUTED_GATE + slot] =
                canonical_routed[slot];
        job.weights[YVEX_MOE_WEIGHT_ROUTED_GATE].kernel_family = "unadmitted-expert-kernel";
        reference_output->is_written = 0;
        memset(&result, 0, sizeof(result));
        rc = operations->execute_rows(backend, &job, &rows, &output, &result, &err);
        YVEX_TEST_ASSERT(
            rc == YVEX_ERR_FORMAT && !result.completed &&
                !yvex_device_tensor_is_written(reference_output),
            "mismatched compiler-selected expert kernels refuse without fallback");
        job.weights[YVEX_MOE_WEIGHT_ROUTED_GATE].kernel_family =
            YVEX_MOE_KERNEL_SM121_TENSORCORE_EXPERT;
    } else {
        YVEX_TEST_ASSERT(
            rc == YVEX_ERR_UNSUPPORTED && !result.completed &&
                !yvex_device_tensor_is_written(small_output),
            "portable bundle refuses compiled encoded MoE without fallback");
    }

    rows.row_count = ROWS;
    rows.device_rows = input;
    rows.device_outputs = encoded_output;
    job.weights[YVEX_MOE_WEIGHT_ROUTED_UP].activation =
        YVEX_EXECUTION_ACTIVATION_DEVICE_F32;
    encoded_output->is_written = 0;
    memset(&result, 0, sizeof(result));
    rc = operations->execute_rows(backend, &job, &rows, &output, &result, &err);
    YVEX_TEST_ASSERT(
        rc == YVEX_ERR_FORMAT && !result.completed &&
            !yvex_device_tensor_is_written(encoded_output),
        "mixed compiled gate/up activation refuses before publication");

    yvex_backend_workspace_detach(backend);
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_release(backend, &workspace, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &one_output, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &one_input, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &small_output, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &encoded_output, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &reference_output, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &small_input, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &input, &err) == YVEX_OK &&
            yvex_backend_resident_detach(backend, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &anchor, &err) == YVEX_OK,
        "release encoded MoE fixture ownership");
    return 0;
}

static int assert_grouped_moe(yvex_backend *backend)
{
    static const float mhc_function[] = {1.0f, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f};
    static const float three_ones[] = {1.0f, 1.0f, 1.0f};
    static const float three_zeroes[] = {0.0f, 0.0f, 0.0f};
    static const float identity[] = {1.0f, 0.0f, 0.0f, 1.0f};
    static const float routed[] = {1.0f, 0.0f, 0.0f, 1.0f,
                                   0.0f, 1.0f, 1.0f, 0.0f};
    yvex_backend_tensor_desc descriptor = {0};
    yvex_device_tensor *anchor = NULL, *input = NULL, *normal_output = NULL;
    yvex_device_tensor *audit_output = NULL, *batch_input = NULL;
    yvex_device_tensor *batch_output = NULL, *reference_output = NULL, *workspace = NULL;
    yvex_backend_moe_execution *execution = NULL;
    yvex_moe_layer_plan layer = {0};
    yvex_moe_layer_job job = {0};
    yvex_moe_layer_result normal, audit;
    yvex_moe_weight_view selected[3];
    const yvex_backend_moe_operations *row_operations;
    yvex_moe_row_batch row_batch = {0};
    yvex_moe_row_batch_output row_output = {0};
    yvex_moe_row_batch_result row_result, completion_result;
    yvex_moe_device_completion device_completion = {0};
    moe_fixture_weights fixture = {0};
    yvex_error err;
    float host_input[2] = {1.0f, 0.5f};
    float host_batch[4] = {1.0f, 0.5f, 0.5f, 1.0f};
    float normal_device[2], audit_device[2];
    float batch_device[4], reference_device[4];
    float combined[2], routed_output[2], shared_output[2], post[1], combination[1];
    float batch_combined[4] = {0}, batch_routed[4] = {0}, batch_shared[4] = {0};
    float batch_post[2] = {0}, batch_combination[2] = {0}, batch_weights[2] = {0};
    unsigned long long batch_selected[2] = {0};
    yvex_moe_device_completion_slot deferred = {0};
    unsigned long long address = 0ull, expert, workspace_bytes = 0ull, slot;
    unsigned long long immediate_active_bytes;
    int rc;
    memset(&descriptor, 0, sizeof(descriptor));
    descriptor.name = "grouped-moe-anchor";
    descriptor.dtype = YVEX_DTYPE_I8;
    descriptor.rank = 1u;
    descriptor.dims[0] = descriptor.bytes = 128ull * sizeof(*fixture.arena);
    YVEX_TEST_ASSERT(yvex_backend_resident_alloc(
                         backend, &descriptor, &anchor,
                         (unsigned char **)&fixture.arena, &err) == YVEX_OK,
                     "allocate grouped MoE managed fixture");
    memset(fixture.arena, 0, (size_t)descriptor.bytes);
    YVEX_TEST_ASSERT(yvex_backend_resident_attach(
                         backend, (const unsigned char *)fixture.arena,
                         128ull * sizeof(*fixture.arena), anchor, 7ull, &err) == YVEX_OK &&
                         yvex_backend_resident_resolve(
                             backend, (const unsigned char *)fixture.arena,
                             128ull * sizeof(*fixture.arena), &address) ==
                             YVEX_BACKEND_RESIDENT_HIT,
                     "register grouped MoE fixture once");
    fixture.device_base = address;
    layer.schema_version = YVEX_MOE_PLAN_SCHEMA_V1;
    layer.router_class = YVEX_MOE_ROUTER_LEARNED_HIDDEN_STATE;
    layer.scoring = YVEX_MOE_SCORING_SQRT_SOFTPLUS;
    layer.topk_policy = YVEX_MOE_TOPK_NOAUX_TC;
    layer.activation = YVEX_MOE_ACTIVATION_SILU;
    layer.hidden_width = layer.expanded_width = 2ull;
    layer.residual_streams = 1ull;
    layer.mhc_mixing_rows = 3ull;
    layer.mhc_sinkhorn_iterations = 1ull;
    layer.routed_experts = 2ull;
    layer.shared_experts = layer.experts_per_token = 1ull;
    layer.expert_intermediate_width = layer.shared_intermediate_width = 2ull;
    layer.correction_bias_width = 2ull;
    layer.rms_epsilon = layer.mhc_epsilon = 0.00001;
    layer.mhc_post_multiplier = layer.routed_scaling_factor = 1.0;
    layer.activation_limit = 10.0;
    layer.requires_correction_bias = layer.normalize_topk_probabilities = 1;
    for (slot = 0ull; slot < YVEX_MOE_WEIGHT_COUNT; ++slot)
        layer.tensor_ids[slot] = YVEX_MOE_NO_TENSOR;
    job.layer = &layer;
    job.expanded_input = host_input;
    job.token_id_present = 1;
    job.evidence_level = YVEX_ATTENTION_EVIDENCE_SUMMARY;
    moe_fixture_weight(&fixture, &job.weights[YVEX_MOE_WEIGHT_MHC_FUNCTION],
                       YVEX_TENSOR_ROLE_HC_FFN_FUNCTION, 3ull, 2ull, mhc_function);
    moe_fixture_weight(&fixture, &job.weights[YVEX_MOE_WEIGHT_MHC_SCALE],
                       YVEX_TENSOR_ROLE_HC_FFN_SCALE, 1ull, 3ull, three_ones);
    moe_fixture_weight(&fixture, &job.weights[YVEX_MOE_WEIGHT_MHC_BASE],
                       YVEX_TENSOR_ROLE_HC_FFN_BASE, 1ull, 3ull, three_zeroes);
    moe_fixture_weight(&fixture, &job.weights[YVEX_MOE_WEIGHT_FFN_NORM],
                       YVEX_TENSOR_ROLE_FFN_NORM, 1ull, 2ull, three_ones);
    moe_fixture_weight(&fixture, &job.weights[YVEX_MOE_WEIGHT_ROUTER],
                       YVEX_TENSOR_ROLE_MOE_ROUTER, 2ull, 2ull, identity);
    moe_fixture_weight(&fixture, &job.weights[YVEX_MOE_WEIGHT_ROUTER_BIAS],
                       YVEX_TENSOR_ROLE_MOE_ROUTER_BIAS, 1ull, 2ull, three_zeroes);
    moe_fixture_weight(&fixture, &job.weights[YVEX_MOE_WEIGHT_ROUTED_GATE],
                       YVEX_TENSOR_ROLE_MOE_EXPERT_GATE, 4ull, 2ull, routed);
    moe_fixture_weight(&fixture, &job.weights[YVEX_MOE_WEIGHT_ROUTED_UP],
                       YVEX_TENSOR_ROLE_MOE_EXPERT_UP, 4ull, 2ull, routed);
    moe_fixture_weight(&fixture, &job.weights[YVEX_MOE_WEIGHT_ROUTED_DOWN],
                       YVEX_TENSOR_ROLE_MOE_EXPERT_DOWN, 4ull, 2ull, routed);
    moe_fixture_weight(&fixture, &job.weights[YVEX_MOE_WEIGHT_SHARED_GATE],
                       YVEX_TENSOR_ROLE_MOE_SHARED_EXPERT_GATE, 2ull, 2ull, identity);
    moe_fixture_weight(&fixture, &job.weights[YVEX_MOE_WEIGHT_SHARED_UP],
                       YVEX_TENSOR_ROLE_MOE_SHARED_EXPERT_UP, 2ull, 2ull, identity);
    moe_fixture_weight(&fixture, &job.weights[YVEX_MOE_WEIGHT_SHARED_DOWN],
                       YVEX_TENSOR_ROLE_MOE_SHARED_EXPERT_DOWN, 2ull, 2ull, identity);
    for (slot = 0ull; slot < YVEX_MOE_WEIGHT_COUNT; ++slot)
        if (job.weights[slot].device_address)
            layer.tensor_ids[slot] = job.weights[slot].tensor_id;
    row_operations = yvex_backend_moe_operations_get(backend);
    YVEX_TEST_ASSERT(row_operations &&
                         row_operations->workspace_required(
                             &layer, 2ull, &workspace_bytes, &err) == YVEX_OK &&
                         workspace_bytes != 0ull,
                     "derive grouped MoE workspace from width and layer geometry");
#define ALLOCATE_TENSOR(OWNER_, NAME_, BYTES_)                                             \
    do {                                                                                   \
        memset(&descriptor, 0, sizeof(descriptor));                                        \
        descriptor.name = (NAME_);                                                         \
        descriptor.dtype = YVEX_DTYPE_F32;                                                 \
        descriptor.rank = 1u;                                                              \
        descriptor.dims[0] = (BYTES_) / sizeof(float);                                    \
        descriptor.bytes = (BYTES_);                                                       \
        YVEX_TEST_ASSERT(yvex_backend_tensor_alloc(backend, &descriptor, &(OWNER_), &err) == \
                             YVEX_OK,                                                       \
                         "allocate grouped MoE device tensor");                           \
    } while (0)
    ALLOCATE_TENSOR(input, "grouped-moe-input", sizeof(host_input));
    ALLOCATE_TENSOR(normal_output, "grouped-moe-normal", sizeof(host_input));
    ALLOCATE_TENSOR(audit_output, "grouped-moe-audit", sizeof(host_input));
    ALLOCATE_TENSOR(batch_input, "grouped-moe-batch-input", sizeof(host_batch));
    ALLOCATE_TENSOR(batch_output, "grouped-moe-batch-output", sizeof(host_batch));
    ALLOCATE_TENSOR(reference_output, "grouped-moe-reference-output", sizeof(host_batch));
    ALLOCATE_TENSOR(workspace, "grouped-moe-workspace", workspace_bytes);
#undef ALLOCATE_TENSOR
    YVEX_TEST_ASSERT(yvex_backend_tensor_write(
                         backend, input, host_input, sizeof(host_input), &err) == YVEX_OK &&
                         yvex_backend_tensor_write(
                             backend, batch_input, host_batch, sizeof(host_batch), &err) == YVEX_OK &&
                         yvex_backend_workspace_attach(backend, workspace, 1ull, &err) == YVEX_OK,
                     "prepare grouped MoE device input and workspace");
    job.device_input = input;
    job.device_output = normal_output;
    moe_fixture_result(&normal, combined, routed_output, shared_output, post, combination);
    rc = yvex_backend_moe_begin(&execution, backend, &job, &normal, &err);
    if (rc == YVEX_OK)
        rc = yvex_backend_moe_add_expert(
            execution, &job.weights[YVEX_MOE_WEIGHT_SHARED_GATE],
            &job.weights[YVEX_MOE_WEIGHT_SHARED_UP],
            &job.weights[YVEX_MOE_WEIGHT_SHARED_DOWN], 1.0f, 1, &err);
    if (rc == YVEX_OK) rc = yvex_backend_moe_finish(execution, &normal, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK &&
                         yvex_backend_moe_close(&execution, &err) == YVEX_OK &&
                         yvex_backend_tensor_read(backend, normal_output, normal_device,
                                                  sizeof(normal_device), &err) == YVEX_OK,
                     "execute grouped direct-address MoE");
    job.evidence_level = YVEX_ATTENTION_EVIDENCE_FULL;
    job.device_output = audit_output;
    moe_fixture_result(&audit, combined, routed_output, shared_output, post, combination);
    rc = yvex_backend_moe_begin(&execution, backend, &job, &audit, &err);
    expert = audit.router.selected_experts[0];
    selected[0] = moe_fixture_expert(
        &job.weights[YVEX_MOE_WEIGHT_ROUTED_GATE], expert, 2ull);
    selected[1] = moe_fixture_expert(
        &job.weights[YVEX_MOE_WEIGHT_ROUTED_UP], expert, 2ull);
    selected[2] = moe_fixture_expert(
        &job.weights[YVEX_MOE_WEIGHT_ROUTED_DOWN], expert, 2ull);
    if (rc == YVEX_OK)
        rc = yvex_backend_moe_add_expert(execution, &selected[0], &selected[1],
                                         &selected[2], audit.router.selected_weights[0],
                                         0, &err);
    if (rc == YVEX_OK)
        rc = yvex_backend_moe_add_expert(
            execution, &job.weights[YVEX_MOE_WEIGHT_SHARED_GATE],
            &job.weights[YVEX_MOE_WEIGHT_SHARED_UP],
            &job.weights[YVEX_MOE_WEIGHT_SHARED_DOWN], 1.0f, 1, &err);
    if (rc == YVEX_OK) rc = yvex_backend_moe_finish(execution, &audit, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK &&
                         yvex_backend_moe_close(&execution, &err) == YVEX_OK &&
                         yvex_backend_tensor_read(backend, audit_output, audit_device,
                                                  sizeof(audit_device), &err) == YVEX_OK,
                     "execute audit selected-expert MoE");
    YVEX_TEST_ASSERT(memcmp(normal_device, audit_device, sizeof(normal_device)) == 0 &&
                         normal.router.selected_experts[0] ==
                             audit.router.selected_experts[0] &&
                         normal.memory.activation_bytes ==
                             2ull * layer.expanded_width * sizeof(float) &&
                         normal.memory.temporary_bytes != 0ull &&
                         normal.upload_count == 0ull &&
                         normal.device_to_host_bytes < audit.device_to_host_bytes &&
                         normal.device_synchronizations == 1ull,
                     "grouped and audit MoE agree with reduced movement and synchronization");
    for (slot = 0ull; slot < 2ull; ++slot) {
        yvex_device_tensor input_view, output_view;
        yvex_moe_layer_result reference;
        YVEX_TEST_ASSERT(
            yvex_backend_tensor_f32_subview(batch_input, slot * 2ull, 2ull, &input_view) &&
                yvex_backend_tensor_f32_subview(reference_output, slot * 2ull, 2ull,
                                                &output_view),
            "form one-row MoE reference views");
        job.device_input = &input_view;
        job.device_output = &output_view;
        job.expanded_input = host_batch + slot * 2ull;
        job.evidence_level = YVEX_ATTENTION_EVIDENCE_SUMMARY;
        moe_fixture_result(&reference, combined, routed_output, shared_output, post, combination);
        rc = yvex_backend_moe_begin(&execution, backend, &job, &reference, &err);
        if (rc == YVEX_OK)
            rc = yvex_backend_moe_add_expert(
                execution, &job.weights[YVEX_MOE_WEIGHT_SHARED_GATE],
                &job.weights[YVEX_MOE_WEIGHT_SHARED_UP],
                &job.weights[YVEX_MOE_WEIGHT_SHARED_DOWN], 1.0f, 1, &err);
        if (rc == YVEX_OK) rc = yvex_backend_moe_finish(execution, &reference, &err);
        YVEX_TEST_ASSERT(rc == YVEX_OK &&
                             yvex_backend_moe_close(&execution, &err) == YVEX_OK,
                         "execute one-row MoE reference for width-N comparison");
    }
    memset(&row_batch, 0, sizeof(row_batch));
    row_batch.schema_version = YVEX_MOE_ROW_BATCH_SCHEMA_V1;
    row_batch.row_count = 2ull;
    row_batch.row_width = row_batch.row_stride = 2ull;
    row_batch.expanded_rows = host_batch;
    row_batch.device_rows = batch_input;
    row_batch.device_outputs = batch_output;
    row_batch.token_ids = (const unsigned int[]){7u, 11u};
    row_batch.token_ids_present = 1;
    row_batch.execution_class = YVEX_EXECUTION_CLASS_DEVICE_NATIVE;
    row_output.combined_rows = batch_combined;
    row_output.combined_capacity = 4ull;
    row_output.routed_rows = batch_routed;
    row_output.routed_capacity = 4ull;
    row_output.shared_rows = batch_shared;
    row_output.shared_capacity = 4ull;
    row_output.post_rows = batch_post;
    row_output.post_capacity = 2ull;
    row_output.combination_rows = batch_combination;
    row_output.combination_capacity = 2ull;
    row_output.selected_experts = batch_selected;
    row_output.selected_weights = batch_weights;
    row_output.selection_capacity = 2ull;
    job.device_input = batch_input;
    job.device_output = batch_output;
    job.expanded_input = host_batch;
    YVEX_TEST_ASSERT(!yvex_device_tensor_is_written(batch_output),
                     "width-N MoE output begins unpublished");
    rc = row_operations->execute_rows(
        backend, &job, &row_batch, &row_output, &row_result, &err);
    YVEX_TEST_ASSERT(
        rc == YVEX_OK && row_result.completed == 1 && row_result.row_count == 2ull &&
            row_result.row_expert_pairs == 2ull && row_result.unique_experts >= 1ull &&
            yvex_device_tensor_is_written(batch_output) &&
            row_result.kernel_launches < 2ull * normal.kernel_launches &&
            row_result.stream_synchronizations == 1ull &&
            row_result.device_synchronizations == 0ull &&
            yvex_backend_tensor_read(backend, batch_output, batch_device,
                                     sizeof(batch_device), &err) == YVEX_OK &&
            yvex_backend_tensor_read(backend, reference_output, reference_device,
                                     sizeof(reference_device), &err) == YVEX_OK &&
            memcmp(batch_device, reference_device, sizeof(batch_device)) == 0,
        "width-N MoE equals two one-row oracles with one grouped transaction");
    immediate_active_bytes = row_result.encoded_bytes_read;
    batch_selected[0] = batch_selected[1] = ULLONG_MAX;
    batch_weights[0] = batch_weights[1] = -99.0f;
    device_completion.defer = 1;
    device_completion.host = &deferred;
    job.device_completion = &device_completion;
    batch_output->is_written = 0;
    rc = row_operations->execute_rows(
        backend, &job, &row_batch, &row_output, &row_result, &err);
    YVEX_TEST_ASSERT(
        rc == YVEX_OK && !row_result.completed &&
            row_result.device_completion_pending &&
            yvex_device_tensor_is_written(batch_output) &&
            !row_result.stream_synchronizations &&
            !row_result.device_synchronizations &&
            row_result.d2h_bytes == sizeof(deferred.status) + sizeof(deferred.unique_experts) &&
            !row_result.memory.complete && row_result.memory.activation_bytes != 0ull &&
            batch_selected[0] == ULLONG_MAX && batch_selected[1] == ULLONG_MAX &&
            batch_weights[0] == -99.0f && batch_weights[1] == -99.0f,
        "deferred width-N MoE keeps routing evidence and completion off the layer path");
    rc = row_operations->complete_rows(
        backend, 2, &completion_result, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_INVALID_ARG,
                     "deferred MoE refuses an unproved completion barrier");
    rc = row_operations->complete_rows(
        backend, 0, &completion_result, &err);
    YVEX_TEST_ASSERT(
        rc == YVEX_OK && completion_result.completed && deferred.status == 0 &&
            deferred.unique_experts >= 1ull &&
            completion_result.stream_synchronizations == 1ull &&
            completion_result.device_synchronizations == 0ull &&
            row_result.active_weight_base_bytes +
                    row_result.active_weight_per_unique_expert_bytes * deferred.unique_experts ==
                immediate_active_bytes,
        "one phase completion validates deferred MoE and restores exact active bytes");
    rc = row_operations->complete_rows(backend, 1, &completion_result, &err);
    YVEX_TEST_ASSERT(
        rc == YVEX_OK && completion_result.completed &&
            !completion_result.stream_synchronizations &&
            !completion_result.device_synchronizations,
        "a proved same-stream barrier adds no redundant MoE synchronization");
    job.device_completion = NULL;
    yvex_backend_workspace_detach(backend);
    YVEX_TEST_ASSERT(yvex_backend_tensor_release(backend, &workspace, &err) == YVEX_OK &&
                         yvex_backend_tensor_release(backend, &reference_output, &err) == YVEX_OK &&
                         yvex_backend_tensor_release(backend, &batch_output, &err) == YVEX_OK &&
                         yvex_backend_tensor_release(backend, &batch_input, &err) == YVEX_OK &&
                         yvex_backend_tensor_release(backend, &audit_output, &err) == YVEX_OK &&
                         yvex_backend_tensor_release(backend, &normal_output, &err) == YVEX_OK &&
                         yvex_backend_tensor_release(backend, &input, &err) == YVEX_OK &&
                         yvex_backend_resident_detach(backend, &err) == YVEX_OK &&
                         yvex_backend_tensor_release(backend, &anchor, &err) == YVEX_OK,
                     "release grouped MoE fixture ownership");
    return 0;
}

static int assert_finite_activation_overflow(yvex_backend *backend)
{
    enum { ARENA_FLOATS = 32, CUDA_BLOCK = 256 };
    yvex_backend_tensor_desc descriptor = {0};
    yvex_device_tensor *arena = NULL;
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    float host[ARENA_FLOATS] = {0.0f}, observed[ARENA_FLOATS] = {0.0f};
    const float large = yvex_quant_bf16_decode(yvex_quant_bf16_encode(1.0e20f));
    CUdeviceptr base, values, weight, status, residual, mix, scale, mhc_base;
    CUdeviceptr collapsed, post, combination;
    unsigned long long count = 4ull, vectors = 1ull;
    unsigned long long streams = 2ull, width = 2ull, mixing_rows = 8ull;
    unsigned long long iterations = 1ull, row_count = 1ull;
    unsigned int qtype = YVEX_GGUF_QTYPE_F32;
    double epsilon = 1.0e-6, mhc_epsilon = 1.0e-6, multiplier = 2.0;
    int device_wide = 0, rc;
    yvex_error err;

    descriptor.name = "finite-rms-overflow-arena";
    descriptor.dtype = YVEX_DTYPE_F32;
    descriptor.rank = 1u;
    descriptor.dims[0] = ARENA_FLOATS;
    descriptor.bytes = sizeof(host);
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_alloc(backend, &descriptor, &arena, &err) == YVEX_OK,
        "allocate finite RMS overflow arena");
    base = yvex_cuda_activation_pointer(backend, arena);
    status = base + 31ull * sizeof(float);
    host[0] = host[1] = host[4] = host[5] = host[6] = host[7] = large;
    host[2] = host[3] = -large;
    YVEX_TEST_ASSERT(
        base && yvex_backend_tensor_write(
                    backend, arena, host, sizeof(host), &err) == YVEX_OK,
        "upload finite matvec overflow fixture");
    {
        CUdeviceptr encoded = base, vector = base + 4ull * sizeof(float);
        CUdeviceptr additive = 0ull, out = base + 8ull * sizeof(float);
        unsigned long long row_bytes = 4ull * sizeof(float), start_row = 0ull;
        int q8_input = 0, block_row = 1, forensic = 0, output_bf16 = 0;
        void *params[] = {
            &encoded, &row_bytes, &count, &start_row, &row_count, &vectors,
            &qtype, &vector, &count, &q8_input, &block_row, &forensic, &additive,
            &out, &vectors, &output_bf16, &status};
        rc = yvex_cuda_launch(
            backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
            state->qtype_matvec_function, 1u, CUDA_BLOCK, 0u, params,
            "cuda.test.matvec-overflow", &err);
    }
    if (rc == YVEX_OK)
        rc = yvex_cuda_launch_synchronize(
            backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED, &device_wide,
            "cuda.test.matvec-overflow", &err);
    YVEX_TEST_ASSERT(
        rc == YVEX_OK &&
            yvex_backend_tensor_read(backend, arena, observed, sizeof(observed), &err) == YVEX_OK &&
            observed[31] == 0.0f && isfinite(observed[8]),
        "finite F32 matvec survives intermediate reduction overflow");

    memset(host, 0, sizeof(host));
    host[0] = large;
    host[1] = -large;
    host[2] = large;
    host[3] = -large;
    host[4] = host[5] = host[6] = host[7] = 1.0f;
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_write(backend, arena, host, sizeof(host), &err) == YVEX_OK,
        "upload finite RMS overflow fixture");
    values = base;
    weight = base + 4ull * sizeof(float);
    {
        void *params[] = {
            &values, &count, &weight, &qtype, &epsilon, &vectors, &status};
        rc = yvex_cuda_launch(
            backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
            state->attention_weighted_norm_function, 1u, CUDA_BLOCK,
            CUDA_BLOCK * sizeof(double), params,
            "cuda.test.weighted-norm-overflow", &err);
    }
    if (rc == YVEX_OK)
        rc = yvex_cuda_launch_synchronize(
            backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED, &device_wide,
            "cuda.test.weighted-norm-overflow", &err);
    YVEX_TEST_ASSERT(
        rc == YVEX_OK &&
            yvex_backend_tensor_read(backend, arena, observed, sizeof(observed), &err) == YVEX_OK &&
            observed[31] == 0.0f && fabsf(observed[0] - 1.0f) <= 0.01f &&
            fabsf(observed[1] + 1.0f) <= 0.01f &&
            fabsf(observed[2] - 1.0f) <= 0.01f &&
            fabsf(observed[3] + 1.0f) <= 0.01f,
        "finite BF16-range values survive weighted RMS square-sum overflow");

    memset(host, 0, sizeof(host));
    host[0] = large;
    host[1] = -large;
    host[4] = large;
    host[5] = -large;
    host[12] = host[13] = host[14] = 1.0f;
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_write(backend, arena, host, sizeof(host), &err) == YVEX_OK,
        "upload finite mHC overflow fixture");
    residual = base;
    mix = base + 4ull * sizeof(float);
    scale = base + 12ull * sizeof(float);
    mhc_base = base + 15ull * sizeof(float);
    collapsed = base + 23ull * sizeof(float);
    post = base + 25ull * sizeof(float);
    combination = base + 27ull * sizeof(float);
    {
        void *params[] = {
            &residual, &mix, &scale, &mhc_base, &streams, &width,
            &mixing_rows, &iterations, &epsilon, &mhc_epsilon, &multiplier,
            &collapsed, &post, &combination, &row_count, &status};
        rc = yvex_cuda_launch(
            backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
            state->residual_mhc_pre_function, 1u, CUDA_BLOCK,
            (unsigned int)((streams + 1ull + CUDA_BLOCK) * sizeof(double)), params,
            "cuda.test.mhc-pre-overflow", &err);
    }
    if (rc == YVEX_OK)
        rc = yvex_cuda_launch_synchronize(
            backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED, &device_wide,
            "cuda.test.mhc-pre-overflow", &err);
    YVEX_TEST_ASSERT(
        rc == YVEX_OK &&
            yvex_backend_tensor_read(backend, arena, observed, sizeof(observed), &err) == YVEX_OK &&
            observed[31] == 0.0f && observed[23] > 0.7f * large &&
            observed[24] < -0.7f * large,
        "finite BF16-range values preserve mHC normalization after square-sum overflow");
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_release(backend, &arena, &err) == YVEX_OK,
        "release finite RMS overflow fixture");
    return 0;
}

/* Prove the sparse-row MoE kernel recovers a finite dot after F32 block overflow. */
static int assert_moe_row_dot_overflow(yvex_backend *backend)
{
    enum {
        Q8_BLOCKS = 2,
        Q8_0_ROW_BYTES = Q8_BLOCKS * 8 * 34,
        Q8_K_BYTES = Q8_BLOCKS * 292,
        SELECTED_OFFSET = Q8_0_ROW_BYTES + Q8_K_BYTES,
        WEIGHT_OFFSET = SELECTED_OFFSET + 8,
        ORDER_OFFSET = WEIGHT_OFFSET + 8,
        OUTPUT_OFFSET = ORDER_OFFSET + 8,
        STATUS_OFFSET = OUTPUT_OFFSET + 4,
        ARENA_BYTES = STATUS_OFFSET + 4
    };
    yvex_backend_tensor_desc descriptor = {0};
    yvex_device_tensor *arena = NULL;
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    unsigned char host[ARENA_BYTES] = {0}, observed[ARENA_BYTES] = {0};
    const float activation_scale = 1.0e34f, route_weight = 1.0f;
    const unsigned short weight_scale = yvex_quant_f16_encode(1.0f);
    CUdeviceptr base, down, selected, weights, order, activation, output, status;
    unsigned long long row_bytes = Q8_0_ROW_BYTES, expert_bytes = Q8_0_ROW_BYTES;
    unsigned long long pair_count = 1ull, topk = 1ull, experts = 1ull;
    unsigned long long intermediate_extent = Q8_BLOCKS, hidden = 1ull;
    unsigned int qtype = YVEX_GGUF_QTYPE_Q8_0;
    int q8_input = 1, device_wide = 0, device_status, rc;
    float result;
    yvex_error err;

    for (unsigned int block = 0u; block < Q8_BLOCKS * 8u; ++block) {
        unsigned char *weight = host + block * 34u;
        memcpy(weight, &weight_scale, sizeof(weight_scale));
        memset(weight + 2u, block < 8u ? 127 : 129, 32u);
    }
    for (unsigned int block = 0u; block < Q8_BLOCKS; ++block) {
        unsigned char *q8 = host + Q8_0_ROW_BYTES + block * 292u;
        memcpy(q8, &activation_scale, sizeof(activation_scale));
        memset(q8 + 4u, 127, 256u);
    }
    memcpy(host + WEIGHT_OFFSET, &route_weight, sizeof(route_weight));
    descriptor.name = "moe-row-dot-overflow";
    descriptor.dtype = YVEX_DTYPE_I8;
    descriptor.rank = 1u;
    descriptor.dims[0] = descriptor.bytes = ARENA_BYTES;
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_alloc(backend, &descriptor, &arena, &err) == YVEX_OK &&
            yvex_backend_tensor_write(backend, arena, host, sizeof(host), &err) == YVEX_OK,
        "allocate sparse-row MoE overflow fixture");
    base = yvex_cuda_activation_pointer(backend, arena);
    down = base;
    activation = base + Q8_0_ROW_BYTES;
    selected = base + SELECTED_OFFSET;
    weights = base + WEIGHT_OFFSET;
    order = base + ORDER_OFFSET;
    output = base + OUTPUT_OFFSET;
    status = base + STATUS_OFFSET;
    {
        void *params[] = {
            &down, &row_bytes, &expert_bytes, &qtype, &selected, &weights,
            &order, &pair_count, &topk, &experts, &activation,
            &intermediate_extent, &q8_input, &hidden, &output, &status};
        rc = yvex_cuda_launch(
            backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
            state->moe_grouped_down_rows_function, 1u, 256u, 0u, params,
            "cuda.test.moe-row-dot-overflow", &err);
    }
    if (rc == YVEX_OK)
        rc = yvex_cuda_launch_synchronize(
            backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED, &device_wide,
            "cuda.test.moe-row-dot-overflow", &err);
    YVEX_TEST_ASSERT(
        rc == YVEX_OK &&
            yvex_backend_tensor_read(backend, arena, observed, sizeof(observed), &err) == YVEX_OK,
        "read sparse-row MoE overflow result");
    memcpy(&result, observed + OUTPUT_OFFSET, sizeof(result));
    memcpy(&device_status, observed + STATUS_OFFSET, sizeof(device_status));
    YVEX_TEST_ASSERT(device_status == 0 && isfinite(result) && fabsf(result) <= 1.0e30f,
                     "sparse-row MoE preserves finite cancellation after block overflow");
    YVEX_TEST_ASSERT(yvex_backend_tensor_release(backend, &arena, &err) == YVEX_OK,
                     "release sparse-row MoE overflow fixture");
    return 0;
}

static int assert_deferred_attention_completion(yvex_backend *backend)
{
    yvex_backend_attention_completion completion;
    float staged = 4.5f, published = -1.0f;
    int status = 0;
    yvex_error err;
    int rc;

    memset(&completion, 0, sizeof(completion));
    completion.pending = 1;
    completion.host_status = &status;
    completion.attention_class = YVEX_BACKEND_ATTENTION_SWA;
    completion.token_count = 1ull;
    completion.transfer_count = 1u;
    completion.transfers[0] = (yvex_backend_attention_completion_transfer){
        .output = &published,
        .staged = &staged,
        .capacity = 1ull,
        .output_capacity = 1ull,
        .used = 1ull,
        .width = sizeof(staged),
        .stage = "test.cuda.attention.complete"};
    YVEX_TEST_ASSERT(published == -1.0f,
                     "deferred attention keeps staged output unpublished");
    rc = yvex_backend_attention_complete(backend, &completion, 0, &err);
    YVEX_TEST_ASSERT(
        rc == YVEX_OK && !completion.pending && published == staged &&
            completion.barrier_observed &&
            completion.output.stream_synchronizations == 1ull &&
            !completion.output.device_synchronizations,
        "first deferred attention completion publishes after one stream barrier");

    published = -1.0f;
    completion.pending = 1;
    completion.barrier_observed = 0;
    completion.output.stream_synchronizations = 0ull;
    rc = yvex_backend_attention_complete(backend, &completion, 1, &err);
    YVEX_TEST_ASSERT(
        rc == YVEX_OK && published == staged && completion.barrier_observed &&
            !completion.output.stream_synchronizations &&
            !completion.output.device_synchronizations,
        "ordered attention completions reuse one already observed barrier");

    completion.pending = 1;
    completion.transfers[0].output_capacity = 0ull;
    rc = yvex_backend_attention_complete(backend, &completion, 1, &err);
    YVEX_TEST_ASSERT(
        rc == YVEX_ERR_BOUNDS && !completion.pending &&
            completion.failure.code == YVEX_BACKEND_ATTENTION_FAILURE_COPY,
        "deferred attention publication refuses an invalid host extent");
    completion.transfers[0].output_capacity = 1ull;

    completion.pending = 1;
    status = 1;
    rc = yvex_backend_attention_complete(backend, &completion, 1, &err);
    YVEX_TEST_ASSERT(
        rc == YVEX_ERR_FORMAT && !completion.pending &&
            completion.failure.code == YVEX_BACKEND_ATTENTION_FAILURE_NUMERIC,
        "deferred attention publication refuses device numerical failure");
    status = 0;

    return 0;
}

static int unreachable_attention_stage(
    void *context, const yvex_attention_publication *publication,
    const yvex_attention_cancellation *cancellation,
    char state_delta_identity[YVEX_SHA256_HEX_CAP],
    yvex_attention_failure *failure, yvex_error *err)
{
    (void)context;
    (void)publication;
    (void)cancellation;
    (void)state_delta_identity;
    (void)failure;
    (void)err;
    return YVEX_ERR_STATE;
}

static int assert_deferred_attention_layer_failure(yvex_backend *backend)
{
    yvex_backend_attention_completion completion = {0};
    yvex_attention_publication publication = {0};
    yvex_attention_cpu_result evidence = {0};
    yvex_attention_probe_state_provider provider = {0};
    yvex_attention_failure failure = {0};
    char state_delta_identity[YVEX_SHA256_HEX_CAP] = {0};
    int provider_context = 1, status = 1, rc;
    yvex_error err;

    completion.pending = 1;
    completion.host_status = &status;
    completion.attention_class = YVEX_BACKEND_ATTENTION_SWA;
    completion.token_count = 1ull;
    publication.layer_index = 45ull;
    publication.device_completion_pending = 1;
    provider.context = &provider_context;
    provider.stage = unreachable_attention_stage;
    rc = yvex_attention_device_completion_resolve(
        backend, &completion, &publication, &evidence, &provider, NULL, 1,
        state_delta_identity, &failure, &err);
    YVEX_TEST_ASSERT(
        rc == YVEX_ERR_FORMAT &&
            failure.code == YVEX_ATTENTION_FAILURE_NUMERIC &&
            strcmp(yvex_error_where(&err), "graph.attention.completion") == 0 &&
            strstr(yvex_error_message(&err), "layer=45 expected=0 actual=1") != NULL,
        "deferred attention numerical refusal identifies its layer and device status");
    return 0;
}

static int assert_deferred_attention_sync_failure(void)
{
    yvex_backend_attention_completion completion = {0};
    yvex_backend_options options = {0};
    yvex_backend *backend = NULL;
    yvex_error err;
    int status = 0;
    int rc;

    options.kind = YVEX_BACKEND_KIND_CUDA;
    YVEX_TEST_ASSERT(yvex_backend_open(&backend, &options, &err) == YVEX_OK,
                     "open isolated CUDA attention completion backend");
    completion.pending = 1;
    completion.host_status = &status;
    completion.attention_class = YVEX_BACKEND_ATTENTION_SWA;
    completion.token_count = 1ull;
    YVEX_TEST_ASSERT(
        setenv("YVEX_TEST_CUDA_SYNC_FAILURE", "encoded-attention", 1) == 0,
        "inject deferred attention completion synchronization failure");
    rc = yvex_backend_attention_complete(backend, &completion, 0, &err);
    YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_CUDA_SYNC_FAILURE") == 0,
                     "clear deferred attention synchronization failure");
    YVEX_TEST_ASSERT(
        rc == YVEX_ERR_BACKEND && !completion.pending &&
            !completion.barrier_observed &&
            completion.failure.code == YVEX_BACKEND_ATTENTION_FAILURE_SYNCHRONIZE,
        "deferred attention completion fails closed before publication");
    yvex_backend_close(backend);
    return 0;
}

int yvex_cuda_test_info(void)
{
    yvex_backend *backend = NULL;
    yvex_backend_options options;
    yvex_backend_device_info info;
    yvex_backend_tensor_desc descriptor;
    yvex_device_tensor *resident = NULL;
    yvex_backend_cuda_attention_graph_summary kernel_summary = {0};
    yvex_backend_bandwidth_evidence bandwidth = {0}, repeated = {0};
    const char *required_native = getenv("YVEX_REQUIRE_NATIVE_CUDA_TEST");
    yvex_error err;
    static const char *attention_symbols[] = {
        "yvex_attention_bf16_round",
        "yvex_qtype_matvec",
        "yvex_encoded_row_decode",
        "yvex_attention_weighted_norm",
        "yvex_attention_unit_norm",
        "yvex_attention_yarn_rope",
        "yvex_attention_activation_quantize",
        "yvex_attention_rolling_state",
        "yvex_attention_topk",
        "yvex_attention_reduce"
    };
    size_t symbol_index;
    unsigned char *imported = NULL, *mapped = NULL, *pageable = NULL;
    unsigned char mapped_readback[4096];
    unsigned long long mapped_address = 0ull, prefetched_bytes = 0ull;
    int rc;
    memset(&options, 0, sizeof(options));
    options.kind = YVEX_BACKEND_KIND_CUDA;
    rc = yvex_backend_open(&backend, &options, &err);
    if (rc == YVEX_ERR_UNSUPPORTED) {
        fprintf(stderr, "SKIP: CUDA unavailable: %s\n", yvex_error_message(&err));
        return 77;
    }
    YVEX_TEST_ASSERT(rc == YVEX_OK, "open cuda backend");
    rc = yvex_backend_get_device_info(backend, &info, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "get cuda device info");
    YVEX_TEST_ASSERT(info.kind == YVEX_BACKEND_KIND_CUDA, "device info kind");
    YVEX_TEST_ASSERT(info.name && info.name[0] != '\0', "device name non-empty");
    YVEX_TEST_ASSERT(info.compute_capability_major >= 1, "compute capability major");
    YVEX_TEST_ASSERT(info.global_memory_bytes > 0, "global memory nonzero");
    YVEX_TEST_ASSERT(info.total_memory_bytes > 0, "total memory nonzero");
    YVEX_TEST_ASSERT(yvex_backend_cuda_attention_graph_summary_get(
                         backend, &kernel_summary, &err) == YVEX_OK &&
                         kernel_summary.kernel_bundle_architecture[0] &&
                         yvex_sha256_hex_valid(kernel_summary.cuda_build_identity),
                     "query admitted CUDA kernel image identity");
    YVEX_TEST_ASSERT(
        strcmp(yvex_cuda_kernel_function_identity(
                   yvex_cuda_state(backend),
                   yvex_cuda_state(backend)->qtype_tensorcore_rows_function),
               "yvex_qtype_tensorcore_rows") == 0,
        "graph identity resolves the admitted Tensor Core kernel from bundle authority");
    if (required_native && required_native[0]) {
        YVEX_TEST_ASSERT(kernel_summary.kernel_bundle_native,
                         "native CUDA validation refuses a PTX-only bundle");
        YVEX_TEST_ASSERT(strcmp(kernel_summary.kernel_bundle_architecture,
                                required_native) == 0,
                         "native CUDA image targets the required architecture");
        fprintf(stderr, "native CUDA kernel bundle: architecture=%s identity=%s\n",
                kernel_summary.kernel_bundle_architecture,
                kernel_summary.cuda_build_identity);
    }
    YVEX_TEST_ASSERT(yvex_backend_bandwidth_probe(backend, &bandwidth, &err) == YVEX_OK &&
                         bandwidth.schema_version ==
                             YVEX_BACKEND_BANDWIDTH_SCHEMA_V1 &&
                         bandwidth.sample_count == YVEX_BACKEND_BANDWIDTH_SAMPLE_COUNT &&
                         bandwidth.working_set_bytes > 0ull && bandwidth.iterations > 0ull &&
                         bandwidth.sustainable_read_bytes_per_second > 0ull &&
                         bandwidth.sustainable_copy_bytes_per_second > 0ull &&
                         bandwidth.sustainable_coherent_host_bytes_per_second > 0ull &&
                         yvex_sha256_hex_valid(bandwidth.identity) &&
                         strcmp(bandwidth.kernel_bundle_identity,
                                kernel_summary.cuda_build_identity) == 0,
                     "measure identity-bound CUDA bandwidth evidence");
    YVEX_TEST_ASSERT(yvex_backend_bandwidth_probe(backend, &repeated, &err) == YVEX_OK &&
                         memcmp(&bandwidth, &repeated, sizeof(bandwidth)) == 0,
                     "CUDA backend reuses one immutable bandwidth measurement");

    memset(&descriptor, 0, sizeof(descriptor));
    descriptor.name = "cuda-addressable-host-fixture";
    descriptor.dtype = YVEX_DTYPE_I8;
    descriptor.rank = 1u;
    descriptor.dims[0] = descriptor.bytes = 4096ull;
    YVEX_TEST_ASSERT(posix_memalign((void **)&imported, 4096u, 4096u) == 0,
                     "allocate page-aligned imported host residency");
    memset(imported, 0x3c, 4096u);
    YVEX_TEST_ASSERT(mlock(imported, 4096u) == 0,
                     "lock imported host residency before CUDA registration");
    mapped = imported;
    YVEX_TEST_ASSERT(yvex_backend_resident_alloc(
                         backend, &descriptor, &resident, &mapped, &err) == YVEX_OK &&
                         mapped == imported,
                     "register existing host residency without replacement");
    YVEX_TEST_ASSERT(yvex_backend_resident_attach(
                         backend, mapped, 4096ull, resident, 1ull, &err) == YVEX_OK &&
                         yvex_backend_resident_resolve(
                             backend, mapped, 4096ull, &mapped_address) ==
                             YVEX_BACKEND_RESIDENT_HIT && mapped_address != 0ull,
                     "resolve registered host residency to its CUDA address");
    YVEX_TEST_ASSERT(yvex_backend_resident_detach(backend, &err) == YVEX_OK &&
                         yvex_backend_tensor_release(backend, &resident, &err) == YVEX_OK,
                     "unregister imported host residency exactly once");
    YVEX_TEST_ASSERT(munlock(imported, 4096u) == 0,
                     "unlock imported host residency after CUDA unregister");
    free(imported);
    imported = mapped = NULL;
    YVEX_TEST_ASSERT(yvex_backend_resident_alloc(
                         backend, &descriptor, &resident, &mapped, &err) == YVEX_OK,
                     "allocate exact managed residency");
    memset(mapped, 0x5a, 4096u);
    YVEX_TEST_ASSERT(setenv("YVEX_TEST_CUDA_PREFETCH_FAILURE", "1", 1) == 0,
                     "install residency prefetch failure injection");
    YVEX_TEST_ASSERT(yvex_backend_resident_prefetch(
                         backend, resident, &prefetched_bytes, &err) == YVEX_ERR_BACKEND &&
                         prefetched_bytes == 0ull,
                     "residency prefetch failure publishes no migrated bytes");
    YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_CUDA_PREFETCH_FAILURE") == 0,
                     "clear residency prefetch failure injection");
    YVEX_TEST_ASSERT(yvex_backend_resident_prefetch(
                         backend, resident, &prefetched_bytes, &err) == YVEX_OK &&
                         prefetched_bytes == 4096ull,
                     "prefetch initialized managed residency to the execution device");
    rc = yvex_backend_resident_attach(backend, mapped, 4096ull, resident, 1ull, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK &&
                         yvex_backend_resident_resolve(
                             backend, mapped, 4096ull, &mapped_address) ==
                             YVEX_BACKEND_RESIDENT_HIT && mapped_address != 0ull,
                     "attach one direct managed CUDA range");
    rc = yvex_backend_resident_attach(backend, mapped, 4096ull, resident, 1ull, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_STATE, "duplicate managed attachment refuses");
    rc = yvex_backend_resident_detach(backend, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK &&
                         yvex_backend_tensor_release(backend, &resident, &err) == YVEX_OK,
                     "release exact CUDA managed residency");
    mapped = NULL;
    if (yvex_backend_resident_map_readonly_supported(backend)) {
        pageable = mmap(NULL, 4096u, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        YVEX_TEST_ASSERT(pageable != MAP_FAILED,
                         "allocate pageable immutable residency fixture");
        memset(pageable, 0x69, 4096u);
        YVEX_TEST_ASSERT(mprotect(pageable, 4096u, PROT_READ) == 0,
                         "seal pageable residency fixture read-only");
        YVEX_TEST_ASSERT(
            yvex_backend_resident_map_readonly(
                backend, &descriptor, pageable, &resident, &err) == YVEX_OK && resident &&
                yvex_backend_resident_attach(
                    backend, pageable, 4096ull, resident, 2ull, &err) == YVEX_OK &&
                yvex_backend_resident_resolve(
                    backend, pageable, 4096ull, &mapped_address) ==
                    YVEX_BACKEND_RESIDENT_HIT && mapped_address != 0ull,
            "register and attach immutable artifact residency without a private copy");
        memset(mapped_readback, 0, sizeof(mapped_readback));
        YVEX_TEST_ASSERT(
            yvex_backend_tensor_read(
                backend, resident, mapped_readback, sizeof(mapped_readback), &err) == YVEX_OK &&
                memcmp(mapped_readback, pageable, sizeof(mapped_readback)) == 0,
            "CUDA reads exact bytes through the host page-table mapping");
        YVEX_TEST_ASSERT(
            yvex_backend_tensor_write(
                backend, resident, mapped_readback, sizeof(mapped_readback), &err) ==
                YVEX_ERR_UNSUPPORTED,
            "immutable pageable residency refuses device writes");
        YVEX_TEST_ASSERT(
            yvex_backend_resident_detach(backend, &err) == YVEX_OK &&
                yvex_backend_tensor_release(backend, &resident, &err) == YVEX_OK &&
                munmap(pageable, 4096u) == 0,
            "release borrowed pageable metadata without freeing artifact bytes");
        pageable = NULL;
    }
    YVEX_TEST_ASSERT(assert_grouped_moe(backend) == 0,
                     "grouped direct-address MoE matches audit execution");
    YVEX_TEST_ASSERT(assert_finite_activation_overflow(backend) == 0,
                     "CUDA normalization retains finite BF16-range activations");
    YVEX_TEST_ASSERT(assert_moe_row_dot_overflow(backend) == 0,
                     "CUDA sparse-row MoE retains finite cancellation");
    YVEX_TEST_ASSERT(assert_encoded_moe(backend) == 0,
                     "compiled encoded MoE is native and fail-closed");
    YVEX_TEST_ASSERT(assert_deferred_attention_completion(backend) == 0,
                     "deferred attention owns one ordered publication barrier");
    YVEX_TEST_ASSERT(assert_deferred_attention_layer_failure(backend) == 0,
                     "deferred attention refusal preserves layer evidence");
    for (rc = 0; rc < (int)YVEX_BACKEND_VARIANT_COUNT; ++rc) {
        YVEX_TEST_ASSERT(assert_supported_variant(
                             backend, (yvex_backend_operation_variant)rc) == 0,
                         "all advertised CUDA variants are exact");
    }
    {
        const yvex_backend_moe_operations *operations =
            yvex_backend_moe_operations_get(backend);
        yvex_moe_row_batch_result completion = {0};
        YVEX_TEST_ASSERT(
            operations && operations->complete_rows &&
                setenv("YVEX_TEST_CUDA_SYNC_FAILURE", "encoded-attention", 1) == 0,
            "inject deferred MoE completion failure");
        rc = operations->complete_rows(backend, 0, &completion, &err);
        YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_CUDA_SYNC_FAILURE") == 0,
                         "clear deferred MoE completion failure");
        YVEX_TEST_ASSERT(
            rc == YVEX_ERR_BACKEND && !completion.completed &&
                strstr(yvex_error_message(&err), "synchronization failure") != NULL,
            "deferred MoE completion fails closed before physical facts publish");
    }
    yvex_backend_close(backend);
    backend = NULL;
    YVEX_TEST_ASSERT(assert_deferred_attention_sync_failure() == 0,
                     "deferred attention synchronization failure is isolated and typed");
    YVEX_TEST_ASSERT(assert_bundle_rollback(
                         "module",
                         YVEX_BACKEND_CAPABILITY_REASON_KERNEL_BUNDLE_REJECTED,
                         YVEX_BACKEND_VARIANT_EMBED_F32_TO_F32) == 0,
                     "module failure rolls back atomically");
    YVEX_TEST_ASSERT(assert_bundle_rollback(
                         "symbol",
                         YVEX_BACKEND_CAPABILITY_REASON_FUNCTION_MISSING,
                         YVEX_BACKEND_VARIANT_EMBED_F32_TO_F32) == 0,
                     "symbol failure rolls back atomically");
    for (symbol_index = 0u;
         symbol_index < sizeof(attention_symbols) / sizeof(attention_symbols[0]);
         ++symbol_index) {
        YVEX_TEST_ASSERT(assert_bundle_rollback(
                             attention_symbols[symbol_index],
                             YVEX_BACKEND_CAPABILITY_REASON_FUNCTION_MISSING,
                             YVEX_BACKEND_VARIANT_ATTENTION_ENCODED) == 0,
                         "each encoded-attention symbol is atomically required");
    }
    backend = NULL;
    rc = yvex_backend_open(&backend, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "CUDA bundle admission retries after rollback");
    YVEX_TEST_ASSERT(assert_supported_variant(
                         backend, YVEX_BACKEND_VARIANT_EMBED_F32_TO_F32) == 0,
                     "retry resolves canonical function");
    yvex_backend_close(backend);
    return 0;
}
