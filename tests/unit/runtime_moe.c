/*
 * Exercises the bounded production MoE contracts and their semantic mutation sensitivity.
 * References do not call production routing, top-k, activation, or accumulation algorithms.
 * Focused internal-ABI evidence; no fixture enters production objects.
 */
#define _GNU_SOURCE
#include "tests/test.h"

#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <yvex/internal/moe.h>
#include <yvex/internal/quant_numeric.h>

static void moe_test_identity(char output[YVEX_SHA256_HEX_CAP], unsigned int value)
{
    (void)snprintf(output, YVEX_SHA256_HEX_CAP, "%064x", value);
}

static int moe_test_family_plan(void)
{
    const yvex_moe_family_api *family = yvex_graph_moe_family_at(0ull);
    yvex_runtime_descriptor_summary runtime = {0};
    yvex_attention_layer_plan attention = {0};
    yvex_error err;
    unsigned long long layer;

    YVEX_TEST_ASSERT(family && family->project_layer,
                     "DeepSeek MoE family adapter is registered");
    runtime.layer_count = 43ull;
    runtime.routed_experts = 256ull;
    runtime.experts_per_token = 6ull;
    runtime.vocabulary_size = 129280ull;
    runtime.model_execution.schema_version = YVEX_MODEL_EXECUTION_DESCRIPTOR_SCHEMA_V1;
    runtime.model_execution.hash_router_layer_count = 3ull;
    runtime.model_execution.shared_experts = 1ull;
    runtime.model_execution.routed_ffn_width = 2048ull;
    runtime.model_execution.shared_ffn_width = 2048ull;
    runtime.model_execution.routed_scaling_factor = 1.5;
    runtime.model_execution.activation_limit = 10.0;
    attention.hidden_dimension = 4096ull;
    attention.residual_stream_count = 4ull;
    attention.residual_expanded_width = 16384ull;
    attention.mhc_mixing_rows = 24ull;
    attention.mhc_sinkhorn_iterations = 20ull;
    attention.rms_norm_epsilon = 1e-6;
    attention.mhc_epsilon = 1e-6;
    attention.mhc_residual_post_multiplier = 2.0;
    attention.tensor_scope = YVEX_TENSOR_SCOPE_MAIN_LAYER;
    attention.predictor_index = YVEX_MATERIALIZATION_NO_INDEX;
    for (layer = 0ull; layer < runtime.layer_count; ++layer) {
        yvex_moe_layer_plan projected;
        attention.ordinal = layer;
        attention.layer_index = layer;
        YVEX_TEST_ASSERT(
            family->project_layer(layer, &runtime, &attention, &projected, &err) == YVEX_OK,
            "every DeepSeek main layer projects an exact MoE policy");
        YVEX_TEST_ASSERT(
            projected.ordinal == layer && projected.layer_index == layer &&
                projected.router_class == (layer < 3ull ? YVEX_MOE_ROUTER_HASH_TOKEN_ID
                                                        : YVEX_MOE_ROUTER_LEARNED_HIDDEN_STATE) &&
                projected.routed_experts == 256ull && projected.shared_experts == 1ull &&
                projected.experts_per_token == 6ull &&
                projected.expert_intermediate_width == 2048ull &&
                projected.routed_scaling_factor == 1.5 && projected.activation_limit == 10.0,
            "projected layer preserves hash/learned schedule and expert geometry");
    }
    memset(&runtime.model_execution, 0, sizeof(runtime.model_execution));
    attention.ordinal = attention.layer_index = 0ull;
    {
        yvex_moe_layer_plan rejected;
        YVEX_TEST_ASSERT(
            family->project_layer(0ull, &runtime, &attention, &rejected, &err) ==
                YVEX_ERR_INVALID_ARG,
            "MoE projection refuses an uncompiled execution descriptor");
    }
    return 0;
}

static yvex_moe_weight_view moe_test_weight(float *values,
                                             unsigned long long width,
                                             unsigned long long rows)
{
    yvex_moe_weight_view view = {0};
    view.qtype = YVEX_GGUF_QTYPE_F32;
    view.encoded = (const unsigned char *)values;
    view.row_width = width;
    view.row_count = rows;
    view.row_bytes = width * sizeof(float);
    view.encoded_bytes = (size_t)(view.row_bytes * rows);
    return view;
}

static double moe_test_score(double value)
{
    double softplus = value > 0.0 ? value + log1p(exp(-value)) : log1p(exp(value));
    return sqrt(softplus);
}

static int moe_test_routing(void)
{
    yvex_moe_layer_plan layer = {0};
    yvex_moe_layer_job job = {0};
    yvex_moe_router_result result;
    float router[] = {1.0f, 0.0f, 0.0f, 0.0f,
                      2.0f, 0.0f, 0.0f, 0.0f,
                      2.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f};
    float bias[] = {5.0f, 0.0f, 0.0f, 0.0f};
    float input[] = {1.0f, 0.0f, 0.0f, 0.0f};
    int32_t table[] = {3, 1};
    double first, second;
    yvex_error err;

    layer.hidden_width = 4ull;
    layer.routed_experts = 4ull;
    layer.experts_per_token = 2ull;
    layer.routed_scaling_factor = 1.5;
    layer.normalize_topk_probabilities = 1;
    job.layer = &layer;
    job.expanded_input = input;
    job.weights[YVEX_MOE_WEIGHT_ROUTER] = moe_test_weight(router, 4ull, 4ull);
    job.weights[YVEX_MOE_WEIGHT_ROUTER_BIAS] = moe_test_weight(bias, 4ull, 1ull);
    layer.router_class = YVEX_MOE_ROUTER_LEARNED_HIDDEN_STATE;
    YVEX_TEST_ASSERT(yvex_moe_route_cpu(&job, input, &result, &err) == YVEX_OK,
                     "learned sqrt-softplus router executes");
    YVEX_TEST_ASSERT(result.selected_experts[0] == 0ull &&
                         result.selected_experts[1] == 1ull,
                     "correction bias selects expert zero and ordinal tie selects expert one");
    first = moe_test_score(1.0);
    second = moe_test_score(2.0);
    YVEX_TEST_ASSERT(fabs(result.selected_weights[0] - first / (first + second) * 1.5) < 1e-6 &&
                         fabs(result.selected_weights[1] - second / (first + second) * 1.5) < 1e-6,
                     "selection bias does not alter normalized routed weights");
    bias[0] = 0.0f;
    YVEX_TEST_ASSERT(yvex_moe_route_cpu(&job, input, &result, &err) == YVEX_OK &&
                         result.selected_experts[0] == 1ull &&
                         result.selected_experts[1] == 2ull,
                     "bias mutation changes selection and preserves deterministic ties");

    layer.router_class = YVEX_MOE_ROUTER_HASH_TOKEN_ID;
    layer.hash_table_rows = 8ull;
    job.token_id = 4u;
    job.token_id_present = 1;
    job.weights[YVEX_MOE_WEIGHT_ROUTER_TABLE] = (yvex_moe_weight_view){
        .qtype = YVEX_GGUF_QTYPE_I32, .encoded = (const unsigned char *)table,
        .encoded_bytes = sizeof(table), .row_bytes = sizeof(table),
        .row_width = 2ull, .row_count = 1ull};
    YVEX_TEST_ASSERT(yvex_moe_route_cpu(&job, input, &result, &err) == YVEX_OK &&
                         result.selected_experts[0] == 3ull &&
                         result.selected_experts[1] == 1ull,
                     "hash router consumes the exact token-selected table row");
    table[1] = 3;
    YVEX_TEST_ASSERT(yvex_moe_route_cpu(&job, input, &result, &err) == YVEX_ERR_FORMAT,
                     "duplicate hash-selected experts refuse");
    job.token_id = 8u;
    YVEX_TEST_ASSERT(yvex_moe_route_cpu(&job, input, &result, &err) == YVEX_ERR_BOUNDS,
                     "out-of-vocabulary hash input refuses");
    return 0;
}

static float moe_test_bf16(float value)
{
    return yvex_quant_bf16_decode(yvex_quant_bf16_encode(value));
}

static int moe_test_expert(void)
{
    yvex_moe_layer_plan layer = {0};
    float gate[] = {1.0f, 2.0f, -1.0f, 1.0f};
    float up[] = {2.0f, 0.0f, 1.0f, 3.0f};
    float down[] = {1.0f, 0.5f, -0.25f, 2.0f};
    float input[] = {1.0f, 2.0f}, output[2], expected[2], intermediate[2];
    yvex_moe_weight_view gate_view = moe_test_weight(gate, 2ull, 2ull);
    yvex_moe_weight_view up_view = moe_test_weight(up, 2ull, 2ull);
    yvex_moe_weight_view down_view = moe_test_weight(down, 2ull, 2ull);
    yvex_error err;
    unsigned int row;

    layer.hidden_width = 2ull;
    layer.activation_limit = 10.0;
    for (row = 0u; row < 2u; ++row) {
        double g = gate[row * 2u] * input[0] + gate[row * 2u + 1u] * input[1];
        double u = up[row * 2u] * input[0] + up[row * 2u + 1u] * input[1];
        intermediate[row] = moe_test_bf16((float)((g / (1.0 + exp(-g))) * u));
    }
    expected[0] = moe_test_bf16(down[0] * intermediate[0] + down[1] * intermediate[1]);
    expected[1] = moe_test_bf16(down[2] * intermediate[0] + down[3] * intermediate[1]);
    YVEX_TEST_ASSERT(yvex_moe_expert_cpu(&layer, &gate_view, &up_view, &down_view,
                                         input, output, &err) == YVEX_OK,
                     "encoded selected expert executes");
    YVEX_TEST_ASSERT(output[0] == expected[0] && output[1] == expected[1],
                     "selected expert matches independent BF16 SwiGLU equation");
    up_view.row_count = 1ull;
    YVEX_TEST_ASSERT(yvex_moe_expert_cpu(&layer, &gate_view, &up_view, &down_view,
                                         input, output, &err) == YVEX_ERR_INVALID_ARG,
                     "malformed expert geometry refuses");
    return 0;
}

typedef struct {
    yvex_moe_input_summary summary;
    yvex_moe_input_layer_record records[2];
    float activations[16];
    unsigned int token_ids[2];
} moe_input_fixture;

static void moe_input_fixture_open(moe_input_fixture *fixture)
{
    unsigned long long index;
    memset(fixture, 0, sizeof(*fixture));
    fixture->summary.schema_version = YVEX_MOE_INPUT_SCHEMA_V1;
    fixture->summary.token_start = 5ull;
    fixture->summary.token_count = 2ull;
    fixture->summary.layer_count = 2ull;
    fixture->summary.activation_payload_bytes = sizeof(fixture->activations);
    fixture->summary.token_id_payload_bytes = sizeof(fixture->token_ids);
    moe_test_identity(fixture->summary.logical_model_identity, 1u);
    moe_test_identity(fixture->summary.runtime_numeric_identity, 2u);
    moe_test_identity(fixture->summary.runtime_descriptor_identity, 3u);
    moe_test_identity(fixture->summary.moe_plan_identity, 4u);
    for (index = 0ull; index < 2ull; ++index) {
        fixture->records[index].ordinal = fixture->records[index].layer_index = index;
        fixture->records[index].width = fixture->records[index].stride = 4ull;
        fixture->records[index].payload_offset = index * 8ull * sizeof(float);
        fixture->records[index].payload_bytes = 8ull * sizeof(float);
        moe_test_identity(fixture->records[index].layer_identity,
                          (unsigned int)(10ull + index));
    }
    for (index = 0ull; index < 16ull; ++index)
        fixture->activations[index] = (float)(index + 1ull) / 32.0f;
    fixture->token_ids[0] = 17u;
    fixture->token_ids[1] = 19u;
}

static int moe_input_append(const char *path)
{
    const unsigned char byte = 0xa5u;
    int fd = open(path, O_WRONLY | O_APPEND | O_CLOEXEC);
    int ok = fd >= 0 && write(fd, &byte, 1u) == 1;
    if (fd >= 0 && close(fd) != 0) ok = 0;
    return ok;
}

/* Prove memory/file identity, bounded views, malformed input, and secure cleanup. */
static int moe_test_input(void)
{
    moe_input_fixture fixture, changed;
    yvex_moe_input *input = NULL;
    const float *view = NULL;
    unsigned long long stride = 0ull;
    yvex_moe_input_limits limits = {64ull * 1024ull};
    char root[] = "/tmp/yvex-runtime-moe.XXXXXX";
    char path[512], trailing[512], link_path[512];
    yvex_error err;
    int rc;

    moe_input_fixture_open(&fixture);
    rc = yvex_moe_input_seal(&fixture.summary, fixture.records, fixture.activations,
                             fixture.token_ids, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && yvex_sha256_hex_valid(fixture.summary.input_identity),
                     "typed MoE input seals with canonical identities");
    rc = yvex_moe_input_open_memory(&input, &fixture.summary, fixture.records,
                                    fixture.activations, fixture.token_ids, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && input, "sealed MoE memory input admits");
    YVEX_TEST_ASSERT(yvex_moe_input_layer_view(input, 1ull, &view, &stride, &err) == YVEX_OK &&
                         stride == 4ull && view[0] == fixture.activations[8],
                     "checked layer view resolves exact activation span");
    YVEX_TEST_ASSERT(yvex_moe_input_token_ids(input)[1] == 19u,
                     "typed token ID payload is preserved");
    yvex_moe_input_close(&input);
    yvex_moe_input_close(&input);
    changed = fixture;
    changed.records[1].ordinal = 0ull;
    YVEX_TEST_ASSERT(yvex_moe_input_seal(&changed.summary, changed.records,
                                         changed.activations, changed.token_ids,
                                         &err) == YVEX_ERR_FORMAT,
                     "duplicate/reordered layers refuse");
    changed = fixture;
    changed.activations[3] = NAN;
    YVEX_TEST_ASSERT(yvex_moe_input_seal(&changed.summary, changed.records,
                                         changed.activations, changed.token_ids,
                                         &err) == YVEX_ERR_FORMAT,
                     "non-finite activation input refuses");

    YVEX_TEST_ASSERT(mkdtemp(root) != NULL &&
                         snprintf(path, sizeof(path), "%s/input%s", root,
                                  YVEX_MOE_INPUT_SUFFIX) < (int)sizeof(path) &&
                         snprintf(trailing, sizeof(trailing), "%s/trailing%s", root,
                                  YVEX_MOE_INPUT_SUFFIX) < (int)sizeof(trailing) &&
                         snprintf(link_path, sizeof(link_path), "%s/link%s", root,
                                  YVEX_MOE_INPUT_SUFFIX) < (int)sizeof(link_path),
                     "MoE temporary paths fit");
    YVEX_TEST_ASSERT(yvex_moe_input_write(path, &fixture.summary, fixture.records,
                                          fixture.activations, fixture.token_ids,
                                          &err) == YVEX_OK &&
                         yvex_moe_input_open_file(&input, path, &limits, &err) == YVEX_OK,
                     "bounded regular MoE tensor file admits");
    yvex_moe_input_close(&input);
    YVEX_TEST_ASSERT(yvex_moe_input_write(trailing, &fixture.summary, fixture.records,
                                          fixture.activations, fixture.token_ids,
                                          &err) == YVEX_OK && moe_input_append(trailing),
                     "trailing-data fixture publishes");
    YVEX_TEST_ASSERT(yvex_moe_input_open_file(&input, trailing, &limits, &err) == YVEX_ERR_FORMAT,
                     "trailing input bytes refuse");
    YVEX_TEST_ASSERT(symlink(path, link_path) == 0 &&
                         yvex_moe_input_open_file(&input, link_path, &limits, &err) == YVEX_ERR_IO,
                     "symlink MoE input refuses");
    YVEX_TEST_ASSERT(unlink(link_path) == 0 && unlink(trailing) == 0 &&
                         unlink(path) == 0 && rmdir(root) == 0,
                     "MoE input fixtures clean exactly");
    return 0;
}

int yvex_test_runtime_moe(void)
{
    if (moe_test_family_plan() != 0) return 1;
    if (moe_test_routing() != 0) return 1;
    if (moe_test_expert() != 0) return 1;
    if (moe_test_input() != 0) return 1;
    return 0;
}
