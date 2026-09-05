/* Falsify selective SSD prefill/decode continuity and common state ownership. */
#include "tests/test.h"
#include <yvex/internal/state_space.h>
#include <yvex/internal/sequence_state.h>
#include <math.h>
#include <stdlib.h>

typedef struct {
    float projection[112], weight[48], bias[16], decay[4], skip[4], dt[4], norm[8];
    float convolution[2][48], recurrent[2][16], workspace[16], output[32];
} ssd_fixture;

static void ssd_fixture_init(ssd_fixture *f)
{
    size_t i;
    memset(f, 0, sizeof(*f));
    for (i = 0; i < 112u; ++i) f->projection[i] = ((float)(i % 17u) - 8.0f) * 0.125f;
    for (i = 0; i < 48u; ++i) f->weight[i] = ((float)(i % 7u) - 3.0f) * 0.125f;
    for (i = 0; i < 16u; ++i) f->bias[i] = ((float)i - 7.0f) * 0.03125f;
    for (i = 0; i < 4u; ++i) {
        f->decay[i] = (float)i * 0.125f;
        f->skip[i] = 0.5f + (float)i * 0.25f;
        f->dt[i] = -0.5f + (float)i * 0.125f;
    }
    for (i = 0; i < 8u; ++i) f->norm[i] = 0.75f + (float)i * 0.0625f;
}

static yvex_selective_ssd_cpu_request ssd_request(ssd_fixture *f)
{
    return (yvex_selective_ssd_cpu_request){
        .token_count = 4u, .projection = f->projection, .projection_capacity = 112u,
        .convolution_weight = f->weight, .convolution_weight_capacity = 48u,
        .convolution_bias = f->bias, .convolution_bias_capacity = 16u,
        .decay_log = f->decay, .decay_log_capacity = 4u,
        .skip = f->skip, .skip_capacity = 4u, .time_bias = f->dt, .time_bias_capacity = 4u,
        .normalization_weight = f->norm, .normalization_weight_capacity = 8u,
        .next_state = {f->convolution[0], 48u, f->recurrent[0], 16u},
        .workspace = f->workspace, .workspace_capacity = 16u,
        .output = f->output, .output_capacity = 32u};
}

static int ssd_near(const float *a, const float *b, size_t count)
{
    size_t i;
    for (i = 0; i < count; ++i)
        if (!isfinite(a[i]) || !isfinite(b[i]) ||
            fabsf(a[i] - b[i]) > 3e-6f + 3e-5f * fabsf(b[i])) return 0;
    return 1;
}

static int ssd_oracle(const ssd_fixture *f)
{
    const char *path = getenv("YVEX_SELECTIVE_SSD_ORACLE");
    float values[96];
    FILE *fp;
    size_t i;
    int valid;
    if (!path || !path[0]) return 0;
    fp = fopen(path, "rb");
    YVEX_TEST_ASSERT(fp != NULL, "open explicit external oracle");
    for (i = 0; i < 96u; ++i) {
        if (fscanf(fp, "%f", &values[i]) != 1) {
            fclose(fp);
            YVEX_TEST_FAIL("oracle output is truncated");
        }
    }
    valid = ssd_near(f->output, values, 32u) &&
        ssd_near(f->convolution[0], values + 32u, 48u) &&
        ssd_near(f->recurrent[0], values + 80u, 16u);
    fclose(fp);
    YVEX_TEST_ASSERT(valid, "prefill output AND both final states match external oracle");
    return 0;
}

static int ssd_cancel(void *context)
{
    unsigned int *calls = context;
    return (*calls)++ != 0u;
}

static float *ssd_read_values(FILE *fp, unsigned long long count)
{
    float *values;
    unsigned long long i;
    if (!count || count > 4u * 1024u * 1024u) return NULL;
    values = malloc((size_t)count * sizeof(float));
    if (!values) return NULL;
    for (i = 0; i < count; ++i)
        if (fscanf(fp, "%f", values + i) != 1 || !isfinite(values[i])) {
            free(values);
            return NULL;
        }
    return values;
}

static void ssd_error_report(const char *label, const float *actual, const float *expected,
                              unsigned long long count)
{
    unsigned long long i;
    double maximum = 0.0, squares = 0.0;
    for (i = 0; i < count; ++i) {
        double error = (double)actual[i] - expected[i];
        maximum = fmax(maximum, fabs(error));
        squares += error * error;
    }
    printf("ssd-real-%s: values=%llu max_abs=%.9g rms=%.9g\n",
           label, count, maximum, sqrt(squares / (double)count));
}

static int ssd_real_component(void)
{
    const char *path = getenv("YVEX_SELECTIVE_SSD_REAL_ORACLE");
    yvex_selective_ssd_requirement q = {.schema_version = YVEX_SELECTIVE_SSD_SCHEMA_V1,
                                       .time_step_unbounded = 1};
    yvex_selective_ssd_geometry g;
    yvex_selective_ssd_cpu_request request = {0};
    yvex_selective_ssd_cpu_result result;
    yvex_error err;
    unsigned long long tokens, sizes[10];
    float *values[10] = {0}, *output = NULL, *conv = NULL, *state = NULL, *workspace = NULL;
    FILE *fp;
    int i, valid = 0;
    if (!path || !path[0]) return 0;
    fp = fopen(path, "rb");
    YVEX_TEST_ASSERT(fp != NULL, "open explicit acquired-tensor oracle fixture");
    if (fscanf(fp, "%llu %llu %llu %llu %llu %llu %llu %d %lf", &q.heads,
            &q.head_dimension, &q.state_dimension, &q.groups, &q.convolution_kernel,
            &tokens, &q.normalization_groups, &q.norm_before_gate, &q.normalization_epsilon) != 9 ||
        tokens > 16u || !tokens || yvex_selective_ssd_geometry_seal(&g, &q, &err) != YVEX_OK ||
        g.projection_width > 262144u || g.recurrent_state_values > 4194304u ||
        g.convolution_state_values > 4194304u) goto cleanup;
    sizes[0] = tokens * g.projection_width; sizes[1] = g.convolution_state_values;
    sizes[2] = g.convolution_width; sizes[3] = sizes[4] = sizes[5] = q.heads;
    sizes[6] = g.width; sizes[7] = tokens * g.width;
    sizes[8] = g.convolution_state_values; sizes[9] = g.recurrent_state_values;
    for (i = 0; i < 10; ++i) if (!(values[i] = ssd_read_values(fp, sizes[i]))) goto cleanup;
    output = calloc((size_t)sizes[7], sizeof(float));
    conv = calloc((size_t)sizes[8], sizeof(float));
    state = calloc((size_t)sizes[9], sizeof(float));
    workspace = calloc((size_t)g.convolution_width, sizeof(float));
    if (!output || !conv || !state || !workspace) goto cleanup;
    request = (yvex_selective_ssd_cpu_request){.token_count = tokens,
        .projection = values[0], .projection_capacity = sizes[0],
        .convolution_weight = values[1], .convolution_weight_capacity = sizes[1],
        .convolution_bias = values[2], .convolution_bias_capacity = sizes[2],
        .decay_log = values[3], .decay_log_capacity = sizes[3],
        .skip = values[4], .skip_capacity = sizes[4],
        .time_bias = values[5], .time_bias_capacity = sizes[5],
        .normalization_weight = values[6], .normalization_weight_capacity = sizes[6],
        .next_state = {conv, sizes[8], state, sizes[9]},
        .workspace = workspace, .workspace_capacity = g.convolution_width,
        .output = output, .output_capacity = sizes[7]};
    valid = yvex_selective_ssd_execute_cpu(&g, &request, &result, &err) == YVEX_OK &&
        ssd_near(output, values[7], (size_t)sizes[7]) &&
        ssd_near(conv, values[8], (size_t)sizes[8]) && ssd_near(state, values[9], (size_t)sizes[9]);
    if (valid) {
        ssd_error_report("output", output, values[7], sizes[7]);
        ssd_error_report("conv", conv, values[8], sizes[8]);
        ssd_error_report("state", state, values[9], sizes[9]);
        /* This measures policy disagreement, not correctness of the alternative. */
        q.normalization_groups = q.groups;
        valid = yvex_selective_ssd_geometry_seal(&g, &q, &err) == YVEX_OK &&
            yvex_selective_ssd_execute_cpu(&g, &request, &result, &err) == YVEX_OK;
        if (valid) ssd_error_report("grouped-policy-difference", output, values[7], sizes[7]);
    }
cleanup:
    fclose(fp);
    for (i = 0; i < 10; ++i) free(values[i]);
    free(output); free(conv); free(state); free(workspace);
    YVEX_TEST_ASSERT(valid, "real tensor full geometry: output AND conv/SSM state match external oracle");
    return 0;
}

static int ssd_transaction(const yvex_selective_ssd_geometry *plan, ssd_fixture *f)
{
    yvex_sequence_state_binding binding;
    yvex_sequence_state_plan state_plan;
    yvex_sequence_state *state = NULL, *other = NULL;
    yvex_sequence_state_summary summary;
    yvex_sequence_state_view committed, independent;
    yvex_selective_ssd_cpu_request request = ssd_request(f);
    yvex_selective_ssd_cpu_result result;
    yvex_runtime_transaction_participant participant;
    yvex_error err;
    unsigned int calls = 0;

    YVEX_TEST_ASSERT(yvex_sequence_state_binding_seal(
        &binding, 0, plan->convolution_state_values, plan->recurrent_state_values,
        plan->identity, &err) == YVEX_OK, "seal SSD allocation without a delta plan");
    state_plan = (yvex_sequence_state_plan){YVEX_SEQUENCE_STATE_SCHEMA_V1, &binding, 1u};
    YVEX_TEST_ASSERT(yvex_sequence_state_open(&state, &state_plan, &err) == YVEX_OK &&
        yvex_sequence_state_open(&other, &state_plan, &err) == YVEX_OK,
        "two SSD owners coexist without KV storage");
    YVEX_TEST_ASSERT(yvex_sequence_state_begin(state, 0, 4, &err) == YVEX_OK &&
        yvex_sequence_state_layer(state, 0, &request.state, &request.next_state, &err) == YVEX_OK &&
        yvex_selective_ssd_execute_cpu(plan, &request, &result, &err) == YVEX_OK &&
        yvex_sequence_state_stage(state, 0, &err) == YVEX_OK &&
        yvex_sequence_state_participant(state, &participant, &err) == YVEX_OK &&
        yvex_runtime_transaction_resolve(&participant, 1u, YVEX_OK, &err) == YVEX_OK,
        "prefill publishes through existing transactions");
    YVEX_TEST_ASSERT(yvex_sequence_state_committed(state, 0, &committed, &err) == YVEX_OK &&
        yvex_sequence_state_committed(other, 0, &independent, &err) == YVEX_OK &&
        independent.recurrent != committed.recurrent && independent.recurrent[0] == 0.0f &&
        ssd_near(committed.recurrent, f->recurrent[0], 16u), "sessions do not alias");
    YVEX_TEST_ASSERT(yvex_sequence_state_begin(state, 4, 4, &err) == YVEX_OK &&
        yvex_sequence_state_layer(state, 0, &request.state, &request.next_state, &err) == YVEX_OK,
        "open candidate after committed prefill");
    request.cancel_requested = ssd_cancel;
    request.cancel_context = &calls;
    YVEX_TEST_ASSERT(yvex_selective_ssd_execute_cpu(plan, &request, &result, &err) ==
        YVEX_ERR_CANCELLED && result.cancelled && result.completed_tokens == 1u &&
        yvex_sequence_state_participant(state, &participant, &err) == YVEX_OK &&
        yvex_runtime_transaction_resolve(&participant, 1u, YVEX_ERR_CANCELLED, &err) ==
        YVEX_ERR_CANCELLED && ssd_near(committed.recurrent, f->recurrent[0], 16u),
        "cancellation after mutation discards only candidate state");
    YVEX_TEST_ASSERT(yvex_sequence_state_summary_copy(state, &summary, &err) == YVEX_OK &&
        summary.committed_position == 4u && summary.convolution_state_bytes == 192u &&
        summary.recurrent_state_bytes == 64u && summary.committed_state_bytes == 256u &&
        summary.candidate_state_bytes == 256u, "state resources reconcile");
    YVEX_TEST_ASSERT(yvex_sequence_state_reset(state, &err) == YVEX_OK &&
        yvex_sequence_state_committed(state, 0, &committed, &err) == YVEX_OK &&
        committed.recurrent[0] == 0.0f && committed.convolution[0] == 0.0f,
        "reset destroys continuity");
    yvex_sequence_state_close(&state);
    yvex_sequence_state_close(&other);
    YVEX_TEST_ASSERT(!state && !other, "cleanup retires both owners");
    return 0;
}

int yvex_test_selective_ssd(void)
{
    ssd_fixture prefill, decode;
    yvex_selective_ssd_requirement requirement = {
        .schema_version = YVEX_SELECTIVE_SSD_SCHEMA_V1,
        .heads = 4u, .head_dimension = 2u, .state_dimension = 2u, .groups = 2u,
        .convolution_kernel = 3u, .normalization_groups = 1u,
        .normalization_epsilon = 1e-5, .time_step_unbounded = 1};
    yvex_selective_ssd_geometry plan, invalid;
    yvex_selective_ssd_cpu_request request;
    yvex_selective_ssd_cpu_result result;
    yvex_error err;
    unsigned int token, bank = 0;
    ssd_fixture_init(&prefill);
    ssd_fixture_init(&decode);
    YVEX_TEST_ASSERT(yvex_selective_ssd_geometry_seal(&plan, &requirement, &err) == YVEX_OK &&
        plan.width == 8u && plan.projection_width == 28u &&
        plan.convolution_state_values == 48u && plan.recurrent_state_values == 16u,
        "derive grouped scalar-transition geometry");
    request = ssd_request(&prefill);
    YVEX_TEST_ASSERT(yvex_selective_ssd_execute_cpu(&plan, &request, &result, &err) == YVEX_OK &&
        result.complete && result.completed_tokens == 4u && result.state_updates == 64u,
        "multi-token scan completes");
    YVEX_TEST_ASSERT(ssd_oracle(&prefill) == 0, "external oracle when explicitly supplied");
    for (token = 0; token < 4u; ++token) {
        request = ssd_request(&decode);
        request.token_count = 1;
        request.projection += token * 28u;
        request.projection_capacity = 28u;
        request.output += token * 8u;
        request.output_capacity = 8u;
        if (token) request.state = (yvex_sequence_state_view){
            decode.convolution[bank ^ 1u], decode.recurrent[bank ^ 1u]};
        request.next_state = (yvex_sequence_state_output){
            decode.convolution[bank], 48u, decode.recurrent[bank], 16u};
        YVEX_TEST_ASSERT(yvex_selective_ssd_execute_cpu(&plan, &request, &result, &err) == YVEX_OK,
            "decode consumes only retained state");
        bank ^= 1u;
    }
    YVEX_TEST_ASSERT(ssd_near(prefill.output, decode.output, 32u) &&
        ssd_near(prefill.recurrent[0], decode.recurrent[bank ^ 1u], 16u) &&
        ssd_near(prefill.convolution[0], decode.convolution[bank ^ 1u], 48u),
        "scan/decode agree on output and both state classes");
    invalid = plan;
    invalid.recurrent_state_values++;
    YVEX_TEST_ASSERT(yvex_selective_ssd_geometry_validate(&invalid, &err) != YVEX_OK,
        "stale geometry rejected");
    requirement.groups = 3u;
    YVEX_TEST_ASSERT(yvex_selective_ssd_geometry_seal(&invalid, &requirement, &err) != YVEX_OK,
        "wrong group geometry rejected");
    request = ssd_request(&decode);
    request.state = (yvex_sequence_state_view){decode.convolution[0], decode.recurrent[0]};
    YVEX_TEST_ASSERT(yvex_selective_ssd_execute_cpu(&plan, &request, &result, &err) != YVEX_OK &&
        !result.completed_tokens, "committed alias rejected before execution");
    YVEX_TEST_ASSERT(ssd_transaction(&plan, &prefill) == 0, "common SSD lifecycle");
    YVEX_TEST_ASSERT(ssd_real_component() == 0, "real acquired tensor oracle when supplied");
    return 0;
}
