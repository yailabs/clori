/* CUDA gated-delta execution is compared with the portable exact F32 semantic owner. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <yvex/api.h>
#include <yvex/internal/sequence_mixer.h>
#include <yvex/internal/transformer.h>

#include "tests/test.h"

typedef struct {
    float *qkv, *output_gate, *beta, *decay;
    float *convolution_weight, *decay_log, *time_bias, *normalization_weight;
    float *committed_convolution, *committed_recurrent;
    float *candidate_convolution, *candidate_recurrent, *output;
    float *observed_convolution, *observed_recurrent, *observed_output;
} gated_delta_host;

typedef struct {
    yvex_device_tensor *qkv, *output_gate, *beta, *decay;
    yvex_device_tensor *convolution_weight, *decay_log, *time_bias;
    yvex_device_tensor *normalization_weight;
    yvex_device_tensor *committed_convolution, *committed_recurrent;
    yvex_device_tensor *candidate_convolution, *candidate_recurrent, *output;
} gated_delta_device;

static int gated_delta_allocate(float **values, unsigned long long count)
{
    if (!values || !count || count > SIZE_MAX / sizeof(float)) return 0;
    *values = (float *)calloc((size_t)count, sizeof(float));
    return *values != NULL;
}

static int gated_delta_host_open(const yvex_gated_delta_plan *plan,
                                 unsigned long long tokens, int committed,
                                 gated_delta_host *host)
{
    unsigned long long qkv_values = tokens * plan->qkv_width;
    unsigned long long output_values = tokens * plan->value_width;
    unsigned long long head_values = tokens * plan->requirement.value_heads;
    unsigned long long weight_values =
        plan->qkv_width * plan->requirement.convolution_kernel;
    memset(host, 0, sizeof(*host));
#define ALLOCATE(field, count) \
    do { if (!gated_delta_allocate(&host->field, (count))) return 0; } while (0)
    ALLOCATE(qkv, qkv_values);
    ALLOCATE(output_gate, output_values);
    ALLOCATE(beta, head_values);
    ALLOCATE(decay, head_values);
    ALLOCATE(convolution_weight, weight_values);
    ALLOCATE(decay_log, plan->requirement.value_heads);
    ALLOCATE(time_bias, plan->requirement.value_heads);
    ALLOCATE(normalization_weight, plan->requirement.value_head_dimension);
    if (committed) {
        ALLOCATE(committed_convolution, plan->convolution_state_values);
        ALLOCATE(committed_recurrent, plan->recurrent_state_values);
    }
    ALLOCATE(candidate_convolution, plan->convolution_state_values);
    ALLOCATE(candidate_recurrent, plan->recurrent_state_values);
    ALLOCATE(output, output_values);
    ALLOCATE(observed_convolution, plan->convolution_state_values);
    ALLOCATE(observed_recurrent, plan->recurrent_state_values);
    ALLOCATE(observed_output, output_values);
#undef ALLOCATE
    return 1;
}

static void gated_delta_host_close(gated_delta_host *host)
{
    if (!host) return;
    free(host->qkv); free(host->output_gate); free(host->beta); free(host->decay);
    free(host->convolution_weight); free(host->decay_log); free(host->time_bias);
    free(host->normalization_weight); free(host->committed_convolution);
    free(host->committed_recurrent); free(host->candidate_convolution);
    free(host->candidate_recurrent); free(host->output);
    free(host->observed_convolution); free(host->observed_recurrent);
    free(host->observed_output);
    memset(host, 0, sizeof(*host));
}

static void gated_delta_seed(const yvex_gated_delta_plan *plan,
                             unsigned long long tokens, gated_delta_host *host)
{
    unsigned long long index, channel, tap;
    for (index = 0ull; index < tokens * plan->qkv_width; ++index)
        host->qkv[index] = (float)((int)((index * 13ull + 5ull) % 31ull) - 15) * 0.0125f;
    for (index = 0ull; index < tokens * plan->value_width; ++index)
        host->output_gate[index] =
            (float)((int)((index * 7ull + 3ull) % 23ull) - 11) * 0.025f;
    for (index = 0ull; index < tokens * plan->requirement.value_heads; ++index) {
        host->beta[index] =
            (float)((int)((index * 5ull + 1ull) % 17ull) - 8) * 0.04f;
        host->decay[index] =
            (float)((int)((index * 11ull + 2ull) % 19ull) - 9) * 0.03f;
    }
    for (channel = 0ull; channel < plan->qkv_width; ++channel)
        for (tap = 0ull; tap < plan->requirement.convolution_kernel; ++tap)
            host->convolution_weight[
                channel * plan->requirement.convolution_kernel + tap] =
                tap + 1ull == plan->requirement.convolution_kernel
                    ? 0.72f : 0.035f * (float)(tap + 1ull);
    for (index = 0ull; index < plan->requirement.value_heads; ++index) {
        host->decay_log[index] = -2.1f + 0.025f * (float)(index % 7ull);
        host->time_bias[index] = -0.8f + 0.04f * (float)(index % 5ull);
    }
    for (index = 0ull; index < plan->requirement.value_head_dimension; ++index)
        host->normalization_weight[index] = 0.9f + 0.002f * (float)(index % 17ull);
    if (host->committed_convolution)
        for (index = 0ull; index < plan->convolution_state_values; ++index)
            host->committed_convolution[index] =
                (float)((int)(index % 13ull) - 6) * 0.003f;
    if (host->committed_recurrent)
        for (index = 0ull; index < plan->recurrent_state_values; ++index)
            host->committed_recurrent[index] =
                (float)((int)(index % 17ull) - 8) * 0.0005f;
}

static int gated_delta_tensor_open(yvex_backend *backend, const char *name,
                                   unsigned long long count, const float *source,
                                   yvex_device_tensor **tensor, yvex_error *err)
{
    yvex_backend_tensor_desc descriptor = {0};
    int rc;
    descriptor.name = name;
    descriptor.dtype = YVEX_DTYPE_F32;
    descriptor.rank = 1u;
    descriptor.dims[0] = count;
    descriptor.bytes = count * sizeof(float);
    rc = yvex_backend_tensor_alloc(backend, &descriptor, tensor, err);
    if (rc == YVEX_OK && source)
        rc = yvex_backend_tensor_write(
            backend, *tensor, source, descriptor.bytes, err);
    return rc;
}

static int gated_delta_device_open(
    yvex_backend *backend, const yvex_gated_delta_plan *plan,
    unsigned long long tokens, const gated_delta_host *host,
    gated_delta_device *device, yvex_error *err)
{
    unsigned long long output_values = tokens * plan->value_width;
    unsigned long long head_values = tokens * plan->requirement.value_heads;
    int rc = YVEX_OK;
    memset(device, 0, sizeof(*device));
#define OPEN(field, count, source) \
    do { \
        if (rc == YVEX_OK) rc = gated_delta_tensor_open( \
            backend, "gated-delta-" #field, (count), (source), &device->field, err); \
    } while (0)
    OPEN(qkv, tokens * plan->qkv_width, host->qkv);
    OPEN(output_gate, output_values, host->output_gate);
    OPEN(beta, head_values, host->beta);
    OPEN(decay, head_values, host->decay);
    OPEN(convolution_weight,
         plan->qkv_width * plan->requirement.convolution_kernel,
         host->convolution_weight);
    OPEN(decay_log, plan->requirement.value_heads, host->decay_log);
    OPEN(time_bias, plan->requirement.value_heads, host->time_bias);
    OPEN(normalization_weight, plan->requirement.value_head_dimension,
         host->normalization_weight);
    if (host->committed_convolution) {
        OPEN(committed_convolution, plan->convolution_state_values,
             host->committed_convolution);
        OPEN(committed_recurrent, plan->recurrent_state_values,
             host->committed_recurrent);
    }
    OPEN(candidate_convolution, plan->convolution_state_values, NULL);
    OPEN(candidate_recurrent, plan->recurrent_state_values, NULL);
    OPEN(output, output_values, NULL);
#undef OPEN
    return rc;
}

static int gated_delta_device_close(yvex_backend *backend,
                                    gated_delta_device *device, yvex_error *err)
{
    yvex_device_tensor **tensors[] = {
        &device->qkv, &device->output_gate, &device->beta, &device->decay,
        &device->convolution_weight, &device->decay_log, &device->time_bias,
        &device->normalization_weight, &device->committed_convolution,
        &device->committed_recurrent, &device->candidate_convolution,
        &device->candidate_recurrent, &device->output};
    size_t index;
    int rc = YVEX_OK;
    for (index = sizeof(tensors) / sizeof(tensors[0]); index > 0u; --index) {
        int release_rc = *tensors[index - 1u]
            ? yvex_backend_tensor_release(backend, tensors[index - 1u], err)
            : YVEX_OK;
        if (rc == YVEX_OK && release_rc != YVEX_OK) rc = release_rc;
    }
    return rc;
}

static double gated_delta_compare(const float *expected, const float *actual,
                                  unsigned long long count, double tolerance)
{
    unsigned long long index;
    double maximum = 0.0;
    for (index = 0ull; index < count; ++index) {
        double difference = fabs((double)expected[index] - (double)actual[index]);
        double bound = tolerance * (1.0 + fabs((double)expected[index]));
        if (!isfinite(actual[index]) || difference > bound) return -1.0;
        if (difference > maximum) maximum = difference;
    }
    return maximum;
}

static int gated_delta_cancelled(void *context)
{
    return context != NULL;
}

static int gated_delta_parity_case(
    yvex_backend *backend, const yvex_gated_delta_requirement *requirement,
    unsigned long long tokens, int committed, unsigned long long expected_chunks,
    double tolerance, double *maximum_difference)
{
    const yvex_backend_transformer_operations *operations =
        yvex_backend_transformer_operations_get(backend);
    yvex_gated_delta_plan plan;
    gated_delta_host host;
    gated_delta_device device;
    yvex_gated_delta_cpu_request cpu_request = {0};
    yvex_gated_delta_cpu_result cpu_result;
    yvex_gated_delta_device_request request = {0};
    yvex_gated_delta_device_result result;
    yvex_backend_operation_facts facts;
    yvex_error err;
    unsigned long long workspace = 0ull;
    double difference;
    int rc;
    YVEX_TEST_ASSERT(
        operations && operations->gated_delta_workspace_required &&
            operations->gated_delta_execute,
        "CUDA exposes admitted gated-delta transformer operations");
    YVEX_TEST_ASSERT(
        yvex_gated_delta_plan_seal(&plan, requirement, &err) == YVEX_OK &&
            gated_delta_host_open(&plan, tokens, committed, &host),
        "seal CUDA gated-delta plan and allocate host fixture");
    gated_delta_seed(&plan, tokens, &host);
    cpu_request.token_count = tokens;
    cpu_request.projected_qkv = host.qkv;
    cpu_request.projected_qkv_capacity = tokens * plan.qkv_width;
    cpu_request.projected_output_gate = host.output_gate;
    cpu_request.projected_output_gate_capacity = tokens * plan.value_width;
    cpu_request.projected_beta = host.beta;
    cpu_request.projected_beta_capacity = tokens * requirement->value_heads;
    cpu_request.projected_decay = host.decay;
    cpu_request.projected_decay_capacity = tokens * requirement->value_heads;
    cpu_request.convolution_weight = host.convolution_weight;
    cpu_request.convolution_weight_capacity =
        plan.qkv_width * requirement->convolution_kernel;
    cpu_request.decay_log = host.decay_log;
    cpu_request.decay_log_capacity = requirement->value_heads;
    cpu_request.time_bias = host.time_bias;
    cpu_request.time_bias_capacity = requirement->value_heads;
    cpu_request.normalization_weight = host.normalization_weight;
    cpu_request.normalization_weight_capacity = requirement->value_head_dimension;
    cpu_request.state.convolution = host.committed_convolution;
    cpu_request.state.recurrent = host.committed_recurrent;
    cpu_request.next_state.convolution = host.candidate_convolution;
    cpu_request.next_state.convolution_capacity = plan.convolution_state_values;
    cpu_request.next_state.recurrent = host.candidate_recurrent;
    cpu_request.next_state.recurrent_capacity = plan.recurrent_state_values;
    cpu_request.output = host.output;
    cpu_request.output_capacity = tokens * plan.value_width;
    YVEX_TEST_ASSERT(
        yvex_gated_delta_execute_cpu(
            &plan, &cpu_request, &cpu_result, &err) == YVEX_OK &&
            cpu_result.complete,
        "portable gated-delta semantic authority executes fixture");
    YVEX_TEST_ASSERT(
        gated_delta_device_open(
            backend, &plan, tokens, &host, &device, &err) == YVEX_OK,
        "allocate complete CUDA gated-delta fixture");
    request.token_count = tokens;
    request.projected_qkv = device.qkv;
    request.projected_output_gate = device.output_gate;
    request.projected_beta = device.beta;
    request.projected_decay = device.decay;
    request.convolution_weight = device.convolution_weight;
    request.decay_log = device.decay_log;
    request.time_bias = device.time_bias;
    request.normalization_weight = device.normalization_weight;
    request.convolution_state = device.committed_convolution;
    request.recurrent_state = device.committed_recurrent;
    request.next_convolution_state = device.candidate_convolution;
    request.next_recurrent_state = device.candidate_recurrent;
    request.output = device.output;
    YVEX_TEST_ASSERT(
        operations->gated_delta_workspace_required(
            &plan, tokens, &workspace, &err) == YVEX_OK && workspace > sizeof(int),
        "derive bounded chunked CUDA gated-delta workspace");
    rc = operations->gated_delta_execute(
        backend, &plan, &request, &result, &facts, &err);
    if (rc != YVEX_OK || !result.complete || result.token_count != tokens ||
        result.execution_chunks != expected_chunks ||
        facts.kernel_launches != expected_chunks * 2ull ||
        facts.download_count != expected_chunks ||
        facts.queue_synchronizations + facts.device_synchronizations != expected_chunks ||
        facts.temporary_bytes > workspace ||
        facts.temporary_bytes <
            (tokens < 64ull ? tokens : 64ull) * plan.qkv_width *
                sizeof(float) + sizeof(int) ||
        facts.state_bytes !=
            (plan.convolution_state_bytes + plan.recurrent_state_bytes) *
                (committed ? 2ull : 1ull))
        fprintf(stderr,
                "gated-delta facts rc=%d where=%s error=%s complete=%d "
                "tokens=%llu chunks=%llu kernels=%llu downloads=%llu sync=%llu "
                "temporary=%llu/%llu state=%llu/%llu\n",
                rc, yvex_error_where(&err), yvex_error_message(&err), result.complete,
                result.token_count, result.execution_chunks, facts.kernel_launches,
                facts.download_count,
                facts.queue_synchronizations + facts.device_synchronizations,
                facts.temporary_bytes, workspace, facts.state_bytes,
                (plan.convolution_state_bytes + plan.recurrent_state_bytes) *
                    (committed ? 2ull : 1ull));
    YVEX_TEST_ASSERT(
        rc == YVEX_OK && result.complete && result.token_count == tokens &&
            result.execution_chunks == expected_chunks &&
            facts.kernel_launches == expected_chunks * 2ull &&
            facts.download_count == expected_chunks &&
            facts.queue_synchronizations + facts.device_synchronizations ==
                expected_chunks &&
            facts.temporary_bytes <= workspace &&
            facts.temporary_bytes >=
                (tokens < 64ull ? tokens : 64ull) * plan.qkv_width *
                    sizeof(float) + sizeof(int) &&
            facts.state_bytes ==
                (plan.convolution_state_bytes + plan.recurrent_state_bytes) *
                    (committed ? 2ull : 1ull),
        "CUDA gated-delta publishes exact chunk, state, and operation facts");
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_read(
            backend, device.candidate_convolution, host.observed_convolution,
            plan.convolution_state_bytes, &err) == YVEX_OK &&
            yvex_backend_tensor_read(
                backend, device.candidate_recurrent, host.observed_recurrent,
                plan.recurrent_state_bytes, &err) == YVEX_OK &&
            yvex_backend_tensor_read(
                backend, device.output, host.observed_output,
                tokens * plan.value_width * sizeof(float), &err) == YVEX_OK,
        "read CUDA gated-delta candidate state and output");
    difference = gated_delta_compare(
        host.candidate_convolution, host.observed_convolution,
        plan.convolution_state_values, tolerance);
    YVEX_TEST_ASSERT(difference >= 0.0,
                     "CUDA causal convolution state matches portable authority");
    if (difference > *maximum_difference) *maximum_difference = difference;
    difference = gated_delta_compare(
        host.candidate_recurrent, host.observed_recurrent,
        plan.recurrent_state_values, tolerance);
    YVEX_TEST_ASSERT(difference >= 0.0,
                     "CUDA recurrent matrix state matches portable authority");
    if (difference > *maximum_difference) *maximum_difference = difference;
    difference = gated_delta_compare(
        host.output, host.observed_output, tokens * plan.value_width, tolerance);
    YVEX_TEST_ASSERT(difference >= 0.0,
                     "CUDA gated-delta output matches portable authority");
    if (difference > *maximum_difference) *maximum_difference = difference;
    if (committed) {
        memset(host.observed_convolution, 0, (size_t)plan.convolution_state_bytes);
        memset(host.observed_recurrent, 0, (size_t)plan.recurrent_state_bytes);
        request.cancel_requested = gated_delta_cancelled;
        request.cancel_context = &request;
        YVEX_TEST_ASSERT(
            operations->gated_delta_execute(
                backend, &plan, &request, &result, &facts, &err) ==
                YVEX_ERR_CANCELLED && result.cancelled &&
                !yvex_device_tensor_is_written(device.candidate_convolution) &&
                !yvex_device_tensor_is_written(device.candidate_recurrent) &&
                yvex_backend_tensor_read(
                    backend, device.committed_convolution,
                    host.observed_convolution, plan.convolution_state_bytes,
                    &err) == YVEX_OK &&
                yvex_backend_tensor_read(
                    backend, device.committed_recurrent,
                    host.observed_recurrent, plan.recurrent_state_bytes,
                    &err) == YVEX_OK &&
                memcmp(host.committed_convolution, host.observed_convolution,
                       (size_t)plan.convolution_state_bytes) == 0 &&
                memcmp(host.committed_recurrent, host.observed_recurrent,
                       (size_t)plan.recurrent_state_bytes) == 0,
            "pre-launch cancellation leaves committed mixed sequence state unchanged");
        request.cancel_requested = NULL;
        request.cancel_context = NULL;
        host.qkv[0] = NAN;
        YVEX_TEST_ASSERT(
            yvex_backend_tensor_write(
                backend, device.qkv, host.qkv,
                tokens * plan.qkv_width * sizeof(float), &err) == YVEX_OK &&
                operations->gated_delta_execute(
                    backend, &plan, &request, &result, &facts, &err) ==
                    YVEX_ERR_FORMAT && !result.complete &&
                !yvex_device_tensor_is_written(device.candidate_convolution) &&
                !yvex_device_tensor_is_written(device.candidate_recurrent) &&
                yvex_backend_tensor_read(
                    backend, device.committed_convolution,
                    host.observed_convolution, plan.convolution_state_bytes,
                    &err) == YVEX_OK &&
                yvex_backend_tensor_read(
                    backend, device.committed_recurrent,
                    host.observed_recurrent, plan.recurrent_state_bytes,
                    &err) == YVEX_OK &&
                memcmp(host.committed_convolution, host.observed_convolution,
                       (size_t)plan.convolution_state_bytes) == 0 &&
                memcmp(host.committed_recurrent, host.observed_recurrent,
                       (size_t)plan.recurrent_state_bytes) == 0,
            "non-finite CUDA recurrence refuses unpublished candidate state");
    }
    YVEX_TEST_ASSERT(
        gated_delta_device_close(backend, &device, &err) == YVEX_OK,
        "release CUDA gated-delta fixture tensors");
    gated_delta_host_close(&host);
    return 0;
}

int yvex_cuda_test_sequence_mixer(void)
{
    const yvex_gated_delta_requirement small = {
        .schema_version = YVEX_SEQUENCE_MIXER_GATED_DELTA_SCHEMA_V2,
        .output_normalization_weight_convention =
            YVEX_NORMALIZATION_WEIGHT_ONE_PLUS,
        .query_heads = 2ull, .key_heads = 2ull, .value_heads = 4ull,
        .key_head_dimension = 4ull, .value_head_dimension = 3ull,
        .convolution_kernel = 3ull,
        .projected_dtype = YVEX_DTYPE_F32,
        .convolution_state_dtype = YVEX_DTYPE_F32,
        .recurrent_state_dtype = YVEX_DTYPE_F32,
        .accumulation_dtype = YVEX_DTYPE_F32,
        .output_dtype = YVEX_DTYPE_F32,
        .numeric_contract = YVEX_SEQUENCE_MIXER_NUMERIC_F32_RECURRENCE,
        .qk_normalization_epsilon = 1e-6,
        .output_normalization_epsilon = 1e-6,
        .query_scale = 0.5,
        .deterministic = 1};
    const yvex_gated_delta_requirement wide = {
        .schema_version = YVEX_SEQUENCE_MIXER_GATED_DELTA_SCHEMA_V2,
        .output_normalization_weight_convention =
            YVEX_NORMALIZATION_WEIGHT_DIRECT,
        .query_heads = 16ull, .key_heads = 16ull, .value_heads = 48ull,
        .key_head_dimension = 128ull, .value_head_dimension = 128ull,
        .convolution_kernel = 4ull,
        .projected_dtype = YVEX_DTYPE_F32,
        .convolution_state_dtype = YVEX_DTYPE_F32,
        .recurrent_state_dtype = YVEX_DTYPE_F32,
        .accumulation_dtype = YVEX_DTYPE_F32,
        .output_dtype = YVEX_DTYPE_F32,
        .numeric_contract = YVEX_SEQUENCE_MIXER_NUMERIC_F32_RECURRENCE,
        .qk_normalization_epsilon = 1e-6,
        .output_normalization_epsilon = 1e-6,
        .query_scale = 0.08838834764831845,
        .deterministic = 1};
    yvex_backend *backend = NULL;
    yvex_backend_options options = {0};
    yvex_error err;
    double maximum_difference = 0.0;
    int rc;
    options.kind = YVEX_BACKEND_KIND_CUDA;
    rc = yvex_backend_open(&backend, &options, &err);
    if (rc == YVEX_ERR_UNSUPPORTED) {
        fprintf(stderr, "SKIP: CUDA unavailable: %s\n", yvex_error_message(&err));
        return 77;
    }
    YVEX_TEST_ASSERT(rc == YVEX_OK, "open CUDA gated-delta backend");
    YVEX_TEST_ASSERT(
        gated_delta_parity_case(
            backend, &small, 65ull, 0, 2ull, 6e-5, &maximum_difference) == 0,
        "chunked causal-convolution and recurrence prefill match CPU authority");
    YVEX_TEST_ASSERT(
        gated_delta_parity_case(
            backend, &small, 1ull, 1, 1ull, 6e-5, &maximum_difference) == 0,
        "cached single-token recurrence consumes immutable committed state");
    YVEX_TEST_ASSERT(
        gated_delta_parity_case(
            backend, &wide, 1ull, 0, 1ull, 8e-5, &maximum_difference) == 0,
        "16/16/48 by 128 production geometry matches portable authority");
    fprintf(stderr, "CUDA gated-delta maximum absolute difference: %.9g\n",
            maximum_difference);
    YVEX_TEST_ASSERT(
        yvex_backend_close_checked(&backend, &err) == YVEX_OK && !backend,
        "close CUDA gated-delta backend cleanly");
    return 0;
}
