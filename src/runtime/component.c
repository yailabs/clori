/* Share checked component buffers and bindings without importing family policy. */
#include <yvex/internal/component.h>

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <yvex/internal/core.h>
#include <yvex/internal/convolution.h>
#include <yvex/internal/runtime.h>

int yvex_component_buffer_open(
    yvex_component_f32_buffer *buffer, unsigned long long count,
    unsigned long long maximum, unsigned long long *live,
    unsigned long long *peak, const char *stage, const char *label, yvex_error *err)
{
    unsigned long long bytes, next;
    if (buffer) memset(buffer, 0, sizeof(*buffer));
    if (!buffer || !live || !peak || !stage || !label || !count ||
        !yvex_core_u64_mul(count, sizeof(float), &bytes) ||
        bytes > (unsigned long long)SIZE_MAX ||
        !yvex_core_u64_add(*live, bytes, &next)) {
        yvex_error_setf(err, YVEX_ERR_BOUNDS, stage,
                        "%s workspace extent overflowed", label);
        return YVEX_ERR_BOUNDS;
    }
    if (next > maximum) {
        yvex_error_setf(err, YVEX_ERR_BOUNDS, stage,
                        "%s workspace budget was exceeded", label);
        return YVEX_ERR_BOUNDS;
    }
    buffer->data = (float *)malloc((size_t)bytes);
    if (!buffer->data) {
        yvex_error_setf(err, YVEX_ERR_NOMEM, stage,
                        "%s workspace allocation failed", label);
        return YVEX_ERR_NOMEM;
    }
    buffer->count = count;
    *live = next;
    if (next > *peak) *peak = next;
    return YVEX_OK;
}

void yvex_component_buffer_close(yvex_component_f32_buffer *buffer,
                                 unsigned long long *live)
{
    unsigned long long bytes;
    if (!buffer || !live) return;
    bytes = buffer->count * sizeof(float);
    if (bytes <= *live) *live -= bytes;
    free(buffer->data);
    memset(buffer, 0, sizeof(*buffer));
}

const yvex_materialized_tensor_binding *yvex_component_binding_find(
    const yvex_materialization_session *session, const char *name)
{
    unsigned long long index;
    if (!session || !name) return NULL;
    for (index = 0ull;; ++index) {
        const yvex_materialized_tensor_binding *binding =
            yvex_materialization_session_tensor_at(session, index);
        if (!binding || strcmp(binding->name, name) == 0) return binding;
    }
}

int yvex_component_weight_bind(
    const yvex_materialization_session *session,
    const yvex_runtime_residency *residency, const char *name,
    yvex_component_encoded_weight *weight, yvex_error *err)
{
    const yvex_materialized_tensor_binding *binding =
        yvex_component_binding_find(session, name);
    if (!binding || !binding->row_count || !residency || !weight) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "runtime.component.binding",
                       "an exact resident component weight binding is unavailable");
        return YVEX_ERR_FORMAT;
    }
    memset(weight, 0, sizeof(*weight));
    if (yvex_runtime_residency_binding_view(
            residency, binding, &weight->encoded, &weight->encoded_bytes, err) != YVEX_OK)
        return yvex_error_code(err);
    weight->qtype = binding->qtype;
    weight->row_count = binding->row_count;
    weight->row_width = binding->row_width;
    weight->row_bytes = binding->encoded_bytes / binding->row_count;
    return YVEX_OK;
}

int yvex_component_weight_bind_sized(
    void *context, const char *name, unsigned long long rows,
    unsigned long long width, yvex_component_encoded_weight *weight, yvex_error *err)
{
    const yvex_runtime_component_session *session =
        (const yvex_runtime_component_session *)context;
    unsigned long long values, expected_bytes;
    int rc;
    if (!session || !weight || !yvex_core_u64_mul(rows, width, &values) ||
        !yvex_core_u64_mul(values, sizeof(float), &expected_bytes)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.component.binding",
                       "component weight binding geometry is invalid");
        return YVEX_ERR_INVALID_ARG;
    }
    rc = yvex_component_weight_bind(
        yvex_runtime_component_session_materialization(session),
        yvex_runtime_component_session_residency(session), name, weight, err);
    if (rc == YVEX_OK && weight->encoded_bytes != expected_bytes) {
        yvex_error_setf(err, YVEX_ERR_FORMAT, "runtime.component.binding",
                        "component tensor %s has %llu bytes, expected %llu",
                        name, weight->encoded_bytes, expected_bytes);
        rc = YVEX_ERR_FORMAT;
    }
    return rc;
}

static int component_load_reject(
    yvex_component_load_failure *failure, yvex_component_load_code code,
    const char *name, unsigned long long expected, unsigned long long actual,
    const char *reason, yvex_status status, const char *stage, yvex_error *err)
{
    failure->code = code;
    failure->expected = expected;
    failure->actual = actual;
    failure->reason = reason;
    yvex_core_text_copy(failure->tensor_name, sizeof(failure->tensor_name), name);
    yvex_error_setf(err, status, stage, "component tensor contract rejected %s", name);
    return status;
}

int yvex_component_f32_load(
    yvex_materialization_session *session, const char *name, unsigned int rank,
    const unsigned long long *dims, yvex_component_f32_buffer *buffer,
    unsigned long long maximum, unsigned long long *live, unsigned long long *peak,
    unsigned long long *reads, unsigned long long *payload,
    yvex_component_load_failure *failure, const char *stage, const char *label,
    yvex_error *err)
{
    const yvex_materialized_tensor_binding *binding;
    yvex_materialization_failure materialization_failure;
    unsigned long long count = 1ull, expected_bytes = 0ull;
    unsigned int dimension;
    int rc;
    if (!session || !name || !dims || !buffer || !live || !peak || !reads ||
        !payload || !failure || !stage || !label) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "runtime.component.load",
                       "complete bounded component load state is required");
        return YVEX_ERR_INVALID_ARG;
    }
    memset(failure, 0, sizeof(*failure));
    binding = yvex_component_binding_find(session, name);
    if (!binding)
        return component_load_reject(failure, YVEX_COMPONENT_LOAD_MISSING, name,
                                     1ull, 0ull, "component tensor is missing",
                                     YVEX_ERR_FORMAT, stage, err);
    if (binding->qtype != YVEX_GGUF_QTYPE_F32 || binding->rank != rank)
        return component_load_reject(failure, YVEX_COMPONENT_LOAD_CONTRACT, name,
                                     rank, binding->rank,
                                     "component tensor rank or qtype differs",
                                     YVEX_ERR_FORMAT, stage, err);
    for (dimension = 0u; dimension < rank; ++dimension)
        if (binding->dims[dimension] != dims[dimension] ||
            !yvex_core_u64_mul(count, dims[dimension], &count))
            return component_load_reject(failure, YVEX_COMPONENT_LOAD_CONTRACT, name,
                                         dims[dimension], binding->dims[dimension],
                                         "component tensor shape differs",
                                         YVEX_ERR_FORMAT, stage, err);
    if (!yvex_core_u64_mul(count, sizeof(float), &expected_bytes) ||
        binding->encoded_bytes != expected_bytes)
        return component_load_reject(failure, YVEX_COMPONENT_LOAD_CONTRACT, name,
                                     expected_bytes, binding->encoded_bytes,
                                     "component tensor byte extent differs",
                                     YVEX_ERR_FORMAT, stage, err);
    rc = yvex_component_buffer_open(
        buffer, count, maximum, live, peak, stage, label, err);
    if (rc != YVEX_OK) {
        failure->code = YVEX_COMPONENT_LOAD_BUDGET;
        failure->reason = yvex_error_message(err);
        return rc;
    }
    rc = yvex_materialization_session_read(
        session, binding, 0ull, buffer->data, (size_t)binding->encoded_bytes,
        &materialization_failure, err);
    if (rc != YVEX_OK) {
        yvex_component_buffer_close(buffer, live);
        failure->code = YVEX_COMPONENT_LOAD_MATERIALIZATION;
        failure->expected = binding->encoded_bytes;
        failure->actual = materialization_failure.actual;
        failure->reason = materialization_failure.reason;
        yvex_core_text_copy(failure->tensor_name, sizeof(failure->tensor_name), name);
        return rc;
    }
    (*reads)++;
    *payload += binding->encoded_bytes;
    return YVEX_OK;
}

typedef struct {
    yvex_materialization_session *session;
    const yvex_alias_decoder_request *request;
    yvex_alias_decoder_result *result;
    yvex_component_execution_failure *failure;
    unsigned long long live_bytes;
    yvex_error *err;
} alias_cpu_run;

static int alias_cpu_refuse(alias_cpu_run *run, yvex_component_execution_code code,
                            const char *name, unsigned long long expected,
                            unsigned long long actual, yvex_status status,
                            const char *reason)
{
    if (run && run->failure) {
        run->failure->code = code;
        run->failure->expected = expected;
        run->failure->actual = actual;
        run->failure->reason = reason;
        yvex_core_text_copy(run->failure->tensor_name,
                            sizeof(run->failure->tensor_name), name);
    }
    yvex_error_set(run ? run->err : NULL, status, "graph.alias-decoder.cpu", reason);
    return status;
}

static int alias_cpu_cancel(alias_cpu_run *run)
{
    if (run->request->cancel_requested &&
        run->request->cancel_requested(run->request->cancel_context))
        return alias_cpu_refuse(run, YVEX_COMPONENT_EXECUTION_CANCELLED, NULL, 0ull, 1ull,
                                YVEX_ERR_CANCELLED,
                                "alias decoder was cancelled between layers");
    return YVEX_OK;
}

static int alias_cpu_buffer_open(alias_cpu_run *run, yvex_component_f32_buffer *buffer,
                                 unsigned long long count)
{
    int rc = yvex_component_buffer_open(
        buffer, count, run->request->maximum_workspace_bytes, &run->live_bytes,
        &run->result->peak_host_bytes, "graph.alias-decoder.cpu", "alias decoder", run->err);
    return rc == YVEX_OK
               ? rc
               : alias_cpu_refuse(run, YVEX_COMPONENT_EXECUTION_BUDGET, NULL,
                                  run->request->maximum_workspace_bytes, count,
                                  (yvex_status)rc, yvex_error_message(run->err));
}

static void alias_cpu_buffer_close(alias_cpu_run *run, yvex_component_f32_buffer *buffer)
{
    yvex_component_buffer_close(buffer, &run->live_bytes);
}

static int alias_cpu_name(alias_cpu_run *run, yvex_alias_decoder_weight_role role,
                          unsigned long long stage, unsigned long long block,
                          unsigned long long layer, char output[256])
{
    return run->request->weight_name(
        run->request->weight_name_context, role, stage, block, layer, output, run->err);
}

static int alias_cpu_load(alias_cpu_run *run, yvex_alias_decoder_weight_role role,
                          unsigned long long stage, unsigned long long block,
                          unsigned long long layer, unsigned int rank,
                          const unsigned long long *dims, yvex_component_f32_buffer *buffer)
{
    yvex_component_load_failure issue = {0};
    char name[256] = {0};
    int rc = alias_cpu_name(run, role, stage, block, layer, name);
    if (rc == YVEX_OK)
        rc = yvex_component_f32_load(
            run->session, name, rank, dims, buffer, run->request->maximum_workspace_bytes,
            &run->live_bytes, &run->result->peak_host_bytes, &run->result->tensor_reads,
            &run->result->payload_bytes_read, &issue, "graph.alias-decoder.cpu",
            "alias decoder", run->err);
    if (rc != YVEX_OK && run->failure && !run->failure->code) {
        run->failure->code = issue.code ? issue.code + 2u
                                        : YVEX_COMPONENT_EXECUTION_TENSOR_CONTRACT;
        run->failure->expected = issue.expected;
        run->failure->actual = issue.actual;
        run->failure->reason = issue.reason ? issue.reason : yvex_error_message(run->err);
        yvex_core_text_copy(run->failure->tensor_name,
                            sizeof(run->failure->tensor_name), name);
    }
    return rc;
}

static int alias_cpu_convolution(
    alias_cpu_run *run, yvex_alias_decoder_weight_role weight_role,
    yvex_alias_decoder_weight_role gain_role, yvex_alias_decoder_weight_role bias_role,
    int normalized, int biased, unsigned long long stage, unsigned long long block,
    unsigned long long layer, const yvex_graph_conv1d_geometry *geometry,
    const float *input, unsigned long long input_count,
    float *output, unsigned long long output_count)
{
    yvex_component_f32_buffer weight = {0}, gain = {0}, bias = {0};
    unsigned long long weight_dims[3], gain_dims[3], bias_dims[1];
    int rc = alias_cpu_cancel(run);
    weight_dims[0] = geometry->transposed ? geometry->input_channels
                                          : geometry->output_channels;
    weight_dims[1] = geometry->transposed ? geometry->output_channels
                                          : geometry->input_channels;
    weight_dims[2] = geometry->kernel_size;
    if (rc == YVEX_OK)
        rc = alias_cpu_load(run, weight_role, stage, block, layer, 3u,
                            weight_dims, &weight);
    if (normalized && rc == YVEX_OK) {
        gain_dims[0] = geometry->transposed ? geometry->input_channels
                                            : geometry->output_channels;
        gain_dims[1] = gain_dims[2] = 1ull;
        rc = alias_cpu_load(run, gain_role, stage, block, layer, 3u, gain_dims, &gain);
    }
    if (biased && rc == YVEX_OK) {
        bias_dims[0] = geometry->output_channels;
        rc = alias_cpu_load(run, bias_role, stage, block, layer, 1u, bias_dims, &bias);
    }
    if (rc == YVEX_OK)
        rc = yvex_graph_conv1d_f32(
            geometry, input, input_count, weight.data, weight.count,
            bias.data, bias.count, gain.data, gain.count, output, output_count, run->err);
    if (rc != YVEX_OK && run->failure && !run->failure->code) {
        run->failure->code = YVEX_COMPONENT_EXECUTION_NUMERIC;
        run->failure->reason = yvex_error_message(run->err);
    }
    alias_cpu_buffer_close(run, &bias);
    alias_cpu_buffer_close(run, &gain);
    alias_cpu_buffer_close(run, &weight);
    return rc;
}

static int alias_cpu_activation(
    alias_cpu_run *run, yvex_alias_decoder_weight_role alpha_role,
    yvex_alias_decoder_weight_role beta_role, yvex_alias_decoder_weight_role up_role,
    yvex_alias_decoder_weight_role down_role, unsigned long long stage,
    unsigned long long block, unsigned long long layer, const float *input,
    unsigned long long batch, unsigned long long channels, unsigned long long length,
    float *output, float *scratch, unsigned long long scratch_count)
{
    yvex_component_f32_buffer alpha = {0}, beta = {0}, up = {0}, down = {0};
    unsigned long long channel_dims[1] = {channels};
    unsigned long long filter_dims[3] = {1ull, 1ull, 12ull};
    int rc = alias_cpu_cancel(run);
    if (rc == YVEX_OK)
        rc = alias_cpu_load(run, alpha_role, stage, block, layer, 1u,
                            channel_dims, &alpha);
    if (rc == YVEX_OK)
        rc = alias_cpu_load(run, beta_role, stage, block, layer, 1u,
                            channel_dims, &beta);
    if (rc == YVEX_OK)
        rc = alias_cpu_load(run, up_role, stage, block, layer, 3u, filter_dims, &up);
    if (rc == YVEX_OK)
        rc = alias_cpu_load(run, down_role, stage, block, layer, 3u, filter_dims, &down);
    if (rc == YVEX_OK)
        rc = yvex_graph_alias_snake_f32(
            input, batch, channels, length, alpha.data, beta.data, up.data, down.data,
            output, scratch, scratch_count, run->err);
    if (rc != YVEX_OK && run->failure && !run->failure->code) {
        run->failure->code = YVEX_COMPONENT_EXECUTION_NUMERIC;
        run->failure->reason = yvex_error_message(run->err);
    }
    alias_cpu_buffer_close(run, &down);
    alias_cpu_buffer_close(run, &up);
    alias_cpu_buffer_close(run, &beta);
    alias_cpu_buffer_close(run, &alpha);
    return rc;
}

static int alias_cpu_residual(
    alias_cpu_run *run, unsigned long long stage, unsigned long long block,
    const float *input, unsigned long long batch, unsigned long long channels,
    unsigned long long length, float *output)
{
    const yvex_alias_decoder_recipe *recipe = run->request->recipe;
    yvex_component_f32_buffer activation = {0}, convolution = {0}, scratch = {0};
    yvex_graph_conv1d_geometry geometry = {0};
    unsigned long long values, layer, index;
    int rc;
    if (!yvex_core_u64_mul(batch, channels, &values) ||
        !yvex_core_u64_mul(values, length, &values))
        return alias_cpu_refuse(run, YVEX_COMPONENT_EXECUTION_BUDGET, NULL, 1ull, 0ull,
                                YVEX_ERR_BOUNDS, "alias residual extent overflowed");
    rc = alias_cpu_buffer_open(run, &activation, values);
    if (rc == YVEX_OK) rc = alias_cpu_buffer_open(run, &convolution, values);
    if (rc == YVEX_OK) rc = alias_cpu_buffer_open(run, &scratch, length * 2ull);
    if (rc != YVEX_OK) goto cleanup;
    memcpy(output, input, (size_t)values * sizeof(float));
    geometry = (yvex_graph_conv1d_geometry){
        batch, channels, channels, length, recipe->residual_kernels[block],
        1ull, 1ull, 0ull, 0ull, 0};
    for (layer = 0ull; layer < recipe->residual_layers && rc == YVEX_OK; ++layer) {
        rc = alias_cpu_activation(
            run, YVEX_ALIAS_DECODER_ACT_ALPHA, YVEX_ALIAS_DECODER_ACT_BETA,
            YVEX_ALIAS_DECODER_ACT_UP_FILTER, YVEX_ALIAS_DECODER_ACT_DOWN_FILTER,
            stage, block, layer * 2ull, output, batch, channels, length,
            activation.data, scratch.data, scratch.count);
        geometry.dilation = recipe->residual_dilations[layer];
        geometry.padding = ((geometry.kernel_size - 1ull) * geometry.dilation) / 2ull;
        if (rc == YVEX_OK)
            rc = alias_cpu_convolution(
                run, YVEX_ALIAS_DECODER_RES1_WEIGHT, YVEX_ALIAS_DECODER_RES1_GAIN,
                YVEX_ALIAS_DECODER_RES1_BIAS, 1, 1, stage, block, layer, &geometry,
                activation.data, activation.count, convolution.data, convolution.count);
        if (rc == YVEX_OK)
            rc = alias_cpu_activation(
                run, YVEX_ALIAS_DECODER_ACT_ALPHA, YVEX_ALIAS_DECODER_ACT_BETA,
                YVEX_ALIAS_DECODER_ACT_UP_FILTER, YVEX_ALIAS_DECODER_ACT_DOWN_FILTER,
                stage, block, layer * 2ull + 1ull, convolution.data, batch, channels,
                length, activation.data, scratch.data, scratch.count);
        geometry.dilation = 1ull;
        geometry.padding = (geometry.kernel_size - 1ull) / 2ull;
        if (rc == YVEX_OK)
            rc = alias_cpu_convolution(
                run, YVEX_ALIAS_DECODER_RES2_WEIGHT, YVEX_ALIAS_DECODER_RES2_GAIN,
                YVEX_ALIAS_DECODER_RES2_BIAS, 1, 1, stage, block, layer, &geometry,
                activation.data, activation.count, convolution.data, convolution.count);
        if (rc == YVEX_OK)
            for (index = 0ull; index < values; ++index) output[index] += convolution.data[index];
    }
cleanup:
    alias_cpu_buffer_close(run, &scratch);
    alias_cpu_buffer_close(run, &convolution);
    alias_cpu_buffer_close(run, &activation);
    return rc;
}

static int alias_cpu_stage(alias_cpu_run *run, unsigned long long stage,
                           const yvex_component_f32_buffer *input,
                           unsigned long long batch, unsigned long long input_channels,
                           unsigned long long input_length,
                           yvex_component_f32_buffer *output,
                           unsigned long long *output_channels,
                           unsigned long long *output_length)
{
    const yvex_alias_decoder_recipe *recipe = run->request->recipe;
    yvex_component_f32_buffer upsampled = {0}, sum = {0}, block = {0};
    yvex_graph_conv1d_geometry geometry = {0};
    unsigned long long values, residual, index;
    int rc;
    *output_channels = input_channels / 2ull;
    geometry = (yvex_graph_conv1d_geometry){
        batch, input_channels, *output_channels, input_length,
        recipe->upsample_kernels[stage], recipe->rates[stage], 1ull,
        (recipe->upsample_kernels[stage] - recipe->rates[stage]) / 2ull, 0ull, 1};
    rc = yvex_graph_conv1d_output_length(&geometry, output_length, run->err);
    if (rc != YVEX_OK || !yvex_core_u64_mul(batch, *output_channels, &values) ||
        !yvex_core_u64_mul(values, *output_length, &values))
        return alias_cpu_refuse(run, YVEX_COMPONENT_EXECUTION_BUDGET, NULL, 1ull, 0ull,
                                YVEX_ERR_BOUNDS, "alias decoder stage extent overflowed");
    rc = alias_cpu_buffer_open(run, &upsampled, values);
    if (rc == YVEX_OK) rc = alias_cpu_buffer_open(run, &sum, values);
    if (rc == YVEX_OK) rc = alias_cpu_buffer_open(run, &block, values);
    if (rc == YVEX_OK)
        rc = alias_cpu_convolution(
            run, YVEX_ALIAS_DECODER_UP_WEIGHT, YVEX_ALIAS_DECODER_UP_GAIN,
            YVEX_ALIAS_DECODER_UP_BIAS, 1, 1, stage, 0ull, 0ull, &geometry,
            input->data, input->count, upsampled.data, upsampled.count);
    if (rc == YVEX_OK) memset(sum.data, 0, (size_t)sum.count * sizeof(float));
    for (residual = 0ull; residual < recipe->residual_blocks && rc == YVEX_OK; ++residual) {
        rc = alias_cpu_residual(run, stage, residual, upsampled.data, batch,
                                *output_channels, *output_length, block.data);
        if (rc == YVEX_OK)
            for (index = 0ull; index < values; ++index) sum.data[index] += block.data[index];
    }
    if (rc == YVEX_OK) {
        for (index = 0ull; index < values; ++index)
            sum.data[index] /= (float)recipe->residual_blocks;
        *output = sum;
        memset(&sum, 0, sizeof(sum));
    }
    alias_cpu_buffer_close(run, &block);
    alias_cpu_buffer_close(run, &sum);
    alias_cpu_buffer_close(run, &upsampled);
    return rc;
}

static int alias_cpu_recipe_valid(const yvex_alias_decoder_request *request,
                                  unsigned long long *output_length)
{
    const yvex_alias_decoder_recipe *recipe = request ? request->recipe : NULL;
    unsigned long long channels, length, values, stage, index;
    if (!recipe || !request->input || !request->output || !request->batch ||
        !request->input_length || !request->maximum_workspace_bytes ||
        !recipe->input_channels || !recipe->projection_channels || !recipe->input_kernel ||
        !recipe->pre_channels || !recipe->pre_kernel || !recipe->stage_count ||
        recipe->stage_count > YVEX_ALIAS_DECODER_MAX_STAGES || !recipe->residual_blocks ||
        recipe->residual_blocks > YVEX_ALIAS_DECODER_MAX_RESBLOCKS ||
        !recipe->residual_layers || recipe->residual_layers > YVEX_ALIAS_DECODER_MAX_LAYERS ||
        !recipe->final_channels || !recipe->final_kernel || !request->weight_name ||
        !yvex_core_u64_mul(request->batch, recipe->input_channels, &values) ||
        !yvex_core_u64_mul(values, request->input_length, &values) ||
        values != request->input_count) return 0;
    channels = recipe->pre_channels;
    length = request->input_length;
    for (stage = 0ull; stage < recipe->stage_count; ++stage) {
        unsigned long long rate = recipe->rates[stage];
        unsigned long long kernel = recipe->upsample_kernels[stage];
        if (!rate || kernel < rate || ((kernel - rate) & 1ull) || channels < 2ull ||
            length > (ULLONG_MAX - kernel) / rate) return 0;
        length = (length - 1ull) * rate + kernel - (kernel - rate);
        channels /= 2ull;
    }
    for (index = 0ull; index < recipe->residual_blocks; ++index)
        if (!recipe->residual_kernels[index] || !(recipe->residual_kernels[index] & 1ull))
            return 0;
    for (index = 0ull; index < recipe->residual_layers; ++index)
        if (!recipe->residual_dilations[index]) return 0;
    if (!yvex_core_u64_mul(request->batch, recipe->final_channels, &values) ||
        !yvex_core_u64_mul(values, length, &values) || request->output_capacity < values)
        return 0;
    *output_length = length;
    return 1;
}

int yvex_runtime_alias_decoder_execute_cpu(
    yvex_materialization_session *session, const yvex_alias_decoder_request *request,
    yvex_alias_decoder_result *result, yvex_component_execution_failure *failure,
    yvex_error *err)
{
    alias_cpu_run run = {session, request, result, failure, 0ull, err};
    const yvex_alias_decoder_recipe *recipe;
    yvex_component_f32_buffer current = {0}, next = {0}, activated = {0}, scratch = {0};
    yvex_graph_conv1d_geometry geometry = {0};
    unsigned long long output_length = 0ull, channels, length, values, stage, index;
    int rc;
    if (result) memset(result, 0, sizeof(*result));
    if (failure) memset(failure, 0, sizeof(*failure));
    if (!session || !result || !alias_cpu_recipe_valid(request, &output_length))
        return alias_cpu_refuse(&run, YVEX_COMPONENT_EXECUTION_INVALID_ARGUMENT, NULL,
                                1ull, 0ull, YVEX_ERR_INVALID_ARG,
                                "one complete bounded alias decoder request is required");
    recipe = request->recipe;
    channels = recipe->projection_channels;
    length = request->input_length;
    rc = alias_cpu_buffer_open(&run, &current, request->batch * channels * length);
    geometry = (yvex_graph_conv1d_geometry){
        request->batch, recipe->input_channels, channels, length,
        recipe->input_kernel, 1ull, 1ull, 0ull, 0ull, 0};
    if (rc == YVEX_OK)
        rc = alias_cpu_convolution(
            &run, YVEX_ALIAS_DECODER_INPUT_WEIGHT, YVEX_ALIAS_DECODER_INPUT_WEIGHT,
            YVEX_ALIAS_DECODER_INPUT_BIAS, 0, 1, 0ull, 0ull, 0ull, &geometry,
            request->input, request->input_count, current.data, current.count);
    geometry.input_channels = channels;
    geometry.output_channels = recipe->pre_channels;
    geometry.kernel_size = recipe->pre_kernel;
    geometry.padding = (recipe->pre_kernel - 1ull) / 2ull;
    channels = recipe->pre_channels;
    if (rc == YVEX_OK)
        rc = alias_cpu_buffer_open(&run, &next, request->batch * channels * length);
    if (rc == YVEX_OK)
        rc = alias_cpu_convolution(
            &run, YVEX_ALIAS_DECODER_PRE_WEIGHT, YVEX_ALIAS_DECODER_PRE_GAIN,
            YVEX_ALIAS_DECODER_PRE_BIAS, 1, 1, 0ull, 0ull, 0ull, &geometry,
            current.data, current.count, next.data, next.count);
    alias_cpu_buffer_close(&run, &current);
    current = next;
    memset(&next, 0, sizeof(next));
    for (stage = 0ull; stage < recipe->stage_count && rc == YVEX_OK; ++stage) {
        unsigned long long next_channels = 0ull, next_length = 0ull;
        rc = alias_cpu_stage(&run, stage, &current, request->batch, channels, length,
                             &next, &next_channels, &next_length);
        if (rc == YVEX_OK) {
            alias_cpu_buffer_close(&run, &current);
            current = next;
            memset(&next, 0, sizeof(next));
            channels = next_channels;
            length = next_length;
        }
    }
    if (rc == YVEX_OK) rc = alias_cpu_buffer_open(&run, &activated, current.count);
    if (rc == YVEX_OK) rc = alias_cpu_buffer_open(&run, &scratch, length * 2ull);
    if (rc == YVEX_OK)
        rc = alias_cpu_activation(
            &run, YVEX_ALIAS_DECODER_POST_ACT_ALPHA, YVEX_ALIAS_DECODER_POST_ACT_BETA,
            YVEX_ALIAS_DECODER_POST_ACT_UP_FILTER, YVEX_ALIAS_DECODER_POST_ACT_DOWN_FILTER,
            0ull, 0ull, 0ull, current.data, request->batch, channels, length,
            activated.data, scratch.data, scratch.count);
    geometry = (yvex_graph_conv1d_geometry){
        request->batch, channels, recipe->final_channels, length, recipe->final_kernel,
        1ull, 1ull, (recipe->final_kernel - 1ull) / 2ull, 0ull, 0};
    if (rc == YVEX_OK)
        rc = alias_cpu_convolution(
            &run, YVEX_ALIAS_DECODER_POST_WEIGHT, YVEX_ALIAS_DECODER_POST_GAIN,
            YVEX_ALIAS_DECODER_POST_GAIN, 1, 0, 0ull, 0ull, 0ull, &geometry,
            activated.data, activated.count, request->output,
            request->batch * recipe->final_channels * output_length);
    values = request->batch * recipe->final_channels * output_length;
    if (rc == YVEX_OK)
        for (index = 0ull; index < values; ++index)
            request->output[index] = fmaxf(-1.0f, fminf(1.0f, request->output[index]));
    if (rc == YVEX_OK) {
        result->output_length = output_length;
        result->output_values = values;
        result->complete = 1;
        yvex_error_clear(err);
    }
    alias_cpu_buffer_close(&run, &scratch);
    alias_cpu_buffer_close(&run, &activated);
    alias_cpu_buffer_close(&run, &next);
    alias_cpu_buffer_close(&run, &current);
    return rc;
}
